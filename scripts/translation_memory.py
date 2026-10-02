"""A translation memory for the Kazakh batches.

The corpus is templated: the same English sentence recurs across conversations.
Translating message slots one at a time therefore does the same work repeatedly
and lets the wording drift, which is exactly what happened -- 66 source strings
picked up more than one Kazakh rendering across the first six batches.

A memory fixes both. Each distinct string is translated once and reused, so the
remaining work shrinks and drift becomes impossible by construction.

Keyed on (source string, style_variant), not on the string alone. Measured on the
finished batches, 51 of the 66 conflicts are between two renderings of the same
string in the *same* style, which is drift; the other 15 differ only across
styles, which may be a deliberate difference of register between, say,
`support_formal` and `discord_casual`. Keying on the pair removes the drift and
keeps the register.

Commands:

    build      read the finished batches into data/translation/memory.json,
               and report every string that has more than one rendering
    normalize  rewrite the finished batches so each string uses the canonical
               rendering the memory chose
    todo       write the strings of a pending batch that the memory cannot fill
    assemble   build batch_NN_kk.json from the memory plus a todo file that has
               been filled in
"""
import argparse
import collections
import json
import pathlib

import pandas as pd

MEMORY_NAME = "memory.json"
CONFLICTS_NAME = "memory_conflicts.json"


def batch_paths(translation_dir):
    for source in sorted(pathlib.Path(translation_dir).glob("batch_*_source.json")):
        yield source, source.with_name(source.name.replace("_source", "_kk"))


def load_styles(selection):
    """source_id -> style_variant, so the memory can be keyed on register."""
    frame = pd.read_parquet(selection)
    return dict(zip(frame.id, frame.style_variant))


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    pathlib.Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def collect(translation_dir, styles):
    """Every (source, style) -> Kazakh rendering seen in the finished batches."""
    seen = collections.defaultdict(collections.Counter)
    for source_path, target_path in batch_paths(translation_dir):
        if not target_path.exists():
            continue
        for src_conv, tgt_conv in zip(read_json(source_path), read_json(target_path)):
            style = styles[src_conv["source_id"]]
            for src_msg, tgt_msg in zip(src_conv["messages"], tgt_conv["messages"]):
                seen[(src_msg["content"], style)][tgt_msg["content"]] += 1
    return seen


def choose(renderings):
    """The canonical rendering: the most frequent, ties broken by first seen.

    A mechanical choice. Every string where it had to choose is written to the
    conflicts file so a native speaker can overrule it in one pass rather than
    hunting through the batches.
    """
    return renderings.most_common(1)[0][0]


def build_memory(seen):
    memory, conflicts = [], []
    for (source, style), renderings in sorted(seen.items()):
        canonical = choose(renderings)
        memory.append({"source": source, "style": style, "target": canonical})
        if len(renderings) > 1:
            conflicts.append({
                "source": source,
                "style": style,
                "chosen": canonical,
                "variants": [{"target": target, "count": count}
                             for target, count in renderings.most_common()],
            })
    return memory, conflicts


def lookup_table(memory):
    """Exact (source, style) matches, plus strings that are unambiguous everywhere."""
    exact = {(entry["source"], entry["style"]): entry["target"] for entry in memory}
    by_source = collections.defaultdict(set)
    for entry in memory:
        by_source[entry["source"]].add(entry["target"])
    unambiguous = {source: next(iter(targets))
                   for source, targets in by_source.items() if len(targets) == 1}
    return exact, unambiguous


def resolve(source, style, exact, unambiguous):
    """The memory's rendering for a string, or None if it must be translated."""
    if (source, style) in exact:
        return exact[(source, style)]
    return unambiguous.get(source)


def cmd_build(args):
    styles = load_styles(args.selection)
    seen = collect(args.translation, styles)
    memory, conflicts = build_memory(seen)

    out_dir = pathlib.Path(args.translation)
    write_json(out_dir / MEMORY_NAME, memory)
    write_json(out_dir / CONFLICTS_NAME, conflicts)

    print(f"{len(memory)} (string, style) entries -> {out_dir / MEMORY_NAME}")
    print(f"{len(conflicts)} needed a choice between renderings -> {out_dir / CONFLICTS_NAME}")

    exact, unambiguous = lookup_table(memory)
    slots = fillable = 0
    needed = set()
    for source_path, target_path in batch_paths(args.translation):
        if target_path.exists():
            continue
        for conv in read_json(source_path):
            style = styles[conv["source_id"]]
            for message in conv["messages"]:
                slots += 1
                if resolve(message["content"], style, exact, unambiguous) is not None:
                    fillable += 1
                else:
                    needed.add((message["content"], style))

    if slots:
        print()
        print(f"pending message slots: {slots}")
        print(f"  fillable from memory: {fillable} ({100 * fillable / slots:.0f}%)")
        print(f"  still to translate:   {len(needed)} distinct strings, "
              f"{sum(len(s) for s, _ in needed):,} characters")


def cmd_normalize(args):
    styles = load_styles(args.selection)
    memory = read_json(pathlib.Path(args.translation) / MEMORY_NAME)
    exact, unambiguous = lookup_table(memory)

    total = 0
    for source_path, target_path in batch_paths(args.translation):
        if not target_path.exists():
            continue
        source = read_json(source_path)
        target = read_json(target_path)
        changed = 0
        for src_conv, tgt_conv in zip(source, target):
            style = styles[src_conv["source_id"]]
            for src_msg, tgt_msg in zip(src_conv["messages"], tgt_conv["messages"]):
                canonical = resolve(src_msg["content"], style, exact, unambiguous)
                if canonical is not None and canonical != tgt_msg["content"]:
                    tgt_msg["content"] = canonical
                    changed += 1
        if changed:
            write_json(target_path, target)
            total += changed
        print(f"{target_path.name}: {changed} messages normalised")
    print(f"\n{total} messages rewritten to the canonical rendering")


def cmd_todo(args):
    styles = load_styles(args.selection)
    memory_path = pathlib.Path(args.translation) / MEMORY_NAME
    memory = read_json(memory_path) if memory_path.exists() else []
    exact, unambiguous = lookup_table(memory)

    source_path = pathlib.Path(args.translation) / f"batch_{args.batch:02d}_source.json"
    entries, seen = [], set()
    for conv in read_json(source_path):
        style = styles[conv["source_id"]]
        for message in conv["messages"]:
            key = (message["content"], style)
            if key in seen:
                continue
            if resolve(message["content"], style, exact, unambiguous) is not None:
                continue
            seen.add(key)
            entries.append({"source": message["content"], "style": style, "target": ""})

    out = pathlib.Path(args.out or
                       pathlib.Path(args.translation) / f"batch_{args.batch:02d}_todo.json")
    write_json(out, entries)
    print(f"{len(entries)} strings to translate ({sum(len(e['source']) for e in entries):,} "
          f"characters) -> {out}")


def cmd_assemble(args):
    styles = load_styles(args.selection)
    memory = read_json(pathlib.Path(args.translation) / MEMORY_NAME)
    exact, unambiguous = lookup_table(memory)

    for path in args.todo:
        for entry in read_json(path):
            if not entry["target"].strip():
                raise SystemExit(f"{path}: empty translation for {entry['source'][:60]!r}")
            exact[(entry["source"], entry["style"])] = entry["target"]

    source_path = pathlib.Path(args.translation) / f"batch_{args.batch:02d}_source.json"
    out = []
    missing = []
    for conv in read_json(source_path):
        style = styles[conv["source_id"]]
        messages = []
        for message in conv["messages"]:
            target = resolve(message["content"], style, exact, unambiguous)
            if target is None:
                missing.append((conv["translation_index"], message["i"]))
                target = ""
            messages.append({"i": message["i"], "role": message["role"],
                             "speaker": message["speaker"], "content": target})
        out.append({
            "translation_index": conv["translation_index"],
            "source_id": conv["source_id"],
            "label_3class": conv["label_3class"],
            "scenario_category": conv["scenario_category"],
            "messages": messages,
        })

    if missing:
        raise SystemExit(f"no translation for {len(missing)} messages, first few: {missing[:5]}")

    target_path = pathlib.Path(args.translation) / f"batch_{args.batch:02d}_kk.json"
    write_json(target_path, out)
    print(f"wrote {len(out)} conversations -> {target_path}")
    print(f"now run: python validate_translation.py --source {source_path} --target {target_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--translation", default="../data/translation")
    parser.add_argument("--selection", default="../data/selection/selected_300.parquet")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build").set_defaults(func=cmd_build)
    sub.add_parser("normalize").set_defaults(func=cmd_normalize)

    todo = sub.add_parser("todo")
    todo.add_argument("--batch", type=int, required=True)
    todo.add_argument("--out")
    todo.set_defaults(func=cmd_todo)

    assemble = sub.add_parser("assemble")
    assemble.add_argument("--batch", type=int, required=True)
    assemble.add_argument("--todo", nargs="+", required=True)
    assemble.set_defaults(func=cmd_assemble)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
