"""Merge the Kazakh translations back into the labelled corpus.

Produces one dataset holding the original 37,421 rows plus the 300 Kazakh ones,
with the three-class label on every row.

Two rules decide how a Kazakh row is built.

**Split.** A Kazakh row inherits its source's split. It is the same conversation
in another language, so putting it in a different split would let a model train
on the English and be scored on the Kazakh. 67 of the 300 belong to validation or
test, so this is not a hypothetical.

**Untranslated fields.** Only `messages` and `system_prompt` were translated.
`response_text`, `explanation` and `chosen_action` are the agent's side of the
record and stay English; `translated_fields` names what is actually Kazakh on
each row, so nothing has to be guessed later.
"""
import argparse
import json
import pathlib

import pandas as pd

SPLITS = ["train", "test", "validation"]
KK_SUFFIX = "::kk"


def load_translations(translation_dir):
    """source_id -> the Kazakh conversation, from every finished batch."""
    translations = {}
    directory = pathlib.Path(translation_dir)
    for source_path in sorted(directory.glob("batch_*_source.json")):
        target_path = source_path.with_name(source_path.name.replace("_source", "_kk"))
        if not target_path.exists():
            print(f"skipping {source_path.name}: no translation yet")
            continue
        for conv in json.loads(target_path.read_text(encoding="utf-8")):
            translations[conv["source_id"]] = conv
    return translations


def build_kazakh_row(row, conv):
    """A Kazakh row: same record, Kazakh conversation, same split and label."""
    messages = json.loads(row.messages)
    if len(messages) != len(conv["messages"]):
        raise SystemExit(f"{row.id}: {len(messages)} messages, {len(conv['messages'])} translated")

    translated = []
    for original, message in zip(messages, conv["messages"]):
        updated = dict(original)
        updated["content"] = message["content"]
        translated.append(updated)

    kk = row._asdict()
    kk.pop("Index", None)
    kk["id"] = row.id + KK_SUFFIX
    kk["language"] = "kk"
    kk["messages"] = json.dumps(translated, ensure_ascii=False)
    kk["source_dataset"] = f"{row.source_dataset}+kk_translation"
    kk["translated_from"] = row.id

    # the system prompt is repeated as messages[0] in this corpus; when that holds,
    # the translated first message is the translated system prompt
    fields = ["messages"]
    first = messages[0]["content"] if messages else ""
    if row.system_prompt and row.system_prompt == first:
        kk["system_prompt"] = conv["messages"][0]["content"]
        fields.append("system_prompt")
    kk["translated_fields"] = ",".join(fields)
    return kk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", required=True)
    parser.add_argument("--translation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    translations = load_translations(args.translation)
    print(f"{len(translations)} Kazakh conversations")

    labeled_dir = pathlib.Path(args.labeled)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {}
    matched = 0
    for split in SPLITS:
        frame = pd.read_parquet(labeled_dir / f"{split}.parquet")
        frame["translated_from"] = ""
        frame["translated_fields"] = ""

        kk_rows = [build_kazakh_row(row, translations[row.id])
                   for row in frame.itertuples() if row.id in translations]
        matched += len(kk_rows)

        combined = pd.concat([frame, pd.DataFrame(kk_rows)], ignore_index=True) \
            if kk_rows else frame
        combined.to_parquet(out_dir / f"{split}.parquet", index=False)
        combined.to_json(out_dir / f"{split}.jsonl", orient="records", lines=True,
                         force_ascii=False)
        totals[split] = (len(frame), len(kk_rows), len(combined))
        print(f"{split}: {len(frame)} original + {len(kk_rows)} Kazakh = {len(combined)}")

    if matched != len(translations):
        raise SystemExit(f"only {matched} of {len(translations)} translations matched a source row")

    combined = pd.concat([pd.read_parquet(out_dir / f"{s}.parquet") for s in SPLITS],
                         ignore_index=True)

    # the checks worth failing on
    if combined.id.duplicated().any():
        raise SystemExit("duplicate ids in the merged dataset")
    kazakh = combined[combined.language == "kk"]
    source_split = dict(zip(combined.id, combined.split))
    leaked = [row.id for row in kazakh.itertuples()
              if source_split.get(row.translated_from) != row.split]
    if leaked:
        raise SystemExit(f"{len(leaked)} Kazakh rows are in a different split than their source")
    print("\nOK: ids unique, every Kazakh row shares its source's split")

    lines = ["# Merged dataset", "",
             f"Total rows: {len(combined):,} "
             f"({len(combined) - len(kazakh):,} original + {len(kazakh)} Kazakh)", "",
             "| Split | Original | Kazakh | Total |", "|---|---:|---:|---:|"]
    for split, (original, kk, total) in totals.items():
        lines.append(f"| {split} | {original:,} | {kk} | {total:,} |")
    lines += ["", "## Kazakh rows by class", "", "| Class | Rows |", "|---|---:|"]
    for label, count in kazakh.label_3class.value_counts().items():
        lines.append(f"| {label} | {count} |")
    lines += ["", "## Fields translated on the Kazakh rows", "", "| Fields | Rows |", "|---|---:|"]
    for fields, count in kazakh.translated_fields.value_counts().items():
        lines.append(f"| `{fields}` | {count} |")
    lines += ["",
              "Every Kazakh row keeps its source's split, so the same conversation never",
              "appears in training in one language and in evaluation in the other.",
              "`translated_from` names the English row each Kazakh row came from.", ""]
    (out_dir / "merge_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir / 'merge_report.md'}")


if __name__ == "__main__":
    main()
