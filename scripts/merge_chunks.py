"""Rebuild a translated batch from chunk files of message contents.

A chunk maps translation_index -> list of translated message contents. Structure
(ids, roles, speakers, order) is copied from the source batch, so a translation
can only ever differ from its source in message text.
"""
import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--chunks", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = json.loads(pathlib.Path(args.source).read_text(encoding="utf-8"))

    contents = {}
    for chunk in args.chunks:
        for index, messages in json.loads(pathlib.Path(chunk).read_text(encoding="utf-8")).items():
            contents[int(index)] = messages

    out = []
    missing = []
    for conv in source:
        index = conv["translation_index"]
        if index not in contents:
            missing.append(index)
            continue
        translated = contents[index]
        if len(translated) != len(conv["messages"]):
            raise SystemExit(
                f"#{index}: {len(conv['messages'])} messages in source, {len(translated)} translated"
            )
        out.append({
            "translation_index": index,
            "source_id": conv["source_id"],
            "label_3class": conv["label_3class"],
            "scenario_category": conv["scenario_category"],
            "messages": [
                {"i": m["i"], "role": m["role"], "speaker": m["speaker"], "content": text}
                for m, text in zip(conv["messages"], translated)
            ],
        })

    if missing:
        raise SystemExit(f"no translation for conversations: {missing}")

    pathlib.Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(out)} conversations to {args.out}")


if __name__ == "__main__":
    main()
