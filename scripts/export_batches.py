"""Split the selected conversations into batch files ready for translation.

Each batch file holds the source text only. A translated file with the same
structure is written next to it, and `validate_translation.py` compares the two.
"""
import argparse
import json
import pathlib

import pandas as pd

BATCH_SIZE = 20


def build_record(row):
    messages = json.loads(row.messages)
    record = {
        "translation_index": int(row.translation_index),
        "source_id": row.id,
        "label_3class": row.label_3class,
        "scenario_category": row.scenario_category,
        "messages": [],
    }

    # the system prompt is repeated as messages[0] in most rows; only carry it
    # separately when it differs, so it does not get translated twice
    first = messages[0]["content"] if messages else ""
    if row.system_prompt and row.system_prompt != first:
        record["system_prompt"] = row.system_prompt

    for position, message in enumerate(messages):
        record["messages"].append({
            "i": position,
            "role": message["role"],
            "speaker": message.get("speaker", ""),
            "content": message["content"],
        })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    df = pd.read_parquet(args.selection)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [build_record(row) for row in df.itertuples()]

    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        number = start // args.batch_size + 1
        path = out_dir / f"batch_{number:02d}_source.json"
        path.write_text(json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
        chars = sum(len(m["content"]) for r in batch for m in r["messages"])
        labels = pd.Series([r["label_3class"] for r in batch]).value_counts().to_dict()
        print(f"{path.name}: {len(batch)} conversations, {chars:,} chars, {labels}")

    print(f"\n{len(records)} conversations in {(len(records) + args.batch_size - 1) // args.batch_size} batches")


if __name__ == "__main__":
    main()
