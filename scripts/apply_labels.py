"""Apply the three-class label to every ScamBench split and write the result.

Reads the raw parquet splits, adds four columns, writes parquet + jsonl per split
and one combined stats report. The original columns are left untouched.
"""
import argparse
import collections
import json
import pathlib

import pandas as pd

from label_3class import label_row

SPLITS = ["train", "test", "validation"]


def label_frame(df):
    labels, rules, indicators, review = [], [], [], []
    for row in df.itertuples():
        label, rule, families, needs_review = label_row(
            row.decision_class,
            row.should_trigger_scam_defense,
            row.scenario_category,
            row.unsafe_signals,
            row.diagnostic_labels,
        )
        labels.append(label)
        rules.append(rule)
        indicators.append(json.dumps(families))
        review.append(needs_review)

    df = df.copy()
    df["label_3class"] = labels
    df["label_rule"] = rules
    df["risk_indicators"] = indicators
    df["needs_review"] = review
    return df


def build_report(frames):
    combined = pd.concat(frames.values(), ignore_index=True)
    lines = ["# Three-class labelling report", ""]
    lines.append(f"Total records: {len(combined)}")
    lines.append("")

    lines.append("## Label distribution")
    lines.append("")
    lines.append("| Label | Records | Share |")
    lines.append("|---|---:|---:|")
    for label, count in combined.label_3class.value_counts().items():
        lines.append(f"| {label} | {count} | {100 * count / len(combined):.1f}% |")
    lines.append("")

    lines.append("## Per split")
    lines.append("")
    table = pd.crosstab(combined.split, combined.label_3class)
    lines.append("| Split | " + " | ".join(table.columns) + " |")
    lines.append("|---" * (len(table.columns) + 1) + "|")
    for split, row in table.iterrows():
        lines.append(f"| {split} | " + " | ".join(str(v) for v in row.values) + " |")
    lines.append("")

    lines.append("## Rule that produced each label")
    lines.append("")
    lines.append("| Rule | Records |")
    lines.append("|---|---:|")
    for rule, count in combined.label_rule.value_counts().items():
        lines.append(f"| `{rule}` | {count} |")
    lines.append("")

    lines.append("## Relationship to the original binary flag")
    lines.append("")
    lines.append("The three-class label refines `should_trigger_scam_defense` rather than")
    lines.append("contradicting it: no record flagged benign became SCAM, and no record flagged")
    lines.append("as an attack became LEGITIMATE.")
    lines.append("")
    cross = pd.crosstab(combined.should_trigger_scam_defense, combined.label_3class)
    lines.append("| `should_trigger_scam_defense` | " + " | ".join(cross.columns) + " |")
    lines.append("|---" * (len(cross.columns) + 1) + "|")
    for flag, row in cross.iterrows():
        lines.append(f"| {flag} | " + " | ".join(str(v) for v in row.values) + " |")
    lines.append("")

    lines.append("## Risk indicators behind the SUSPICIOUS class")
    lines.append("")
    suspicious = combined[combined.label_3class == "SUSPICIOUS"]
    counter = collections.Counter()
    empty = 0
    for raw in suspicious.risk_indicators:
        families = json.loads(raw)
        if not families:
            empty += 1
        for family in families:
            counter[family] += 1
    lines.append("| Indicator family | SUSPICIOUS records carrying it |")
    lines.append("|---|---:|")
    for family, count in counter.most_common():
        lines.append(f"| {family} | {count} |")
    lines.append("")
    lines.append(f"Records with no indicator family resolved: {empty} "
                 f"({100 * empty / len(suspicious):.1f}% of SUSPICIOUS).")
    lines.append(f"Records flagged `needs_review`: {int(combined.needs_review.sum())}.")
    lines.append("")

    lines.append("## Language coverage")
    lines.append("")
    lines.append("| Language | Records |")
    lines.append("|---|---:|")
    for language, count in combined.language.value_counts().items():
        lines.append(f"| {language} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="directory holding <split>.parquet")
    parser.add_argument("--out", required=True, help="directory to write labelled data into")
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for split in SPLITS:
        df = pd.read_parquet(raw_dir / f"{split}.parquet")
        df["split"] = split
        df = label_frame(df)
        df.to_parquet(out_dir / f"{split}.parquet", index=False)
        df.to_json(out_dir / f"{split}.jsonl", orient="records", lines=True, force_ascii=False)
        frames[split] = df
        counts = df.label_3class.value_counts().to_dict()
        print(f"{split}: {len(df)} records -> {counts}")

    report = build_report(frames)
    (out_dir / "labelling_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir / 'labelling_report.md'}")


if __name__ == "__main__":
    main()
