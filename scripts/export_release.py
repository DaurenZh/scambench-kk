"""Package the dataset for release: parquet files plus a dataset card.

Three things come out of this project that someone else can use, and none of them
is useful locked in a working directory:

- **the three-class label** over all 37,421 ScamBench rows, derived rather than
  invented, with the rule that produced each one recorded;
- **300 human Kazakh translations**, structurally verified, each paired with its
  English source so cross-lingual transfer can be measured rather than assumed;
- **a conversation-grouped split**, because the published splits leak -- 83% of
  validation rows have a restyling of the same conversation in training.

The card states the limitations in the same place as the numbers, including the
one that matters most: no native speaker has reviewed the Kazakh.

Source corpus is CC-BY-SA-4.0, so the release carries the same licence and
attribution.

Usage:

    python export_release.py --merged ../data/merged --splits ../data/splits \\
        --out ../data/release
"""
import argparse
import json
import pathlib

import pandas as pd

SPLITS = ["train", "test", "validation"]
SOURCE = "shaw/scambench-training"
LICENCE = "cc-by-sa-4.0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged = pd.concat([pd.read_parquet(pathlib.Path(args.merged) / f"{s}.parquet")
                        for s in SPLITS], ignore_index=True)
    grouped = pd.read_csv(pathlib.Path(args.splits) / "grouped_split.csv")
    merged = merged.merge(grouped[["id", "conversation", "split_grouped"]],
                          on="id", how="left")
    if merged.split_grouped.isna().any():
        raise SystemExit("some rows have no grouped split; re-run make_group_split.py")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        part = merged[merged.split_grouped == split]
        part.to_parquet(out_dir / f"{split}.parquet", index=False)
        print(f"{split}: {len(part):,} rows")

    kazakh = merged[merged.language == "kk"]
    pairs = kazakh.merge(merged[["id", "messages"]], left_on="translated_from",
                         right_on="id", suffixes=("_kk", "_en"))
    pairs[["id_kk", "id_en", "label_3class", "scenario_category", "style_variant",
           "split_grouped", "messages_en", "messages_kk"]].to_parquet(
        out_dir / "kazakh_pairs.parquet", index=False)
    print(f"kazakh_pairs: {len(pairs)} aligned EN/KK conversations")

    counts = merged.label_3class.value_counts()
    card = f"""---
license: {LICENCE}
language:
- en
- kk
task_categories:
- text-classification
tags:
- scam-detection
- prompt-injection
- cross-lingual
- kazakh
---

# ScamBench-KK

Two additions to [{SOURCE}](https://huggingface.co/datasets/{SOURCE}):
a **three-class risk label** over the whole corpus, and **300 human Kazakh
translations** paired with their English sources. Plus a **conversation-grouped
split**, because the published splits leak.

{len(merged):,} rows ({len(merged) - len(kazakh):,} original + {len(kazakh)} Kazakh).

## The three-class label

ScamBench has no SCAM/LEGITIMATE column. It has `should_trigger_scam_defense`
(was this an attack?) and `decision_class` (what should a correct agent do?).
`label_3class` is derived from both, so no original annotation is overwritten and
`label_rule` records which rule fired.

| Label | Rows | Meaning |
|---|---:|---|
| LEGITIMATE | {counts.get('LEGITIMATE', 0):,} | normal engagement, no attack flag, no risk signal |
| SUSPICIOUS | {counts.get('SUSPICIOUS', 0):,} | verify before acting, or a risk signal without proof |
| SCAM | {counts.get('SCAM', 0):,} | attack confirmed, correct action is defensive |

It never contradicts the original flag: no benign-flagged row became SCAM, no
attack-flagged row became LEGITIMATE. Every SUSPICIOUS row carries named risk
indicators in `risk_indicators` — nothing is suspicious for merely mentioning
money or passwords.

**Known limitation.** SCAM and SUSPICIOUS both carry
`should_trigger_scam_defense = True` and differ only by `decision_class`, which
is a judgement about what an agent should *do* and is never stated in the
conversation. A text-only classifier separates them at 0.537 macro-F1 against
0.500 chance. Report the binary task (SCAM+SUSPICIOUS merged) alongside the
three-class one; the same features reach 0.879 there.

## The Kazakh subset

300 conversations, 100 per class, stratified across scenario categories, drawn
from the deduplicated base pool. `kazakh_pairs.parquet` holds them aligned with
their English sources.

Every translation is machine-verified: message count, roles and speakers
unchanged, and every URL, email, handle, phone number, amount, code and hash
preserved exactly. Wording is kept consistent by a translation memory keyed on
(source string, style variant), so there is no drift within a register.

Translation decisions: leetspeak is rendered as readable Kazakh (19 of 300
conversations); jailbreak and injection payloads are translated as prose,
preserving intent, because that is the content a classifier must learn; data is
never translated; day and month names are translated, clock times and numerals
are not.

**`response_text`, `explanation` and `chosen_action` remain English** on the
Kazakh rows. `translated_fields` records what is actually Kazakh.

**No native speaker has reviewed these translations for fluency.** They are
verified structurally, not stylistically. This is the dataset's main limitation.

## The grouped split

A row id is `<pool>::<conversation>::<row>`, and the augmented pool rewrites one
conversation in up to 76 voices. The published splits assign those rewrites
independently, so **83% of validation rows have a twin of the same conversation
in training**. Scores measured against the original splits are inflated: on our
own task, 0.783 against the published split and 0.650 against a grouped one.

`split_grouped` keeps a whole conversation family on one side. Same sizes, class
balance identical across splits to within 0.1pp, and every Kazakh row stays with
its English source. The original `split` column is preserved unchanged.

**Use `split_grouped`.**

## Columns added to the original schema

| Column | Meaning |
|---|---|
| `label_3class` | SCAM / SUSPICIOUS / LEGITIMATE |
| `label_rule` | which rule produced the label |
| `risk_indicators` | named risk families behind a SUSPICIOUS label |
| `needs_review` | the rule flagged this row as uncertain |
| `split_grouped` | conversation-grouped split — use this one |
| `conversation` | conversation family id |
| `translated_from` | for Kazakh rows, the English row they came from |
| `translated_fields` | which fields are actually Kazakh |

## Reproducing

Everything is built by scripts, not notebooks:
https://github.com/DaurenZh/scambench-kk

## Licence and attribution

{LICENCE}, inherited from [{SOURCE}](https://huggingface.co/datasets/{SOURCE}).
Derivative works must carry the same licence and attribute the original corpus.
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    print(f"\nwrote {out_dir / 'README.md'} (dataset card)")
    print(f"upload with:  huggingface-cli upload <user>/scambench-kk {out_dir} .")


if __name__ == "__main__":
    main()
