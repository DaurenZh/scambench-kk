---
license: cc-by-sa-4.0
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

Two additions to [shaw/scambench-training](https://huggingface.co/datasets/shaw/scambench-training):
a **three-class risk label** over the whole corpus, and **300 human Kazakh
translations** paired with their English sources. Plus a **conversation-grouped
split**, because the published splits leak.

37,721 rows (37,421 original + 300 Kazakh).

## The three-class label

ScamBench has no SCAM/LEGITIMATE column. It has `should_trigger_scam_defense`
(was this an attack?) and `decision_class` (what should a correct agent do?).
`label_3class` is derived from both, so no original annotation is overwritten and
`label_rule` records which rule fired.

| Label | Rows | Meaning |
|---|---:|---|
| LEGITIMATE | 18,764 | normal engagement, no attack flag, no risk signal |
| SUSPICIOUS | 11,079 | verify before acting, or a risk signal without proof |
| SCAM | 7,878 | attack confirmed, correct action is defensive |

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

cc-by-sa-4.0, inherited from [shaw/scambench-training](https://huggingface.co/datasets/shaw/scambench-training).
Derivative works must carry the same licence and attribute the original corpus.
