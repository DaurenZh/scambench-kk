# ScamBench: Kazakh extension and a third class

Two additions to the [ScamBench training corpus](https://huggingface.co/datasets/shaw/scambench-training):

1. A third classification label, **SUSPICIOUS**, alongside SCAM and LEGITIMATE.
2. Natural Kazakh translations of a stratified sample of conversations.

## Layout

```
scambench_kk/
  data/
    raw/           original parquet splits (train/test/validation)
    labeled/       every record + 3-class label (parquet, jsonl, report)
    selection/     the 300 conversations chosen for translation
    translation/   batch_NN_source.json  = English source
                   batch_NN_kk.json      = Kazakh translation
  scripts/
    label_3class.py           the labelling rule
    apply_labels.py           runs the rule over all splits
    select_for_translation.py picks 300 conversations, stratified
    export_batches.py         splits them into translation batches
    validate_translation.py   checks a translation against its source
  review/
    review_batches_1_and_6.html   side-by-side EN/KK for human review
```

## Task 2: the SUSPICIOUS class

ScamBench has no SCAM/LEGITIMATE column. It has `should_trigger_scam_defense`
(was this an attack?) and `decision_class` (what should a correct agent do?).
The three-class label is derived from those, so no original annotation is
overwritten. See `data/labeled/labelling_report.md` for the full breakdown.

| Label | Meaning | Records |
|---|---|---|
| SCAM | attack confirmed, correct action is defensive (refuse/escalate/block) | 7,778 |
| SUSPICIOUS | correct action is verify-first, or a real risk signal with incomplete evidence | 10,979 |
| LEGITIMATE | normal engagement, no attack flag, no risk signal | 18,664 |

Two guarantees:

- The label never contradicts the original flag: no benign row became SCAM, no
  attack row became LEGITIMATE.
- Every SUSPICIOUS row carries named risk indicators (social engineering,
  instruction override, credential request, urgency, suspicious link, and so
  on), stored in the `risk_indicators` column. Nothing is marked suspicious for
  merely mentioning money, banking or passwords.

Reproduce:

```
cd scripts
python apply_labels.py --raw ../data/raw --out ../data/labeled
```

## Task 1: Kazakh translation

300 conversations, 100 per class, stratified across all seven scenario
categories, drawn only from the deduplicated `base` pool. Translations are
faithful: names, handles, URLs, phone numbers, amounts, codes, hashes and code
snippets are kept exactly; only natural-language content is rendered into modern
Kazakh. Speaker roles and message order are preserved, and each translated row
keeps the label of its source.

Progress: batches 1 (legitimate) and 6 (scam) are done and validated; the rest
are pending human review of those two.

Reproduce the selection and validation:

```
cd scripts
python select_for_translation.py --labeled ../data/labeled --out ../data/selection/selected_300.parquet
python export_batches.py --selection ../data/selection/selected_300.parquet --out ../data/translation
python validate_translation.py --source ../data/translation/batch_01_source.json --target ../data/translation/batch_01_kk.json
```

The validator checks that every message is present, roles and speakers are
unchanged, and every URL, amount, handle and code from the source survives in
the translation.

### Two translation decisions

- Leetspeak source (`H3ll0`) is rendered as readable Kazakh. The obfuscation is
  already recorded in the label and the `style_variant` metadata, and the task
  asks for natural Kazakh.
- Jailbreak and prompt-injection payloads (e.g. DUDE, LiveGPT) are translated as
  prose, preserving the manipulative intent, because they are the content a
  classifier must learn from.
