# ScamBench: Kazakh extension and a third class

Two additions to the [ScamBench training corpus](https://huggingface.co/datasets/shaw/scambench-training):

1. A third classification label, **SUSPICIOUS**, alongside SCAM and LEGITIMATE.
2. Natural Kazakh translations of a stratified sample of conversations.

Both are complete. The merged dataset is **37,721 rows** — 37,421 original plus 300
Kazakh — with the three-class label on every row. `EXPLANATION.md` is the full
account; its Part 2 covers the finishing work and the open risks.

## Layout

```
scambench_kk/
  data/
    raw/           original parquet splits (train/test/validation)
    labeled/       every record + 3-class label (parquet, jsonl, report)
    selection/     the 300 conversations chosen for translation
    translation/   batch_NN_source.json  = English source
                   batch_NN_kk.json      = Kazakh translation
                   memory.json           = (string, style) -> Kazakh
                   entity_waivers.json   = recorded validator waivers
    merged/        original + Kazakh rows, 3-class labelled (+ merge_report.md)
  scripts/
    label_3class.py           the labelling rule
    apply_labels.py           runs the rule over all splits
    select_for_translation.py picks 300 conversations, stratified
    export_batches.py         splits them into translation batches
    translation_memory.py     build / normalize / todo / assemble
    validate_translation.py   checks a translation against its source
    test_validate_translation.py  tests for the validator itself
    merge_kk.py               merges the Kazakh rows back in
    build_features.py         features from conversation text only
    make_group_split.py       a split that never divides a conversation family
    run_crosslingual.py       train on English, evaluate on Kazakh
    features/      54 features per conversation (+ feature_report.md)
    splits/        grouped_split.csv -- conversation-level split assignment
    experiments/   cross-lingual results (+ crosslingual_report.md)
  notebooks/
    01_dataset_audit.ipynb    distributions, split-leakage audit, coverage
    02_feature_engineering.ipynb  label-leakage audit, features, EN/KK parity
    03_crosslingual_baseline.ipynb  train EN, evaluate KK: what translation costs
  review/
    review_batches_1_and_6.html   side-by-side EN/KK for human review
```

Scripts build, notebooks look. The pipeline stays in `scripts/` because its outputs
are gitignored and regenerated, so the scripts are the only record of how the dataset
was built.

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

**All 300 are translated and all fifteen batches pass validation.** A native-speaker
fluency review is still outstanding — that is the project's main open risk.

Reproduce the selection, translation memory and validation:

```
cd scripts
python select_for_translation.py --labeled ../data/labeled --out ../data/selection/selected_300.parquet
python export_batches.py --selection ../data/selection/selected_300.parquet --out ../data/translation
python translation_memory.py build
python test_validate_translation.py
python validate_translation.py --source ../data/translation/batch_01_source.json --target ../data/translation/batch_01_kk.json
```

The validator checks that every message is present, roles and speakers are
unchanged, and every URL, amount, handle and code from the source survives in
the translation. It knows leetspeak: `455157` is the word "assist", not a code.
Four entities it cannot decide by rule are waived individually, with reasons, in
`data/translation/entity_waivers.json`.

Wording is kept consistent by a translation memory keyed on
`(source string, style_variant)`, so an English sentence gets one Kazakh rendering
within a register. There are **0** strings drifting within a style. 45 strings are
rendered differently across styles, which is register kept on purpose — `I'm
listening.` is `Тыңдап отырмын.` in `short_professional` and `Тыңдап тұрмын.` in
`slightly_slangy`.

## Features

```
cd scripts
python build_features.py --merged ../data/merged --out ../data/features
```

54 features from the conversation text alone. **13 of the 25 columns are excluded as
leakage**: `label_3class` was derived from `decision_class` and
`should_trigger_scam_defense`, and that pair determines the label 99.99% of the time,
so those columns reproduce the answer rather than predicting it. The script fails if
any of them reaches the feature table; `build_features.LEAKING_COLUMNS` records the
reason for each.

Macro-F1 on validation, under two splits:

| | original split | grouped split |
|---|---:|---:|
| majority class | 0.222 | 0.221 |
| text features only | 0.783 | **0.709** |
| + `decision_class` (leaked) | 0.986 | 0.962 |

**Read the second column.** ScamBench's own splits leak: the augmented pool rewrites a
conversation in up to 25 voices and assigns the rewrites to splits independently, so
93% of validation rows have a twin of the same conversation in training.
`scripts/make_group_split.py` produces `split_grouped`, which keeps a conversation
family on one side of the line — same sizes, same class balance, zero families split.
The honest baseline is **0.709**; the last row is what this dataset looks like when the
label-leakage audit is skipped.

The 300 EN/KK pairs make translation-invariance measurable rather than assumed:
`struct_` matches on 300/300 pairs, `entity_` on 292/300, `surface_` on 0/300 (they
measure the writing system) and `lex_` on 39/300.

## Cross-lingual baseline

```
cd scripts
python make_group_split.py --merged ../data/merged --out ../data/splits
python run_crosslingual.py --features ../data/features --merged ../data/merged --out ../data/experiments
```

The 300 conversations exist in both languages with identical structure, so the gap
between a model's English and Kazakh score is the cost of the language change and
nothing else. Trained on 12,435 English conversations, with the 300 **and every
restyling of them** held out.

| Model | EN-300 | KK-300 | gap |
|---|---:|---:|---:|
| majority class | 0.167 | 0.167 | 0.000 |
| hand features (all 54) | 0.563 | **0.539** | +0.024 |
| hand features (struct+entity, 25) | 0.343 | 0.346 | −0.002 |
| char n-gram TF-IDF (3-5) | **0.736** | 0.491 | +0.245 |
| word TF-IDF (1-2) | 0.690 | 0.416 | +0.274 |

The feature grouping predicted in notebook 02 holds: the translation-invariant groups
transfer with no loss, and the lexical models fall off the script boundary. Char
n-grams are the best English model and among the worst Kazakh ones — on the language
this project is about, the interpretable features win.

## Expanding the Kazakh set

```
cd scripts
python select_expanded.py --merged ../data/merged --out ../data/selection --per-class 100
python export_batches.py --selection ../data/selection/expanded_300.parquet --out ../data/translation_v2
```

The original selection drew only from the `base` pool, assuming the rest was
style-variant augmentation. Split by conversation family that is half right: 13,899
augmented rows are restylings of base conversations and were rightly excluded, but
**5,067 are not tied to any base conversation** and were excluded for no reason.

Counting each conversation family once and removing what is already translated,
**326 more conversations per class are available — 978 total, 3.3x the current 300** —
reaching seven scenario categories the original 300 never contained
(`cascading-failure`, `hitl-bypass`, `inter-agent-communication`, `state-persistence`
and others). A first 300 are selected and exported, ready to translate.

### Translation decisions

- Leetspeak source (`H3ll0`) is rendered as readable Kazakh. The obfuscation is
  already recorded in the label and the `style_variant` metadata, and the task
  asks for natural Kazakh. This applies to 19 of the 300 conversations.
- Jailbreak and prompt-injection payloads (e.g. DUDE, LiveGPT) are translated as
  prose, preserving the manipulative intent, because they are the content a
  classifier must learn from.
- Data is never translated: URLs, emails, handles, phone numbers, amounts, hashes,
  code and JSON payloads are kept byte-for-byte, including leetspeak email
  addresses, which are addresses rather than words.
- Day and month names are translated; clock times, dates and numerals are not.

## Merging

```
cd scripts
python merge_kk.py --labeled ../data/labeled --translation ../data/translation --out ../data/merged
```

Each Kazakh row inherits **its English source's split**. A translation is the same
conversation in another language, so splitting them apart would let a model train on
the English and be scored on the Kazakh. 67 of the 300 belong to validation or test,
so this matters; the merge asserts it and fails if it is ever violated.

Kazakh rows carry `language=kk`, an id of `<source id>::kk`, `translated_from`
naming the English row, and `translated_fields` recording what is actually Kazakh
(`messages,system_prompt` — `response_text` and `explanation` stay English).
