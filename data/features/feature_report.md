# Feature report

37,721 rows, 54 features

## Columns excluded as leakage

`label_3class` was derived from `decision_class` and
`should_trigger_scam_defense`. Knowing that pair pins the label down
99.99% of the time, so these columns reproduce the answer rather than
predicting it and are kept out of the feature table.

| Column | Why |
|---|---|
| `label_rule` | names the rule that produced the label (100%) |
| `decision_class` | half of the pair the label was derived from (91%) |
| `should_trigger_scam_defense` | the other half; the pair gives 99.99% |
| `chosen_action` | the action the decision implies (91%) |
| `unsafe_signals` | annotation the risk indicators were read from (92%) |
| `diagnostic_labels` | annotation the risk indicators were read from (96%) |
| `risk_indicators` | derived by our own labelling rule (81%) |
| `needs_review` | set by the labelling rule |
| `scenario_category` | the attack type, an annotation rather than an observation (73%) |
| `response_text` | the agent's reply, written knowing the right answer |
| `explanation` | the rationale for the label |
| `reasoning_trace` | chain of thought toward the labelled decision |
| `source_dataset` | provenance; which corpus a row came from predicts its label (79%) |

## Feature groups

| Prefix | Count | Survives translation? |
|---|---:|---|
| `struct_` | 8 | yes, by construction (message count and roles are preserved) |
| `entity_` | 17 | yes, by construction (links, amounts and codes are preserved) |
| `surface_` | 13 | no, it measures the writing system |
| `lex_` | 16 | only where the lexicon has that language |

## Class balance

| Class | Rows |
|---|---:|
| LEGITIMATE | 18,764 |
| SUSPICIOUS | 11,079 |
| SCAM | 7,878 |

## English / Kazakh parity

300 conversations exist in both languages. For a feature to be
usable across languages, a pair should agree on it.

| Feature group | Pairs agreeing exactly |
|---|---:|
| `struct_` | 300 / 300 |
| `entity_` | 292 / 300 |
| `surface_` | 0 / 300 |
| `lex_` | 39 / 300 |
