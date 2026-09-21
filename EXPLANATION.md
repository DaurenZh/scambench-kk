# Multilingual Scam Conversation Detection: Kazakh Extension and a Third Class

A step-by-step account of what we did, in plain language, for presentation and defense.

Dataset: ScamBench Training Corpus, https://huggingface.co/datasets/shaw/scambench-training
(37,421 multi-turn conversations, 14 languages, CC-BY-SA-4.0).

We did two things:

1. **Task 2 — a third class.** Extended the labels from two categories to three:
   SCAM, LEGITIMATE, and a new **SUSPICIOUS** class. Applied to all 37,421 records.
2. **Task 1 — Kazakh.** Translated a stratified sample of conversations into natural
   Kazakh. 40 of a planned 300 are finished and validated; the rest are paused for a
   native-speaker quality check.

---

## Step 0. Understanding the dataset (and one key discovery)

Before writing any code we inspected the columns. The important finding that shaped
the whole project:

**ScamBench has no SCAM / LEGITIMATE column.** There is nothing to copy. Instead the
corpus carries two service fields:

- `should_trigger_scam_defense` — a boolean: was this conversation an attack? (yes/no)
- `decision_class` — what a correct agent should *do* (13 values: `engage_legitimate`,
  `request_verification`, `refuse`, `audit`, `escalate`, `ignore`, `block_actor`,
  `execute_transaction`, and so on).

Because there was no ready label, we had to **derive** the class from these fields
rather than take it for granted. This is the core of Task 2.

---

## Step 1 (Task 2). Why we need our own class even though `decision_class` exists

This is the first question a reviewer asks, so it is answered directly.

`decision_class` and our label answer **different questions**:

| | `decision_class` | our `label_3class` |
|---|---|---|
| Question | "What action should the agent take?" | "What is this conversation?" |
| Nature | operational / agent-side | risk category / classification target |
| Values | 13, imbalanced | 3 (SCAM / SUSPICIOUS / LEGITIMATE) |
| Given in dataset | yes | no — we build it |

The decisive point: **one `decision_class` value does not correspond to one risk class.**
The same action is used for both harmless and dangerous conversations. Measured on the
data:

| `decision_class` | benign (`should_trigger=False`) | attack (`should_trigger=True`) |
|---|---:|---:|
| `request_verification` | 2,426 | 6,697 |
| `refuse` | 54 | 5,209 |
| `audit` | 343 | 2,037 |
| `ignore` | 13 | 65 |

`request_verification` alone spans 2,426 safe conversations and 6,697 attacks. If we
used `decision_class` as the label, safe and malicious conversations would land in the
same category, which is useless for a scam classifier. That is exactly why we combine
**two** fields to recover the true risk level, and why a dedicated 3-class label is
needed.

(An analogy: `decision_class` is the doctor's *prescription*; our label is the
*diagnosis*. "Run more tests" can be prescribed both to a healthy patient and to a sick
one, so the prescription alone does not tell you who is sick.)

---

## Step 2 (Task 2). How the three classes are defined

The rule (implemented in `scripts/label_3class.py`):

- **SCAM** — the attack is confirmed (`should_trigger = True`) **and** the correct
  action is defensive (`refuse`, `escalate`, `block_actor`, `warn_actor`,
  `deny_privileged_action`, `ignore`). Confirmed malicious behavior.
- **SUSPICIOUS** — the correct action is "verify before acting"
  (`request_verification`, `audit`), **or** a real risk signal is present but the
  evidence is not strong enough to confirm a scam. Warning signs without full proof.
- **LEGITIMATE** — normal engagement (`engage_legitimate`, `allow_safe_action`,
  `accept`, ...), no attack flag, no risk signal.

Result across all 37,421 records:

| Class | Records | Share |
|---|---:|---:|
| LEGITIMATE | 18,664 | 49.9% |
| SUSPICIOUS | 10,979 | 29.3% |
| SCAM | 7,778 | 20.8% |

### Two properties to state in the defense

1. **The new label never contradicts the original data.** No benign-flagged
   conversation became SCAM; no attack-flagged conversation became LEGITIMATE. We
   refined the existing annotation, we did not overwrite it.

2. **Every SUSPICIOUS conversation is backed by named risk indicators**, stored in the
   `risk_indicators` column. The indicator families are:

   - social engineering / manipulation
   - instruction override (prompt injection)
   - credential or verification-code request
   - financial / payment request
   - urgency and pressure
   - suspicious link
   - impersonation
   - privilege escalation
   - identity inconsistency

   Nothing is marked SUSPICIOUS just for mentioning money, banking, or passwords —
   this was an explicit requirement of the assignment, and it is satisfied by
   construction.

Output: `data/labeled/` (parquet + jsonl per split) and a full breakdown in
`data/labeled/labelling_report.md`.

Reproduce:

```
cd scripts
python apply_labels.py --raw ../data/raw --out ../data/labeled
```

---

## Step 3 (Task 1). Translating into Kazakh

We did not translate everything. Most of the corpus is style-variant copies of the same
base conversations, so translating all of it would create near-duplicate Kazakh rows.
Instead we followed a principled sampling and translation pipeline.

1. **Selection.** 300 conversations, 100 per class, spread across all 7 scenario
   categories (scam, phishing, prompt-injection, ...), drawn only from the deduplicated
   `base` pool. Script: `select_for_translation.py`.

2. **Translation into natural, modern Kazakh**, preserving exactly:
   - message order and conversation structure;
   - speaker roles and labels;
   - meaning, intent, and the scammer's persuasion tactics;
   - URLs, emails, handles, phone numbers, amounts, dates, codes, hashes, and code
     snippets (kept verbatim, never translated);
   - the original class label of each conversation.

   Example:

   ```
   Original:  SCAMMER: Your bank account has been suspended.
   Kazakh:    SCAMMER: Сіздің банк шотыңыз бұғатталды.
   ```

3. **Automatic validation.** Script `validate_translation.py` compares each translation
   against its source and fails on any of: a missing or merged message, a changed
   speaker or role, a dropped URL / amount / code, or a line left untranslated.

### Current status

- **Batch 1** (20 LEGITIMATE) — done, validation passed.
- **Batch 6** (20 SCAM, including leetspeak and prompt-injection payloads) — done,
  validation passed.
- Remaining 260 conversations are paused for a native-speaker quality review of these
  first 40, so any wording preference can be applied consistently before scaling up.

A human-readable side-by-side (English vs Kazakh) is in
`review/review_batches_1_and_6.html`.

### Two translation decisions worth mentioning

- **Leetspeak** source (`H3ll0`) is rendered as readable Kazakh. The obfuscation is
  already recorded in the label and in the `style_variant` metadata, and the task asks
  for natural Kazakh.
- **Jailbreak / prompt-injection payloads** (e.g. "DUDE", "LiveGPT") are translated as
  prose, preserving their manipulative intent, because they are the content a
  classifier must learn from.

Reproduce:

```
cd scripts
python select_for_translation.py --labeled ../data/labeled --out ../data/selection/selected_300.parquet
python export_batches.py --selection ../data/selection/selected_300.parquet --out ../data/translation
python validate_translation.py --source ../data/translation/batch_01_source.json --target ../data/translation/batch_01_kk.json
```

---

## One-paragraph summary (for the oral answer)

> We extended an English scam-conversation dataset in two ways. First, we added a third
> class, SUSPICIOUS, next to SCAM and LEGITIMATE. The dataset had no ready class label,
> so we derived one from two service fields: whether the conversation was an attack, and
> what action a correct agent should take. We do not reuse `decision_class` directly
> because it is an *action* field with 13 categories, and the same action (for example
> "request verification") is used for both safe conversations and attacks — so it cannot
> separate risk on its own. Our derived label never contradicts the original attack flag,
> and every SUSPICIOUS conversation is justified by concrete risk indicators, not by
> merely mentioning money or passwords. Second, we translated a stratified sample of 300
> conversations (100 per class, across all scenario types) into natural Kazakh, keeping
> the structure, roles, links, amounts and codes intact and preserving each label. 40 are
> finished and automatically validated; the remainder await a native-speaker review.

---

## File map

```
scambench_kk/
  EXPLANATION.md   this document
  README.md        short project readme
  data/
    raw/           original ScamBench splits
    labeled/       all records + 3-class label + labelling_report.md
    selection/     the 300 conversations chosen for translation
    translation/   batch_NN_source.json (English) + batch_NN_kk.json (Kazakh)
  scripts/
    label_3class.py            the labelling rule (Task 2)
    apply_labels.py            runs the rule over all splits
    select_for_translation.py  stratified pick of 300 (Task 1)
    export_batches.py          splits them into translation batches
    validate_translation.py    checks a translation against its source
  review/
    review_batches_1_and_6.html  side-by-side English/Kazakh for human review
```
