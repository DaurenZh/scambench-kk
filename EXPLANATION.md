# Multilingual Scam Conversation Detection: Kazakh Extension and a Third Class

A step-by-step account of what we did, in plain language, for presentation and defense.

Dataset: ScamBench Training Corpus, https://huggingface.co/datasets/shaw/scambench-training
(37,421 multi-turn conversations, 14 languages, CC-BY-SA-4.0).

We did two things:

1. **Task 2 — a third class.** Extended the labels from two categories to three:
   SCAM, LEGITIMATE, and a new **SUSPICIOUS** class. Applied to all 37,421 records.
2. **Task 1 — Kazakh.** Translated a stratified sample of conversations into natural
   Kazakh. All 300 are finished and validated.

> **Part 1 below is the original account and some of it is now out of date.**
> **[Part 2](#part-2-finishing-the-work) is current** and corrects three claims made
> here: the translation is finished rather than paused, two batches were failing
> validation, and the `decision_class` table just below was computed on the train
> split only.

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

> **These four rows are the train split only, not all 37,421 records.** Quote the
> full-corpus figures in [Part 2](#what-the-state-actually-was) instead. The argument
> is unchanged and slightly stronger there.

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

**Superseded — see [Part 2](#step-4-the-remaining-180-conversations).** All 300
conversations are translated and all fifteen batches pass validation. A
native-speaker fluency review is still outstanding.

A human-readable side-by-side (English vs Kazakh) for the first two batches is in
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

---

# Part 2. Finishing the work

Part 1 above describes the project as it stood with 120 of 300 conversations
translated and the pipeline paused. This part records what changed, and is written
to be defended: every claim below is a number produced by a script in this repo,
and the command that produces it is given.

## What the state actually was

The first thing done was to check the documentation against the data. Three of its
statements were wrong, and all three are corrected here.

| Claim in Part 1 | Reality |
|---|---|
| 40 of 300 translated, batches 1 and 6 | **120** of 300, batches 1, 2, 6, 7, 8, 12 |
| Both finished batches validate | Batches 2 and 7 **failed** validation |
| The `decision_class` table is "measured on the data" | It was measured on the **train split only** |

The third is the one a reviewer is most likely to catch, so it is worth stating
plainly. The table in Part 1 (`request_verification` 2,426 benign / 6,697 attack,
and so on) reproduces exactly on the train split and matches nothing on the full
corpus. Recomputed over all 37,421 records the numbers are:

| `decision_class` | benign | attack |
|---|---:|---:|
| `request_verification` | 3,015 | 8,383 |
| `refuse` | 72 | 6,474 |
| `audit` | 442 | 2,560 |
| `ignore` | 16 | 81 |

The argument is unchanged and in fact slightly stronger: `request_verification`
alone spans 3,015 safe conversations and 8,383 attacks. But the figures quoted
should be these, not the ones in Part 1.

## Step 1. The pipeline is reproducible, and this was verified rather than assumed

`data/raw/`, `data/labeled/*.parquet` and `data/selection/selected_300.parquet` are
all gitignored, and none of them were on disk. Only the scripts, the batch files and
the reports survived. Since the scripts are the project's reproducibility claim, the
first job was to prove they still produce the same thing.

The three source parquet files were downloaded from the Hub and match its published
byte counts exactly (22,448,446 / 3,028,215 / 3,004,060). Then:

```
python scripts/apply_labels.py --raw data/raw --out data/labeled
python scripts/select_for_translation.py --labeled data/labeled --out data/selection/selected_300.parquet
```

Two results are worth stating at the defense:

- The regenerated `labelling_report.md` is **byte-identical** to the committed one
  (`git diff` on it is empty).
- The regenerated selection reproduces **exactly the same 300 conversations**, in
  the same order: the `id` sets are identical, `translation_index` maps to the same
  `id` in every case, and every label matches. So the 300 batch files and the
  selection script agree, and the merge can safely join on `id`.

This also settles a discrepancy: the Hugging Face dataset card says 37,423 rows. The
live dataset has **37,421**, which is what this project has always reported. The card
is wrong, not the project.

## Step 2. Four real bugs in the validator

Batches 2 and 7 were failing on entities like `455157` and `455355`. Neither is a
verification code: in leetspeak they spell **"assist"** and **"assess"**. The Kazakh
renders them as verbs, correctly, and the validator was wrong to fail.

Rather than pass a flag each run, leetspeak detection was made automatic. A run of
digits can only decode using the letters `o i e a s t b g`, which makes the set of
reachable words small and enumerable: `LEET_WORDS` in `validate_translation.py` is
every word of five letters or more in the system dictionary spelled from those
letters, written back as digits — 350 entries, including both offenders. Runs of
four digits or fewer are deliberately left alone, since they fall below every
extractor threshold anyway and a real four-digit code must stay checked.

Three more false-positive classes surfaced while translating and were fixed the same
way — by narrowing what counts as data, never by loosening what must survive:

- `Yes/No`, `Malware/Hacking`, `dividends/share` were being read as file paths. A
  single-slash, dot-free match whose sides are both plain words is prose. `8080/chat`
  and `results/gpt2_heads` are still checked.
- `月付199新辦/攜碼享三重好康` likewise. Paths and URLs in this corpus are ASCII, so a
  slashed match carrying non-ASCII letters is prose.
- `H4r455m3nt/D15cr1m1n4t10n` is the same word pair in leetspeak, so inside a
  leetspeak message the decoded form is tested too.

### The fix that was rejected, and why it matters

Two leetspeak tokens — `P4R1B45` ("PARIBAS") and `PWN3D` ("PWNED") — are uppercase
words that the code pattern matches. The obvious rule is to blank any leetspeak token
that decodes to a word. **That rule was implemented, tested, and thrown away**,
because it also blanked `9QYOJ9`, which decodes to "gqyojg" and is a real secret key
in batch 10. A dictionary does not separate them either: "pwned" is not in the
dictionary. There is no rule here that is safe, so the two tokens are recorded
individually in `data/translation/entity_waivers.json` with the reason written out,
alongside two leetspeak digit runs mistaken for phone numbers — four waivers in
total, each visible and reviewable.

This is the honest answer to "did you just make the validator pass?" **No.** The
opposite: a heuristic that made things pass was deleted because it silently stopped
checking real data. `scripts/test_validate_translation.py` pins the behaviour in both
directions — leetspeak prose is waived, and a dropped code, phone number, link or
amount is still reported. It passes, and so do all fifteen batches.

## Step 3. A translation memory, because the corpus is templated

Across the 300 conversations there are only about 1,500 distinct message strings, and
176 of them occur three times or more. Translating message slots one at a time
therefore repeats work and lets wording drift — and it already had. Measured on the
six finished batches, **66 English strings had picked up more than one Kazakh
rendering**, some as many as four:

```
EN  Ops said this should happen before the next handoff.
KK  Ops бұл келесі тапсырудан бұрын орындалуы керек деді.
KK  Ops мұны келесі тапсырудан бұрын жасалуы керек деді.
KK  Ops мұны келесі тапсырудан бұрын істеу керек деді.
KK  Операциялық бөлім бұл келесі ауысымға дейін орындалуы керек деді.
```

`scripts/translation_memory.py` fixes this. Each distinct string is translated once
and reused, so drift becomes impossible by construction rather than by care.

**The memory is keyed on (string, style_variant), not on the string alone.** This was
a measurement, not a preference: of the 66 conflicts, **51 were between two renderings
of the same string in the same style**, which is drift, while only 15 differed purely
across styles, which may be a deliberate difference of register between
`support_formal` and `discord_casual`. Keying on the pair removes the drift and keeps
the register.

Running `normalize` rewrote 103 messages in the finished batches to one canonical
rendering each. Every choice it had to make is recorded in `memory_conflicts.json`
for review. 3 of those 103 were in the human-reviewed batches 1 and 6; the other 100
were in batches that had never been reviewed.

The effect on the remaining work was large. At the start, 180 conversations and 1,877
message slots were outstanding. **996 of those slots (53%) could be filled from the
memory**, leaving 143k characters of genuinely new text instead of 208k — and each
finished batch shrank the next, until the last batch had only 72 new strings.

## Step 4. The remaining 180 conversations

All fifteen batches are now translated and validated. The final state:

| | Before | After |
|---|---|---|
| Conversations translated | 120 / 300 | **300 / 300** |
| Batches passing validation | 4 of 6 | **15 of 15** |
| Strings drifting *within* a style | 51 | **0** |
| Translation memory | — | 1,906 (string, style) entries |

Be precise about that third row, because the two numbers must be measured the same
way to be comparable. Of the 66 strings that had more than one rendering, **51 were
conflicts within a single style** — that is drift, and it is now 0. Across the
finished 300 there remain **45 strings rendered differently in different styles**,
which is register and is kept on purpose:

```
EN  I'm listening.
  [short_professional] Тыңдап отырмын.
  [slightly_slangy]    Тыңдап тұрмын.
```

The audit notebook reports both numbers side by side so the distinction cannot be
misread, and `translation_memory.py build` reports 0 conflicting (string, style)
pairs.

Structure cannot drift either: `translation_memory.py assemble` copies ids, roles,
speakers and message order from the source file, so a translation can differ from its
source in message text and in nothing else.

Translation decisions, extending the two recorded in Part 1:

- **Data is never translated.** URLs, emails, handles, phone numbers, amounts, hashes,
  code, JSON payloads, matrices and placeholders such as
  `<HASH_OR_ENCRYPTION_KEY_8021>` are kept byte-for-byte, and where a message is prose
  wrapped around a payload the payload is spliced from the source rather than retyped.
  Leetspeak email addresses like `4ndy@3m4t.4t` are kept in their leetspeak form,
  because they are addresses, not words.
- **Injection and jailbreak payloads are translated as prose**, preserving their
  manipulative intent, because that is the content a classifier has to learn. The same
  applies to abusive and harmful-request samples: they carry the labels that make the
  dataset useful.
- **Day and month names are translated; clock times, dates and numerals are not.**
  `Thursday at 2 PM` becomes `бейсенбі күні 2 PM`.

## Step 5. The merged dataset, and the leakage rule

`scripts/merge_kk.py` produces the combined corpus:

```
python scripts/merge_kk.py --labeled data/labeled --translation data/translation --out data/merged
```

**37,721 rows: 37,421 original plus 300 Kazakh**, every row carrying the three-class
label. Full breakdown in `data/merged/merge_report.md`.

The rule that matters here, and the one most likely to be asked about:

> **A Kazakh row inherits the split of its English source.**

A translation is the *same conversation* in another language. If a Kazakh row landed
in a different split than its source, a model could train on the English and be
evaluated on the Kazakh — the score would be inflated and the benchmark would be
worthless. This is not hypothetical: the 300 are spread **233 train / 35 validation /
32 test**, so 67 of them belong outside training. The merge enforces this and then
asserts it, failing loudly if any Kazakh row disagrees with its source's split. It
also fails on duplicate ids. Both checks pass.

Two more properties of the merged rows:

- `language` is `kk`, the id is the source id plus `::kk`, and `translated_from`
  names the English row it came from, so the pairing is never guesswork.
- `translated_fields` records what is actually Kazakh. For all 300 rows it is
  `messages,system_prompt`. `response_text`, `explanation` and `chosen_action` are the
  agent's side of the record and **remain English** — stated on the row rather than
  left to be discovered.

## Step 6. Where the analysis lives

`notebooks/01_dataset_audit.ipynb` holds the exploratory work: the distributions, the
evidence for the third class, the leakage audit, the leetspeak count and the
translation-coverage numbers. It runs end to end against the regenerated data.

The division is deliberate and worth defending: **scripts build, notebooks look.** The
pipeline stays in `scripts/` because its outputs are gitignored and regenerated, which
makes the scripts the only record of how the dataset was built — a reproducibility
claim that a notebook's hidden execution order is the wrong place for. The notebook
imports from `scripts/` and never writes into `data/`.

## Anticipated questions

**"Is the Kazakh actually good?"** Automatically, it is verified to be *structurally*
correct: every message present, every role and speaker unchanged, every link, amount,
code and hash preserved, and Kazakh-specific letters present throughout. Its
*fluency* has **not** been verified by a native speaker. That is the largest open risk
in the project and is listed as such below rather than glossed over.

**"Did the validator get weaker?"** No — see Step 2, including the rule that was
implemented and then deleted for hiding a real secret key. The behaviour is pinned by
tests and the four waivers are individually justified in a tracked file.

**"Is a translation memory just find-and-replace?"** It reuses a rendering only for an
identical source string *in the same style variant*, and the assembler copies all
structure from the source. It cannot merge messages, change a speaker or reorder a
conversation.

**"Why is the leetspeak rendered as plain Kazakh?"** Because the task asks for natural
Kazakh and the obfuscation is already recorded in `style_variant`. The scope of this
is small and was measured: **19 of the 300** conversations are `style_variant =
leetspeak`, not the larger number a naive digit-in-word heuristic suggests. It does
mean the Kazakh subset cannot support style-variant experiments on those 19 rows.

**"Why is harmful content translated at all?"** Because the labels are the point. A
scam detector learns from scam text; a prompt-injection classifier learns from
injection payloads. Every such row keeps its original class label and its risk
indicators.

## Open risks, stated plainly

1. **No native-speaker review.** 300 conversations are translated and structurally
   validated; none has been checked for fluency by a Kazakh speaker. This is the
   blocker Part 1 named, and it is still open.
2. **103 messages were normalised by a mechanical choice** — most frequent rendering,
   ties broken by first seen. Every one is listed in `memory_conflicts.json` and a
   reviewer can overrule them in a single pass.
3. **The Kazakh rows are half-translated by design.** `response_text` and
   `explanation` are still English. This is fine for classification and not fine for
   generation; `translated_fields` makes it explicit.
4. **Four entity waivers are judgement calls.** Each is written out with its reasoning
   in `entity_waivers.json`.
5. **Licence.** The source corpus is CC-BY-SA-4.0, so any release must carry
   attribution and the same licence.

## Reproducing all of it

```
python -m venv .venv && .venv/bin/pip install pandas pyarrow
for s in train test validation; do
  curl -sL -o data/raw/$s.parquet \
    "https://huggingface.co/datasets/shaw/scambench-training/resolve/main/data/$s/$s-00000-of-00001.parquet"
done
python scripts/apply_labels.py          --raw ../data/raw --out ../data/labeled
python scripts/select_for_translation.py --labeled ../data/labeled --out ../data/selection/selected_300.parquet
python scripts/export_batches.py         --selection ../data/selection/selected_300.parquet --out ../data/translation
python scripts/translation_memory.py build
python scripts/test_validate_translation.py
python scripts/merge_kk.py --labeled ../data/labeled --translation ../data/translation --out ../data/merged
```
