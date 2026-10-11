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

---

# Part 3. Features, and what to present

## Step 7. Feature engineering

`scripts/build_features.py` produces the feature table; `notebooks/02_feature_engineering.ipynb`
establishes why it is built that way and checks the result. 54 features over 37,721
rows, in `data/features/`.

### The leakage audit is the whole story

`label_3class` was **derived** from `decision_class` and `should_trigger_scam_defense`.
Measured on the merged corpus, that pair determines the label **99.99%** of the time.
So those columns do not predict the label — they are the label, rewritten. The same
holds for everything produced by or alongside the annotation.

| Column | Determines the label |
|---|---:|
| `label_rule` | 100.0% |
| `diagnostic_labels` | 96.3% |
| `unsafe_signals` | 92.1% |
| `decision_class` | 90.5% |
| `chosen_action` | 90.5% |
| `risk_indicators` | 81.1% |
| `source_dataset` | 78.9% |
| `scenario_category` | 73.4% |

**13 of 25 columns are excluded.** The cut is made on provenance, not on a threshold:
`needs_review` scores only 49.7% and is still excluded, because our own labelling rule
sets it. The question is always "could a system know this at runtime, before the label
existed?" `build_features.py` fails if any excluded column reaches the feature table.

What remains is what a detector actually sees: the conversation, plus `language`,
`style_variant` and `source_pool`.

### The features

| Prefix | Count | What it measures |
|---|---:|---|
| `struct_` | 8 | message count, roles, who speaks last |
| `entity_` | 17 | links, emails, handles, phones, amounts, codes |
| `surface_` | 13 | lengths, case, digits, punctuation, leetspeak |
| `lex_` | 16 | urgency, credentials, payment, authority, secrecy, instruction override, verification pushback |

The keyword lexicon is **bilingual**. English matches on word boundaries; Kazakh is
agglutinative, so it matches on stems — `раста` covers растау, растаңыз, расталды.

### What the numbers say

Signal check on the validation split, macro-F1:

| | macro-F1 |
|---|---:|
| Majority class | 0.222 |
| Text features only | 0.783 |
| Text features + `decision_class` (leaked) | 0.986 |

**Both of those numbers are inflated, and Part 4 explains why.** Measured on a split
that never divides a conversation family they become 0.221 / **0.709** / 0.962. The
argument is unchanged — the last row is still the label being read back — but quote
0.709, not 0.783.

Top features by mutual information are `surface_max_msg_chars`, `struct_last_is_user`,
`lex_families_present` and `struct_n_messages`. **No `entity_` feature reaches the top
ranks** — whether a conversation contains a link or an amount separates the classes far
less than how it is shaped. Scam and legitimate conversations both contain URLs and money.

### What the 300 Kazakh pairs made measurable

Each of the 300 exists in both languages, so "does this feature survive translation?"
has an answer rather than an assumption:

| Group | Pairs identical |
|---|---:|
| `struct_` | 300 / 300 |
| `entity_` | 292 / 300 |
| `surface_` | 0 / 300 |
| `lex_` | 39 / 300 |

`entity_` started at 286/300, and the gap was treated as a bug rather than rounded away.
It was two defects in our own feature code, pointing in opposite directions:

- leetspeak digits counted as entities in the English — `455157` ("assist") read as a
  six-digit code — so leetspeak rows looked like they carried data they did not. Fixed
  by blanking leetspeak before counting, the same way the validator does;
- Kazakh morphology created fake codes — `$6370-тен` makes `6370-` match the code
  pattern. Fixed by requiring a code to contain a letter.

The residual **8 are genuine**: English `1920s` has no word boundary, Kazakh
`1920 жылдар` does. They were left rather than tuned away.

`surface_` scoring 0/300 is not a failure — those features measure the writing system,
so they are usable within a language and never across one. And an English-only lexicon
run over both languages scores 54 hits on the English rows against 8 on the Kazakh, with
52 of 60 Kazakh rows scoring zero. It does not error; it quietly reports that Kazakh scam
conversations contain no urgency and no payment demands.

---

## How to present this

### Running order (about 10 minutes)

1. **The problem** — ScamBench has no SCAM/LEGITIMATE column. There was nothing to copy.
2. **Task 2, the third class** — derived from two service fields. Show the
   `request_verification` row: 3,015 safe conversations and 8,383 attacks share one
   action, so `decision_class` cannot be the label.
3. **Task 1, Kazakh** — 300 conversations, 100 per class, stratified; structure, roles,
   links, amounts and codes all preserved and automatically verified.
4. **The finishing work** — reproducibility proven, four validator bugs fixed, drift
   eliminated, merged with the split-leakage rule enforced.
5. **Features** — the leakage audit, then the three-row score table.
6. **What is not done** — say it before you are asked.

### The two slides that carry the argument

- **`decision_class` benign-vs-attack** (notebook 01): one action, both risk levels.
  This justifies the entire third-class task.
- **The three-row macro-F1 table** (notebook 02): 0.221 / 0.709 / 0.962. This justifies
  the entire leakage audit, in three numbers.

### Questions to expect

**"Why not just use `decision_class`?"** It is an action, not a diagnosis. The same
action is prescribed to safe and malicious conversations. Analogy: it is the
prescription, not the diagnosis — "run more tests" is given to healthy and sick patients
alike.

**"Isn't dropping 13 columns throwing away information?"** It is throwing away the
answer. Those columns exist because an annotator already decided what the conversation
was.

**"Why not use `scenario_category`?"** It is the attack type — an annotation. A detector
at runtime never sees it.

**"Did you weaken the validator to make batches pass?"** The opposite. A rule that made
things pass was written and deleted because it also silently stopped checking `9QYOJ9`, a
real secret key. Four entities are waived individually, each with its reason recorded.

**"How do you know the Kazakh is any good?"** Structurally, it is verified: every
message, role, speaker, link, amount and code. For fluency, it has not been checked by a
native speaker — that is the project's main open risk and it is listed first.

**"Why hand-built features instead of a transformer?"** A transformer is the obvious next
step and would likely be stronger. Hand-built features are interpretable, which is what
let us catch two measurement bugs an embedding would have hidden.

### Say these before you are asked

1. No native-speaker review of the Kazakh.
2. The classes are imbalanced (50 / 29 / 21), so macro-F1, never accuracy.
3. The Kazakh rows are 300 of 37,721 — under 1%. They are an evaluation slice, not
   training material. The experiment they enable is train on English, evaluate zero-shot
   on the Kazakh pairs.
4. A table in Part 1 was computed on the train split only; Part 2 has the corrected
   full-corpus figures.

---

# Part 4. Train English, evaluate Kazakh — and two contaminations found doing it

## Step 8. The experiment the Kazakh subset was built for

The 300 translated conversations exist in both languages with identical structure and
identical entities. The gap between a model's score on the English 300 and the Kazakh
300 is therefore the cost of the language change and nothing else — same
conversations, same labels, one variable. `scripts/run_crosslingual.py`;
`notebooks/03_crosslingual_baseline.ipynb`.

### Two contaminations had to be removed first

Both were found by disbelieving a good-looking number, and both are worth presenting.

**1. The split rule puts English twins in training.** Each Kazakh row inherits its
source's split — right for the merged dataset, wrong for this experiment. 233 of the
300 English twins sit in the training split, so training on all English rows means the
model has read the English version of most conversations it is then tested on in
Kazakh. That is not zero-shot.

**2. Restylings are worse, and they were nearly missed.** Holding out the 300 source
rows is not enough. An id is `<pool>::<conversation>::<row>`, and the augmented pool
rewrites one conversation in up to 25 voices. **3,473 training rows are restylings of
273 of the 300**, a median of 12 each. With those left in, a character n-gram model
scored **0.973** on the "held-out" English — by recognising text it had already read in
another register. Removing them dropped it to 0.736, and dropped the hand-feature model
from 0.858 to 0.563.

The fix is to hold out the whole conversation family. Training set: 12,435 English
conversations.

### Results

| Model | EN-300 | KK-300 | gap |
|---|---:|---:|---:|
| majority class | 0.167 | 0.167 | 0.000 |
| hand features (all 54) | 0.563 | **0.539** | +0.024 |
| hand features (struct+entity, 25) | 0.343 | 0.346 | −0.002 |
| char n-gram TF-IDF (3-5) | **0.736** | 0.491 | +0.245 |
| word TF-IDF (1-2) | 0.690 | 0.416 | +0.274 |

**The feature grouping from notebook 02 was a prediction, and it held.** The
translation-invariant groups transfer with no measurable loss (−0.002); the lexical
models lose a quarter of their score crossing the script boundary.

**There is a real trade between accuracy and transferability.** Char n-grams are the
best English model and among the worst Kazakh ones. The hand features are worse in
English and better in Kazakh. On the language this project is about, the interpretable
language-aware features win.

**Perfect transfer of a weak signal is not a solution.** The invariant-only model
transfers flawlessly and scores 0.343 against a 0.167 baseline.

**SCAM is where it breaks.** Per-class on Kazakh: LEGITIMATE F1 0.72, SUSPICIOUS 0.58,
**SCAM 0.31 with recall 0.24**. The model finds roughly one Kazakh scam in four. That
is the single most useful number in the project, and it is the one that would matter if
any of this were deployed.

## Step 9. ScamBench's own splits leak

This came out of contamination 2, and it is bigger than our experiment.

The restyling problem is not confined to the 300. Across the corpus, **981 of 3,963
conversation families span more than one split**, which means **93% of validation rows
have a twin of the same conversation sitting in training**. A model does not have to
generalise to score well on the official splits; it only has to recognise a sentence it
already read in another voice.

This is a property of the published corpus, not of anything built here, and it means
**every score measured against ScamBench's own splits is inflated — including ours.**

`scripts/make_group_split.py` produces a second assignment, `split_grouped`, that keeps
a conversation family on one side of the line. The original `split` column is left
untouched, because the merged dataset should keep the corpus's own splits; results are
reported against both.

| | original split | grouped split |
|---|---:|---:|
| majority class | 0.222 | 0.221 |
| text features only | **0.783** | **0.709** |
| + `decision_class` (leaked) | 0.986 | 0.962 |

The grouped split has the same sizes (30,176 / 3,773 / 3,772), identical class balance
across all three (49.7 / 20.9 / 29.4), no family spanning two splits, and every Kazakh
row still with its source. Validation rows with a training twin: **93% → 0%**.

**Quote 0.709, not 0.783.**

## What this adds to the defence

One more slide, and it is a strong one: *"we found that the dataset we were given has
leaking splits, measured it at 93%, and supplied a fixed split."* That is a
contribution to the corpus, not just a use of it.

Two more questions to expect:

**"Your Kazakh scores are low — isn't that a failure?"** 0.539 against a 0.167 baseline
is far better than chance and nowhere near usable. These are deliberately simple models
on hand-built features. The honest conclusion is that the task is not solved, not that
it is unsolvable — and we now have the baseline a better model has to beat.

**"Can you tell these models apart?"** Mostly no, and we say so. With 300 conversations
the 95% interval is about ±5.6 points, so the 0.245 and 0.274 gaps are real and the
difference between 0.539 and 0.491 is not. That is the argument for more Kazakh data,
stated as a measurement rather than a wish.

### Next, in order of value

1. ~~**A multilingual encoder**~~ — **tried, and it lost. See Step 11.**
2. **More Kazakh** — `scripts/select_expanded.py` corrects the eligible pool and
   shows the corpus supports **326 more conversations per class (978 total, 3.3x the
   current 300)** without any external data. Worth doing *after* step 1, because its
   value is narrowing a comparison and there is not yet a comparison worth narrowing.
3. **Translate-train** — machine-translate English training data into Kazakh, keep the
   300 human translations as the gold test set. The grouped split makes that safe to
   measure.

## Step 10. The pool is bigger than the original selection assumed

`select_for_translation.py` drew only from the `base` pool, reasoning that the rest is
style-variant augmentation whose translation would yield near-duplicate Kazakh. Split
by conversation family, that reasoning is half right:

| | Conversations |
|---|---:|
| `base`, distinct | 1,166 |
| augmented, a restyling of a base conversation | 13,899 — rightly excluded |
| augmented, not tied to any base conversation | 5,067 — **wrongly excluded** |

The 5,067 come from 2,954 distinct generators and are not restylings of anything.
`scripts/select_expanded.py` includes them, while enforcing two rules the original did
not need: one conversation family is picked at most once, and families already
translated are excluded entirely.

**Available: 326 more per class, 978 total — 3.3x the current 300.**

A number worth correcting out loud, because an earlier estimate of 747 per class was
wrong in two ways: it counted several voices of the same conversation as separate
candidates (747 → 422) and it did not exclude what was already translated (422 → 326).

The gain is not only volume. The expansion reaches **seven scenario categories the
original 300 never touched** — `cascading-failure`, `extended-multi-turn`,
`format-recovery`, `hitl-bypass`, `inter-agent-communication`, `state-persistence`,
`verbose-context-injection` — because the `base` pool does not contain them at all. A
first 300 are selected and exported to `data/translation_v2/`, ready to translate.

## Step 11. The multilingual encoder was tried, and it lost

The plan written at the end of Step 9 was that a multilingual encoder would beat char
n-grams, since char n-grams cannot cross a script boundary by construction. That
prediction was tested with two encoders and was wrong.

| Model | EN-300 | KK-300 | gap |
|---|---:|---:|---:|
| hand features (all 54) | 0.563 | **0.539** | +0.024 |
| LaBSE + hand features | 0.586 | 0.492 | +0.093 |
| LaBSE | 0.534 | 0.484 | +0.050 |
| multilingual-e5-small | 0.522 | 0.462 | +0.059 |
| char n-gram TF-IDF | **0.736** | 0.491 | +0.245 |

**The encoders do what they were brought in to do.** A conversation's own translation
is its nearest neighbour 265/300 of the time under e5-small and 286/300 under LaBSE,
and their gaps are a third of the lexical models'. The cross-lingual capability is
real and measurable.

**It just does not pay for itself.** Neither beats 54 hand-built numbers on Kazakh.

**And the obvious explanation is only half right**, which is worth saying because it
is the first thing a reviewer will suggest. Under e5-small two *unrelated*
conversations sit at cosine 0.85 — it compresses the whole corpus into a narrow cone,
which does explain that model's weakness. But LaBSE's space is healthy at 0.54 and it
gains barely two points. Narrow spread is a defect in the small model, not the reason
the method falls short. A probe sweep (C from 1 to 256, LinearSVC, gradient boosting)
moved nothing, so it is not underfitting either.

The better reading: a frozen sentence embedding summarises *what a conversation is
about*, and every conversation in this corpus is about the same thing — a user asking
an assistant for something. What separates SCAM from LEGITIMATE is a turn of phrase, a
request slipped into a system prompt, who speaks last. Mean-pooling 512 tokens into one
vector throws exactly that away.

**This is a result about frozen embeddings with a linear probe, not about
transformers.** Fine-tuning XLM-R end-to-end is the version with a real chance; it
needs a GPU and was not run.

### What to say about it

This is the strongest methodological moment in the project and should be presented as
one: *we wrote down a prediction, tested it with two models, were wrong, ruled out the
two obvious explanations, and said so.* A result that contradicts the plan is worth
more than one that confirms it, because nobody has to wonder whether it was fitted
after the fact.

The revised ordering:

1. **Fine-tune, do not freeze** — needs a GPU.
2. **Translate-train** — machine-translate English training data into Kazakh, keep the
   300 human translations as the gold test set. Cheap, and it attacks the real problem:
   the model has never seen Kazakh.
3. **More Kazakh** (326 per class available) — once there is a comparison worth
   resolving. At ±5.6 points the top Kazakh scores are currently tied.
