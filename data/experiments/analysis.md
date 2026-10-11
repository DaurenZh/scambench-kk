# Are the gaps real, and where does it fail?

EN-300 and KK-300 are the same 300 conversations, so the comparison is
paired and McNemar's test applies. Confidence intervals are a percentile
bootstrap over conversations (2,000 resamples), not a normal
approximation.

## Is the English/Kazakh gap real?

| Model | EN macro-F1 (95% CI) | KK macro-F1 (95% CI) | right in EN only | right in KK only | McNemar p |
|---|---|---|---:|---:|---:|
| hand features | 0.563 (0.517–0.612) | 0.539 (0.481–0.592) | 44 | 21 | 0.0059 |
| char n-gram | 0.736 (0.685–0.785) | 0.491 (0.434–0.545) | 84 | 11 | 4.5e-15 |

## Hand features versus char n-grams, on Kazakh

Notebook 03 called this one a tie on a +/- 5.6 point interval. Paired, it
is a sharper question: of the conversations where the two models differ,
how lopsided is it?

- hand features right, char n-grams wrong: **65**
- char n-grams right, hand features wrong: **51**
- McNemar p = **0.227**

so this evaluation set still cannot separate them.

## Where the Kazakh loss concentrates

`lost` counts conversations the model gets right in English and wrong in
Kazakh -- the translation cost, conversation by conversation.

### By class

| class | n | EN acc | KK acc | lost |
|---|---:|---:|---:|---:|
| LEGITIMATE | 100 | 0.97 | 0.78 | 20 |
| SCAM | 100 | 0.11 | 0.24 | 5 |
| SUSPICIOUS | 100 | 0.84 | 0.67 | 19 |

### By scenario category

| scenario category | n | EN acc | KK acc | lost |
|---|---:|---:|---:|---:|
| social-engineering | 117 | 0.51 | 0.50 | 11 |
| legitimate | 100 | 0.97 | 0.78 | 20 |
| prompt-injection | 45 | 0.31 | 0.29 | 7 |
| phishing-link | 18 | 0.50 | 0.50 | 3 |
| research-assisted | 18 | 0.56 | 0.44 | 3 |

### By style variant

| style variant | n | EN acc | KK acc | lost |
|---|---:|---:|---:|---:|
| support_formal | 46 | 0.61 | 0.52 | 8 |
| slightly_slangy | 44 | 0.50 | 0.36 | 7 |
| internet | 37 | 0.76 | 0.62 | 7 |
| plain | 36 | 0.83 | 0.67 | 7 |
| discord_casual | 35 | 0.60 | 0.66 | 2 |
| short_professional | 30 | 0.77 | 0.73 | 2 |
| operator_brief | 29 | 0.62 | 0.48 | 6 |
| broken_english | 24 | 0.58 | 0.54 | 3 |
| leetspeak | 19 | 0.42 | 0.53 | 2 |

### By conversation length

| length | n | EN acc | KK acc | lost |
|---|---:|---:|---:|---:|
| shortest | 75 | 0.71 | 0.64 | 10 |
| short | 76 | 0.61 | 0.61 | 7 |
| long | 74 | 0.62 | 0.54 | 11 |
| longest | 75 | 0.63 | 0.47 | 16 |

## The missed Kazakh scams

76 of 100 SCAM conversations are misclassified in Kazakh. What the model called them instead:

| predicted | n |
|---|---:|
| SUSPICIOUS | 56 |
| LEGITIMATE | 20 |

Of those 76, **5** are classified correctly in English, so the translation is what lost them; the remaining 71 the model cannot do in either language.


## Which boundary is hard?

SCAM and SUSPICIOUS both carry `should_trigger_scam_defense = True`. What
separates them is `decision_class`: refuse / escalate / block against
request_verification / audit. That is a judgement about what a correct
agent should **do**, and the conversation never states it. Splitting the
task shows how much that costs.

| Task | EN | KK |
|---|---:|---:|
| 3-class, as defined | 0.563 | 0.539 |
| attack vs not (SCAM and SUSPICIOUS merged) | **0.879** | **0.764** |
| SCAM vs SUSPICIOUS alone | 0.537 | 0.578 |

Chance on the last row is 0.500.

Read together: the text supports attack detection well, and barely
supports the SCAM/SUSPICIOUS split at all. The three-class score is
dominated by a boundary a text-only model has almost no evidence for,
which is a property of how the label was defined rather than of the
features. It does not invalidate the label -- it still refines the binary
flag and every SUSPICIOUS row carries named risk indicators -- but any
three-class number should be reported next to the binary one.
