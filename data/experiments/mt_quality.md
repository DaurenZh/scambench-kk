# How good is the machine translation?

The 300 human translations are the reference. The same English was run
through NLLB-200-distilled-600M with the entity masking used for the
silver training data.

Conversations compared: 300

| Measure | Value |
|---|---:|
| chrF against the human translation | 0.549 |
| LaBSE cosine against the human translation | 0.857 |
| conversations with every entity preserved | 259 / 300 |
| entities dropped in total | 99 |

chrF runs 0 to 1. Machine translation is not expected to reproduce a
human translation word for word, so a middling score is normal and the
number is here to be reported honestly, not to be passed.
