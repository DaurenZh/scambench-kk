# Train English, evaluate Kazakh

Trained on 12,435 English conversations. The 300 translated
conversations were held out together with every restyling of them -- the
augmented pool rewrites the same conversation up to 25 times, and leaving
those in let a character n-gram model reach 0.973 by recognising text it had
already read in another voice.
The two evaluation sets are the same 300 conversations, in English and in
Kazakh, so the gap is the cost of the language change alone.

Two scores per model. The three-class task is the one defined in
`label_3class`; attack-vs-not merges SCAM and SUSPICIOUS, which are
separated only by `decision_class` -- a judgement about what a correct
agent should do, never stated in the conversation. The three-class number
alone understates what the text supports.

| Model | 3-class EN | 3-class KK | gap | binary EN | binary KK | gap |
|---|---:|---:|---:|---:|---:|---:|
| majority class | 0.167 | 0.167 | +0.000 | 0.250 | 0.250 | +0.000 |
| hand features (all 54) | 0.563 | 0.539 | +0.024 | 0.853 | 0.783 | +0.070 |
| hand features (struct+entity, 25) | 0.343 | 0.346 | -0.002 | 0.493 | 0.489 | +0.004 |
| char n-gram TF-IDF (3-5) | 0.736 | 0.491 | +0.245 | 0.942 | 0.632 | +0.310 |
| word TF-IDF (1-2) | 0.690 | 0.416 | +0.274 | 0.953 | 0.553 | +0.399 |
| multilingual encoder (multilingual-e5-small) | 0.522 | 0.462 | +0.059 | 0.805 | 0.626 | +0.180 |
| encoder + hand features (multilingual-e5-small) | 0.528 | 0.495 | +0.034 | 0.831 | 0.735 | +0.096 |
| multilingual encoder (LaBSE) | 0.534 | 0.484 | +0.050 | 0.838 | 0.816 | +0.023 |
| encoder + hand features (LaBSE) | 0.586 | 0.492 | +0.093 | 0.857 | 0.779 | +0.079 |
| hand features + silver KK | 0.528 | 0.564 | -0.037 | 0.821 | 0.838 | -0.016 |
| char n-gram TF-IDF + silver KK | 0.736 | 0.687 | +0.050 | 0.945 | 0.847 | +0.099 |
| word TF-IDF + silver KK | 0.690 | 0.661 | +0.029 | 0.953 | 0.898 | +0.055 |

Macro-F1. With 300 conversations per evaluation set the 95% interval is
roughly +/- 5 points, so small differences between rows are not meaningful;
the large ones are.

## Per-class on Kazakh, best model (char n-gram TF-IDF + silver KK)

```
              precision    recall  f1-score   support

  LEGITIMATE       0.73      0.90      0.81       100
        SCAM       0.71      0.45      0.55       100
  SUSPICIOUS       0.66      0.75      0.70       100

    accuracy                           0.70       300
   macro avg       0.70      0.70      0.69       300
weighted avg       0.70      0.70      0.69       300

```
