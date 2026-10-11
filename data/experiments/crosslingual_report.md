# Train English, evaluate Kazakh

Trained on 12,435 English conversations. The 300 translated
conversations were held out together with every restyling of them -- the
augmented pool rewrites the same conversation up to 25 times, and leaving
those in let a character n-gram model reach 0.973 by recognising text it had
already read in another voice.
The two evaluation sets are the same 300 conversations, in English and in
Kazakh, so the gap is the cost of the language change alone.

| Model | EN-300 | KK-300 | gap |
|---|---:|---:|---:|
| majority class | 0.167 | 0.167 | +0.000 |
| hand features (all 54) | 0.563 | 0.539 | +0.024 |
| hand features (struct+entity, 25) | 0.343 | 0.346 | -0.002 |
| char n-gram TF-IDF (3-5) | 0.736 | 0.491 | +0.245 |
| word TF-IDF (1-2) | 0.690 | 0.416 | +0.274 |
| multilingual encoder (multilingual-e5-small) | 0.522 | 0.462 | +0.059 |
| encoder + hand features (multilingual-e5-small) | 0.528 | 0.495 | +0.034 |
| multilingual encoder (LaBSE) | 0.534 | 0.484 | +0.050 |
| encoder + hand features (LaBSE) | 0.586 | 0.492 | +0.093 |

Macro-F1. With 300 conversations per evaluation set the 95% interval is
roughly +/- 5 points, so small differences between rows are not meaningful;
the large ones are.

## Per-class on Kazakh, best model (hand features (all 54))

```
              precision    recall  f1-score   support

  LEGITIMATE       0.67      0.78      0.72       100
        SCAM       0.44      0.24      0.31       100
  SUSPICIOUS       0.52      0.67      0.58       100

    accuracy                           0.56       300
   macro avg       0.54      0.56      0.54       300
weighted avg       0.54      0.56      0.54       300

```
