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
| hand features + silver KK | 0.528 | 0.564 | -0.037 |
| char n-gram TF-IDF + silver KK | 0.736 | 0.687 | +0.050 |
| word TF-IDF + silver KK | 0.690 | 0.661 | +0.029 |

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
