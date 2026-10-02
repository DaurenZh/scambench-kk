# Merged dataset

Total rows: 37,721 (37,421 original + 300 Kazakh)

| Split | Original | Kazakh | Total |
|---|---:|---:|---:|
| train | 29,951 | 233 | 30,184 |
| test | 3,734 | 32 | 3,766 |
| validation | 3,736 | 35 | 3,771 |

## Kazakh rows by class

| Class | Rows |
|---|---:|
| LEGITIMATE | 100 |
| SUSPICIOUS | 100 |
| SCAM | 100 |

## Fields translated on the Kazakh rows

| Fields | Rows |
|---|---:|
| `messages,system_prompt` | 300 |

Every Kazakh row keeps its source's split, so the same conversation never
appears in training in one language and in evaluation in the other.
`translated_from` names the English row each Kazakh row came from.
