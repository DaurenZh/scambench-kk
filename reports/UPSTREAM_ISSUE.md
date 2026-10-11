# Train/validation/test splits share conversations

Thanks for publishing ScamBench — the three-class Kazakh work we are doing would
not be possible without it. We hit something while building an evaluation and
think you would want to know.

## What we found

A `base` row id has the form `base::<conversation hash>::<row>`, and the
augmented pool contains several stylistic rewrites of the same conversation — up
to 76 of them. Those rewrites are assigned to splits independently of
one another.

The result is that most of validation and test has a near-twin in training:

| | Share with a same-conversation twin in `train` |
|---|---:|
| validation rows | **83%** |
| test rows | **83%** |

967 of 7,382 conversation families (13%) span more than one split.

(Worth noting if you reproduce this: the middle segment is a conversation hash
only for `base`-derived rows. A generated row is `generated::<generator name>::<row>`,
where the middle is a label such as `exfil` shared by 730 unrelated conversations.
Grouping on it without checking merges those into one family and overstates the
problem — we made exactly that mistake first and measured 93% instead of 83%.)

## Why it matters

A model does not have to generalise to score well; recognising a sentence it
already read in another register is enough. On our task — 3-class risk
classification, gradient boosting over 54 hand-built features:

| Split | macro-F1 |
|---|---:|
| as published | 0.783 |
| conversation-grouped | **0.650** |

Roughly 13 points of our score was memorisation. We assume the same holds for
anyone else benchmarking on these splits.

## Reproducing it

```python
import re, pandas as pd

HASH = re.compile(r"^[0-9a-f]{16}$")

def family(identifier):
    parts = identifier.split("::")
    return parts[1] if len(parts) > 1 and HASH.match(parts[1]) else identifier

frames = []
for split in ["train", "validation", "test"]:
    df = pd.read_parquet(f"data/{split}/{split}-00000-of-00001.parquet")
    df["split"] = split
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
df["family"] = df.id.map(family)

train = set(df.loc[df.split == "train", "family"])
print((df.loc[df.split == "validation", "family"].isin(train)).mean())
```

## Suggested fix

Assign whole conversation families to a split rather than individual rows. A
greedy stratified assignment keeps your split sizes and class balance almost
exactly:

| | published | grouped |
|---|---:|---:|
| train | 29,951 | 30,176 |
| validation | 3,736 | 3,773 |
| test | 3,734 | 3,772 |

Class balance is identical across all three splits to within 0.1pp, and no family
spans two splits. Implementation:
https://github.com/DaurenZh/scambench-kk/blob/main/scripts/make_group_split.py

Happy to open a PR adding a `split_grouped` column if that would help.

