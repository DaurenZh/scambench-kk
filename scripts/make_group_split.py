"""A split that keeps every restyling of a conversation on the same side.

ScamBench's own train/validation/test assignment leaks. An id is
`<pool>::<conversation>::<row>`, and the augmented pool rewrites one conversation
in up to 25 voices; those rewrites are assigned to splits independently, so **86%
of validation rows have a twin of the same conversation sitting in training**. A
model does not have to generalise to score well, it only has to recognise a
sentence it already read in another register.

That is a property of the upstream corpus, not of anything added here, and it is
not fixed by changing our own code -- any score measured against the original
splits is inflated. So this script produces a second assignment, `split_grouped`,
which puts a whole conversation family on one side of the line. The original
`split` column is left untouched, because the merged dataset should keep the
corpus's own splits; results are simply reported against both.

Kazakh rows need no special handling: `base::<conversation>::<row>::kk` shares the
conversation segment with its English source, so a pair lands in the same split
for free, which is the rule merge_kk.py already enforces.

Usage:

    python make_group_split.py --merged ../data/merged --out ../data/splits
"""
import argparse
import collections
import pathlib
import random

import pandas as pd

SPLITS = ["train", "test", "validation"]
TARGET = {"train": 0.80, "validation": 0.10, "test": 0.10}
SEED = 42


def conversation_of(identifier):
    """The family a row belongs to: the middle segment of its id."""
    parts = identifier.split("::")
    return parts[1] if len(parts) > 1 else identifier


def assign(families, labels, rng):
    """Give each family a split, keeping the class mix close to target.

    Families are placed largest first, each going to whichever split is furthest
    behind its quota for that family's dominant class. Greedy, deterministic given
    the seed, and good enough: the point is that a family is never divided.
    """
    order = sorted(families, key=lambda f: (-len(families[f]), f))
    quota = {s: collections.Counter() for s in SPLITS}
    totals = collections.Counter()
    for rows in families.values():
        for row in rows:
            totals[labels[row]] += 1

    wanted = {s: {c: TARGET[s] * n for c, n in totals.items()} for s in SPLITS}
    placement = {}
    for family in order:
        rows = families[family]
        dominant = collections.Counter(labels[r] for r in rows).most_common(1)[0][0]
        deficit = {s: wanted[s][dominant] - quota[s][dominant] for s in SPLITS}
        best = max(SPLITS, key=lambda s: (deficit[s], rng.random()))
        placement[family] = best
        for row in rows:
            quota[best][labels[row]] += 1
    return placement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged = pd.concat([pd.read_parquet(pathlib.Path(args.merged) / f"{s}.parquet")
                        for s in SPLITS], ignore_index=True)
    merged["conversation"] = merged.id.map(conversation_of)

    families = collections.defaultdict(list)
    for row in merged.itertuples():
        families[row.conversation].append(row.id)
    labels = dict(zip(merged.id, merged.label_3class))

    placement = assign(families, labels, random.Random(SEED))
    merged["split_grouped"] = merged.conversation.map(placement)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged[["id", "conversation", "split", "split_grouped"]].to_csv(
        out_dir / "grouped_split.csv", index=False)

    # the property the whole thing exists for
    by_split = merged.groupby("split_grouped").conversation.apply(set)
    for a in SPLITS:
        for b in SPLITS:
            if a < b and by_split[a] & by_split[b]:
                raise SystemExit(f"a conversation family spans {a} and {b}")

    kazakh = merged[merged.language == "kk"]
    source_split = dict(zip(merged.id, merged.split_grouped))
    bad = [r.id for r in kazakh.itertuples()
           if source_split.get(r.translated_from) != r.split_grouped]
    if bad:
        raise SystemExit(f"{len(bad)} Kazakh rows separated from their source")

    print(f"{len(families):,} conversation families -> {len(merged):,} rows\n")
    comparison = pd.DataFrame({
        "original": merged.split.value_counts(),
        "grouped": merged.split_grouped.value_counts(),
    }).reindex(SPLITS)
    print(comparison.to_string())
    print("\nclass balance under the grouped split:")
    print(pd.crosstab(merged.split_grouped, merged.label_3class,
                      normalize="index").round(3).to_string())
    print("\nno conversation family spans two splits")
    print("every Kazakh row shares its source's split")

    leaked_before = merged[merged.split == "validation"].conversation.isin(
        set(merged[merged.split == "train"].conversation)).mean()
    print(f"\nvalidation rows with a training twin -- original: {leaked_before:.0%}, grouped: 0%")
    print(f"wrote {out_dir / 'grouped_split.csv'}")


if __name__ == "__main__":
    main()
