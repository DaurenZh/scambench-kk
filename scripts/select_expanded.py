"""Choose the next conversations to translate, from a corrected eligible pool.

`select_for_translation.py` drew only from the `base` pool, on the reasoning that
the rest of the corpus is style-variant augmentation and translating it would
produce near-duplicate Kazakh rows. That reasoning is half right, and the half
that is wrong costs a lot of data.

Splitting the augmented pool by conversation family -- the middle segment of an
id -- shows what is actually in it:

    base, distinct conversations            1,166
    augmented, restyling of a base conv    13,899   near-duplicates, rightly excluded
    augmented, not tied to any base conv    5,067   genuinely different conversations

So the near-duplicate worry is real for 13,899 rows and groundless for 5,067. Those
5,067 come from 2,954 distinct generators (`advancefeefraud`, `exfil`,
`adminoverride`, ...) and are not restylings of anything already translated.

Including them raises the ceiling from 154 conversations per class -- the `base` pool
cannot even supply 300 of SCAM -- to **326 per class, 978 in total**, once each family
is counted once and the 300 already translated are removed.

Two rules this script keeps that the original did not have to:

- **one conversation family is picked at most once**, so the new material can never
  be a restyling of itself or of the 300 already done;
- **families already translated are excluded entirely**, not just the 300 rows.

Usage:

    python select_expanded.py --merged ../data/merged --out ../data/selection \\
        --per-class 100
"""
import argparse
import json
import pathlib

import pandas as pd

SPLITS = ["train", "test", "validation"]
SEED = 42
MAX_CHARS = 3500


def conversation_of(identifier):
    parts = identifier.split("::")
    return parts[1] if len(parts) > 1 else identifier


def eligible_pool(merged):
    """Distinct English conversations that are not restylings of each other."""
    english = merged[merged.language == "en"].copy()
    english["conversation"] = english.id.map(conversation_of)

    base_families = set(english.loc[english.source_pool == "base", "conversation"])
    is_restyle = (english.source_pool != "base") & english.conversation.isin(base_families)
    pool = english[~is_restyle].copy()

    pool["n_chars"] = pool.messages.map(
        lambda m: sum(len(x["content"]) for x in json.loads(m)))
    pool = pool[pool.n_chars <= MAX_CHARS]
    pool = pool.drop_duplicates(subset="messages")
    # one row per conversation family: never two voices of the same conversation
    pool = pool.sort_values("id").drop_duplicates(subset="conversation")
    return pool


def pick(pool, n, seed):
    """Spread the quota across scenario categories in proportion to the pool."""
    quotas = (pool.scenario_category.value_counts(normalize=True) * n).round().astype(int)
    chosen = []
    for category, quota in quotas.items():
        if quota <= 0:
            continue
        candidates = pool[pool.scenario_category == category]
        chosen.append(candidates.sample(min(quota, len(candidates)), random_state=seed))
    picked = pd.concat(chosen) if chosen else pool.head(0)

    if len(picked) < n:
        remaining = pool.drop(index=picked.index)
        picked = pd.concat([picked, remaining.sample(min(n - len(picked), len(remaining)),
                                                     random_state=seed)])
    elif len(picked) > n:
        picked = picked.sample(n, random_state=seed)
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    merged = pd.concat([pd.read_parquet(pathlib.Path(args.merged) / f"{s}.parquet")
                        for s in SPLITS], ignore_index=True)

    already = set(merged.loc[merged.language == "kk", "translated_from"].map(conversation_of))
    pool = eligible_pool(merged)
    pool = pool[~pool.conversation.isin(already)]

    counts = pool.label_3class.value_counts()
    print(f"eligible, not yet translated: {len(pool):,} conversations")
    for label in ["LEGITIMATE", "SUSPICIOUS", "SCAM"]:
        print(f"  {label:<12} {counts.get(label, 0):>5}")
    ceiling = counts.min()
    print(f"  -> ceiling for a balanced set: {ceiling} per class ({3 * ceiling:,} total)")

    if args.per_class > ceiling:
        raise SystemExit(f"\nasked for {args.per_class} per class, only {ceiling} available")

    selected = []
    for label in ["SCAM", "SUSPICIOUS", "LEGITIMATE"]:
        picked = pick(pool[pool.label_3class == label], args.per_class, SEED)
        selected.append(picked)
        print(f"\n{label}: selected {len(picked)}")

    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(["label_3class", "scenario_category", "id"]).reset_index(drop=True)
    result["translation_index"] = range(301, 301 + len(result))

    assert result.conversation.nunique() == len(result), "a conversation family picked twice"
    assert not (set(result.conversation) & already), "overlaps work already done"

    out = pathlib.Path(args.out) / f"expanded_{len(result)}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.drop(columns=["conversation"]).to_parquet(out, index=False)

    print(f"\n{len(result)} conversations -> {out}")
    print(f"characters to translate: {int(result.n_chars.sum()):,}")
    print(pd.crosstab(result.scenario_category, result.label_3class).to_string())
    print("\nnext: python export_batches.py --selection "
          f"{out} --out ../data/translation_v2")


if __name__ == "__main__":
    main()
