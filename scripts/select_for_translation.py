"""Choose the English conversations that will be translated into Kazakh.

Only the `base` pool is eligible. The rest of the corpus is style-variant
augmentation of those same conversations, so translating it would produce near
duplicates in Kazakh.
"""
import argparse
import json
import pathlib

import pandas as pd

SEED = 42
PER_CLASS = 100
# conversations longer than this are dropped: a handful run past 20k characters and
# would dominate the set without adding scenario variety
MAX_CHARS = 3500


def conversation_chars(raw_messages):
    return sum(len(m["content"]) for m in json.loads(raw_messages))


def conversation_turns(raw_messages):
    return len(json.loads(raw_messages))


def pick_for_class(pool, n, seed):
    """Spread the quota across scenario categories in proportion to the pool."""
    quotas = (pool.scenario_category.value_counts(normalize=True) * n).round().astype(int)

    chosen = []
    for category, quota in quotas.items():
        if quota <= 0:
            continue
        candidates = pool[pool.scenario_category == category]
        chosen.append(candidates.sample(min(quota, len(candidates)), random_state=seed))

    picked = pd.concat(chosen) if chosen else pool.head(0)

    # rounding can leave the quota short or long, so top up or trim
    if len(picked) < n:
        remaining = pool.drop(index=picked.index)
        shortfall = min(n - len(picked), len(remaining))
        picked = pd.concat([picked, remaining.sample(shortfall, random_state=seed)])
    elif len(picked) > n:
        picked = picked.sample(n, random_state=seed)

    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    args = parser.parse_args()

    labeled_dir = pathlib.Path(args.labeled)
    frames = []
    for split in ["train", "test", "validation"]:
        frames.append(pd.read_parquet(labeled_dir / f"{split}.parquet"))
    df = pd.concat(frames, ignore_index=True)

    pool = df[(df.source_pool == "base") & (df.language == "en")].copy()
    pool["n_turns"] = pool.messages.apply(conversation_turns)
    pool["n_chars"] = pool.messages.apply(conversation_chars)
    pool = pool[pool.n_chars <= MAX_CHARS]
    pool = pool.drop_duplicates(subset="messages")

    print(f"eligible pool: {len(pool)} conversations")
    print(pool.label_3class.value_counts().to_string())

    selected = []
    for label in ["SCAM", "SUSPICIOUS", "LEGITIMATE"]:
        subset = pool[pool.label_3class == label]
        picked = pick_for_class(subset, args.per_class, SEED)
        picked = picked.assign(selection_label=label)
        selected.append(picked)
        print(f"{label}: selected {len(picked)} of {len(subset)} available")

    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(["label_3class", "scenario_category", "id"]).reset_index(drop=True)
    result["translation_index"] = range(1, len(result) + 1)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)

    print()
    print(f"total selected: {len(result)} -> {out_path}")
    print(pd.crosstab(result.scenario_category, result.label_3class).to_string())
    print()
    print(f"total characters to translate: {int(result.n_chars.sum()):,}")
    print(f"turns per conversation: median {int(result.n_turns.median())}, max {int(result.n_turns.max())}")


if __name__ == "__main__":
    main()
