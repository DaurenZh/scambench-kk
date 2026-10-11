"""Embed every conversation with a multilingual sentence encoder, and cache it.

Notebook 03 showed the lexical models losing a quarter of their score crossing
into Kazakh, which is what character n-grams must do: a Cyrillic string shares
almost no character 3-grams with its Latin source. The repair is a representation
built to put a sentence and its translation in the same place, so this script
produces one and caches it for the experiment to reuse.

Default model is `intfloat/multilingual-e5-small` -- 118M parameters, 100
languages including Kazakh, and small enough to run on CPU in minutes. `LaBSE` is
the heavier, better-known alternative and works with `--model sentence-
transformers/LaBSE`.

The e5 family expects a `query: ` prefix on each input; it is applied here.

Conversations are truncated at the model's 512-token limit. A conversation of
3,500 characters is roughly 900 tokens, so the longest ones lose their tail --
worth knowing before reading too much into a score.

Usage:

    python encode_corpus.py --merged ../data/merged --out ../data/embeddings
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

SPLITS = ["train", "test", "validation"]
DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def conversation_text(messages):
    return "\n".join(x.get("content") or "" for x in json.loads(messages))


def cache_path(out_dir, model_name):
    return pathlib.Path(out_dir) / f"{model_name.replace('/', '__')}.npz"


def load_cached(out_dir, model_name):
    """ids -> embedding matrix, or None if not computed yet."""
    path = cache_path(out_dir, model_name)
    if not path.exists():
        return None
    cached = np.load(path, allow_pickle=True)
    return pd.Series(list(cached["vectors"]), index=cached["ids"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--experiment-only", action="store_true",
                        help="encode only the rows run_crosslingual.py needs: the English "
                             "training pool and the 600 paired evaluation rows. A heavier "
                             "model on CPU does not need to embed all 14 languages to "
                             "answer this question.")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    merged = pd.concat([pd.read_parquet(pathlib.Path(args.merged) / f"{s}.parquet")
                        for s in SPLITS], ignore_index=True)
    merged["text"] = merged.messages.map(conversation_text)

    if args.experiment_only:
        kazakh_ids = merged.loc[merged.language == "kk", "id"]
        sources = set(merged.loc[merged.language == "kk", "translated_from"])
        conversation = merged.id.str.split("::").str[1]
        held = set(conversation[merged.id.isin(sources)])
        keep = (merged.id.isin(kazakh_ids) | merged.id.isin(sources)
                | ((merged.language == "en") & (merged.split == "train")
                   & (~conversation.isin(held))))
        merged = merged[keep].reset_index(drop=True)
        print(f"encoding {len(merged):,} rows (experiment subset)")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(out_dir, args.model)
    if path.exists():
        print(f"{path} already exists; delete it to re-encode")
        return

    print(f"loading {args.model}")
    model = SentenceTransformer(args.model)
    prefix = "query: " if "e5" in args.model else ""

    start = time.time()
    vectors = model.encode([prefix + t for t in merged.text],
                           batch_size=args.batch_size,
                           normalize_embeddings=True,
                           show_progress_bar=True)
    elapsed = time.time() - start

    np.savez_compressed(path, ids=merged.id.to_numpy(), vectors=vectors.astype(np.float32))
    print(f"\n{len(vectors):,} conversations, {vectors.shape[1]} dimensions, "
          f"{elapsed / 60:.1f} min")
    print(f"wrote {path}")

    # the property the whole thing depends on: a translation lands near its source
    kazakh = merged[merged.language == "kk"]
    index = pd.Series(range(len(merged)), index=merged.id)
    kk_vectors = vectors[index.loc[kazakh.id].to_numpy()]
    en_vectors = vectors[index.loc[kazakh.translated_from].to_numpy()]
    similarity = en_vectors @ kk_vectors.T
    paired = np.diag(similarity)
    rank_one = (similarity > paired[:, None]).sum(1) == 0
    print(f"\nEN/KK alignment on the 300 pairs:")
    print(f"  cosine with own translation : {paired.mean():.3f}")
    print(f"  cosine with a random one    : "
          f"{similarity[~np.eye(len(similarity), dtype=bool)].mean():.3f}")
    print(f"  translation retrieved first : {rank_one.sum()}/{len(rank_one)}")


if __name__ == "__main__":
    main()
