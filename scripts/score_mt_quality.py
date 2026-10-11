"""How good is the machine translation? The 300 human translations can say.

Silver data is only worth training on if it is recognisably Kazakh and keeps the
meaning. Most projects have to take that on faith. This one does not: the same 300
conversations exist in a careful human translation, so the machine can be run over
the same English and the two compared directly.

Three measures, none of which needs a reference metric library:

- **chrF**, character n-gram F-score, the standard choice for morphologically rich
  languages where BLEU's word matching is too brittle for Kazakh;
- **embedding similarity** against the human translation, using the LaBSE vectors
  already cached;
- **entity survival**, reusing `validate_translation.py` -- the same check the human
  translations had to pass.

Usage:

    python score_mt_quality.py --merged ../data/merged --out ../data/experiments
"""
import argparse
import collections
import json
import pathlib
import re

import numpy as np
import pandas as pd

import translate_train as tt
from validate_translation import check_entities

MODEL = "facebook/nllb-200-distilled-600M"
WORD = re.compile(r"\w+", re.UNICODE)


def char_ngrams(text, n):
    text = re.sub(r"\s+", " ", text.strip())
    return collections.Counter(text[i:i + n] for i in range(len(text) - n + 1))


def chrf(hypothesis, reference, max_n=6, beta=2.0):
    """Character n-gram F-score, averaged over orders 1..max_n."""
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        hyp, ref = char_ngrams(hypothesis, n), char_ngrams(reference, n)
        overlap = sum((hyp & ref).values())
        precisions.append(overlap / max(sum(hyp.values()), 1))
        recalls.append(overlap / max(sum(ref.values()), 1))
    precision, recall = np.mean(precisions), np.mean(recalls)
    if precision + recall == 0:
        return 0.0
    return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--embeddings", default="../data/embeddings")
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    merged = pd.concat([pd.read_parquet(pathlib.Path(args.merged) / f"{s}.parquet")
                        for s in ["train", "test", "validation"]], ignore_index=True)
    by_id = merged.set_index("id")
    kazakh = merged[merged.language == "kk"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    kazakh_token = tokenizer.convert_tokens_to_ids("kaz_Cyrl")

    rows = []
    pending = []           # (conversation index, message index, masked text, entities)
    machine = {}

    for position, row in enumerate(kazakh.itertuples()):
        english = json.loads(by_id.loc[row.translated_from, "messages"])
        for index, message in enumerate(english):
            content = message.get("content") or ""
            if content.strip() and len(content) <= 600:
                masked, found = tt.mask_entities(content)
                pending.append((position, index, masked, found))

    print(f"translating {len(pending):,} messages from the 300 gold conversations")
    for begin in range(0, len(pending), args.batch_size):
        chunk = pending[begin:begin + args.batch_size]
        with torch.inference_mode():
            encoded = tokenizer([m for _, _, m, _ in chunk], return_tensors="pt",
                                padding=True, truncation=True, max_length=256)
            generated = model.generate(**encoded, forced_bos_token_id=kazakh_token,
                                       max_new_tokens=256, num_beams=1)
        for (position, index, _, found), decoded in zip(
                chunk, tokenizer.batch_decode(generated, skip_special_tokens=True)):
            machine[(position, index)] = tt.restore_entities(decoded, found)
        if begin % (args.batch_size * 40) == 0:
            print(f"  {begin:,}/{len(pending):,}", flush=True)

    import encode_corpus
    cached = encode_corpus.load_cached(args.embeddings, "sentence-transformers/LaBSE")

    for position, row in enumerate(kazakh.itertuples()):
        english = json.loads(by_id.loc[row.translated_from, "messages"])
        human = json.loads(row.messages)
        mt_messages, hu_messages, en_messages = [], [], []
        for index, (source, gold) in enumerate(zip(english, human)):
            if (position, index) in machine:
                mt_messages.append(machine[(position, index)])
                hu_messages.append(gold.get("content") or "")
                en_messages.append(source.get("content") or "")
        if not mt_messages:
            continue
        mt_text, hu_text = "\n".join(mt_messages), "\n".join(hu_messages)
        missing = sum(len(check_entities(e, m)) for e, m in zip(en_messages, mt_messages))
        rows.append({
            "id": row.id,
            "chrf_vs_human": chrf(mt_text, hu_text),
            "entities_missing": missing,
            "mt_text": mt_text,
            "human_text": hu_text,
        })

    scores = pd.DataFrame(rows)

    if cached is not None:
        import numpy as np
        human_vectors = np.vstack(cached.loc[kazakh.id].to_numpy())
        print("\n(embedding similarity uses cached LaBSE vectors for the human side only;"
              "\n machine side is encoded fresh)")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/LaBSE")
        mt_vectors = encoder.encode(scores.mt_text.tolist(), normalize_embeddings=True,
                                    batch_size=16, show_progress_bar=False)
        index = {identifier: i for i, identifier in enumerate(kazakh.id)}
        scores["cosine_vs_human"] = [
            float(mt_vectors[i] @ human_vectors[index[row.id]])
            for i, row in enumerate(scores.itertuples())]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.drop(columns=["mt_text", "human_text"]).to_csv(
        out_dir / "mt_quality.csv", index=False)

    lines = ["# How good is the machine translation?", "",
             "The 300 human translations are the reference. The same English was run",
             "through NLLB-200-distilled-600M with the entity masking used for the",
             "silver training data.", "",
             f"Conversations compared: {len(scores)}", "",
             "| Measure | Value |", "|---|---:|",
             f"| chrF against the human translation | {scores.chrf_vs_human.mean():.3f} |"]
    if "cosine_vs_human" in scores:
        lines.append(f"| LaBSE cosine against the human translation "
                     f"| {scores.cosine_vs_human.mean():.3f} |")
    lines += [f"| conversations with every entity preserved "
              f"| {(scores.entities_missing == 0).sum()} / {len(scores)} |",
              f"| entities dropped in total | {int(scores.entities_missing.sum())} |", "",
              "chrF runs 0 to 1. Machine translation is not expected to reproduce a",
              "human translation word for word, so a middling score is normal and the",
              "number is here to be reported honestly, not to be passed.", ""]
    (out_dir / "mt_quality.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nchrF vs human            {scores.chrf_vs_human.mean():.3f}")
    if "cosine_vs_human" in scores:
        print(f"LaBSE cosine vs human    {scores.cosine_vs_human.mean():.3f}")
    print(f"entities fully preserved {(scores.entities_missing == 0).sum()}/{len(scores)}")
    print(f"wrote {out_dir / 'mt_quality.md'}")


if __name__ == "__main__":
    main()
