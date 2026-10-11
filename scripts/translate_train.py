"""Machine-translate English training conversations into Kazakh (silver data).

Notebook 03 ended with every model scoring far worse on Kazakh than on the same
conversations in English, and with the multilingual encoders failing to close the
gap. The remaining explanation is the simplest one: the model has never seen
Kazakh. Translate-train tests that directly -- machine-translate training data,
train on the mixture, and keep the 300 human translations as the gold test set.

Three things make this cheap enough to run on a laptop.

**Translate distinct strings, not message slots.** The corpus is templated: the
12,435 training conversations contain 131,851 messages but only 38,837 distinct
strings, so 71% of the work is repeats. This is the same observation that drove
`translation_memory.py`, reused.

**Protect the entities.** Machine translation rewrites `$199` as `199 доллар` and
drops codes. Every URL, email, amount, phone number and code is replaced with a
placeholder before translation and restored after, using the same patterns
`validate_translation.py` checks. NLLB leaves the placeholders alone and even
attaches Kazakh case endings to them, which survive restoration.

**Leave long strings in English.** Anything past `--max-chars` is kept as-is
rather than truncated, since a half-translated message is worse than an
untranslated one. The report says what fraction that was.

This is silver data: good enough to train on, not good enough to evaluate on. The
300 human translations remain the only gold Kazakh in the project.

Usage:

    python translate_train.py --merged ../data/merged --features ../data/features \\
        --out ../data/silver --conversations 4000
"""
import argparse
import json
import pathlib
import re
import time

import pandas as pd

from validate_translation import (EMAIL, HEX_ADDRESS, LONG_NUMBER, MONEY, PHONE,
                                  UPPER_CODE, URL)

MODEL = "facebook/nllb-200-distilled-600M"
SEED = 42
PATTERNS = [URL, EMAIL, HEX_ADDRESS, MONEY, PHONE, UPPER_CODE, LONG_NUMBER]
# a shape the tokenizer keeps whole and the translator does not try to translate
PLACEHOLDER = re.compile(r"\s*Z\s*(\d+)\s*Q\s*")


def mask_entities(text):
    """Replace data with placeholders, returning the text and what was removed."""
    found = []

    def take(match):
        found.append(match.group(0))
        return f" Z{len(found) - 1}Q "

    for pattern in PATTERNS:
        text = pattern.sub(take, text)
    return text, found


def restore_entities(text, found):
    def put(match):
        index = int(match.group(1))
        return f" {found[index]} " if index < len(found) else " "

    return re.sub(r"\s+", " ", PLACEHOLDER.sub(put, text)).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conversations", type=int, default=4000)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    import run_crosslingual as rx

    features, merged = rx.load(args.features, args.merged)
    train, _, _ = rx.split_sets(features, merged)

    # stratified by label, so silver data does not skew the class mix
    per_class = args.conversations // 3
    chosen = pd.concat([
        train[train.label_3class == label].sample(
            min(per_class, (train.label_3class == label).sum()), random_state=SEED)
        for label in ["LEGITIMATE", "SUSPICIOUS", "SCAM"]])
    print(f"{len(chosen):,} training conversations "
          f"({dict(chosen.label_3class.value_counts())})")

    by_id = merged.set_index("id")
    conversations = {i: json.loads(by_id.loc[i, "messages"]) for i in chosen.id}

    slots = [m.get("content") or "" for msgs in conversations.values() for m in msgs]
    distinct = sorted({s for s in slots if s.strip()})
    short = [s for s in distinct if len(s) <= args.max_chars]
    covered = sum(1 for s in slots if len(s) <= args.max_chars and s.strip())
    print(f"{len(slots):,} message slots, {len(distinct):,} distinct strings")
    print(f"translating {len(short):,} of them (<= {args.max_chars} chars), "
          f"covering {100 * covered / len(slots):.0f}% of slots")
    print(f"{sum(len(s) for s in short):,} characters\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    kazakh_token = tokenizer.convert_tokens_to_ids("kaz_Cyrl")

    masked = [mask_entities(s) for s in short]
    translations = {}
    start = time.time()
    for begin in range(0, len(short), args.batch_size):
        chunk = masked[begin:begin + args.batch_size]
        with torch.inference_mode():
            encoded = tokenizer([m for m, _ in chunk], return_tensors="pt",
                                padding=True, truncation=True, max_length=256)
            generated = model.generate(**encoded, forced_bos_token_id=kazakh_token,
                                       max_new_tokens=256, num_beams=1)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for source, (_, found), target in zip(short[begin:begin + args.batch_size],
                                              chunk, decoded):
            translations[source] = restore_entities(target, found)

        done = begin + len(chunk)
        if done % (args.batch_size * 20) == 0 or done >= len(short):
            rate = done / (time.time() - start)
            print(f"  {done:,}/{len(short):,}  {rate:.1f} strings/sec  "
                  f"{(len(short) - done) / rate / 60:.0f} min left", flush=True)

    rows = []
    for identifier, messages in conversations.items():
        source = by_id.loc[identifier]
        translated = []
        for message in messages:
            content = message.get("content") or ""
            updated = dict(message)
            updated["content"] = translations.get(content, content)
            translated.append(updated)
        row = source.to_dict()
        row["id"] = f"{identifier}::kk_mt"
        row["language"] = "kk"
        row["messages"] = json.dumps(translated, ensure_ascii=False)
        row["translated_from"] = identifier
        row["translated_fields"] = "messages(machine)"
        rows.append(row)

    silver = pd.DataFrame(rows)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / "silver_kk.parquet", index=False)

    elapsed = (time.time() - start) / 60
    print(f"\n{len(silver):,} silver Kazakh conversations in {elapsed:.0f} min")
    print(f"wrote {out_dir / 'silver_kk.parquet'}")


if __name__ == "__main__":
    main()
