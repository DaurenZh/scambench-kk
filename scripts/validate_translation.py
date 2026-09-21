"""Check a translated batch against its source batch.

Catches the failure modes that matter for this dataset: a dropped or merged
message, a changed speaker, a mangled URL or amount, or a line left in English.
"""
import argparse
import json
import pathlib
import re

URL = re.compile(r"https?://[^\s\"'<>)\]]+|(?<![\w@.])(?:[a-z0-9-]+\.)+(?:com|net|org|io|co|ru|kz|me|info|app|xyz|link|site|top|dev|ai|gov|edu)(?:/[^\s\"'<>)\]]*)?", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
HEX_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")
HANDLE = re.compile(r"(?<![\w])@[A-Za-z0-9_]{3,}")
PHONE = re.compile(r"\+?\d[\d\-\s().]{7,}\d")
MONEY = re.compile(r"[$€£₸¥]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:USD|EUR|GBP|KZT|RUB|BTC|ETH|SOL|USDT|тг|₸)\b", re.I)
LONG_NUMBER = re.compile(r"\b\d{4,}\b")
UPPER_CODE = re.compile(r"\b(?=[A-Z0-9-]{5,}\b)(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b")
PATHY = re.compile(r"(?:[\w.-]+/){1,}[\w.-]+|\b[A-Z_]{4,}=[^\s]+|\$[A-Z_]{3,}")

EXTRACTORS = {
    "url": URL,
    "email": EMAIL,
    "hex_address": HEX_ADDRESS,
    "handle": HANDLE,
    "phone": PHONE,
    "money": MONEY,
    "number": LONG_NUMBER,
    "code": UPPER_CODE,
    "path": PATHY,
}

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
KAZAKH_SPECIFIC = set("әғқңөұүһі")
LATIN_WORD = re.compile(r"\b[A-Za-z]{4,}\b")


def is_data_payload(text):
    """True for JSON blobs, hashes and similar machine content that has no prose."""
    stripped = text.strip()
    if stripped.startswith(("{", "[")) and ('":' in stripped or "': " in stripped):
        return True
    return bool(re.fullmatch(r"[\s\w:,.\-\"'{}\[\]/+=]*0x[0-9a-fA-F]{20,}[\s\S]*", stripped))


def entities(text):
    found = {}
    for name, pattern in EXTRACTORS.items():
        values = {m.group(0).strip().rstrip(".,;:!?") for m in pattern.finditer(text)}
        if values:
            found[name] = values
    return found


def normalise_number(value):
    return value.replace(",", "").replace(" ", "").replace("-", "").replace("(", "").replace(")", "")


def check_entities(source_text, target_text):
    """Return a list of entities present in the source but missing from the translation."""
    missing = []
    for name, values in entities(source_text).items():
        for value in values:
            if value in target_text:
                continue
            # digits may be regrouped by spacing, so compare digits only
            if name in {"phone", "number", "money"}:
                digits = normalise_number(value)
                if digits and digits in normalise_number(target_text):
                    continue
            missing.append(f"{name}:{value}")
    return missing


def check_conversation(source, target):
    problems = []
    tag = f"#{source['translation_index']} ({source['source_id'][:28]})"

    if target.get("source_id") != source["source_id"]:
        problems.append(f"{tag}: source_id mismatch")
        return problems

    src_messages = source["messages"]
    tgt_messages = target.get("messages", [])
    if len(src_messages) != len(tgt_messages):
        problems.append(f"{tag}: {len(src_messages)} messages in source, {len(tgt_messages)} in translation")
        return problems

    if ("system_prompt" in source) != ("system_prompt" in target):
        problems.append(f"{tag}: system_prompt present in one file only")

    for src, tgt in zip(src_messages, tgt_messages):
        where = f"{tag} msg[{src['i']}]"
        if src["role"] != tgt.get("role"):
            problems.append(f"{where}: role changed {src['role']} -> {tgt.get('role')}")
        if src["speaker"] != tgt.get("speaker"):
            problems.append(f"{where}: speaker changed {src['speaker']!r} -> {tgt.get('speaker')!r}")

        content = (tgt.get("content") or "").strip()
        if not content:
            problems.append(f"{where}: empty translation")
            continue

        for item in check_entities(src["content"], content):
            problems.append(f"{where}: entity missing from translation -> {item}")

        # a message with real words but no Cyrillic was probably left untranslated;
        # machine payloads carry no prose to translate, so they are exempt
        if (not CYRILLIC.search(content)
                and len(LATIN_WORD.findall(src["content"])) >= 3
                and not is_data_payload(src["content"])):
            problems.append(f"{where}: no Cyrillic text, looks untranslated")

    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    source = json.loads(pathlib.Path(args.source).read_text(encoding="utf-8"))
    target = json.loads(pathlib.Path(args.target).read_text(encoding="utf-8"))

    if len(source) != len(target):
        print(f"FAIL: {len(source)} conversations in source, {len(target)} in translation")
        raise SystemExit(1)

    problems = []
    for src, tgt in zip(source, target):
        problems.extend(check_conversation(src, tgt))

    text = " ".join(m["content"] for conv in target for m in conv["messages"])
    kazakh_letters = sorted(set(text.lower()) & KAZAKH_SPECIFIC)

    print(f"conversations checked: {len(source)}")
    print(f"Kazakh-specific letters seen: {''.join(kazakh_letters) or 'NONE'}")

    if problems:
        print(f"\n{len(problems)} problems:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("OK: structure, speakers and entities all preserved")


if __name__ == "__main__":
    main()
