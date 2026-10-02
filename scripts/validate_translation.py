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
# `Yes/No`, `Malware/Hacking`, `dividends/share`: prose written with a slash, not a path.
# Only a single-slash, dot-free match whose sides are both plain words is dropped, so
# `8080/chat` and `results/gpt2_heads` are still checked.
WORD_PAIR = re.compile(r"^[A-Za-z][A-Za-z'-]*/[A-Za-z][A-Za-z'-]*$")
# Paths and URLs in this corpus are ASCII, so a match carrying non-ASCII letters is prose
# that happens to use a slash as a separator, e.g. the Chinese `月付199新辦/攜碼享三重好康`.
NON_ASCII = re.compile(r"[^\x00-\x7f]")


def is_prose_slash(value, leet=False):
    """True when a slashed match is prose rather than a path.

    In a leetspeak message the same check is applied to the decoded form, so
    `H4r455m3nt/D15cr1m1n4t10n` is recognised as the word pair it spells. The
    decoding is only consulted for leetspeak messages: outside one, `8080/chat`
    would decode to a word pair and wrongly stop being checked.
    """
    if WORD_PAIR.match(value) or NON_ASCII.search(value):
        return True
    return bool(leet and WORD_PAIR.match(value.lower().translate(LEET_MAP)))

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


# Leetspeak prose, not data. In a leetspeak message a run of digits can be a word:
# `455157` is "assist" and `455355` is "assess". Such a run must not be mistaken for a
# verification code, so it is blanked out before entities are extracted.
#
# A digit run can only decode using the letters o i e a s t b g, which makes the set of
# reachable words small and enumerable. This list is every word of five letters or more
# in /usr/share/dict/words spelled only from those letters, written back as digits.
# Runs of four digits or fewer are left in place: they fall below every extractor
# threshold anyway, so nothing is lost, and a real four-digit code still gets checked.
LEET_WORDS = frozenset((
    "005739173", "04515", "04535", "05493", "05537", "0570515",
    "0571473", "0571715", "05730515", "05731715", "070515", "071053",
    "071715", "07473", "080157", "08353", "085355", "087357",
    "1503735", "15053157", "1505745157", "1508453", "154715", "1547473",
    "1549093", "155173", "15531", "170157", "18074", "305473",
    "309434", "315393515", "3451357", "347493", "35513", "357473",
    "390157", "3907157", "39357", "393574", "40734", "40735",
    "4173515", "435711", "43915", "451714", "45515", "455153",
    "455157", "455355", "45535533", "45537", "455375", "4553941",
    "45541", "455473", "4554941", "455844", "457173", "4573470515",
    "4574514", "458357", "45835705", "4583570515", "470554", "47384",
    "474839", "477357", "4810515", "48135", "48137473", "4817181",
    "4819347", "484514", "48453", "484591", "484715", "48473",
    "48517", "48807", "48813", "488355", "48845", "488451",
    "4884551", "49093", "49107493", "49157", "49174", "4917473",
    "49473", "49913", "501353773", "50514", "50713", "5077493",
    "508317", "50937", "513574", "51393", "515500", "51710",
    "51740", "517733", "51774", "53153", "533533", "53393",
    "534833", "53514", "53571", "535737", "537053", "537180",
    "53743", "537733", "538417", "538473", "54173", "54194",
    "5471473", "54807", "54807493", "54814", "5484157", "548847",
    "54884714", "548847157", "5491774", "549177473", "54941", "549413",
    "57007", "570093", "57047", "57055", "570913", "57094",
    "57173", "57181473", "573470515", "57347173", "57410", "57414",
    "57448", "574515", "574535", "5747157", "57473", "57493",
    "5749353", "5749913", "70153", "703703", "70457", "7045733",
    "704704", "707713", "708145", "709473", "709474", "7170157",
    "717135", "717713", "717817", "71814", "718143", "71837",
    "718813", "719749", "73357", "73453", "735715", "73573",
    "735733", "735735", "73574", "7357473", "7357474", "73837",
    "738837", "740157", "740741", "74153", "74194", "7437514",
    "745500", "745513", "74553", "745537", "7455490", "74573",
    "74713", "74735", "747700", "747700157", "747700493", "74774",
    "747838", "74800", "74800157", "748007", "74809", "74835",
    "74837", "7493735", "749454573", "74947053", "75355383", "753753",
    "80053", "80057", "800733", "800735", "80075", "800801513",
    "80157", "80457", "804713", "8047493", "805537", "8055493",
    "8080713", "808813", "809074", "80913", "810515", "81053",
    "8107173", "81074", "81480", "81537053", "81571", "8157473",
    "815833173", "81715", "817713", "81810", "81907", "8199357",
    "83154", "83193", "83357", "83457", "8345713", "834733",
    "83474", "834743", "835007", "83507", "8351393", "83517",
    "83533", "83537", "83551", "835513", "835748", "837055",
    "83715", "837153", "83749", "83750", "83774", "838055",
    "83809", "838173", "8383457", "838417", "8384573", "83847",
    "83908", "839085", "83937", "83947", "839914704", "840848",
    "84505", "845073", "845094", "8451473", "84515", "84535",
    "84550", "845513", "845514", "8455157", "845537", "845537173",
    "84553774", "84554", "84570", "8457173", "84573", "84574",
    "84715", "8471573", "84734", "847474", "8474745", "84774",
    "84800", "848007", "848157", "848173", "8483514", "8483514515",
    "84841", "848813", "8488177", "8488177355", "849080", "8494553",
    "849913", "849917", "8499493", "90053", "90053909", "903714",
    "903743", "904733", "90810", "90814", "90883", "908837",
    "90994", "9177173", "9188053", "91881", "918837", "91885173",
    "91907", "919917", "9308105", "93353", "93357", "93453",
    "934745", "93550", "93573", "9357473", "93743", "938813",
    "939933", "9414554", "945849", "9473493", "948948", "9493173",
    "94933", "949473",
))

LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                         "5": "s", "7": "t", "8": "b", "9": "g"})
LEET_ADJACENCY = re.compile(r"[A-Za-z][0-9]|[0-9][A-Za-z]")
DIGIT_RUN = re.compile(r"(?<![A-Za-z0-9])[0-9]{5,}(?![A-Za-z0-9])")
# a letter, so that a blanked word cannot let a phone or number pattern span the gap
LEET_PLACEHOLDER = "x"
TOKEN = re.compile(r"[A-Za-z0-9]+")
VOWELS = set("aeiou")


def is_leet_message(text):
    """True when a message is written in leetspeak rather than merely containing digits.

    Substitution puts digits directly against letters inside words. Real codes and
    amounts are separated by spaces or punctuation, so they do not produce this signal.
    """
    return len(LEET_ADJACENCY.findall(text)) >= 3


def mask_leet_prose(text):
    """Blank digit runs that are leetspeak words, so they are not read as data.

    Only bare runs of digits are handled, and only those in LEET_WORDS. Deciding by
    shape instead was tried and rejected: `PWN3D` decodes to "pwned" and should be
    blanked, but `9QYOJ9` decodes to "gqyojg" and is a real secret key, and no rule
    separated them -- "pwned" is not in the dictionary either. Tokens like those are
    recorded in entity_waivers.json one at a time, where the choice stays visible.
    """
    if not is_leet_message(text):
        return text
    return DIGIT_RUN.sub(
        lambda m: LEET_PLACEHOLDER if m.group(0) in LEET_WORDS else m.group(0), text)


def is_data_payload(text):
    """True for JSON blobs, hashes and similar machine content that has no prose."""
    stripped = text.strip()
    if stripped.startswith(("{", "[")) and ('":' in stripped or "': " in stripped):
        return True
    return bool(re.fullmatch(r"[\s\w:,.\-\"'{}\[\]/+=]*0x[0-9a-fA-F]{20,}[\s\S]*", stripped))


def entities(text, leet=False):
    found = {}
    for name, pattern in EXTRACTORS.items():
        values = {m.group(0).strip().rstrip(".,;:!?") for m in pattern.finditer(text)}
        if name == "path":
            values = {v for v in values if not is_prose_slash(v, leet)}
        if values:
            found[name] = values
    return found


def normalise_number(value):
    return value.replace(",", "").replace(" ", "").replace("-", "").replace("(", "").replace(")", "")


def check_entities(source_text, target_text, waived=()):
    """Return a list of entities present in the source but missing from the translation."""
    missing = []
    for name, values in entities(mask_leet_prose(source_text), is_leet_message(source_text)).items():
        for value in values:
            if value in waived:
                continue
            if value in target_text:
                continue
            # digits may be regrouped by spacing, so compare digits only
            if name in {"phone", "number", "money"}:
                digits = normalise_number(value)
                if digits and digits in normalise_number(target_text):
                    continue
            missing.append(f"{name}:{value}")
    return missing


def check_conversation(source, target, waived=()):
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

        for item in check_entities(src["content"], content, waived):
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
    parser.add_argument(
        "--waive-entity", action="append", default=[],
        help="entity value that is not real data and need not survive translation "
             "(e.g. a leetspeak word like 455355 = 'assess', which is prose, not a code)")
    parser.add_argument(
        "--waivers",
        help="JSON file of recorded waivers; defaults to entity_waivers.json next to "
             "the source batch. Recording them in the repo keeps a run reproducible, "
             "instead of depending on flags someone has to remember.")
    args = parser.parse_args()

    source = json.loads(pathlib.Path(args.source).read_text(encoding="utf-8"))
    target = json.loads(pathlib.Path(args.target).read_text(encoding="utf-8"))

    if len(source) != len(target):
        print(f"FAIL: {len(source)} conversations in source, {len(target)} in translation")
        raise SystemExit(1)

    waivers_path = pathlib.Path(args.waivers) if args.waivers else \
        pathlib.Path(args.source).with_name("entity_waivers.json")
    recorded = []
    if waivers_path.exists():
        recorded = json.loads(waivers_path.read_text(encoding="utf-8"))
    waived = frozenset(args.waive_entity) | {entry["value"] for entry in recorded}
    problems = []
    for src, tgt in zip(source, target):
        problems.extend(check_conversation(src, tgt, waived))

    text = " ".join(m["content"] for conv in target for m in conv["messages"])
    kazakh_letters = sorted(set(text.lower()) & KAZAKH_SPECIFIC)

    print(f"conversations checked: {len(source)}")
    if recorded:
        print(f"recorded waivers applied: {len(recorded)} (from {waivers_path.name})")
    print(f"Kazakh-specific letters seen: {''.join(kazakh_letters) or 'NONE'}")

    if problems:
        print(f"\n{len(problems)} problems:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("OK: structure, speakers and entities all preserved")


if __name__ == "__main__":
    main()
