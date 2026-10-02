"""Tests for the translation validator, focused on the leetspeak/data distinction.

The validator must stay loud about real data and quiet about leetspeak prose. A
false pass is the dangerous direction: it silently accepts a translation that
dropped a verification code. Run with:

    python test_validate_translation.py
"""
import sys

from validate_translation import check_entities, is_leet_message, mask_leet_prose

# (description, source, translation, entity kinds that MUST be reported)
CASES = [
    # leetspeak words are prose: the Kazakh renders them as verbs, so nothing is missing
    ("leet 'assist' (455157) rendered as a Kazakh verb",
     "H1, w0uld y0u b3 4bl3 70 455157 w17h 50m37h1ng?",
     "Сәлем, бір нәрсеге көмектесе аласыз ба?", set()),
    ("leet 'assess' (455355) rendered as a Kazakh verb",
     "5h4r3 th3 d3t41l5 4nd 1 w1ll 455355 th3 r3qu35t c4r3fully.",
     "Мәліметтерді жіберіңіз, мен сұранысты мұқият бағалаймын.", set()),

    # real data inside a leetspeak message must still be required to survive
    ("phone number dropped from a leetspeak message",
     "Pl3453 c4ll +1 555 019 4477 r1ght n0w", "Қазір қоңырау шалыңыз", {"phone"}),
    ("account number dropped from a leetspeak message",
     "s3nd 1t t0 4cc0unt 18908758 r1ght n0w pl3453", "Қазір жіберіңіз", {"number"}),
    ("link and amount dropped from a leetspeak message",
     "53nd $1000 70 http://bad.link/x n0w, c0d3 is 887421",
     "Ақша жіберіңіз", {"url", "money", "number"}),

    # ordinary messages are untouched by the leetspeak handling
    ("six-digit code dropped from an ordinary message",
     "Your verification code is 455157.", "Растау кодыңыз.", {"number", "code"}),
    ("six-digit code kept in an ordinary message",
     "Your verification code is 455157.", "Растау кодыңыз 455157.", set()),
    ("amount kept with Kazakh spacing",
     "Transfer $1,200 today.", "Бүгін $1,200 аударыңыз.", set()),
]

# messages that must not be treated as leetspeak at all
NOT_LEET = [
    "Your verification code is 455157.",
    "Transfer $1,200 to account 18908758 today.",
    "[session] actor=Ari Young handle=ariyoung verified_admin=false",
]


def main():
    failures = []

    for name, source, target, required in CASES:
        reported = {item.split(":")[0] for item in check_entities(source, target)}
        if required - reported:
            failures.append(f"{name}: expected {sorted(required - reported)} to be reported, "
                            f"got {sorted(reported)}")
        if not required and reported:
            failures.append(f"{name}: expected a clean pass, got {sorted(reported)}")

    for text in NOT_LEET:
        if is_leet_message(text):
            failures.append(f"treated as leetspeak but is not: {text!r}")
        if mask_leet_prose(text) != text:
            failures.append(f"ordinary message was masked: {text!r}")

    print(f"{len(CASES)} entity cases, {len(NOT_LEET)} non-leetspeak cases")
    if failures:
        print(f"\n{len(failures)} failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("OK: leetspeak prose is waived, real data is still required")
    return 0


if __name__ == "__main__":
    sys.exit(main())
