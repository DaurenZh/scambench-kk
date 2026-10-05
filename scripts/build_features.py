"""Build features for the three-class task, from conversation text only.

The hard constraint here is leakage. `label_3class` was derived from
`decision_class` and `should_trigger_scam_defense`, and that pair determines the
label 99.99% of the time. Anything downstream of the annotation -- the decision,
the chosen action, the unsafe signals, the diagnostic labels, the risk
indicators, the rule that fired -- reproduces the label rather than predicting
it. A model given those columns scores perfectly and has learned nothing.

So the only honest inputs are what a detector would actually see at runtime: the
conversation itself, plus metadata that carries no annotation. `LEAKING_COLUMNS`
names what is excluded and why; `build_report` prints the measurement.

Features are grouped by whether they survive translation, because this corpus has
300 conversations in both English and Kazakh and that question can be answered
rather than assumed:

- `struct_*` and `entity_*` are language-independent by construction. The
  translation preserved message count, roles and every URL, amount and code, so
  these should be identical across a pair.
- `surface_*` depends on the writing system, so Latin-vs-Cyrillic shifts them.
- `lex_*` counts keyword families and only works in a language it has words for,
  which is why the lexicon is bilingual. Monolingual lexicons silently score
  zero on the other language, and the notebook shows that happening.

Usage:

    python build_features.py --merged ../data/merged --out ../data/features
"""
import argparse
import json
import pathlib
import re

import pandas as pd

from validate_translation import (EMAIL, HANDLE, HEX_ADDRESS, LONG_NUMBER, MONEY, PHONE,
                                  UPPER_CODE, URL, is_leet_message, mask_leet_prose)

SPLITS = ["train", "test", "validation"]

# Columns that reproduce the label instead of predicting it. The share in brackets
# is how often knowing that column alone pins the label down, measured on the
# merged corpus; see build_report().
LEAKING_COLUMNS = {
    "label_rule": "names the rule that produced the label (100%)",
    "decision_class": "half of the pair the label was derived from (91%)",
    "should_trigger_scam_defense": "the other half; the pair gives 99.99%",
    "chosen_action": "the action the decision implies (91%)",
    "unsafe_signals": "annotation the risk indicators were read from (92%)",
    "diagnostic_labels": "annotation the risk indicators were read from (96%)",
    "risk_indicators": "derived by our own labelling rule (81%)",
    "needs_review": "set by the labelling rule",
    "scenario_category": "the attack type, an annotation rather than an observation (73%)",
    "response_text": "the agent's reply, written knowing the right answer",
    "explanation": "the rationale for the label",
    "reasoning_trace": "chain of thought toward the labelled decision",
    "source_dataset": "provenance; which corpus a row came from predicts its label (79%)",
}

# Keyword families, in both languages the corpus now contains. English matches on
# word boundaries; Kazakh is agglutinative, so it matches on stems -- "раста"
# covers растау, растаңыз, расталды.
LEXICON = {
    "urgency": {
        "en": ["urgent", "urgently", "immediately", "right away", "asap", "hurry",
               "deadline", "expires", "expiring", "last chance", "act fast", "act now",
               "time-sensitive", "don't delay", "no time"],
        "kk": ["шұғыл", "дереу", "жедел", "асығ", "кідірме", "мерзім", "тез"],
    },
    "credential": {
        "en": ["password", "verification code", "one-time code", "otp", "2fa", "mfa",
               "pin code", "social security", "ssn", "credentials", "seed phrase",
               "private key", "security code", "cvv"],
        "kk": ["құпия сөз", "растау коды", "парол", "кілт", "жеке код", "cvv",
               "әлеуметтік сақтандыру", "тіркелгі дерек"],
    },
    "payment": {
        "en": ["payment", "transfer", "wire", "invoice", "deposit", "fee", "refund",
               "bitcoin", "crypto", "gift card", "prepaid", "billing"],
        "kk": ["төле", "аудар", "шот", "депозит", "комиссия", "қайтар", "криптовалюта",
               "сыйлық картас", "алдын ала төлен", "ақша"],
    },
    "authority": {
        "en": ["officer", "department", "support team", "administrator", "admin",
               "bank", "police", "irs", "customer support", "security team"],
        "kk": ["қызметкер", "бөлім", "қолдау қызмет", "әкімші", "банк", "полиция",
               "басқарма", "қауіпсіздік қызмет"],
    },
    "secrecy": {
        "en": ["don't tell", "do not tell", "confidential", "between us",
               "do not share", "keep this secret", "never reveal"],
        "kk": ["ешкімге айтпа", "құпия", "бөліспе", "ешқашан ашпа"],
    },
    "instruction_override": {
        "en": ["ignore previous", "ignore all previous", "disregard", "forget all",
               "forget about", "new instructions", "system override", "you are now",
               "unrestricted mode", "developer mode"],
        "kk": ["алдыңғы нұсқаулар", "ұмыт", "елеме", "жаңа тапсырма", "жаңа нұсқаулар",
               "жүйені айналып", "енді сіз", "шектеусіз режим"],
    },
    "verification_pushback": {
        "en": ["verify your identity", "trusted channel", "independent verification",
               "cannot proceed without", "additional verification", "proper verification"],
        "kk": ["жеке басыңызды раста", "сенімді арна", "тәуелсіз раста",
               "қосымша раста", "тиісті раста"],
    },
}

EN_PATTERNS = {family: re.compile("|".join(rf"\b{re.escape(t)}" for t in terms["en"]), re.I)
               for family, terms in LEXICON.items()}
KK_PATTERNS = {family: re.compile("|".join(re.escape(t) for t in terms["kk"]), re.I)
               for family, terms in LEXICON.items()}

ENTITY_PATTERNS = {
    "url": URL, "email": EMAIL, "handle": HANDLE, "phone": PHONE,
    "money": MONEY, "code": UPPER_CODE, "hex": HEX_ADDRESS, "long_number": LONG_NUMBER,
}

WORD = re.compile(r"\w+", re.UNICODE)
SENTENCE_END = re.compile(r"[.!?]")


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def conversation_features(messages, system_prompt=""):
    """Features for one conversation. `messages` is the parsed list."""
    contents = [m.get("content") or "" for m in messages]
    joined = "\n".join(contents)
    roles = [m.get("role", "") for m in messages]

    user_text = "\n".join(c for c, r in zip(contents, roles) if r == "user")
    assistant_text = "\n".join(c for c, r in zip(contents, roles) if r == "assistant")

    lengths = [len(c) for c in contents] or [0]
    words = WORD.findall(joined)
    letters = [ch for ch in joined if ch.isalpha()]

    out = {
        # structure: survives translation by construction
        "struct_n_messages": len(messages),
        "struct_n_user": roles.count("user"),
        "struct_n_assistant": roles.count("assistant"),
        "struct_n_system": roles.count("system"),
        "struct_n_tool": sum(1 for r in roles if r not in {"user", "assistant", "system"}),
        "struct_user_share": safe_ratio(roles.count("user"), len(roles)),
        "struct_last_is_user": float(roles[-1] == "user") if roles else 0.0,
        "struct_has_system_prompt": float(bool(system_prompt)),

        # size: shifts with the language, so grouped apart from structure
        "surface_total_chars": len(joined),
        "surface_mean_msg_chars": sum(lengths) / len(lengths),
        "surface_max_msg_chars": max(lengths),
        "surface_n_words": len(words),
        "surface_mean_word_len": safe_ratio(sum(len(w) for w in words), len(words)),
        "surface_user_char_share": safe_ratio(len(user_text), len(user_text) + len(assistant_text)),
        "surface_upper_ratio": safe_ratio(sum(1 for ch in letters if ch.isupper()), len(letters)),
        "surface_digit_ratio": safe_ratio(sum(1 for ch in joined if ch.isdigit()), len(joined)),
        "surface_punct_ratio": safe_ratio(sum(1 for ch in joined if ch in "!?.,;:"), len(joined)),
        "surface_exclamations": joined.count("!"),
        "surface_questions": joined.count("?"),
        "surface_mean_sentence_len": safe_ratio(len(words), len(SENTENCE_END.findall(joined)) or 1),
        "surface_leet_messages": sum(1 for c in contents if is_leet_message(c)),
    }

    # Entities, counted on text with leetspeak prose blanked first. Without that,
    # `455157` -- the word "assist" -- counts as a six-digit code in the English and
    # disappears in the Kazakh, so a leetspeak row looks like it carries data it does
    # not. The validator already masks this way; the counts must agree with it.
    masked = "\n".join(mask_leet_prose(c) for c in contents)
    for name, pattern in ENTITY_PATTERNS.items():
        found = pattern.findall(masked)
        if name == "code":
            # a code has a letter in it; `6370-` out of `$6370-тен` is Kazakh
            # morphology, and bare digits are already counted as long_number
            found = [m for m in found if any(ch.isalpha() for ch in m)]
        out[f"entity_n_{name}"] = len(found)
        out[f"entity_has_{name}"] = float(bool(found))
    out["entity_n_total"] = sum(out[f"entity_n_{n}"] for n in ENTITY_PATTERNS)

    # keyword families, matched in whichever language the text is written in
    for family in LEXICON:
        hits = len(EN_PATTERNS[family].findall(joined)) + len(KK_PATTERNS[family].findall(joined))
        out[f"lex_{family}"] = hits
        out[f"lex_{family}_in_user"] = (len(EN_PATTERNS[family].findall(user_text))
                                        + len(KK_PATTERNS[family].findall(user_text)))
    out["lex_total"] = sum(out[f"lex_{f}"] for f in LEXICON)
    out["lex_families_present"] = sum(1 for f in LEXICON if out[f"lex_{f}"] > 0)
    return out


def featurise(frame):
    """Feature table for a dataframe of conversations, keeping only safe columns."""
    rows = [conversation_features(json.loads(row.messages), row.system_prompt or "")
            for row in frame.itertuples()]
    features = pd.DataFrame(rows, index=frame.index)

    # carried through: identity, the target, and metadata with no annotation in it
    keep = ["id", "split", "language", "style_variant", "source_pool", "label_3class"]
    for column in ("translated_from", "translated_fields"):
        if column in frame.columns:
            keep.append(column)
    return pd.concat([frame[keep].reset_index(drop=True),
                      features.reset_index(drop=True)], axis=1)


def feature_columns(frame):
    return [c for c in frame.columns
            if c.startswith(("struct_", "surface_", "entity_", "lex_"))]


def build_report(frame, features):
    lines = ["# Feature report", "",
             f"{len(features):,} rows, {len(feature_columns(features))} features", ""]

    lines += ["## Columns excluded as leakage", "",
              "`label_3class` was derived from `decision_class` and",
              "`should_trigger_scam_defense`. Knowing that pair pins the label down",
              "99.99% of the time, so these columns reproduce the answer rather than",
              "predicting it and are kept out of the feature table.", "",
              "| Column | Why |", "|---|---|"]
    for column, reason in LEAKING_COLUMNS.items():
        lines.append(f"| `{column}` | {reason} |")

    lines += ["", "## Feature groups", "", "| Prefix | Count | Survives translation? |",
              "|---|---:|---|"]
    groups = {
        "struct_": "yes, by construction (message count and roles are preserved)",
        "entity_": "yes, by construction (links, amounts and codes are preserved)",
        "surface_": "no, it measures the writing system",
        "lex_": "only where the lexicon has that language",
    }
    for prefix, note in groups.items():
        lines.append(f"| `{prefix}` | {sum(1 for c in feature_columns(features) if c.startswith(prefix))} | {note} |")

    lines += ["", "## Class balance", "", "| Class | Rows |", "|---|---:|"]
    for label, count in features.label_3class.value_counts().items():
        lines.append(f"| {label} | {count:,} |")

    kazakh = features[features.language == "kk"]
    if len(kazakh):
        lines += ["", "## English / Kazakh parity", "",
                  f"{len(kazakh)} conversations exist in both languages. For a feature to be",
                  "usable across languages, a pair should agree on it.", "",
                  "| Feature group | Pairs agreeing exactly |", "|---|---:|"]
        english = features.set_index("id")
        for prefix in ("struct_", "entity_", "surface_", "lex_"):
            columns = [c for c in feature_columns(features) if c.startswith(prefix)]
            agree = 0
            for row in kazakh.itertuples():
                source = english.loc[row.translated_from]
                if all(getattr(row, c) == source[c] for c in columns):
                    agree += 1
            lines.append(f"| `{prefix}` | {agree} / {len(kazakh)} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged_dir = pathlib.Path(args.merged)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    built = []
    for split in SPLITS:
        frame = pd.read_parquet(merged_dir / f"{split}.parquet")
        features = featurise(frame)
        features.to_parquet(out_dir / f"{split}.parquet", index=False)
        built.append(features)
        print(f"{split}: {len(features):,} rows, {len(feature_columns(features))} features")

    combined = pd.concat(built, ignore_index=True)
    leaked = [c for c in LEAKING_COLUMNS if c in combined.columns]
    if leaked:
        raise SystemExit(f"leaking columns present in the feature table: {leaked}")

    (out_dir / "feature_report.md").write_text(
        build_report(None, combined), encoding="utf-8")
    print(f"\nno leaking column reached the feature table")
    print(f"wrote {out_dir / 'feature_report.md'}")


if __name__ == "__main__":
    main()
