"""Train on English, evaluate on Kazakh: what does translation cost a classifier?

The 300 translated conversations exist in both languages with identical structure
and identical entities, so the gap between a model's score on the English 300 and
the Kazakh 300 is the cost of the language change and nothing else. Same
conversations, same labels, one variable.

Getting that comparison clean needs one precaution. Each Kazakh row inherits its
source's split, which is right for the merged dataset but wrong here: 233 of the
300 English twins sit in the training split, so a model trained on all English
rows would have read the English version of most conversations it is then tested
on in Kazakh. That is not zero-shot. **All 300 English sources are therefore held
out of training**, leaving 15,908 English rows to train on and two evaluation sets
that are both unseen:

    EN-300   unseen conversations, seen language
    KK-300   unseen conversations, unseen language

Four models, chosen to test the claim notebook 02 makes about feature groups --
that `struct_` and `entity_` survive translation while `surface_` and `lex_` do
not:

    hand features (all 54)        the full table
    hand features (struct+entity) only the translation-invariant groups
    char n-gram TF-IDF            surface lexical; Latin vs Cyrillic
    word TF-IDF                   surface lexical, whole words

If the feature grouping is right, the struct+entity model should lose least across
the language boundary and the TF-IDF models should collapse.

Usage:

    python run_crosslingual.py --features ../data/features --merged ../data/merged \\
        --out ../data/experiments
"""
import argparse
import json
import pathlib

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import build_features as bf

SPLITS = ["train", "test", "validation"]
SEED = 0


def load(features_dir, merged_dir):
    features = pd.concat([pd.read_parquet(pathlib.Path(features_dir) / f"{s}.parquet")
                          for s in SPLITS], ignore_index=True)
    merged = pd.concat([pd.read_parquet(pathlib.Path(merged_dir) / f"{s}.parquet")
                        for s in SPLITS], ignore_index=True)
    merged["text"] = merged.messages.apply(
        lambda m: "\n".join(x.get("content") or "" for x in json.loads(m)))
    return features, merged


def split_sets(features, merged):
    """Training pool, and the two paired evaluation sets.

    Holding out the 300 source rows is not enough. An id looks like
    `base::<conversation>::<row>`, and the augmented pool restyles the same
    conversation many times over -- 3,473 training rows are restylings of 273 of
    the 300, a median of 12 each. Leaving those in let a character n-gram model
    score 0.973 on the "held-out" English by recognising text it had already read
    in another voice. The whole conversation family goes out together.
    """
    kazakh = features[features.language == "kk"]
    source_ids = set(kazakh.translated_from)

    conversation = features.id.str.split("::").str[1]
    held_out = set(conversation[features.id.isin(source_ids)])

    train = features[(features.language == "en")
                     & (features.split == "train")
                     & (~features.id.isin(source_ids))
                     & (~conversation.isin(held_out))]
    english_300 = features[features.id.isin(source_ids)]

    # align the pairs so both evaluation sets are in the same order
    english_300 = english_300.set_index("id").loc[kazakh.translated_from].reset_index()
    assert len(english_300) == len(kazakh) == 300
    assert (english_300.label_3class.values == kazakh.label_3class.values).all()
    return train, english_300, kazakh


def evaluate(name, model, fit_X, fit_y, eval_sets):
    model.fit(fit_X, fit_y)
    row = {"model": name}
    for label, (X, y) in eval_sets.items():
        row[label] = f1_score(y, model.predict(X), average="macro")
    row["gap"] = row["EN-300"] - row["KK-300"]
    return row, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    features, merged = load(args.features, args.merged)
    train, english_300, kazakh = split_sets(features, merged)
    text = merged.set_index("id").text

    print(f"train: {len(train):,} English rows (the 300 translated conversations "
          f"and every restyling of them held out)")
    print(f"eval:  {len(english_300)} EN  /  {len(kazakh)} KK, the same conversations\n")

    all_features = bf.feature_columns(features)
    invariant = [c for c in all_features if c.startswith(("struct_", "entity_"))]

    y_train = train.label_3class
    y_eval = english_300.label_3class

    results = []

    numeric_sets = {
        "EN-300": (english_300[all_features].to_numpy(float), y_eval),
        "KK-300": (kazakh[all_features].to_numpy(float), kazakh.label_3class),
    }
    row, _ = evaluate("majority class",
                      DummyClassifier(strategy="most_frequent"),
                      train[all_features].to_numpy(float), y_train, numeric_sets)
    results.append(row)

    row, hand_model = evaluate("hand features (all 54)",
                               HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
                               train[all_features].to_numpy(float), y_train, numeric_sets)
    results.append(row)

    invariant_sets = {
        "EN-300": (english_300[invariant].to_numpy(float), y_eval),
        "KK-300": (kazakh[invariant].to_numpy(float), kazakh.label_3class),
    }
    row, _ = evaluate(f"hand features (struct+entity, {len(invariant)})",
                      HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
                      train[invariant].to_numpy(float), y_train, invariant_sets)
    results.append(row)

    text_sets = {
        "EN-300": (text.loc[english_300.id].tolist(), y_eval),
        "KK-300": (text.loc[kazakh.id].tolist(), kazakh.label_3class),
    }
    train_text = text.loc[train.id].tolist()

    row, _ = evaluate("char n-gram TF-IDF (3-5)",
                      make_pipeline(
                          TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                          min_df=3, max_features=200_000, sublinear_tf=True),
                          LinearSVC(C=1.0, random_state=SEED)),
                      train_text, y_train, text_sets)
    results.append(row)

    row, _ = evaluate("word TF-IDF (1-2)",
                      make_pipeline(
                          TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                          max_features=200_000, sublinear_tf=True),
                          LinearSVC(C=1.0, random_state=SEED)),
                      train_text, y_train, text_sets)
    results.append(row)

    table = pd.DataFrame(results).set_index("model")
    print(table.round(3).to_string())

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "crosslingual_results.csv")

    per_class = classification_report(
        kazakh.label_3class, hand_model.predict(kazakh[all_features].to_numpy(float)),
        zero_division=0)

    lines = ["# Train English, evaluate Kazakh", "",
             f"Trained on {len(train):,} English conversations. The 300 translated",
             "conversations were held out together with every restyling of them -- the",
             "augmented pool rewrites the same conversation up to 25 times, and leaving",
             "those in let a character n-gram model reach 0.973 by recognising text it had",
             "already read in another voice.",
             "The two evaluation sets are the same 300 conversations, in English and in",
             "Kazakh, so the gap is the cost of the language change alone.", "",
             "| Model | EN-300 | KK-300 | gap |", "|---|---:|---:|---:|"]
    for name, row in table.iterrows():
        lines.append(f"| {name} | {row['EN-300']:.3f} | {row['KK-300']:.3f} | {row['gap']:+.3f} |")
    lines += ["", "Macro-F1. With 300 conversations per evaluation set the 95% interval is",
              "roughly +/- 5 points, so small differences between rows are not meaningful;",
              "the large ones are.", "",
              "## Per-class, hand features on Kazakh", "", "```", per_class, "```", ""]
    (out_dir / "crosslingual_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_dir / 'crosslingual_report.md'}")


if __name__ == "__main__":
    main()
