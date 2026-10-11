"""Score what a human produced on the two review tasks.

Both pages from `make_human_tasks.py` export a JSON file of
`{row id: {verdict, note}}`. This reads one back and says what it means.

**labels** -- the human ceiling. Compares a person's class judgements against the
derived label and reports macro-F1 on the same scale as every model in notebook
03. Without it, the best Kazakh model's 0.539 has nothing to be measured against:
a person at 0.75 means the model is weak, a person at 0.60 means the task is hard.
Disagreement is also evidence about the labelling rule itself, so the confusion
matrix is printed rather than summarised.

**review** -- the fluency check. Counts how the 300 translations were judged and
lists everything marked wrong, which is the queue of things to fix.

Usage:

    python score_human.py labels --answers labels.json --key ../review/label_kk_answers.csv
    python score_human.py review --answers review.json --merged ../data/merged
"""
import argparse
import json
import pathlib

import pandas as pd

CLASSES = ["LEGITIMATE", "SUSPICIOUS", "SCAM"]


def load_answers(path):
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    rows = [{"id": identifier,
             "verdict": value.get("verdict"),
             "note": value.get("note", "")}
            for identifier, value in raw.items() if value.get("verdict")]
    return pd.DataFrame(rows)


def score_labels(args):
    from sklearn.metrics import classification_report, cohen_kappa_score, f1_score

    answers = load_answers(args.answers)
    key = pd.read_csv(args.key)
    merged = answers.merge(key, on="id", how="inner")
    if merged.empty:
        raise SystemExit("no overlap between the answers and the key")

    print(f"{len(merged)} of {len(key)} labelled\n")
    truth, human = merged.label_3class, merged.verdict

    macro = f1_score(truth, human, average="macro", labels=CLASSES, zero_division=0)
    print(f"human macro-F1 against the derived label: {macro:.3f}")
    print(f"agreement (Cohen's kappa):                {cohen_kappa_score(truth, human):.3f}\n")
    print(classification_report(truth, human, labels=CLASSES, zero_division=0))
    print("rows are the derived label, columns what the person chose:")
    print(pd.crosstab(truth, human).reindex(index=CLASSES, columns=CLASSES).fillna(0)
          .astype(int).to_string())

    print(f"\nFor comparison, the best model on Kazakh scored 0.539 macro-F1")
    print("(notebook 03). Read the two together:")
    print("  person well above the model  -> the model has room, keep working on it")
    print("  person close to the model    -> the task is hard, or the labels are noisy")
    print("  person below the model       -> suspect the labelling rule, not the person")

    disagreements = merged[truth != human]
    if len(disagreements):
        print(f"\n{len(disagreements)} disagreements; the first few:")
        for row in disagreements.head(8).itertuples():
            print(f"  {row.id[:46]:<46} label {row.label_3class:<11} person {row.verdict}"
                  + (f"  -- {row.note[:50]}" if row.note else ""))


def score_review(args):
    answers = load_answers(args.answers)
    merged = pd.concat([pd.read_parquet(pathlib.Path(args.merged) / f"{s}.parquet")
                        for s in ["train", "test", "validation"]], ignore_index=True)
    kazakh = merged[merged.language == "kk"].set_index("id")

    print(f"{len(answers)} of {len(kazakh)} reviewed\n")
    counts = answers.verdict.value_counts()
    for verdict in ["good", "awkward", "wrong"]:
        n = int(counts.get(verdict, 0))
        print(f"  {verdict:<9} {n:>4}  ({100 * n / max(len(answers), 1):.0f}%)")

    acceptable = int(counts.get("good", 0)) + int(counts.get("awkward", 0))
    print(f"\nreads naturally or close enough: {acceptable}/{len(answers)} "
          f"({100 * acceptable / max(len(answers), 1):.0f}%)")

    for verdict in ["wrong", "awkward"]:
        flagged = answers[answers.verdict == verdict]
        if not len(flagged):
            continue
        print(f"\n--- marked {verdict} ({len(flagged)}) ---")
        for row in flagged.head(20).itertuples():
            style = kazakh.loc[row.id, "style_variant"] if row.id in kazakh.index else "?"
            print(f"  {row.id[:44]:<44} [{style}]"
                  + (f"  {row.note[:60]}" if row.note else ""))

    if len(answers):
        by_style = answers.join(kazakh[["style_variant", "label_3class"]], on="id")
        print("\nverdict by style variant:")
        print(pd.crosstab(by_style.style_variant, by_style.verdict).to_string())


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="task", required=True)

    labels = sub.add_parser("labels", help="score the human ceiling task")
    labels.add_argument("--answers", required=True)
    labels.add_argument("--key", required=True)
    labels.set_defaults(func=score_labels)

    review = sub.add_parser("review", help="summarise the fluency review")
    review.add_argument("--answers", required=True)
    review.add_argument("--merged", required=True)
    review.set_defaults(func=score_review)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
