"""Are the gaps real, and where does the Kazakh model actually fail?

Notebook 03 reports macro-F1 and a +/- 5.6 point interval from a normal
approximation. That interval is the wrong tool twice over: it assumes a binomial
on accuracy rather than macro-F1, and it ignores that EN-300 and KK-300 are *the
same 300 conversations*. Paired data deserves a paired test, which is both
correct and more powerful.

Three things here.

**Is the gap real?** McNemar's test on the paired correctness of one model across
the two languages. It asks only about conversations the model gets right in one
language and wrong in the other, which is exactly the question.

**How wide is the interval, really?** A percentile bootstrap over conversations,
resampling macro-F1 rather than assuming a distribution for it.

**Where does it fail?** The headline number is that SCAM recall on Kazakh is
0.24. This breaks that down by scenario category, style variant and conversation
length, and writes out the missed conversations so they can be read.

Usage:

    python analyse_results.py --features ../data/features --merged ../data/merged \\
        --embeddings ../data/embeddings --out ../data/experiments
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import build_features as bf
import run_crosslingual as rx

SEED = 0
CLASSES = ["LEGITIMATE", "SUSPICIOUS", "SCAM"]
BOOTSTRAP = 2000


def mcnemar(correct_a, correct_b):
    """Exact McNemar on two paired correctness vectors. Returns (b, c, p)."""
    from scipy.stats import binomtest

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    if only_a + only_b == 0:
        return only_a, only_b, 1.0
    return only_a, only_b, binomtest(only_a, only_a + only_b, 0.5).pvalue


def bootstrap_f1(truth, predicted, rng, rounds=BOOTSTRAP):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    scores = []
    for _ in range(rounds):
        pick = rng.integers(0, len(truth), len(truth))
        if len(set(truth[pick])) < 2:
            continue
        scores.append(f1_score(truth[pick], predicted[pick], average="macro",
                               labels=CLASSES, zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    features, merged = rx.load(args.features, args.merged)
    train, english_300, kazakh = rx.split_sets(features, merged)
    text = merged.set_index("id").text
    columns = bf.feature_columns(features)
    rng = np.random.default_rng(SEED)

    truth = english_300.label_3class.to_numpy()
    assert (truth == kazakh.label_3class.to_numpy()).all()

    print("fitting the two headline models")
    hand = HistGradientBoostingClassifier(max_iter=200, random_state=SEED).fit(
        train[columns].to_numpy(float), train.label_3class)
    chars = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        max_features=200_000, sublinear_tf=True),
        LinearSVC(C=1.0, random_state=SEED)).fit(
        text.loc[train.id].tolist(), train.label_3class)

    predictions = {
        ("hand features", "EN"): hand.predict(english_300[columns].to_numpy(float)),
        ("hand features", "KK"): hand.predict(kazakh[columns].to_numpy(float)),
        ("char n-gram", "EN"): chars.predict(text.loc[english_300.id].tolist()),
        ("char n-gram", "KK"): chars.predict(text.loc[kazakh.id].tolist()),
    }

    lines = ["# Are the gaps real, and where does it fail?", "",
             "EN-300 and KK-300 are the same 300 conversations, so the comparison is",
             "paired and McNemar's test applies. Confidence intervals are a percentile",
             f"bootstrap over conversations ({BOOTSTRAP:,} resamples), not a normal",
             "approximation.", "",
             "## Is the English/Kazakh gap real?", "",
             "| Model | EN macro-F1 (95% CI) | KK macro-F1 (95% CI) | right in EN only | right in KK only | McNemar p |",
             "|---|---|---|---:|---:|---:|"]

    print()
    for model in ["hand features", "char n-gram"]:
        en_pred, kk_pred = predictions[(model, "EN")], predictions[(model, "KK")]
        en_f1 = f1_score(truth, en_pred, average="macro", labels=CLASSES, zero_division=0)
        kk_f1 = f1_score(truth, kk_pred, average="macro", labels=CLASSES, zero_division=0)
        en_lo, en_hi = bootstrap_f1(truth, en_pred, rng)
        kk_lo, kk_hi = bootstrap_f1(truth, kk_pred, rng)
        only_en, only_kk, p = mcnemar(en_pred == truth, kk_pred == truth)
        lines.append(f"| {model} | {en_f1:.3f} ({en_lo:.3f}–{en_hi:.3f}) | "
                     f"{kk_f1:.3f} ({kk_lo:.3f}–{kk_hi:.3f}) | {only_en} | {only_kk} | "
                     f"{p:.2g} |")
        print(f"{model:<16} EN {en_f1:.3f} [{en_lo:.3f}, {en_hi:.3f}]  "
              f"KK {kk_f1:.3f} [{kk_lo:.3f}, {kk_hi:.3f}]  McNemar p={p:.2g}")

    # the comparison notebook 03 could not resolve
    hand_kk, char_kk = predictions[("hand features", "KK")], predictions[("char n-gram", "KK")]
    only_hand, only_char, p = mcnemar(hand_kk == truth, char_kk == truth)
    lines += ["", "## Hand features versus char n-grams, on Kazakh", "",
              "Notebook 03 called this one a tie on a +/- 5.6 point interval. Paired, it",
              "is a sharper question: of the conversations where the two models differ,",
              "how lopsided is it?", "",
              f"- hand features right, char n-grams wrong: **{only_hand}**",
              f"- char n-grams right, hand features wrong: **{only_char}**",
              f"- McNemar p = **{p:.3g}**", ""]
    verdict = ("so the difference is real" if p < 0.05 else
               "so this evaluation set still cannot separate them")
    lines.append(f"{verdict}.")
    print(f"\nhand vs char on KK: {only_hand} / {only_char}, McNemar p={p:.3g}")

    # where the loss concentrates
    frame = pd.DataFrame({
        "id": kazakh.id.to_numpy(),
        "label": truth,
        "en_right": predictions[("hand features", "EN")] == truth,
        "kk_right": predictions[("hand features", "KK")] == truth,
        "kk_pred": predictions[("hand features", "KK")],
        # scenario_category is excluded from the features as leakage, which is right
        # for modelling and irrelevant here: this is analysis, not an input
        "category": merged.set_index("id").loc[english_300.id, "scenario_category"].to_numpy(),
        "style": english_300.style_variant.to_numpy(),
        "chars": english_300.surface_total_chars.to_numpy(),
        "messages": english_300.struct_n_messages.to_numpy(),
    })
    frame["lost"] = frame.en_right & ~frame.kk_right

    lines += ["", "## Where the Kazakh loss concentrates", "",
              "`lost` counts conversations the model gets right in English and wrong in",
              "Kazakh -- the translation cost, conversation by conversation.", ""]
    for column, title in [("label", "class"), ("category", "scenario category"),
                          ("style", "style variant")]:
        grouped = frame.groupby(column).agg(
            n=("id", "size"), en=("en_right", "mean"),
            kk=("kk_right", "mean"), lost=("lost", "sum")).sort_values("n", ascending=False)
        grouped = grouped[grouped.n >= 5]
        lines += [f"### By {title}", "", "| " + title + " | n | EN acc | KK acc | lost |",
                  "|---|---:|---:|---:|---:|"]
        for name, row in grouped.iterrows():
            lines.append(f"| {name} | {int(row.n)} | {row.en:.2f} | {row.kk:.2f} | "
                         f"{int(row.lost)} |")
        lines.append("")

    quartile = pd.qcut(frame.chars, 4, labels=["shortest", "short", "long", "longest"])
    by_length = frame.groupby(quartile, observed=True).agg(
        n=("id", "size"), en=("en_right", "mean"), kk=("kk_right", "mean"),
        lost=("lost", "sum"))
    lines += ["### By conversation length", "",
              "| length | n | EN acc | KK acc | lost |", "|---|---:|---:|---:|---:|"]
    for name, row in by_length.iterrows():
        lines.append(f"| {name} | {int(row.n)} | {row.en:.2f} | {row.kk:.2f} | "
                     f"{int(row.lost)} |")

    missed = frame[(frame.label == "SCAM") & ~frame.kk_right]
    lines += ["", "## The missed Kazakh scams", "",
              f"{len(missed)} of {int((frame.label == 'SCAM').sum())} SCAM conversations "
              "are misclassified in Kazakh. What the model called them instead:", "",
              "| predicted | n |", "|---|---:|"]
    for name, count in missed.kk_pred.value_counts().items():
        lines.append(f"| {name} | {count} |")
    lines += ["",
              f"Of those {len(missed)}, **{int(missed.en_right.sum())}** are classified "
              "correctly in English, so the translation is what lost them; the remaining "
              f"{int((~missed.en_right).sum())} the model cannot do in either language.", ""]

    # Which boundary is actually hard? SCAM and SUSPICIOUS both have
    # should_trigger_scam_defense = True and are separated only by decision_class --
    # refuse/escalate/block against request_verification/audit. That is a statement
    # about what a correct agent should DO, and the conversation never says it.
    binary_train = train.label_3class.replace({"SCAM": "ATTACK", "SUSPICIOUS": "ATTACK"})
    binary_truth = pd.Series(truth).replace({"SCAM": "ATTACK",
                                             "SUSPICIOUS": "ATTACK"}).to_numpy()
    binary = HistGradientBoostingClassifier(max_iter=200, random_state=SEED).fit(
        train[columns].to_numpy(float), binary_train)
    binary_en = f1_score(binary_truth, binary.predict(english_300[columns].to_numpy(float)),
                         average="macro")
    binary_kk = f1_score(binary_truth, binary.predict(kazakh[columns].to_numpy(float)),
                         average="macro")

    attack_rows = train.label_3class.isin(["SCAM", "SUSPICIOUS"]).to_numpy()
    attack_eval = np.isin(truth, ["SCAM", "SUSPICIOUS"])
    hard = HistGradientBoostingClassifier(max_iter=200, random_state=SEED).fit(
        train[columns].to_numpy(float)[attack_rows],
        train.label_3class[attack_rows])
    hard_en = f1_score(truth[attack_eval],
                       hard.predict(english_300[columns].to_numpy(float)[attack_eval]),
                       average="macro")
    hard_kk = f1_score(truth[attack_eval],
                       hard.predict(kazakh[columns].to_numpy(float)[attack_eval]),
                       average="macro")

    three_en = f1_score(truth, predictions[("hand features", "EN")], average="macro",
                        labels=CLASSES, zero_division=0)
    three_kk = f1_score(truth, predictions[("hand features", "KK")], average="macro",
                        labels=CLASSES, zero_division=0)

    lines += ["", "## Which boundary is hard?", "",
              "SCAM and SUSPICIOUS both carry `should_trigger_scam_defense = True`. What",
              "separates them is `decision_class`: refuse / escalate / block against",
              "request_verification / audit. That is a judgement about what a correct",
              "agent should **do**, and the conversation never states it. Splitting the",
              "task shows how much that costs.", "",
              "| Task | EN | KK |", "|---|---:|---:|",
              f"| 3-class, as defined | {three_en:.3f} | {three_kk:.3f} |",
              f"| attack vs not (SCAM and SUSPICIOUS merged) | **{binary_en:.3f}** | "
              f"**{binary_kk:.3f}** |",
              f"| SCAM vs SUSPICIOUS alone | {hard_en:.3f} | {hard_kk:.3f} |", "",
              "Chance on the last row is 0.500.", "",
              "Read together: the text supports attack detection well, and barely",
              "supports the SCAM/SUSPICIOUS split at all. The three-class score is",
              "dominated by a boundary a text-only model has almost no evidence for,",
              "which is a property of how the label was defined rather than of the",
              "features. It does not invalidate the label -- it still refines the binary",
              "flag and every SUSPICIOUS row carries named risk indicators -- but any",
              "three-class number should be reported next to the binary one.", ""]
    print(f"\n3-class       EN {three_en:.3f}  KK {three_kk:.3f}")
    print(f"attack vs not EN {binary_en:.3f}  KK {binary_kk:.3f}")
    print(f"SCAM vs SUSP  EN {hard_en:.3f}  KK {hard_kk:.3f}   (chance 0.500)")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "per_conversation.csv", index=False)
    (out_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nlost in translation: {int(frame.lost.sum())} of 300 conversations")
    print(f"missed Kazakh scams: {len(missed)} "
          f"({int(missed.en_right.sum())} of them correct in English)")
    print(f"wrote {out_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
