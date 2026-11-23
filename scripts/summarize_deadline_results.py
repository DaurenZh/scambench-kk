"""Summarize completed local experiments; never fit models or read GPU test results."""
import argparse
import csv
import html
import json
import pathlib
import zipfile

import numpy as np


def macro_from_confusions(matrix, supported):
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(-2) + matrix.sum(-1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator != 0)
    return f1[..., supported].mean(-1)


def paired_family_bootstrap(rows, classes, rounds=2000):
    families = sorted({r["family"] for r in rows})
    index = {family: i for i, family in enumerate(families)}
    supported = sorted({r["label"] for r in rows})
    rng = np.random.default_rng(42)
    weights = rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=rounds)
    values = {}
    for model in ["mlp", "diffusion"]:
        counts = np.zeros((len(families), classes, classes), dtype=float)
        for row in rows:
            counts[index[row["family"]], row["label"], row[model]] += 1
        sampled = np.einsum("rf,fij->rij", weights, counts)
        values[model] = macro_from_confusions(sampled, supported)
    delta = values["diffusion"] - values["mlp"]
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return {"families": len(families), "rows": len(rows), "rounds": rounds,
            "metric": "macro-F1 over the fixed set of classes present in the evaluation set",
            "contrast": "diffusion minus matched MLP", "ci95": [float(lo), float(hi)],
            "interval_includes_zero": bool(lo <= 0 <= hi)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("data/experiments/deadline_2026-11-23"))
    args = parser.parse_args()
    rows, comparisons, per_class = [], {}, {}
    for target in ["risk", "action"]:
        baselines = json.loads((args.runs / f"{target}_tfidf/baselines.json").read_text())
        for name, languages in baselines.items():
            for language, metrics in languages.items():
                rows.append(dict(target=target, model=f"TF-IDF {name}", language=language,
                                 macro_f1=metrics["macro_f1"], supported_macro_f1=metrics["macro_f1_supported"],
                                 accuracy=metrics["accuracy"], binary_macro_f1=metrics.get("binary_macro_f1", "")))
        folder = args.runs / f"diffusion_{target}_mixed_seed42"
        run = json.loads((folder / "run.json").read_text())
        metrics = json.loads((folder / "test_metrics.json").read_text())
        if run["smoke"] or run["epochs"] != 30:
            raise ValueError("Expected completed 30-epoch non-smoke runs")
        for language, result in metrics.items():
            for model in ["linear", "mlp", "diffusion"]:
                m = result[model]
                rows.append(dict(target=target, model=f"SVD {model} mixed", language=language,
                                 macro_f1=m["macro_f1"], supported_macro_f1=m["macro_f1_supported"],
                                 accuracy=m["accuracy"], binary_macro_f1=m.get("binary_macro_f1", "")))
                per_class[f"{target}/{language}/{model}"] = m["per_class"]
            predictions = json.loads((folder / f"{language}_predictions.json").read_text())
            expected = json.loads((args.runs / f"{target}_tfidf/word_mixed_{language}_predictions.json").read_text())
            if [(r["id"], r["label"]) for r in predictions] != [(r["id"], r["label"]) for r in expected]:
                raise ValueError("Evaluation IDs or labels differ between compared models")
            comparisons[f"{target}/{language}"] = paired_family_bootstrap(predictions, len(run["classes"]))
    with (args.runs / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.runs / "paired_family_bootstrap.json").write_text(json.dumps(comparisons, indent=2))
    (args.runs / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2))
    table = []
    for row in rows:
        if row["language"] != "test_kk":
            continue
        primary = row["macro_f1"] if row["target"] == "risk" else row["supported_macro_f1"]
        table.append(f"<tr><td>{html.escape(row['target'])}</td><td>{html.escape(row['model'])}</td>"
                     f"<td>{primary:.3f}</td><td>{row['accuracy']:.3f}</td></tr>")
    intervals = []
    for name, comparison in comparisons.items():
        lo, hi = comparison["ci95"]
        intervals.append(f"<li>{name}: diffusion − MLP, 95% interval [{lo:.3f}, {hi:.3f}].</li>")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><title>ScamBench experiment results</title>
<style>body{font:16px/1.5 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#17202a}
table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}
.note{background:#f2f4f7;padding:16px}h1{font-size:28px}@media print{body{margin:10px}}</style>
<h1>ScamBench: completed action prediction and diffusion experiments</h1>
<p>23 November 2026. Full local CPU runs: seed 42; 30 MLP epochs and 30 diffusion epochs;
100 diffusion steps; 8 samples per prediction. Both heads use identical train-fitted TF-IDF/SVD features.</p>
<p>Mixed training: 11,227 English plus 3,600 silver Kazakh rows. Validation: 1,208 English plus 399 silver rows.
Evaluation: 300 paired conversations per language, belonging to 273 families. Families do not cross these partitions.</p>
<h2>Kazakh evaluation</h2><p>Risk macro-F1 covers 3 classes. Action macro-F1 below covers the 5 actions present
in evaluation; the trained output schema contains 13 actions. The CSV also reports fixed-schema scores.</p>
<table><thead><tr><th>Target</th><th>Model</th><th>Macro-F1</th><th>Accuracy</th></tr></thead><tbody>"""
    document += "".join(table) + "</tbody></table><h2>Paired family bootstrap</h2><ul>" + "".join(intervals)
    document += """</ul><p>Intervals use 2,000 paired resamples of conversation families. This preserves dependencies
between restylings. Intervals containing zero do not establish an improvement.</p>
<div class="note"><strong>Limits:</strong> one seed; a historical benchmark that informed earlier decisions;
no native-speaker review; only five action types evaluated, including just one <code>ignore</code> example.
Action prediction measures agreement with recorded actions, not execution safety. The diffusion implementation
adapts CARD and is not a reproduction. XLM-R is not reported here: the remote Windows machine was still
installing CUDA PyTorch at the last update. These new baselines reserve development families and differ
from the earlier 0.687/0.898 experiments.</div>
<p>Keep the negative results: diffusion need not beat its matched MLP to be a completed experiment.
Report per-class support and inspect the saved predictions before making broad claims.</p></html>"""
    (args.runs / "results.html").write_text(document, encoding="utf-8")
    archive = args.runs / "completed_cpu_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.runs.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".html"}:
                z.write(path, path.relative_to(args.runs))
    for row in rows:
        if row["language"] == "test_kk":
            print(row["target"], row["model"], "macro-F1", round(row["macro_f1"], 4),
                  "supported", round(row["supported_macro_f1"], 4))
    print(json.dumps(comparisons, indent=2))
    print("Report:", args.runs / "results.html")
    print("Archive:", archive)


if __name__ == "__main__":
    main()
