"""Build printable slides, error-review tasks and figures from saved results."""
import argparse
import collections
import csv
import json
import pathlib
import textwrap
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from train_transformer import read


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("data/experiments/deadline_2026-11-23"))
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/gpu_bundle"))
    args = parser.parse_args()
    gpu_verified = (args.runs / "xlmr_mixed_risk/verification.json").exists()
    destination = args.runs / "defense"
    destination.mkdir(exist_ok=True)
    risk = load(args.runs / "risk_tfidf/baselines.json")
    action = load(args.runs / "action_tfidf/baselines.json")
    dense = {target: load(args.runs / f"diffusion_{target}_mixed_seed42/test_metrics.json") for target in ["risk", "action"]}
    intervals = load(args.runs / "paired_family_bootstrap.json")
    manifest = load(args.data / "manifest.json")
    examples = {}
    en, kk = read(args.data, "test_en"), read(args.data, "test_kk")
    for a, b in zip(en, kk):
        assert a["family"] == b["family"]
        examples[b["id"]] = (a["text"], b["text"])
    review, error_counts, changes = [], {}, {}
    for target in ["risk", "action"]:
        folder = args.runs / f"diffusion_{target}_mixed_seed42"
        labels = load(folder / "run.json")["classes"]
        predictions = load(folder / "test_kk_predictions.json")
        tfidf = {r["id"]: r for r in load(args.runs / f"{target}_tfidf/word_mixed_test_kk_predictions.json")}
        changes[target] = dict(
            fixes=sum(r["mlp"] != r["label"] and r["diffusion"] == r["label"] for r in predictions),
            regressions=sum(r["mlp"] == r["label"] and r["diffusion"] != r["label"] for r in predictions))
        for model in ["mlp", "diffusion", "tfidf"]:
            counts = collections.Counter()
            for row in predictions:
                guess = tfidf[row["id"]]["predicted"] if model == "tfidf" else row[model]
                if row["label"] != guess:
                    counts[(labels[row["label"]], labels[guess])] += 1
            error_counts[f"{target}/{model}"] = [dict(reference=a, predicted=b, count=n) for (a, b), n in counts.most_common()]
        # One short example per distinct prediction pattern; fixes, regressions,
        # shared errors and correct controls are represented without reading test text
        # to choose new model settings. These are review tasks, not reviewed findings.
        used = set()
        for category in ["diffusion_fix", "diffusion_regression", "shared_error", "correct_control"]:
            selected = 0
            for row in sorted(predictions, key=lambda r: len(examples[r["id"]][1])):
                a, b, y = row["mlp"], row["diffusion"], row["label"]
                matches = {"diffusion_fix": a != y and b == y,
                           "diffusion_regression": a == y and b != y,
                           "shared_error": a != y and b != y,
                           "correct_control": a == y and b == y}[category]
                pattern = (y, a, b)
                if not matches or pattern in used:
                    continue
                used.add(pattern)
                english, kazakh = examples[row["id"]]
                review.append(dict(target=target, category=category, id=row["id"],
                    reference_label=labels[y], mlp=labels[a], diffusion=labels[b],
                    word_tfidf=labels[tfidf[row["id"]]["predicted"]], english=english, kazakh=kazakh,
                    translation_review="", label_review="", reviewer_notes=""))
                selected += 1
                if selected == 2:
                    break
    with (destination / "examples_for_review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    (destination / "error_analysis.json").write_text(json.dumps(dict(confusions=error_counts, changes=changes), indent=2))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    blue, orange, ink = "#2563a6", "#d36a28", "#172331"
    def slide(title, number):
        fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
        fig.text(.055, .91, title, fontsize=25, weight="bold", color=ink)
        fig.text(.055, .035, "ScamBench KK | 23 November 2026 | " + ("verified XLM-R results included" if gpu_verified else "CPU results; GPU pending"), fontsize=9, color="#596474")
        fig.text(.95, .035, str(number), ha="right", fontsize=10, color="#596474")
        return fig
    def bullets(fig, lines, y=.75, spacing=.115, size=19):
        for line in lines:
            wrapped = textwrap.fill(line, 92)
            fig.text(.065, y, wrapped, fontsize=size, color=ink, va="top", linespacing=1.5)
            y -= spacing + .035 * wrapped.count("\n")
    with PdfPages(destination / "defense_slides.pdf") as pdf:
        fig = slide("Kazakh scam detection: what we built and tested", 1)
        bullets(fig, ["Dataset extension: three risk labels and 300 Kazakh reference conversations.",
            "Evaluation: identify split leakage and keep conversation families together.",
            "Experiments: lexical baselines, action prediction and conditional label diffusion.",
            ("XLM-R: three epochs completed; final model and all 600 test predictions verified." if gpu_verified else "XLM-R: evaluation pending.")], spacing=.13)
        pdf.savefig(fig); plt.close(fig)

        fig = slide("Two prediction tasks; conversation text is the input", 2)
        bullets(fig, ["Risk classification: LEGITIMATE / SUSPICIOUS / SCAM. Binary reporting merges the last two predictions as ATTACK.",
            "Action prediction: predict the original agent decision, such as refuse, audit or request_verification.",
            "The 300-pair test contains five of the thirteen action types. Action results use supported-class macro-F1; full-schema scores are also saved.",
            "Action labels and recorded responses are targets/metadata, never input features."], spacing=.115, size=18)
        pdf.savefig(fig); plt.close(fig)

        fig = slide("Family-safe evaluation and a fixed development set", 3)
        bullets(fig, ["Earlier audit: 83% of original validation rows had a conversation-family counterpart in training. A corrected split keeps families intact.",
            "Current mixed training: 11,227 English + 3,600 silver Kazakh rows.",
            "Development: 1,208 English + 399 silver Kazakh rows; used for checkpoint selection.",
            "Historical evaluation: 300 EN/KK pairs from 273 families. No training/development family overlaps evaluation."], size=18)
        pdf.savefig(fig); plt.close(fig)

        fig = slide("Initial risk results: giving the model Kazakh helps", 4)
        ax = fig.add_axes([.12,.22,.82,.56])
        names = ["Word TF-IDF\nEnglish only", "Word TF-IDF\nEnglish + silver", "MLP\nSVD + silver", "Diffusion\nSVD + silver"]
        en_values = [risk["word_en"]["test_en"]["macro_f1"],risk["word_mixed"]["test_en"]["macro_f1"],dense["risk"]["test_en"]["mlp"]["macro_f1"],dense["risk"]["test_en"]["diffusion"]["macro_f1"]]
        kk_values = [risk["word_en"]["test_kk"]["macro_f1"],risk["word_mixed"]["test_kk"]["macro_f1"],dense["risk"]["test_kk"]["mlp"]["macro_f1"],dense["risk"]["test_kk"]["diffusion"]["macro_f1"]]
        x = np.arange(4)
        for offset, vals, color, label in [(-.18,en_values,blue,"English"),(.18,kk_values,orange,"Kazakh")]:
            bars=ax.bar(x+offset,vals,.35,color=color,label=label)
            ax.bar_label(bars,fmt="%.3f",padding=4,fontsize=11)
        ax.set_xticks(x,names); ax.set_ylim(0,.85); ax.set_ylabel("Three-class macro-F1"); ax.legend(frameon=False)
        fig.text(.12,.11,"Matched MLP/diffusion heads share SVD features. Full TF-IDF uses a different representation.",fontsize=11)
        pdf.savefig(fig); fig.savefig(destination/"risk_results.png",dpi=170); plt.close(fig)

        fig = slide("Initial action results: five supported classes", 5)
        ax=fig.add_axes([.12,.23,.82,.55])
        en_values=[action["word_en"]["test_en"]["macro_f1_supported"],action["word_mixed"]["test_en"]["macro_f1_supported"],dense["action"]["test_en"]["mlp"]["macro_f1_supported"],dense["action"]["test_en"]["diffusion"]["macro_f1_supported"]]
        kk_values=[action["word_en"]["test_kk"]["macro_f1_supported"],action["word_mixed"]["test_kk"]["macro_f1_supported"],dense["action"]["test_kk"]["mlp"]["macro_f1_supported"],dense["action"]["test_kk"]["diffusion"]["macro_f1_supported"]]
        for offset,vals,color,label in [(-.18,en_values,blue,"English"),(.18,kk_values,orange,"Kazakh")]:
            bars=ax.bar(x+offset,vals,.35,color=color,label=label); ax.bar_label(bars,fmt="%.3f",padding=4,fontsize=11)
        ax.set_xticks(x,names); ax.set_ylim(0,.65); ax.set_ylabel("Supported-class macro-F1"); ax.legend(frameon=False)
        fig.text(.12,.115,"Test support: refuse 99 | engage_legitimate 88 | request_verification 87 | audit 25 | ignore 1",fontsize=10)
        fig.text(.12,.08,"These scores measure agreement with recorded actions, not the consequences of executing them.",fontsize=10)
        pdf.savefig(fig); fig.savefig(destination/"action_results.png",dpi=170); plt.close(fig)

        fig = slide("Initial conditional diffusion comparison", 6)
        bullets(fig,["Adaptation of CARD (Han, Zheng and Zhou, NeurIPS 2022): generate label vectors conditioned on text features and a frozen classifier.",
            "Train-fitted word TF-IDF -> 128-dimensional SVD -> standardization. MLP and diffusion receive identical features.",
            "30 epochs per neural head, 100 diffusion steps and eight samples per prediction. Best checkpoints selected on development data.",
            "No measured improvement over the matched MLP is established. Diffusion-vs-MLP family-bootstrap intervals include zero in both languages."],size=18,spacing=.11)
        fig.text(.065,.08,"Reference: arxiv.org/abs/2206.07275 | Adaptation, not a reproduction of the paper's results.",fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        fig = slide("Initial error analysis: explain the scores", 7)
        tf_conf = {(r["reference"],r["predicted"]):r["count"] for r in error_counts["risk/tfidf"]}
        lines=[f"Word TF-IDF on Kazakh: {tf_conf.get(('SCAM','SUSPICIOUS'),0)} SCAM examples become SUSPICIOUS; {tf_conf.get(('SCAM','LEGITIMATE'),0)} become LEGITIMATE.",
               "The compressed-feature neural models also miss attacks as LEGITIMATE. The representation is a possible bottleneck; this is a hypothesis, not an isolated cause.",
               f"Diffusion vs MLP on Kazakh risk: {changes['risk']['fixes']} examples fixed, {changes['risk']['regressions']} previously correct examples broken.",
               f"On actions: {changes['action']['fixes']} fixed, {changes['action']['regressions']} broken. Review examples are exported separately; human review has not been performed."]
        bullets(fig,lines,size=18,spacing=.12)
        pdf.savefig(fig); plt.close(fig)
        fig = slide("What is complete, what remains uncertain", 8)
        bullets(fig,["Complete: dataset extension, leakage analysis, lexical baselines, offline action prediction and conditional diffusion experiments.",
            "Initial comparison: mixed-language TF-IDF was strongest on Kazakh; diffusion did not establish an improvement over its matched MLP. Follow-up experiments are shown next when available.",
            "Limits: one seed; reused historical benchmark; only five action types evaluated; no native-speaker review.",
            "Next: inspect model errors, repeat fixed settings across seeds, review translations/labels, and build a fresh evaluation set."],size=18,spacing=.11)
        pdf.savefig(fig); plt.close(fig)

        improvements = args.runs / "improvements/comparison.csv"
        if improvements.exists():
            with improvements.open(newline="") as f:
                improved_rows = list(csv.DictReader(f))
            lookup = {(r["target"],r["model"]):r for r in improved_rows if r["language"]=="test_kk"}
            fig = slide("Improvement experiments: retain more lexical detail", 9)
            ax=fig.add_axes([.055,.34,.89,.40]); ax.axis("off")
            cells=[]
            for label,before,after in [("Sparse classifier","Word TF-IDF original","Sparse selected on dev"),
                                       ("Matched MLP","MLP word128","MLP hybrid512"),
                                       ("Conditional diffusion","Diffusion word128","Diffusion hybrid512")]:
                cells.append([label,
                    f"{float(lookup['risk',before]['macro_f1']):.3f} → {float(lookup['risk',after]['macro_f1']):.3f}",
                    f"{float(lookup['action',before]['supported_macro_f1']):.3f} → {float(lookup['action',after]['supported_macro_f1']):.3f}"])
            table=ax.table(cellText=cells,colLabels=["Model","Risk macro-F1 (3 classes)","Action macro-F1 (5 supported)"],
                           cellLoc="center",loc="center",colWidths=[.32,.33,.35])
            table.auto_set_font_size(False); table.set_fontsize(14); table.scale(1,2.7)
            for (row,column),cell in table.get_celld().items():
                cell.set_edgecolor("#dddddd")
                if row==0: cell.set_facecolor("#eaf0f7"); cell.set_text_props(weight="bold")
            fig.text(.065,.24,"Sparse search: features, regularization and class weights selected using development data only.",fontsize=12)
            fig.text(.065,.18,"Dense feature change: word-only 128D → word + character 512D; other settings unchanged.",fontsize=12)
            fig.text(.065,.12,"Same 300-pair historical test; exploratory follow-up, one seed. Fresh evaluation is still needed.",fontsize=12)
            pdf.savefig(fig); fig.savefig(destination/"improvement_results.png",dpi=170); plt.close(fig)
            effects=load(args.runs/"improvements/family_bootstrap.json")
            fig=slide("Improvement evidence and remaining uncertainty",10)
            lines=[]
            for target in ["risk","action"]:
                effect=effects[f"{target}/test_kk/diffusion_hybrid512_minus_word128"]
                lo,hi=effect["ci95"]
                lines.append(f"{target.title()}: richer-feature diffusion minus initial diffusion, paired-family 95% interval [{lo:+.3f}, {hi:+.3f}].")
            lines += ["Improving the diffusion system does not by itself isolate a diffusion-head advantage: retain the matched MLP comparison.",
                      "The test informed the follow-up hypotheses. Treat these as exploratory findings; collect more seeds and a fresh reviewed evaluation set."]
            bullets(fig,lines,size=18,spacing=.12)
            pdf.savefig(fig); plt.close(fig)

        if gpu_verified:
            fig=slide("XLM-R: completed and verified",11)
            bullets(fig,["Three epochs completed; best checkpoint is epoch 2, update 928. Development macro-F1: 0.9248.",
                "English test: three-class macro-F1 0.6873; merged-binary 0.9338.",
                "Kazakh test: 0.6341 / 0.8772; word TF-IDF: 0.6675 / 0.8983. Paired-family intervals include zero.",
                "Kazakh SCAM: 25 correct, 62 called SUSPICIOUS, 13 LEGITIMATE. No established improvement; this is one configuration."],size=18,spacing=.115)
            pdf.savefig(fig); plt.close(fig)

    notes = """SPEAKING NOTES — use alongside defense_slides.pdf

1. We extended ScamBench with a third risk class and a 300-conversation Kazakh evaluation slice, then used it to investigate cross-lingual transfer.
2. Risk and action prediction answer different questions. The action field is a prediction target, never an input. We predict recorded actions; we are not evaluating an executing agent.
3. Conversation restylings leak across the original splits. Current experiments hold whole families out, including their translated copies. New scores use a smaller training pool than the old 0.687 experiment.
4. In the current matched split, word TF-IDF rises from 0.409 to 0.667 Kazakh three-class macro-F1 with silver data. Its merged-binary macro-F1 is 0.898. These are different tasks, not interchangeable scores.
5. Word TF-IDF mixed training reaches 0.415 macro-F1 over the five action types present in the Kazakh test. We trained a 13-action schema, but eight actions cannot be evaluated on this slice. There is only one ignore example.
6. Diffusion learns to remove noise from a label vector while conditioning on text features and a frozen neural classifier. It is a CARD-inspired adaptation. On the same features it has not established an improvement over the MLP. A negative result is part of the experiment.
7. Word TF-IDF often calls SCAM examples SUSPICIOUS; weaker compressed-feature models also miss attacks as legitimate. The exported CSV supplies concrete examples for human inspection. Do not claim that those examples have already been reviewed.
8. XLM-R has no completed evaluation received yet. Native-speaker review, more seeds and a fresh test set remain future work. Report these limits plainly.

REVIEW TASK
Open examples_for_review.csv in Excel. Inspect meaning preservation and source-label plausibility independently. Fill translation_review, label_review and reviewer_notes. Record reviewer identity and number actually reviewed. A selected review sample is diagnostic; it does not estimate whole-dataset quality without a suitable sampling design.
"""
    if gpu_verified:
        notes=notes.replace("XLM-R has no completed evaluation received yet.", "XLM-R completed three epochs and restores the best epoch-2 checkpoint. Verified English/Kazakh macro-F1 is 0.6873/0.6341; Kazakh binary macro-F1 is 0.8772. The matched sparse baseline remains higher in point estimates; paired intervals include zero. Supplied model weights reproduce all 600 predictions.")
    (destination/"speaking_notes.txt").write_text(notes,encoding="utf-8")
    if (args.runs/"improvements/comparison.csv").exists():
        with (destination/"speaking_notes.txt").open("a",encoding="utf-8") as f:
            f.write("\nFOLLOW-UP SLIDES 9–10\nWe tested two explicit follow-ups: sparse model tuning on development data, and a richer word/character representation for matched MLP/diffusion heads. Show both original and follow-up scores; do not silently replace the first experiments. Distinguish gains over the initial diffusion system from evidence that diffusion beats its matched MLP. Test reuse makes these exploratory findings. Read the family-bootstrap intervals on slide 10.\n")
    with zipfile.ZipFile(destination/"defense_pack.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(destination.iterdir()):
            if path.is_file() and path.suffix != ".zip":
                z.write(path,path.name)
        for name in ["results.csv","results.html","paired_family_bootstrap.json"]:
            z.write(args.runs/name,name)
        if (args.runs/"improvements/comparison.csv").exists():
            for name in ["comparison.csv","comparison.html","family_bootstrap.json","selection_protocol.json"]:
                z.write(args.runs/"improvements"/name,"improvements/"+name)
    print("Slides:",destination/"defense_slides.pdf")
    print("Review examples:",len(review))
    print("Pack:",destination/"defense_pack.zip")


if __name__ == "__main__":
    main()
