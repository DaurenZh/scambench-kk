"""Compare predeclared development-selected improvements with saved first runs."""
import argparse
import csv
import json
import pathlib
import zipfile

from summarize_deadline_results import paired_family_bootstrap


def load(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=pathlib.Path,default=pathlib.Path("data/experiments/deadline_2026-11-23"))
    args=parser.parse_args()
    root=args.runs
    output=root/"improvements"
    output.mkdir(exist_ok=True)
    tuned=load(root/"tuned_sparse/test_metrics.json")
    selected=load(root/"tuned_sparse/run.json")
    rows=[]
    contrasts={}
    for target in ["risk","action"]:
        baseline=load(root/f"{target}_tfidf/baselines.json")["word_mixed"]
        old_folder=root/f"diffusion_{target}_mixed_seed42"
        new_folder=root/f"diffusion_{target}_hybrid512_seed42"
        old=load(old_folder/"test_metrics.json")
        new=load(new_folder/"test_metrics.json")
        config=load(new_folder/"run.json")
        semantic_folder=root/f"diffusion_{target}_semantic_seed42"
        semantic=load(semantic_folder/"test_metrics.json") if (semantic_folder/"test_metrics.json").exists() else None
        for language in ["test_en","test_kk"]:
            systems={"Word TF-IDF original":baseline[language],
                     "Sparse selected on dev":tuned[target][language],
                     "MLP word128":old[language]["mlp"],
                     "MLP hybrid512":new[language]["mlp"],
                     "Diffusion word128":old[language]["diffusion"],
                     "Diffusion hybrid512":new[language]["diffusion"]}
            if semantic:
                for model in ["linear","mlp","diffusion"]:
                    systems[f"Semantic+hand {model}"]=semantic[language][model]
            for name,metrics in systems.items():
                rows.append(dict(target=target,language=language,model=name,
                    macro_f1=metrics["macro_f1"],supported_macro_f1=metrics["macro_f1_supported"],
                    accuracy=metrics["accuracy"],binary_macro_f1=metrics.get("binary_macro_f1","")))
            old_predictions=load(old_folder/f"{language}_predictions.json")
            new_predictions=load(new_folder/f"{language}_predictions.json")
            original_sparse=load(root/f"{target}_tfidf/word_mixed_{language}_predictions.json")
            tuned_predictions=load(root/f"tuned_sparse/{target}_{language}_predictions.json")
            for pair in [old_predictions,original_sparse,tuned_predictions]:
                assert [(r["id"],r["label"]) for r in pair]==[(r["id"],r["label"]) for r in new_predictions]
            pairs={
                "sparse_selected_minus_original":(original_sparse,tuned_predictions,"predicted","predicted"),
                "diffusion_hybrid512_minus_word128":(old_predictions,new_predictions,"diffusion","diffusion"),
                "mlp_hybrid512_minus_word128":(old_predictions,new_predictions,"mlp","mlp"),
                "diffusion_minus_mlp_hybrid512":(new_predictions,new_predictions,"mlp","diffusion")}
            if semantic:
                semantic_predictions=load(semantic_folder/f"{language}_predictions.json")
                assert [(r["id"],r["label"]) for r in semantic_predictions]==[(r["id"],r["label"]) for r in new_predictions]
                pairs["diffusion_semantic_minus_hybrid512"]=(new_predictions,semantic_predictions,"diffusion","diffusion")
                pairs["diffusion_minus_mlp_semantic"]=(semantic_predictions,semantic_predictions,"mlp","diffusion")
            for name,(before,after,a,b) in pairs.items():
                combined=[dict(family=x["family"],label=x["label"],mlp=x[a],diffusion=y[b]) for x,y in zip(before,after)]
                ci=paired_family_bootstrap(combined,len(config["classes"]))
                ci["contrast"]=name
                contrasts[f"{target}/{language}/{name}"]=ci
    with (output/"comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/"family_bootstrap.json").write_text(json.dumps(contrasts,indent=2))
    (output/"selection_protocol.json").write_text(json.dumps(dict(
        sparse=selected,
        dense="One fixed feature ablation: word-only 128D -> word+character 512D. Same 30 epochs, seed, split, hidden width, diffusion steps and development checkpoint rule.",
        semantic="Frozen LaBSE 768D + 54 text/role measurements, train-fitted standardization, otherwise same neural protocol. Labels excluded from features.",
        caveat="Historical gold scores were inspected before planning these experiments. Hyperparameters and checkpoints use development data, but confirmatory claims need a fresh test set. All tested changes are reported."),indent=2))
    table=[]
    for row in rows:
        if row["language"]!="test_kk": continue
        value=row["macro_f1"] if row["target"]=="risk" else row["supported_macro_f1"]
        table.append(f"<tr><td>{row['target']}</td><td>{row['model']}</td><td>{value:.4f}</td><td>{row['accuracy']:.4f}</td></tr>")
    page="""<!doctype html><meta charset="utf-8"><title>Development-selected improvement experiments</title>
<style>body{font:16px/1.5 system-ui;max-width:1000px;margin:36px auto;padding:0 24px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}</style>
<h1>Development-selected improvement experiments</h1><p>23 November 2026. Same family-safe training, development and historical evaluation sets as the first runs.</p>
<p>Sparse search: word/character/hybrid features, C in {0.25,1,4}, class weights absent or balanced.
Choose silver-Kazakh development macro-F1 with pooled development scores as tie-break. Gold is excluded from fitting and parameter selection.</p>
<p>Dense ablation: retain 512 word+character components instead of 128 word-only components.
Keep all other training settings and the pooled-development checkpoint criterion fixed.</p>
<p>A semantic follow-up uses frozen 768D LaBSE embeddings plus 54 safe text/role measurements. It uses the same split and neural training protocol.</p>
<h2>Kazakh results</h2><p>Risk macro-F1 uses three classes. Action macro-F1 uses the five supported evaluation classes from a thirteen-class schema.</p>
<table><tr><th>Target</th><th>Model</th><th>Macro-F1</th><th>Accuracy</th></tr>"""
    page+="".join(table)+"</table><h2>Interpretation limits</h2><p>Report every tested change, including regressions. Paired family-bootstrap intervals are saved separately. One seed and a previously inspected benchmark do not establish broad superiority. Native-speaker review is pending. XLM-R results remain separate.</p>"
    (output/"comparison.html").write_text(page,encoding="utf-8")
    with zipfile.ZipFile(output/"improvement_results.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in {".json",".csv",".html"}: z.write(path,path.name)
        for folder in [root/"tuned_sparse",root/"diffusion_risk_hybrid512_seed42",root/"diffusion_action_hybrid512_seed42",
                       root/"diffusion_risk_semantic_seed42",root/"diffusion_action_semantic_seed42"]:
            for path in folder.glob("*.json"): z.write(path,path.relative_to(root))
    for row in rows:
        if row["language"]=="test_kk": print(row)
    print("Report:",output/"comparison.html")


if __name__=="__main__":
    main()
