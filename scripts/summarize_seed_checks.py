"""Summarize fixed semantic-head repeats without selecting the best test seed."""
import argparse
import csv
import json
import pathlib

import joblib
import numpy as np

from train_transformer import score, write


def load(path): return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=pathlib.Path,default=pathlib.Path("data/experiments/deadline_2026-11-23"))
    args=parser.parse_args()
    seeds=[0,1,42]; output=args.runs/"seed_stability"; output.mkdir(exist_ok=True)
    rows=[]; summary=[]; ensembles={}
    fixed=["target","variant","epochs","steps","samples","eval_every","dimensions","feature_kind","hidden","batch_size","classes","train_rows","dev_rows","manifest_sha256"]
    for target in ["risk","action"]:
        runs={seed:args.runs/f"diffusion_{target}_semantic_seed{seed}" for seed in seeds}
        reference=load(runs[42]/"run.json")
        cache_hash=None
        for seed,folder in runs.items():
            run=load(folder/"run.json")
            if run["seed"]!=seed or any(run[k]!=reference[k] for k in fixed):
                raise ValueError("Only the seed may change in a stability check")
            digest=joblib.load(folder/"features.joblib")["semantic_sha256"]
            if cache_hash is not None and digest!=cache_hash: raise ValueError("Feature caches differ")
            cache_hash=digest
            metrics=load(folder/"test_metrics.json")
            for language in ["test_en","test_kk"]:
                for model in ["mlp","diffusion"]:
                    m=metrics[language][model]
                    rows.append(dict(target=target,language=language,model=model,seed=seed,
                        macro_f1=m["macro_f1"],supported_macro_f1=m["macro_f1_supported"],accuracy=m["accuracy"],
                        binary_macro_f1=m.get("binary_macro_f1","")))
        for language in ["test_en","test_kk"]:
            predictions={seed:load(folder/f"{language}_predictions.json") for seed,folder in runs.items()}
            identity=[(r["id"],r["family"],r["label"]) for r in predictions[42]]
            for seed in seeds:
                assert [(r["id"],r["family"],r["label"]) for r in predictions[seed]]==identity
            for model in ["mlp","diffusion"]:
                selected=[r for r in rows if r["target"]==target and r["language"]==language and r["model"]==model]
                primary="macro_f1" if target=="risk" else "supported_macro_f1"
                values=np.array([r[primary] for r in selected])
                summary.append(dict(target=target,language=language,model=model,metric=primary,seeds=seeds,
                    mean=float(values.mean()),sample_std=float(values.std(ddof=1)),minimum=float(values.min()),maximum=float(values.max())))
                if model=="diffusion":
                    votes=np.mean([np.array([r["sample_votes"] for r in predictions[seed]]) for seed in seeds],axis=0)
                else:
                    labels=np.array([[r["mlp"] for r in predictions[seed]] for seed in seeds])
                    votes=np.eye(len(reference["classes"]))[labels].mean(axis=0)
                predicted=votes.argmax(-1).tolist()
                metrics=score([label for _,_,label in identity],predicted,reference["classes"])
                key=f"{target}/{language}/{model}"
                ensembles[key]=metrics
                write(output/f"{target}_{language}_{model}_ensemble_predictions.json",[
                    dict(id=identifier,family=family,label=label,predicted=p)
                    for (identifier,family,label),p in zip(identity,predicted)])
    with (output/"per_seed.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(output/"summary.json",dict(
        protocol="Fixed semantic+hand features, split and hyperparameters; seeds 0,1,42. Every seed is retained; no best-test-seed selection.",
        interpretation="Sample standard deviation across three seeds is descriptive, not a confidence interval. Other architectures and XLM-R retain their original seed counts.",
        ensemble_rule="Equal weights. MLP: average one-hot votes; diffusion: average saved sample-vote vectors. Fixed class order breaks ties. No test-fitted weights.",
        summary=summary,ensembles=ensembles))
    for row in summary:
        if row["language"]=="test_kk": print(row)
    for key,metrics in ensembles.items():
        if "/test_kk/" in key: print("Equal ensemble",key,metrics["macro_f1"],metrics["macro_f1_supported"])


if __name__=="__main__": main()
