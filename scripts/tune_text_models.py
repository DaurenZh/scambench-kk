"""Select sparse classifiers on development data; evaluate only selected models.

Fixed search: word/character/hybrid TF-IDF, C in {0.25,1,4}, and optional
inverse-frequency class weights. Primary selection is silver-Kazakh dev macro-F1;
pooled EN+KK dev macro-F1 breaks ties. Gold data is read only by evaluate.
"""
import argparse
import hashlib
import json
import pathlib
import time

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from train_transformer import CLASSES, read, score, target_rows, write


def train(args):
    root, output = pathlib.Path(args.data), pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise SystemExit("Choose a fresh output directory")
    manifest = json.loads((root / "manifest.json").read_text())
    training = read(root, "train_en") + read(root, "train_silver")
    development = read(root, "dev_en") + read(root, "dev_silver")
    if {r["family"] for r in training} & {r["family"] for r in development}:
        raise ValueError("Training and development families overlap")
    texts = [r["text"] for r in training]
    dev_texts = [r["text"] for r in development]
    kk = np.array([r["language"] == "kk" for r in development])
    vectors, matrices, dev_matrices = {}, {}, {}
    for name, settings in [("word",dict(ngram_range=(1,2))),
                           ("char",dict(analyzer="char_wb",ngram_range=(3,5)))]:
        vectors[name] = TfidfVectorizer(min_df=3,max_features=200_000,sublinear_tf=True,**settings)
        matrices[name] = vectors[name].fit_transform(texts)
        dev_matrices[name] = vectors[name].transform(dev_texts)
    matrices["hybrid"] = hstack([matrices["word"],matrices["char"]],format="csr")
    dev_matrices["hybrid"] = hstack([dev_matrices["word"],dev_matrices["char"]],format="csr")
    search, chosen = [], {}
    start = time.perf_counter()
    for target in ["risk","action"]:
        classes = CLASSES if target == "risk" else manifest["decision_classes"]
        y = np.array([r["label"] for r in target_rows(training,target,classes)])
        dy = np.array([r["label"] for r in target_rows(development,target,classes)])
        best, best_model, selected = None, None, None
        for kind in ["word","char","hybrid"]:
            for c in [.25,1.0,4.0]:
                for weight in [None,"balanced"]:
                    model = LinearSVC(C=c,class_weight=weight,random_state=42,max_iter=5000)
                    model.fit(matrices[kind],y)
                    pred = model.predict(dev_matrices[kind])
                    row = dict(target=target,features=kind,C=c,class_weight=weight,
                               dev_kk_macro_f1=f1_score(dy[kk],pred[kk],average="macro",labels=list(range(len(classes))),zero_division=0),
                               dev_pooled_macro_f1=f1_score(dy,pred,average="macro",labels=list(range(len(classes))),zero_division=0))
                    search.append(row)
                    rank = (row["dev_kk_macro_f1"],row["dev_pooled_macro_f1"])
                    if best is None or rank > best:
                        best,best_model,selected = rank,model,row
                    print(target,kind,c,weight,round(row["dev_kk_macro_f1"],4),flush=True)
        chosen[target] = dict(selected,classes=classes)
        joblib.dump(dict(model=best_model,word=vectors["word"],char=vectors["char"],kind=selected["features"]),output/f"{target}.joblib")
    write(output/"search.json",search)
    write(output/"run.json",dict(selected=chosen,train_rows=len(training),dev_rows=len(development),
         selection="silver-Kazakh dev macro-F1, pooled dev tie-break; gold excluded from search",
         search_seconds=time.perf_counter()-start,
         manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()))
    print("Selected on development data only:",json.dumps(chosen,indent=2))


def evaluate(args):
    root,output = pathlib.Path(args.data),pathlib.Path(args.output)
    run = json.loads((output/"run.json").read_text())
    if run["manifest_sha256"] != hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest():
        raise ValueError("Bundle differs from training")
    results = {}
    for target,config in run["selected"].items():
        artifact = joblib.load(output/f"{target}.joblib")
        results[target] = {}
        for name in ["test_en","test_kk"]:
            rows = target_rows(read(root,name),target,config["classes"])
            texts = [r["text"] for r in rows]
            if artifact["kind"] == "hybrid":
                x = hstack([artifact["word"].transform(texts),artifact["char"].transform(texts)],format="csr")
            else:
                x = artifact[artifact["kind"]].transform(texts)
            predictions = artifact["model"].predict(x).tolist()
            results[target][name] = score([r["label"] for r in rows],predictions,config["classes"])
            write(output/f"{target}_{name}_predictions.json",[dict(id=r["id"],family=r["family"],label=r["label"],predicted=p) for r,p in zip(rows,predictions)])
    write(output/"test_metrics.json",results)
    for target,languages in results.items():
        print(target,{k:{metric:round(v[metric],4) for metric in ["macro_f1","macro_f1_supported","accuracy"]} for k,v in languages.items()})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["train","evaluate"])
    parser.add_argument("--data",default="data/gpu_bundle")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    {"train":train,"evaluate":evaluate}[args.command](args)
