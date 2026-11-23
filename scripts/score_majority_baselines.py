"""Measure constant baselines selected from training labels, never test labels."""
import argparse
import collections
import json
import pathlib

from sklearn.metrics import f1_score

from train_transformer import CLASSES, read, score, target_rows, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=pathlib.Path,default=pathlib.Path("data/gpu_bundle"))
    parser.add_argument("--output",type=pathlib.Path,default=pathlib.Path("data/experiments/deadline_2026-11-23/majority_baselines.json"))
    args=parser.parse_args()
    manifest=json.loads((args.data/"manifest.json").read_text())
    training=read(args.data,"train_en")+read(args.data,"train_silver")
    result={}
    for target in ["risk","action"]:
        classes=CLASSES if target=="risk" else manifest["decision_classes"]
        labels=[r["label"] for r in target_rows(training,target,classes)]
        counts=collections.Counter(labels)
        majority=max(range(len(classes)),key=lambda i:counts[i])
        values=dict(majority_label=classes[majority],train_counts={c:counts[i] for i,c in enumerate(classes)},
                    selection="Most frequent target in mixed training data; fixed class order breaks ties")
        for language in ["test_en","test_kk"]:
            rows=target_rows(read(args.data,language),target,classes)
            values[language]=score([r["label"] for r in rows],[majority]*len(rows),classes)
        result[target]=values
    binary_counts=collections.Counter(int(r["label"]!=0) for r in training)
    binary_majority=max([0,1],key=lambda i:binary_counts[i])
    result["dedicated_binary_majority"]={}
    for language in ["test_en","test_kk"]:
        truth=[int(r["label"]!=0) for r in read(args.data,language)]
        result["dedicated_binary_majority"][language]=dict(
            predicted_label="ATTACK" if binary_majority else "LEGITIMATE",
            macro_f1=f1_score(truth,[binary_majority]*len(truth),labels=[0,1],average="macro",zero_division=0))
    write(args.output,result)
    for target in ["risk","action"]:
        print(target,result[target]["majority_label"],result[target]["test_kk"]["macro_f1"],result[target]["test_kk"]["macro_f1_supported"])


if __name__=="__main__": main()
