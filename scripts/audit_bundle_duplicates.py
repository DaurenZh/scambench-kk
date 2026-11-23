"""Audit full-text duplicates across fixed experiment partitions without changing them."""
import argparse
import collections
import hashlib
import pathlib

from train_transformer import read, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=pathlib.Path,default=pathlib.Path("data/gpu_bundle"))
    parser.add_argument("--output",type=pathlib.Path,default=pathlib.Path("data/experiments/deadline_2026-11-23/full_text_duplicate_audit.json"))
    args=parser.parse_args()
    groups={"train":read(args.data,"train_en")+read(args.data,"train_silver"),
            "development":read(args.data,"dev_en")+read(args.data,"dev_silver"),
            "test":read(args.data,"test_en")+read(args.data,"test_kk")}
    results={}
    for normalized in [False,True]:
        def fingerprint(text):
            if normalized: text=" ".join(text.split())
            return hashlib.sha256(text.encode()).hexdigest()
        training=collections.defaultdict(list)
        for row in groups["train"]: training[fingerprint(row["text"])].append(row)
        splits={}
        for name in ["development","test"]:
            overlap=[]
            for row in groups[name]:
                matches=training.get(fingerprint(row["text"]),[])
                if matches:
                    overlap.append(dict(id=row["id"],family=row["family"],language=row["language"],label=row["label"],
                        train_ids=[r["id"] for r in matches],train_families=sorted({r["family"] for r in matches}),
                        train_labels=sorted({r["label"] for r in matches})))
            splits[name]=dict(rows=len(groups[name]),overlapping_rows=len(overlap),
                overlapping_by_language=dict(collections.Counter(r["language"] for r in overlap)),overlap=overlap)
        results["whitespace_normalized" if normalized else "exact"]=splits
    results["interpretation"]="Family IDs are disjoint, but identical full texts can cross families. Do not alter the active run; report development-selection caveat and keep final test evaluation separate. This audit does not rule out near-duplicates."
    args.output.parent.mkdir(parents=True,exist_ok=True); write(args.output,results)
    for mode in ["exact","whitespace_normalized"]:
        print(mode,{name:(value["overlapping_rows"],value["rows"]) for name,value in results[mode].items()})


if __name__=="__main__": main()
