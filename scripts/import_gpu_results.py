"""Verify completed Windows XLM-R outputs and compare them on the paired test.

Input: a run folder, or a ZIP containing run.json, training_log.json,
validation_metrics.json, test_metrics.json and both prediction JSON files.
Weights are neither needed nor extracted. Metrics are recomputed from predictions.
"""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import zipfile

from summarize_deadline_results import paired_family_bootstrap
from train_transformer import CLASSES, read, score, write

REQUIRED=["run.json","training_log.json","validation_metrics.json","test_metrics.json",
          "test_en_predictions.json","test_kk_predictions.json"]


def source_files(source):
    source=pathlib.Path(source)
    if source.is_dir():
        candidates=[]
        for path in source.rglob("run.json"):
            folder=path.parent
            if all((folder/name).is_file() for name in REQUIRED): candidates.append(folder)
        if len(candidates)!=1:
            raise ValueError("Point to one completed run folder; expected exactly one complete result set")
        return {name:(candidates[0]/name).read_bytes() for name in REQUIRED}
    with zipfile.ZipFile(source) as archive:
        names={}
        for info in archive.infolist():
            name=info.filename.replace("\\","/")
            if name in names: raise ValueError("Duplicate ZIP entry")
            names[name]=info
        candidates=[]
        for name in names:
            if name.endswith("/run.json") or name=="run.json":
                prefix=name[:-len("run.json")]
                if all(prefix+required in names for required in REQUIRED): candidates.append(prefix)
        if len(candidates)!=1:
            raise ValueError("ZIP must contain exactly one completed result set; omit smoke and other runs")
        result={}
        for name in REQUIRED:
            info=names[candidates[0]+name]
            if info.file_size>20_000_000: raise ValueError("Unexpectedly large result JSON")
            result[name]=archive.read(info)
        return result


def align_predictions(rows, expected):
    if len(rows)!=len(expected) or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Prediction count or uniqueness mismatch")
    index={r["id"]:r for r in rows}
    if set(index)!={r["id"] for r in expected}: raise ValueError("Wrong evaluation IDs")
    aligned=[]
    for reference in expected:
        row=index[reference["id"]]
        if row["label"]!=reference["label"] or row["family"]!=reference["family"]:
            raise ValueError("Reference labels or family assignment differ")
        if type(row["predicted"]) is not int or row["predicted"] not in range(3):
            raise ValueError("Invalid predicted class index")
        aligned.append(row)
    return aligned


def verify_metrics(rows, reported):
    recomputed=score([r["label"] for r in rows],[r["predicted"] for r in rows],CLASSES)
    for name in ["macro_f1","binary_macro_f1","accuracy"]:
        value=reported.get(name)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value-recomputed[name])>1e-8:
            raise ValueError(f"Reported {name} disagrees with predictions")
    return recomputed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=pathlib.Path)
    parser.add_argument("--data",type=pathlib.Path,default=pathlib.Path("data/gpu_bundle"))
    parser.add_argument("--baselines",type=pathlib.Path,default=pathlib.Path("data/experiments/deadline_2026-11-23/risk_tfidf"))
    parser.add_argument("--output",type=pathlib.Path,default=pathlib.Path("data/experiments/deadline_2026-11-23/xlmr_mixed_risk"))
    args=parser.parse_args()
    files=source_files(args.source)
    payload={name:json.loads(content.decode("utf-8-sig")) for name,content in files.items()}
    run=payload["run.json"]
    if run.get("smoke") or run.get("target")!="risk" or run.get("variant")!="mixed":
        raise ValueError("Expected a full mixed risk experiment, not smoke/action/English-only")
    if run.get("classes")!=CLASSES or "xlm-roberta" not in run.get("model","").lower():
        raise ValueError("Unexpected label mapping or model identity")
    digest=hashlib.sha256((args.data/"manifest.json").read_bytes()).hexdigest()
    if run.get("manifest_sha256")!=digest: raise ValueError("Data bundle differs from the local comparison")
    if run.get("train_rows")!=len(read(args.data,"train_en"))+len(read(args.data,"train_silver")):
        raise ValueError("Training row count differs")
    epochs=[r.get("epoch",0) for r in payload["training_log.json"]]
    if not epochs or max(epochs)<float(run["epochs"])-.01: raise ValueError("Logs do not confirm completion of the configured epochs")
    verified,comparisons,aligned={}, {}, {}
    for language in ["test_en","test_kk"]:
        expected=read(args.data,language)
        if len(expected)!=300: raise ValueError("Expected the paired 300-conversation benchmark")
        rows=align_predictions(payload[f"{language}_predictions.json"],expected)
        aligned[language]=rows
        verified[language]=verify_metrics(rows,payload["test_metrics.json"][language])
        baseline=align_predictions(json.loads((args.baselines/f"word_mixed_{language}_predictions.json").read_text()),expected)
        for binary in [False,True]:
            combined=[dict(family=a["family"],label=int(a["label"]!=0) if binary else a["label"],
                mlp=int(b["predicted"]!=0) if binary else b["predicted"],
                diffusion=int(a["predicted"]!=0) if binary else a["predicted"])
                for a,b in zip(rows,baseline)]
            result=paired_family_bootstrap(combined,2 if binary else 3)
            result["contrast"]="XLM-R minus original mixed word TF-IDF"
            comparisons[language+("/binary" if binary else "/three_class")]=result
    if args.output.exists() and any(args.output.iterdir()): raise ValueError("Output already contains results; choose a fresh output folder")
    args.output.mkdir(parents=True,exist_ok=True)
    for name,content in files.items(): (args.output/name).write_bytes(content)
    write(args.output/"verified_metrics.json",verified)
    write(args.output/"paired_comparisons.json",comparisons)
    write(args.output/"verification.json",dict(status="verified completed run; metrics recomputed",manifest_sha256=digest,
        configured_epochs=run["epochs"],observed_final_epoch=max(epochs),test_rows_per_language=300,
        families=len({r["family"] for r in aligned["test_kk"]}),
        caveat="Historical benchmark, one seed. These checks verify alignment and metric arithmetic, not source annotation correctness."))
    for language,rows in aligned.items():
        matrix=[[0]*3 for _ in range(3)]
        for row in rows: matrix[row["label"]][row["predicted"]]+=1
        with (args.output/f"{language}_confusion.csv").open("w",newline="") as f:
            writer=csv.writer(f); writer.writerow(["reference/predicted",*CLASSES])
            for label,values in zip(CLASSES,matrix): writer.writerow([label,*values])
    print(json.dumps(verified,indent=2))
    print("Verified results ready for presentation/report integration:",args.output)


if __name__=="__main__": main()
