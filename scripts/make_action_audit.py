"""Build a blinded diagnostic action-label task; score an exported answer file."""
import argparse
import collections
import csv
import html
import json
import pathlib
import random

from make_human_tasks import page, verdict_buttons
from train_transformer import read


def build(args):
    rng=random.Random(42)
    rows=read(args.data,"test_kk")
    groups=collections.defaultdict(list)
    for row in rows: groups[row["decision"]].append(row)
    quotas={"refuse":8,"engage_legitimate":8,"request_verification":7,"audit":6,"ignore":1}
    selected=[]; families=set()
    for action,count in quotas.items():
        candidates=groups[action].copy(); rng.shuffle(candidates)
        picked=0
        for row in candidates:
            if row["family"] in families: continue
            selected.append(row); families.add(row["family"]); picked+=1
            if picked==count: break
        if picked!=count: raise ValueError("Not enough distinct action families")
    rng.shuffle(selected)
    schema=json.loads((pathlib.Path(args.data)/"manifest.json").read_text())["decision_classes"]
    definitions={"accept":"accept an offered interaction or request", "allow_safe_action":"permit a safe action",
        "audit":"inspect or audit the situation", "block_actor":"block the actor", "deny_privileged_action":"deny the requested privileged action",
        "engage_legitimate":"continue ordinary legitimate interaction", "escalate":"refer for additional intervention",
        "execute_transaction":"carry out a transaction", "ignore":"ignore the interaction", "refuse":"refuse the request",
        "request_verification":"ask for verification before proceeding", "share_safe_info":"provide non-sensitive information",
        "warn_actor":"issue a warning to the actor"}
    cards=[]; key=[]
    for i,row in enumerate(selected,1):
        identifier=f"A{i:02d}"
        # No reference labels, predictions, scenario metadata or source IDs in HTML.
        cards.append(f'<div class="card"><h2>{identifier}</h2><div class="txt kk">{html.escape(row["text"])}</div>'+
            verdict_buttons(identifier,[(a,a.replace("_"," ")) for a in schema]+[("cannot_determine","Cannot determine from text")])+"</div>")
        key.append(dict(task_id=identifier,id=row["id"],family=row["family"],reference_action=row["decision"]))
    glossary="<details><summary>Action glossary</summary><ul>"+"".join(f"<li><strong>{a}</strong>: {definitions[a]}</li>" for a in schema)+"</ul></details>"
    instructions=("Choose the action best supported by each Kazakh conversation. <strong>Reference labels and model predictions are hidden.</strong> "
        "Use <em>Cannot determine from text</em> when the evidence does not distinguish actions; do not guess just to match an expected answer. "
        "Use the note for your reasoning or a translation issue. There may be several defensible actions. "
        "The page shows the flattened message text used by the classifiers, without role segmentation; missing role context may contribute to uncertainty. "
        "The glossary gives plain-language meanings, not the source annotator's full policy. "
        "This is a stratified diagnostic sample, not an estimate of overall annotation quality. Download answers when finished. "
        '<br>Source: <a href="https://huggingface.co/datasets/shaw/scambench-training">ScamBench</a>; Kazakh extension, CC-BY-SA-4.0. '+glossary)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"blind_action_audit_30.html").write_text(page("Independent action-label audit — 30 conversations",instructions,"".join(cards),"scambench_blind_actions_30",len(selected),extra_class="single"),encoding="utf-8")
    with (args.out/"action_audit_private_key.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(key[0])); writer.writeheader(); writer.writerows(key)
    print("Reviewer page:",args.out/"blind_action_audit_30.html")
    print("Private scoring key stays separate. No human answers have been collected.")


def score_answers(args):
    from sklearn.metrics import classification_report
    answers=json.loads(args.answers.read_text())
    with (args.out/"action_audit_private_key.csv").open(newline="") as f: key=list(csv.DictReader(f))
    valid={r["task_id"]:r["reference_action"] for r in key}
    schema=json.loads((pathlib.Path(args.data)/"manifest.json").read_text())["decision_classes"]
    unknown=[]; truth=[]; chosen=[]
    for task,reference in valid.items():
        answer=answers.get(task,{}).get("verdict")
        if not answer: continue
        if answer=="cannot_determine": unknown.append(task); continue
        if answer not in schema: raise ValueError("Unknown action answer")
        truth.append(reference); chosen.append(answer)
    result=dict(tasks=len(valid),answered=len(truth)+len(unknown),cannot_determine=len(unknown),
                caveat="Diagnostic stratified sample; agreement with source annotation is not objective correctness or a human performance ceiling.")
    if truth:
        result["agreement_on_determinate_answers"]=sum(a==b for a,b in zip(truth,chosen))/len(truth)
        result["per_class"]=classification_report(truth,chosen,labels=schema,output_dict=True,zero_division=0)
    (args.out/"action_audit_results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["build","score"])
    parser.add_argument("--data",default="data/gpu_bundle")
    parser.add_argument("--out",type=pathlib.Path,default=pathlib.Path("review/action_audit"))
    parser.add_argument("--answers",type=pathlib.Path)
    args=parser.parse_args()
    if args.command=="score" and not args.answers: parser.error("score requires --answers")
    {"build":build,"score":score_answers}[args.command](args)
