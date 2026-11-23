"""Replay fictional conversations through saved classifiers, on CPU only.

This is a diagnostic demo, not a chatbot and not an execution agent. Authored
expectations are not independently validated benchmark labels.
"""
import argparse
import csv
import hashlib
import html
import json
import pathlib

from portable_text_model import PortableClassifier, flatten


class ConversationPredictor:
    def __init__(self, models, risk_model="risk"):
        self.risk=PortableClassifier(models,risk_model)
        self.action=PortableClassifier(models,"action")

    def predict(self, messages):
        text=flatten(messages)
        risk=self.risk.predict([text])[0]
        action=self.action.predict([text])[0]
        return dict(risk=risk,merged_binary="LEGITIMATE" if risk=="LEGITIMATE" else "ATTACK",action=action)

    def replay(self, messages):
        flatten(messages)
        texts=[flatten(messages[:i]) for i in range(1,len(messages)+1)]
        risk=self.risk.predict(texts)
        action=self.action.predict(texts)
        return [dict(turn=i,risk=r,merged_binary="LEGITIMATE" if r=="LEGITIMATE" else "ATTACK",action=a)
                for i,(r,a) in enumerate(zip(risk,action),1)]


def xlmr_predictions(path, texts):
    """Optional completed XLM-R risk model; no GPU competition with training."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from train_transformer import EncodedRows
    path=pathlib.Path(path)
    run=json.loads((path/"run.json").read_text(encoding="utf-8"))
    if run.get("smoke") or run.get("target","risk")!="risk":
        raise ValueError("Use a completed full risk run, not a smoke or action checkpoint")
    if not (path/"config.json").exists():
        raise ValueError("Final model is not saved yet; leave the active training run alone")
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to("cpu").eval()
    encoded=EncodedRows([dict(text=t,label=0) for t in texts],tokenizer,run["max_length"])
    loader=DataLoader(encoded,batch_size=2,collate_fn=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8))
    labels=run["classes"]; result=[]
    with torch.inference_mode():
        for batch in loader:
            logits=model(**{k:v for k,v in batch.items() if k!="labels"}).logits
            result.extend(labels[i] for i in logits.argmax(-1).tolist())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",default="models")
    parser.add_argument("--risk-model",choices=["risk","risk_baseline"],default="risk")
    parser.add_argument("--cases",type=pathlib.Path)
    parser.add_argument("--conversation",type=pathlib.Path,help="Your JSON message list, or an object containing messages")
    parser.add_argument("--output",type=pathlib.Path,default=pathlib.Path("demo_results"))
    parser.add_argument("--xlmr-run",help="Optional final XLM-R model folder; inference stays on CPU")
    args=parser.parse_args()
    if bool(args.cases)==bool(args.conversation): parser.error("Choose --cases or --conversation")
    if args.cases:
        data=json.loads(args.cases.read_text(encoding="utf-8")); cases=data["cases"]
    else:
        data=json.loads(args.conversation.read_text(encoding="utf-8"))
        messages=data["messages"] if isinstance(data,dict) else data
        cases=[dict(id="custom",language="custom",messages=messages)]
    predictor=ConversationPredictor(args.models,args.risk_model)
    results=[]
    for case in cases:
        timeline=predictor.replay(case["messages"])
        final=timeline[-1]
        item=dict(id=case["id"],family=case.get("family",case["id"]),language=case.get("language","custom"),
                  prediction=final,timeline=timeline,authored_expected_risk=case.get("expected_risk"),
                  authored_acceptable_actions=case.get("acceptable_actions",[]))
        if case.get("expected_risk"):
            item["matches_authored_risk"]=final["risk"]==case["expected_risk"]
            item["matches_authored_binary"]=(final["risk"]=="LEGITIMATE")== (case["expected_risk"]=="LEGITIMATE")
            item["matches_authored_action_set"]=final["action"] in case["acceptable_actions"]
        results.append(item)
        print(case["id"],final["risk"],"|",final["action"],flush=True)
    if args.xlmr_run:
        texts=[flatten(c["messages"][:i]) for c in cases for i in range(1,len(c["messages"])+1)]
        labels=xlmr_predictions(args.xlmr_run,texts)
        offset=0
        for case,result in zip(cases,results):
            count=len(case["messages"])
            result["xlmr_timeline"]=[dict(turn=i+1,risk=label) for i,label in enumerate(labels[offset:offset+count])]
            result["xlmr_final_risk"]=labels[offset+count-1]
            offset+=count
        run=json.loads((pathlib.Path(args.xlmr_run)/"run.json").read_text(encoding="utf-8-sig"))
        bundle_model=dict(model=run["model"],configured_epochs=run["epochs"],max_length=run["max_length"],
                          device="cpu",scope="risk only; action predictions remain from the sparse CPU classifier")
    by_id={r["id"]:r for r in results}; checks=[]
    for case in cases:
        if case.get("same_intent_as"):
            a,b=by_id[case["same_intent_as"]]["prediction"],by_id[case["id"]]["prediction"]
            checks.append(dict(base=case["same_intent_as"],variant=case["id"],kind=case["variant"],
                               risk_unchanged=a["risk"]==b["risk"],binary_unchanged=a["merged_binary"]==b["merged_binary"],
                               action_unchanged=a["action"]==b["action"]))
    args.output.mkdir(parents=True,exist_ok=True)
    source=args.cases or args.conversation
    bundle=dict(status="authored diagnostic challenge; not gold or population-performance evaluation",
                cases_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                limitations=["Examples and expected labels are authored for testing, not independently adjudicated.",
                             "Kazakh wording has not had native-speaker review.",
                             "Related language pairs and variants are dependent examples.",
                             "Prefixes lack future evidence and are not graded against final labels.",
                             "SVM outputs are labels, not calibrated confidence; no action is executed."],
                models={"risk":predictor.risk.metadata,"action":predictor.action.metadata},
                results=results,invariance_checks=checks)
    if args.xlmr_run: bundle["xlmr_model"]=bundle_model
    (args.output/"challenge_results.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding="utf-8")
    cards=[]
    for case,result in zip(cases,results):
        turns=[]
        for i,(message,prediction) in enumerate(zip(case["messages"],result["timeline"])):
            xlmr=(" | XLM-R risk: "+html.escape(result["xlmr_timeline"][i]["risk"])) if "xlmr_timeline" in result else ""
            turns.append(f'<div class="turn"><strong>{html.escape(message.get("role","message"))}</strong>'
                         f'<pre>{html.escape(message["content"])}</pre><small>After turn {prediction["turn"]}: '
                         f'CPU risk: {html.escape(prediction["risk"])} | CPU action: {html.escape(prediction["action"])}{xlmr}</small></div>')
        xlmr=(" | XLM-R: "+html.escape(result["xlmr_final_risk"])) if "xlmr_final_risk" in result else ""
        cards.append(f'<details><summary>{html.escape(case["id"])} — authored: {html.escape(str(result["authored_expected_risk"]))}; '
                     f'model: {html.escape(result["prediction"]["risk"])} / {html.escape(result["prediction"]["action"])}{xlmr}</summary>'+"".join(turns)+"</details>")
    page='''<!doctype html><meta charset="utf-8"><title>Conversation challenge replay</title>
<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:36px auto;padding:0 24px}details{border:1px solid #ccd4df;border-radius:8px;padding:14px;margin:12px 0}summary{cursor:pointer;font-weight:600}.turn{border-top:1px solid #ddd;padding:12px 0}pre{white-space:pre-wrap;font:inherit}small{color:#435a76}</style>
<h1>Conversation challenge replay</h1><p>Saved CPU classifiers; no actions executed. Author-intended labels are diagnostic expectations, not validated gold annotations. Language pairs and variants are dependent. Prefixes lack later evidence and are not graded against final labels. Open a case to inspect every turn.</p>'''
    page+='<p>Risk model: '+html.escape(predictor.risk.metadata["source"])+"; features: "+html.escape(predictor.risk.metadata["features"])+".</p>"+"".join(cards)
    (args.output/"challenge_replay.html").write_text(page,encoding="utf-8")
    with (args.output/"challenge_results.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["id","language","expected_risk","predicted_risk","predicted_action","risk_matches_authored","binary_matches_authored"]
        if args.xlmr_run: fields.append("xlmr_predicted_risk")
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in results:
            record=dict(id=r["id"],language=r["language"],expected_risk=r["authored_expected_risk"],
                predicted_risk=r["prediction"]["risk"],predicted_action=r["prediction"]["action"],
                risk_matches_authored=r.get("matches_authored_risk",""),binary_matches_authored=r.get("matches_authored_binary",""))
            if args.xlmr_run: record["xlmr_predicted_risk"]=r["xlmr_final_risk"]
            writer.writerow(record)
    print("Saved",args.output/"challenge_results.json")


if __name__=="__main__": main()
