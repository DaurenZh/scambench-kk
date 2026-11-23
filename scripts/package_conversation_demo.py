"""Package the verified CPU demo, notebook, frozen cases and example outputs."""
import hashlib
import json
import pathlib
import shutil
import zipfile

import nbformat


def main():
    root=pathlib.Path(__file__).resolve().parents[1]
    bundle=root/"data/conversation_demo_bundle"
    runs=root/"data/experiments/deadline_2026-11-23"
    shutil.copy2(root/"notebooks/08_conversation_demo.ipynb",bundle/"08_conversation_demo.ipynb")
    for name in ["conversation_demo.py","portable_text_model.py","train_transformer.py"]:
        shutil.copy2(root/"scripts"/name,bundle/name)
    nbformat.validate(nbformat.read(bundle/"08_conversation_demo.ipynb",as_version=4))
    source=json.loads((bundle/"synthetic_conversations.json").read_text(encoding="utf-8"))
    summary=dict(status="authored diagnostic cases, not an independent benchmark",cases=len(source["cases"]),
                 base_families=len({c["family"] for c in source["cases"]}),models={})
    for name,folder in [("selected",runs/"challenge_evaluation"),("baseline",runs/"challenge_evaluation_baseline")]:
        result=json.loads((folder/"challenge_results.json").read_text(encoding="utf-8"))
        summary["models"][name]=dict(
            source=result["models"]["risk"]["source"],
            author_risk_matches=sum(r["matches_authored_risk"] for r in result["results"]),
            author_binary_matches=sum(r["matches_authored_binary"] for r in result["results"]),
            same_intent_variants=len(result["invariance_checks"]),
            variants_changing_risk=sum(not r["risk_unchanged"] for r in result["invariance_checks"]),
            variants_changing_action=sum(not r["action_unchanged"] for r in result["invariance_checks"]))
        shutil.copytree(folder,bundle/"example_outputs"/name,dirs_exist_ok=True)
    (runs/"challenge_evaluation/diagnostic_summary.json").write_text(json.dumps(summary,indent=2))
    (bundle/"example_outputs/diagnostic_summary.json").write_text(json.dumps(summary,indent=2))
    xlmr=runs/"challenge_evaluation_xlmr"
    if xlmr.exists():
        shutil.copytree(xlmr,bundle/"example_outputs/xlmr",dirs_exist_ok=True)
    manifest={}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name!="checksums.json":
            manifest[str(path.relative_to(bundle))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (bundle/"checksums.json").write_text(json.dumps(manifest,indent=2))
    archive=bundle.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(bundle.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts: z.write(path,path.relative_to(bundle))
    print(json.dumps(summary,indent=2))
    print("Portable ZIP:",archive,round(archive.stat().st_size/1e6,2),"MB")


if __name__=="__main__": main()
