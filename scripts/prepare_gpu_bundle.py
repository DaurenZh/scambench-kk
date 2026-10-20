"""Package the existing experiment for a separate Windows/Linux GPU machine.

Run from the repository root: python scripts/prepare_gpu_bundle.py
No model download or GPU work happens here.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from make_group_split import conversation_of

CLASSES = ["LEGITIMATE", "SUSPICIOUS", "SCAM"]


def family(row):
    source = row.get("translated_from") if row["language"] == "kk" else row["id"]
    return conversation_of(source)


def records(frame):
    result = []
    for row in frame.to_dict("records"):
        # Match the historical text baselines: messages only, no answer metadata.
        text = "\n".join(m.get("content") or "" for m in json.loads(row["messages"]))
        result.append(dict(id=row["id"], family=family(row), text=text,
                           label=CLASSES.index(row["label_3class"]),
                           decision=row["decision_class"].replace("-", "_"),
                           language=row["language"]))
    return result


def make_sets(merged, silver, seed=42):
    gold = merged[merged.language.eq("kk")].copy()
    sources = merged.set_index("id").loc[gold.translated_from].reset_index()
    if len(gold) != 300 or not (sources.label_3class.to_numpy() == gold.label_3class.to_numpy()).all():
        raise ValueError("Expected 300 aligned English/Kazakh evaluation pairs")
    held = set(gold.apply(family, axis=1))
    english = merged[merged.language.eq("en") & merged.split.eq("train")].copy()
    english["family"] = english.id.map(conversation_of)
    english = english[~english.family.isin(held)]
    # Stratify family-level majority labels; all restylings stay together.
    families = english.groupby("family").label_3class.agg(lambda x: x.mode().iloc[0])
    _, dev_ids = train_test_split(families.index, test_size=0.1,
                                  stratify=families.values, random_state=seed)
    dev_ids = set(dev_ids)
    train_en = english[~english.family.isin(dev_ids)]
    dev_en = english[english.family.isin(dev_ids)]
    silver = silver.copy()
    silver["family"] = silver.apply(family, axis=1)
    permitted = set(english.family)
    if not set(silver.family) <= permitted:
        raise ValueError("Silver contains a source family outside the eligible English pool")
    if not set(silver.translated_from) <= set(english.id):
        raise ValueError("Silver source IDs must belong to the eligible English pool")
    source_labels = english.set_index("id").label_3class
    if not (silver.translated_from.map(source_labels).to_numpy() == silver.label_3class.to_numpy()).all():
        raise ValueError("Silver labels differ from their English sources")
    source_actions = english.set_index("id").decision_class.str.replace("-", "_", regex=False)
    if not (silver.translated_from.map(source_actions).to_numpy() ==
            silver.decision_class.str.replace("-", "_", regex=False).to_numpy()).all():
        raise ValueError("Silver actions differ from their English sources")
    sets = {
        "train_en": records(train_en),
        "train_silver": records(silver[~silver.family.isin(dev_ids)]),
        "dev_en": records(dev_en),
        "dev_silver": records(silver[silver.family.isin(dev_ids)]),
        "test_en": records(sources),
        "test_kk": records(gold),
    }
    check_sets(sets)
    return sets


def check_sets(sets):
    for name, rows in sets.items():
        if not rows or {r["label"] for r in rows} != {0, 1, 2}:
            raise ValueError(f"{name} must be nonempty and contain all three labels")
    groups = [{r["family"] for name in names for r in sets[name]} for names in
              [("train_en", "train_silver"), ("dev_en", "dev_silver"), ("test_en", "test_kk")]]
    if any(groups[a] & groups[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("A conversation family crosses train/dev/test")
    en, kk = sets["test_en"], sets["test_kk"]
    if len(en) != len(kk) or any(a["family"] != b["family"] or a["label"] != b["label"]
                                 for a, b in zip(en, kk)):
        raise ValueError("Test pairs are not aligned")
    if any(a["decision"] != b["decision"] for a, b in zip(en, kk)):
        raise ValueError("Test action labels are not aligned")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/gpu_bundle"))
    args = parser.parse_args()
    merged = pd.concat([pd.read_parquet(args.root / "data/merged" / f"{s}.parquet")
                        for s in ["train", "validation", "test"]], ignore_index=True)
    silver = pd.read_parquet(args.root / "data/silver/silver_kk.parquet")
    sets = make_sets(merged, silver)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": 42, "classes": CLASSES,
                "decision_classes": sorted(merged.decision_class.str.replace("-", "_", regex=False).unique()),
                "counts": {}, "sha256": {},
                "validation": "Same EN + silver KK development data for both variants; no gold KK.",
                "limitations": ["Historical test 300 has informed earlier experiments.",
                                "Kazakh references have not had native-speaker review.",
                                "Old published baselines used more training families; rerun matched baselines."]}
    for name, rows in sets.items():
        path = args.out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        manifest["counts"][name] = {"rows": len(rows), "families": len({r["family"] for r in rows}),
                                  "labels": {c: sum(r["label"] == i for r in rows) for i, c in enumerate(CLASSES)},
                                  "decisions": {c: sum(r["decision"] == c for r in rows)
                                                for c in manifest["decision_classes"]}}
        manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(args.root / "scripts/train_transformer.py", args.out / "train_transformer.py")
    shutil.copy2(args.root / "scripts/run_label_diffusion.py", args.out / "run_label_diffusion.py")
    shutil.copy2(args.root / "notebooks/05_gpu_finetuning.ipynb", args.out / "05_gpu_finetuning.ipynb")
    shutil.copy2(args.root / "notebooks/06_decisions_and_diffusion.ipynb", args.out / "06_decisions_and_diffusion.ipynb")
    shutil.copy2(args.root / "reports/DECISIONS_AND_DIFFUSION.md", args.out / "DECISIONS_AND_DIFFUSION.md")
    shutil.copy2(args.root / "data/release/README.md", args.out / "DATASET_CARD.md")
    archive = args.out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.out.iterdir()):
            if path.is_file():
                z.write(path, path.name)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Transfer {archive} ({archive.stat().st_size / 1e6:.1f} MB) to the GPU machine.")


if __name__ == "__main__":
    main()
