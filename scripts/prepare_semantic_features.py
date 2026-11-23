"""Reuse cached LaBSE and encode missing silver conversations; append safe features.

Frozen encoding only. Labels are never passed to the encoder. This operation does
not fit a classifier or choose parameters. It records a text hash for each row.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

import build_features as bf
from train_transformer import read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=pathlib.Path,default=pathlib.Path("data/gpu_bundle"))
    parser.add_argument("--out",type=pathlib.Path,default=pathlib.Path("data/embeddings/labse_hand_bundle.npz"))
    args=parser.parse_args()
    if args.out.exists():
        raise SystemExit("Feature cache exists; choose a new output path")
    root=pathlib.Path(__file__).resolve().parents[1]
    rows=[r for name in ["train_en","train_silver","dev_en","dev_silver","test_en","test_kk"] for r in read(args.data,name)]
    assert len({r["id"] for r in rows})==len(rows)
    cached=np.load(root/"data/embeddings/sentence-transformers__LaBSE.npz",allow_pickle=True)
    by_id=dict(zip(cached["ids"].tolist(),cached["vectors"]))
    missing=[r for r in rows if r["id"] not in by_id]
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    # A local snapshot path avoids unnecessary online model-metadata probes.
    path=snapshot_download("sentence-transformers/LaBSE",local_files_only=True)
    model=SentenceTransformer(path,local_files_only=True,device="cpu")
    unique=sorted({r["text"] for r in missing})
    print(f"Reusing {len(rows)-len(missing)} cached embeddings; encoding {len(missing)} missing rows / {len(unique)} distinct texts; max tokens {model.max_seq_length}.",flush=True)
    vectors=model.encode(unique,batch_size=16,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
    by_text=dict(zip(unique,vectors))
    for row in missing: by_id[row["id"]]=by_text[row["text"]]
    original=pd.concat([pd.read_parquet(root/"data/features"/f"{s}.parquet") for s in ["train","validation","test"]],ignore_index=True)
    silver=bf.featurise(pd.read_parquet(root/"data/silver/silver_kk.parquet"))
    names=bf.feature_columns(original)
    assert len(names)==54 and all(c not in bf.LEAKING_COLUMNS for c in names)
    table=pd.concat([original[["id",*names]],silver[["id",*names]]],ignore_index=True).set_index("id")
    if not table.index.is_unique: raise ValueError("Duplicate feature IDs")
    semantic=np.vstack([by_id[r["id"]] for r in rows]).astype(np.float32)
    hand=table.loc[[r["id"] for r in rows],names].to_numpy(dtype=np.float32)
    features=np.concatenate([semantic,hand],axis=1)
    if not np.isfinite(features).all(): raise ValueError("Nonfinite features")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,ids=np.array([r["id"] for r in rows]),features=features,
                        text_sha256=np.array([hashlib.sha256(r["text"].encode()).hexdigest() for r in rows]),
                        feature_names=np.array([f"semantic_{i}" for i in range(semantic.shape[1])]+names),
                        metadata=np.array(json.dumps(dict(model="sentence-transformers/LaBSE",max_tokens=model.max_seq_length,
                            reused=len(rows)-len(missing),new=len(missing),source_cache="existing encode_corpus.py output",
                            hand_features="conversation text/roles only; no annotation inputs",frozen=True))))
    print("Saved",args.out,"shape",features.shape,flush=True)


if __name__=="__main__": main()
