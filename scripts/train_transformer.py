"""Portable GPU runner. Use the accompanying notebook, or --help.

Text inputs contain only message contents. Gold test data is read only in the
explicit evaluate/baselines commands, never during training/checkpoint selection.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib

CLASSES = ["LEGITIMATE", "SUSPICIOUS", "SCAM"]


def read(root, name):
    root = pathlib.Path(root)
    path = root / f"{name}.jsonl"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][path.name]:
        raise ValueError(f"Checksum mismatch: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def score(truth, predicted, classes=None):
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    classes = CLASSES if classes is None else classes
    labels = list(range(len(classes)))
    present = sorted(set(truth))
    values = {"macro_f1": f1_score(truth, predicted, average="macro", labels=labels, zero_division=0),
              "macro_f1_supported": f1_score(truth, predicted, average="macro", labels=present, zero_division=0),
              "accuracy": accuracy_score(truth, predicted),
              "per_class": classification_report(truth, predicted, labels=labels,
                                                   target_names=classes, output_dict=True, zero_division=0)}
    if classes == CLASSES:
        values["binary_macro_f1"] = f1_score([int(y != 0) for y in truth], [int(y != 0) for y in predicted],
                                              average="macro", labels=[0, 1], zero_division=0)
    return values


def target_rows(rows, target, classes):
    """The decision is an output target only; text stays byte-for-byte unchanged."""
    if target == "risk":
        return rows
    mapping = {label: i for i, label in enumerate(classes)}
    return [dict(row, label=mapping[row["decision"]]) for row in rows]


def write(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class EncodedRows:
    def __init__(self, rows, tokenizer, max_length):
        # Explicit head+tail policy: retain both the initial request and final turns.
        # No answer metadata or derived annotation enters the tokenizer.
        encoded = tokenizer([r["text"] for r in rows], add_special_tokens=False,
                            truncation=False, verbose=False)["input_ids"]
        budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        if budget < 2:
            raise ValueError("max_length is too small")
        self.truncated = sum(len(ids) > budget for ids in encoded)
        self.rows = []
        for row, ids in zip(rows, encoded):
            if len(ids) > budget:
                head = budget // 2
                ids = ids[:head] + ids[-(budget - head):]
            ids = tokenizer.build_inputs_with_special_tokens(ids)
            self.rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": row["label"]})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Select the CUDA PyTorch kernel on your GPU machine.")
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; CUDA {torch.version.cuda}")
    return torch


def train(args):
    torch = gpu()
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments, set_seed)
    set_seed(args.seed)
    root, output = pathlib.Path(args.data), pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((output / "config.json").exists() or list(output.glob("checkpoint-*"))):
        raise SystemExit("Output already contains a model. Choose a new output directory.")
    if args.resume:
        previous = json.loads((output / "run.json").read_text(encoding="utf-8"))
        keys = ["variant", "model", "seed", "max_length", "batch_size", "accumulation", "epochs", "smoke", "target",
                "optimizer", "save_steps"]
        if any(previous.get(k, {"optimizer": "adamw_torch", "save_steps": None}.get(k)) != getattr(args, k) for k in keys):
            raise ValueError("Resume settings must match the interrupted run")
        if previous["manifest_sha256"] != hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest():
            raise ValueError("Resume data differs from the interrupted run")
    rows = read(root, "train_en")
    if args.variant == "mixed":
        rows += read(root, "train_silver")
    dev_rows = read(root, "dev_en") + read(root, "dev_silver")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    classes = CLASSES if args.target == "risk" else manifest["decision_classes"]
    rows = target_rows(rows, args.target, classes)
    dev_rows = target_rows(dev_rows, args.target, classes)
    # Exercise long inputs, rather than only an easy short prefix.
    if args.smoke:
        rows = sorted(rows, key=lambda r: len(r["text"]), reverse=True)[:32]
        dev_rows = sorted(dev_rows, key=lambda r: len(r["text"]), reverse=True)[:16]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    training = EncodedRows(rows, tokenizer, args.max_length)
    development = EncodedRows(dev_rows, tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(classes), id2label=dict(enumerate(classes)),
        label2id={c: i for i, c in enumerate(classes)})
    model.config.use_cache = False
    use_bf16 = torch.cuda.is_bf16_supported()
    settings = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        max_steps=2 if args.smoke else -1,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        gradient_checkpointing=True,
        eval_strategy="steps", save_strategy="steps",
        eval_steps=1 if args.smoke else args.save_steps,
        save_steps=1 if args.smoke else args.save_steps, save_total_limit=2,
        eval_accumulation_steps=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, bf16=use_bf16, fp16=not use_bf16,
        seed=args.seed, data_seed=args.seed, report_to="none", logging_steps=5,
        disable_tqdm=True, dataloader_num_workers=0, optim=args.optimizer,
    )

    class FirstCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # Save a resumable checkpoint immediately after the first update.
            # Later checkpoints follow the regular save/evaluation interval.
            if state.global_step == 1:
                control.should_save = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            # Also evaluate/save the final partial interval of each epoch.
            control.should_evaluate = True
            control.should_save = True
            return control

    def metrics(result):
        logits, labels = result
        values = score(labels.tolist(), logits.argmax(axis=-1).tolist(), classes)
        return {k: values[k] for k in ["macro_f1", "binary_macro_f1", "accuracy"] if k in values}

    trainer = Trainer(model=model, args=settings, train_dataset=training,
                      eval_dataset=development, processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
                      callbacks=[FirstCheckpoint()],
                      compute_metrics=metrics)
    print(f"Train: {len(training)}; dev: {len(development)}; gold test unused.", flush=True)
    run = vars(args).copy()
    run.update(classes=classes, train_rows=len(training), dev_rows=len(development),
               train_truncated=training.truncated, dev_truncated=development.truncated,
               truncation_policy="head+tail", model_revision=getattr(model.config, "_commit_hash", None),
               manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
               packages={p: importlib.metadata.version(p) for p in ["torch", "transformers", "accelerate", "scikit-learn"]})
    write(output / "run.json", run)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    write(output / "validation_metrics.json", trainer.evaluate())
    write(output / "training_log.json", trainer.state.log_history)
    print(f"Saved best validation checkpoint to {output}. Gold test remains unused.")


def evaluate(args):
    torch = gpu()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from torch.utils.data import DataLoader
    root, output = pathlib.Path(args.data), pathlib.Path(args.output)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if run["smoke"]:
        raise SystemExit("A smoke checkpoint is not an experiment. Train a full run first.")
    if hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() != run["manifest_sha256"]:
        raise ValueError("Evaluation bundle differs from the training bundle")
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForSequenceClassification.from_pretrained(output).to("cuda").eval()
    results = {}
    for name in ["test_en", "test_kk"]:
        rows = target_rows(read(root, name), run["target"], run["classes"])
        encoded = EncodedRows(rows, tokenizer, run["max_length"])
        predicted = []
        loader = DataLoader(encoded, batch_size=args.batch_size,
                            collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
        with torch.inference_mode():
            for batch in loader:
                inputs = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
                predicted.extend(model(**inputs).logits.argmax(-1).cpu().tolist())
        results[name] = score([r["label"] for r in rows], predicted, run["classes"])
        results[name]["truncated"] = encoded.truncated
        write(output / f"{name}_predictions.json", [dict(id=r["id"], family=r["family"],
                    label=r["label"], predicted=p) for r, p in zip(rows, predicted)])
    write(output / "test_metrics.json", results)
    print(json.dumps(results, indent=2))


def baselines(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    manifest = json.loads((pathlib.Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    classes = CLASSES if args.target == "risk" else manifest["decision_classes"]
    for variant in ["en", "mixed"]:
        rows = read(args.data, "train_en")
        if variant == "mixed":
            rows += read(args.data, "train_silver")
        rows = target_rows(rows, args.target, classes)
        for kind in ["char", "word"]:
            model = make_pipeline(TfidfVectorizer(
                analyzer="char_wb" if kind == "char" else "word",
                ngram_range=(3, 5) if kind == "char" else (1, 2),
                min_df=3, max_features=200_000, sublinear_tf=True), LinearSVC(C=1.0, random_state=0))
            model.fit([r["text"] for r in rows], [r["label"] for r in rows])
            results[f"{kind}_{variant}"] = {}
            for name in ["test_en", "test_kk"]:
                test = target_rows(read(args.data, name), args.target, classes)
                predicted = model.predict([r["text"] for r in test]).tolist()
                results[f"{kind}_{variant}"][name] = score([r["label"] for r in test], predicted, classes)
                write(output / f"{kind}_{variant}_{name}_predictions.json", [
                    dict(id=r["id"], family=r["family"], label=r["label"], predicted=p)
                    for r, p in zip(test, predicted)])
            print(kind, variant, {k: round(v["macro_f1"], 4) for k, v in results[f"{kind}_{variant}"].items()}, flush=True)
    write(output / "baselines.json", results)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "baselines"])
    parser.add_argument("--data", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["en", "mixed"], default="en")
    parser.add_argument("--target", choices=["risk", "action"], default="risk")
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw_torch", "adafactor"], default="adamw_torch")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint/evaluation interval in optimizer updates")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", help="Path to an interrupted run's checkpoint directory")
    args = parser.parse_args()
    if args.batch_size < 1 or args.accumulation < 1 or args.epochs <= 0 or args.save_steps < 1:
        parser.error("batch-size, accumulation, and epochs must be positive")
    if not 4 <= args.max_length <= 512:
        parser.error("max-length must be between 4 and 512")
    {"train": train, "evaluate": evaluate, "baselines": baselines}[args.command](args)


if __name__ == "__main__":
    main()
