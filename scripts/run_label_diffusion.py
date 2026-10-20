"""CARD-inspired conditional label diffusion and matched dense-feature baselines.

Adaptation, not a reproduction: train-only TF-IDF/SVD text features, a frozen MLP
conditional mean, cosine noise schedule, and DDPM on centered one-hot labels.
Source: https://arxiv.org/abs/2206.07275 . No text generation or interactive RL.
"""
import argparse
import copy
import hashlib
import json
import math
import pathlib
import time

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_transformer import CLASSES, read, score, target_rows, write


class Guidance(nn.Module):
    def __init__(self, features, classes, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, classes))

    def forward(self, x):
        return self.net(x)


class Denoiser(nn.Module):
    def __init__(self, features, classes, hidden):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(features, hidden), nn.SiLU())
        self.net = nn.Sequential(nn.Linear(hidden + 2 * classes + 32, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, classes))

    def forward(self, x, residual, prior, t):
        frequency = torch.exp(torch.arange(16, device=x.device) * (-math.log(10000) / 15))
        angle = t.float()[:, None] * frequency[None, :]
        clock = torch.cat([angle.sin(), angle.cos()], dim=-1)
        return self.net(torch.cat([self.condition(x), residual, prior, clock], dim=-1))


class Schedule(nn.Module):
    def __init__(self, steps):
        super().__init__()
        if steps < 2:
            raise ValueError("Diffusion needs at least two steps")
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
        curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        beta = (1 - curve[1:] / curve[:-1]).clamp(1e-6, 0.999).float()
        alpha = 1 - beta
        bar = alpha.cumprod(0)
        previous = torch.cat([torch.ones(1), bar[:-1]])
        for key, value in dict(beta=beta, bar=bar,
                               posterior_variance=beta * (1 - previous) / (1 - bar),
                               coefficient0=beta * previous.sqrt() / (1 - bar),
                               coefficient1=alpha.sqrt() * (1 - previous) / (1 - bar)).items():
            self.register_buffer(key, value)

    def noisy_residual(self, onehot, prior, t, noise):
        a = self.bar[t, None]
        return a.sqrt() * (onehot - prior) + (1 - a).sqrt() * noise

    def reverse_mean(self, residual, prior, epsilon, step):
        a = self.bar[step]
        reconstructed = ((residual - (1 - a).sqrt() * epsilon) / a.sqrt() + prior).clamp(0, 1)
        return self.coefficient0[step] * (reconstructed - prior) + self.coefficient1[step] * residual


@torch.no_grad()
def sample_votes(model, guidance, schedule, x, classes, samples, seed, batch_size=128):
    """Return Monte Carlo class votes; agreement is not calibrated confidence."""
    model.eval()
    guidance.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    all_votes = []
    for chunk in x.split(batch_size):
        chunk = chunk.to(device).repeat_interleave(samples, dim=0)
        prior = guidance(chunk).softmax(-1)
        residual = torch.randn(prior.shape, device=device, generator=generator)
        for step in reversed(range(len(schedule.bar))):
            t = torch.full((len(chunk),), step, device=device, dtype=torch.long)
            epsilon = model(chunk, residual, prior, t)
            residual = schedule.reverse_mean(residual, prior, epsilon, step)
            if step:
                residual += schedule.posterior_variance[step].sqrt() * torch.randn(
                    residual.shape, device=device, generator=generator)
        labels = (residual + prior).argmax(-1).reshape(-1, samples)
        votes = F.one_hot(labels, classes).float().mean(1)
        all_votes.append(votes.cpu())
    return torch.cat(all_votes).numpy()


@torch.no_grad()
def mlp_predictions(model, x, batch_size=512):
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(part.to(device)).argmax(-1).cpu() for part in x.split(batch_size)]).numpy()


def selection_score(y, pred, classes):
    return score(y.tolist(), pred.tolist(), classes)["macro_f1"]


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this kernel. Use --device cpu only for a local smoke test.")
    return torch.device(name)


def train(args):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    root, output = pathlib.Path(args.data), pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise SystemExit("Output already contains a run. Choose a fresh output directory.")
    device = device_for(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Avoid oversubscribing CPU laptops during the explicit smoke check.
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    classes = CLASSES if args.target == "risk" else manifest["decision_classes"]
    training = read(root, "train_en")
    if args.variant == "mixed":
        training += read(root, "train_silver")
    development = read(root, "dev_en") + read(root, "dev_silver")
    training = target_rows(training, args.target, classes)
    development = target_rows(development, args.target, classes)
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        training = [training[i] for i in rng.permutation(len(training))[:256]]
        development = [development[i] for i in rng.permutation(len(development))[:64]]
    train_groups = {r["family"] for r in training}
    if train_groups & {r["family"] for r in development}:
        raise ValueError("Training and development families overlap")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)
    sparse = vectorizer.fit_transform([r["text"] for r in training])
    dimensions = min(args.dimensions, sparse.shape[0] - 1, sparse.shape[1] - 1)
    if dimensions < 2:
        raise ValueError("Not enough distinct text to fit the feature representation")
    svd = TruncatedSVD(n_components=dimensions, random_state=args.seed)
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(svd.fit_transform(sparse)), dtype=torch.float32)
    dx = torch.tensor(scaler.transform(svd.transform(vectorizer.transform(
        [r["text"] for r in development]))), dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(dx).all():
        raise ValueError("Non-finite text features; inspect the SVD preprocessing")
    y = torch.tensor([r["label"] for r in training], dtype=torch.long)
    dy = np.array([r["label"] for r in development])
    linear = LinearSVC(C=1.0, random_state=args.seed).fit(x.numpy(), y.numpy())
    joblib.dump(dict(vectorizer=vectorizer, svd=svd, scaler=scaler, linear=linear), output / "features.joblib")
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    guidance = Guidance(dimensions, len(classes), args.hidden).to(device)
    optimizer = torch.optim.AdamW(guidance.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1, None
    history = []
    for epoch in range(1, args.epochs + 1):
        guidance.train()
        total = 0.0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(guidance(bx.to(device)), by.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(bx)
        value = selection_score(dy, mlp_predictions(guidance, dx), classes)
        history.append(dict(model="mlp", epoch=epoch, loss=total / len(x), dev_macro_f1=value))
        if value > best:
            best, best_state = value, copy.deepcopy(guidance.state_dict())
        print(f"MLP epoch {epoch}: dev macro-F1={value:.4f}", flush=True)
    guidance.load_state_dict(best_state)
    guidance.eval().requires_grad_(False)
    torch.save(guidance.state_dict(), output / "guidance.pt")

    denoiser = Denoiser(dimensions, len(classes), args.hidden).to(device)
    schedule = Schedule(args.steps).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1, None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        denoiser.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                prior = guidance(bx).softmax(-1)
            onehot = F.one_hot(by, len(classes)).float()
            noise = torch.randn_like(onehot)
            t = torch.randint(args.steps, (len(bx),), device=device)
            residual = schedule.noisy_residual(onehot, prior, t, noise)
            loss = F.mse_loss(denoiser(bx, residual, prior, t), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(bx)
        row = dict(model="diffusion", epoch=epoch, noise_mse=total / len(x))
        # Fixed sampling seed makes validation checkpoints comparable.
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            votes = sample_votes(denoiser, guidance, schedule, dx, len(classes), args.samples, args.seed + 1000)
            value = selection_score(dy, votes.argmax(-1), classes)
            row["dev_macro_f1"] = value
            if value > best:
                best, best_state = value, copy.deepcopy(denoiser.state_dict())
            print(f"Diffusion epoch {epoch}: noise MSE={row['noise_mse']:.4f}; dev macro-F1={value:.4f}", flush=True)
        history.append(row)
    torch.save(best_state, output / "diffusion.pt")
    config = vars(args).copy()
    config.update(classes=classes, dimensions=dimensions, train_rows=len(x), dev_rows=len(dx),
                  torch_version=str(torch.__version__), numpy_version=np.__version__,
                  validation_best_macro_f1=best, diffusion_training_seconds=time.perf_counter() - start,
                  representation="word TF-IDF -> train-fitted SVD -> train-fitted standardization",
                  train_support={c: int((y == i).sum()) for i, c in enumerate(classes)},
                  manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())
    write(output / "run.json", config)
    write(output / "training_log.json", history)
    write(output / "validation_metrics.json", dict(
        linear=score(dy.tolist(), linear.predict(dx.numpy()).tolist(), classes),
        mlp=score(dy.tolist(), mlp_predictions(guidance, dx).tolist(), classes),
        diffusion_best_macro_f1=best))
    print("Saved validation-selected models. Gold test was not loaded.")


def evaluate(args):
    root, output = pathlib.Path(args.data), pathlib.Path(args.output)
    config = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if config["smoke"]:
        raise SystemExit("Smoke runs cannot be reported as full experiments")
    if config["manifest_sha256"] != hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest():
        raise ValueError("Evaluation bundle differs from training bundle")
    device = device_for(args.device)
    classes = config["classes"]
    # Load only artifacts produced locally by the training command.
    features = joblib.load(output / "features.joblib")
    guidance = Guidance(config["dimensions"], len(classes), config["hidden"]).to(device)
    guidance.load_state_dict(torch.load(output / "guidance.pt", map_location=device, weights_only=True))
    denoiser = Denoiser(config["dimensions"], len(classes), config["hidden"]).to(device)
    denoiser.load_state_dict(torch.load(output / "diffusion.pt", map_location=device, weights_only=True))
    schedule = Schedule(config["steps"]).to(device)
    results = {}
    for name in ["test_en", "test_kk"]:
        rows = target_rows(read(root, name), config["target"], classes)
        x = torch.tensor(features["scaler"].transform(features["svd"].transform(
            features["vectorizer"].transform([r["text"] for r in rows]))), dtype=torch.float32)
        truth = [r["label"] for r in rows]
        times = {}
        start = time.perf_counter()
        linear_prediction = features["linear"].predict(x.numpy())
        times["linear"] = time.perf_counter() - start
        start = time.perf_counter()
        mlp_prediction = mlp_predictions(guidance, x)
        times["mlp"] = time.perf_counter() - start
        start = time.perf_counter()
        votes = sample_votes(denoiser, guidance, schedule, x, len(classes), config["samples"], config["seed"] + 2000)
        times["diffusion"] = time.perf_counter() - start
        predictions = dict(linear=linear_prediction, mlp=mlp_prediction, diffusion=votes.argmax(-1))
        results[name] = {model: score(truth, pred.tolist(), classes) for model, pred in predictions.items()}
        results[name]["prediction_seconds_excluding_features"] = times
        results[name]["class_support"] = {c: truth.count(i) for i, c in enumerate(classes)}
        results[name]["unseen_training_target_rows"] = sum(config["train_support"][classes[y]] == 0 for y in truth)
        write(output / f"{name}_predictions.json", [dict(
            id=row["id"], family=row["family"], label=row["label"],
            linear=int(predictions["linear"][i]), mlp=int(predictions["mlp"][i]),
            diffusion=int(predictions["diffusion"][i]), sample_votes=votes[i].tolist())
            for i, row in enumerate(rows)])
    write(output / "test_metrics.json", results)
    for name, metrics in results.items():
        print(name, {m: round(metrics[m]["macro_f1"], 4) for m in ["linear", "mlp", "diffusion"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--data", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", choices=["risk", "action"], default="risk")
    parser.add_argument("--variant", choices=["en", "mixed"], default="mixed")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.samples, args.eval_every, args.hidden, args.batch_size) < 1 or args.dimensions < 2 or args.steps < 2:
        parser.error("Positive settings required; dimensions and steps must be >= 2")
    if args.smoke:
        args.epochs, args.steps, args.samples, args.dimensions, args.eval_every = 2, 8, 2, 16, 1
    {"train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
