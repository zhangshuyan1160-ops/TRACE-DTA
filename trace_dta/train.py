"""Fit the base predictor, transfer molecular relations and fit the target-conditioned adapter."""

import argparse
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from .features import ChainStore
from .model import TraceDTA
from .prepare import table


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def snapshot(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def affinity_loss(z, y):
    loss = 2 * F.huber_loss(z, y, delta=1.0)
    if len(z) > 1:
        i, j = torch.triu_indices(len(z), len(z), 1, device=z.device)
        difference = y[i] - y[j]
        ranking = difference.abs().clamp_max(1) * F.softplus(-difference.sign() * (z[i] - z[j]))
        loss = loss + 0.1 * ranking.mean()
    return loss


def normalized_distances(vectors):
    i, j = torch.triu_indices(len(vectors), len(vectors), 1, device=vectors.device)
    unit = F.normalize(vectors, dim=-1)
    distances = torch.linalg.vector_norm(unit[i] - unit[j], dim=-1)
    return distances


def relation_loss(student, teacher):
    desired = normalized_distances(teacher.detach())
    if len(desired) == 0 or desired.mean().item() <= 1e-8:
        raise ValueError("A relation batch must contain at least two distinct teacher embeddings")
    distance = normalized_distances(student)
    return F.mse_loss(distance / distance.mean().clamp_min(1e-8), desired / desired.mean())


def optimizer(model):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=1e-5,
        foreach=False,
        fused=False,
    )


def update(model, opt, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    opt.step()


def batch_values(model, rows, store):
    z = model.normalized_affinity(
        [r["smiles"] for r in rows], [r["chains"].split(":") for r in rows], store
    )
    y = torch.tensor([float(r["pkd"]) for r in rows], dtype=torch.float64, device=z.device)
    return z, y


@torch.no_grad()
def validation_mse(model, rows, store, batch):
    model.eval()
    errors = []
    for start in range(0, len(rows), batch):
        z, y = batch_values(model, rows[start : start + batch], store)
        errors.extend((model.training_pkd_mean + model.training_pkd_std * z - y).square().tolist())
    score = float(np.mean(errors))
    if not math.isfinite(score):
        raise ValueError("Nonfinite validation MSE")
    return score


def fit_affinity(model, phase, train, val, store, full_training_count, args):
    set_seed(args.seed)
    model.set_phase(phase)
    opt = optimizer(model)
    score = validation_mse(model, val, store, args.batch) if phase == "affinity" else math.inf
    best = snapshot(model) if phase == "affinity" else None
    interval = math.ceil(full_training_count / args.batch)
    step, cursor, order = 0, 0, []
    for _ in range(args.epochs):
        model.train()
        for _ in range(interval):
            if cursor >= len(order):
                order = list(range(len(train)))
                random.shuffle(order)
                cursor = 0
            rows = [train[i] for i in order[cursor : cursor + args.batch]]
            cursor += len(rows)
            z, y = batch_values(model, rows, store)
            update(
                model, opt, affinity_loss(z, (y - model.training_pkd_mean) / model.training_pkd_std)
            )
            step += 1
        current = validation_mse(model, val, store, args.batch)
        if current < score:
            score, best = current, snapshot(model)
        print(f"{phase}  step {step}  validation MSE {current:.6f}", flush=True)
    model.load_state_dict(best, strict=True)
    model.clear_cache()


def fit_relations(model, pool, teacher_path, args):
    set_seed(args.seed)
    model.set_phase("relation")
    pool_smiles = {r["smiles"] for r in pool}
    with np.load(teacher_path, allow_pickle=False) as stored:
        available = {str(s): i for i, s in enumerate(stored["smiles"])}
        excluded = set(stored["excluded"].tolist())
        if (set(available) & excluded) or not pool_smiles <= (set(available) | excluded):
            raise ValueError("The teacher cache does not cover this training pool")
        smiles = sorted(pool_smiles & set(available))
        vectors = torch.tensor(
            stored["embedding"][[available[s] for s in smiles]], device=args.device
        )
    if len(smiles) < 2:
        raise ValueError("Too few eligible molecules for relation training")
    ids = np.linspace(0, len(smiles) - 1, min(len(smiles), 128), dtype=int).tolist()

    def loss_for(selection):
        return relation_loss(
            model.relation_vectors([smiles[i] for i in selection]), vectors[selection]
        )

    opt = optimizer(model)
    with torch.no_grad():
        score = loss_for(ids).item()
    best = snapshot(model.relation_encoder)
    deadline = time.monotonic() + args.relation_seconds
    for epoch in range(args.relation_epochs):
        order = list(range(len(smiles)))
        random.shuffle(order)
        start = 0
        while start < len(order) and time.monotonic() < deadline:
            end = min(start + args.batch, len(order))
            if len(order) - end == 1:
                end += 1
            update(model, opt, loss_for(order[start:end]))
            start = end
        if start != len(order):
            break
        with torch.no_grad():
            current = loss_for(ids).item()
        if current < score:
            score, best = current, snapshot(model.relation_encoder)
        print(f"relation  epoch {epoch + 1}  distance loss {current:.6f}", flush=True)
    model.relation_encoder.load_state_dict(best, strict=True)
    model.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--relation-epochs", type=int, default=80)
    parser.add_argument("--relation-seconds", type=int, default=3600)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.relation_epochs < 1 or args.relation_seconds < 1 or args.batch < 2:
        parser.error("Use positive training limits and a batch size of at least 2")
    if args.output.exists():
        parser.error("Choose a new output path for the trained weights")
    train, val, pool = (
        table(args.split / (name + ".tsv")) for name in ("train", "validation", "pool")
    )
    groups = [{r["sample"] for r in rr} for rr in (train, val, pool)]
    if (
        not train
        or not val
        or not groups[0] <= groups[2]
        or groups[1] & groups[2]
        or any(len(g) != len(rr) for g, rr in zip(groups, (train, val, pool)))
    ):
        raise ValueError("Invalid membership in training, validation or unlabelled training pool")
    set_seed(args.seed)
    model = TraceDTA().to(args.device)
    store = ChainStore(args.features)
    labels = np.asarray([float(r["pkd"]) for r in train], dtype=np.float64)
    if not np.isfinite(labels).all() or labels.std() <= 0:
        raise ValueError("Training requires finite, variable pKd measurements")
    model.training_pkd_mean.fill_(labels.mean())
    model.training_pkd_std.fill_(labels.std())
    fit_affinity(model, "base", train, val, store, len(pool), args)
    fit_relations(model, pool, args.teacher, args)
    fit_affinity(model, "affinity", train, val, store, len(pool), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        path = Path(tmp) / "model.safetensors"
        save_file(snapshot(model), str(path))
        path.replace(args.output)


if __name__ == "__main__":
    main()
