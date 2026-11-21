"""Predict pKd for drug–protein pairs with a trained TRACE-DTA model."""

import argparse
import csv
from pathlib import Path

from safetensors.torch import load_file
import torch

from .features import ChainStore
from .model import TraceDTA
from .prepare import table


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("Use a positive batch size")
    rows = table(args.input)
    if not rows or len({r["sample"] for r in rows}) != len(rows):
        parser.error("Supply a nonempty input with unique sample IDs")
    model = TraceDTA()
    model.load_state_dict(load_file(str(args.weights)), strict=True)
    model.to(args.device).eval()
    store = ChainStore(args.features)
    results = []
    for i in range(0, len(rows), args.batch):
        batch = rows[i : i + args.batch]
        values = model([r["smiles"] for r in batch], [r["chains"].split(":") for r in batch], store)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite affinity prediction")
        results.extend((r["sample"], format(v, ".12g")) for r, v in zip(batch, values.tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(("sample", "predicted_pkd"))
        writer.writerows(results)


if __name__ == "__main__":
    main()
