"""Prepare full-chain ESM-2 inputs and molecular teacher embeddings."""

import argparse
import csv
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .esm_features import load_model, stream_contacts, tiled_contacts

TEACHER_REPO = "seyonec/PubChem10M_SMILES_BPE_450k"
TEACHER_REVISION = "c18fccd09b3326bf2d4633412c256d7db872156d"
TEACHER_SHA = "f471de1ad73de11d47e1bbdfddb93aeace0b8aeab31ec03bcde0eabb17a97264"
ESM_FILES = {
    "esm2_t33_650M_UR50D.pt": "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
    "esm2_t33_650M_UR50D-contact-regression.pt": "8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575",
}


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class ChainStore:
    def __init__(self, folder):
        self.folder = Path(folder)

    def path(self, sequence):
        return self.folder / (hashlib.sha256(sequence.encode("ascii")).hexdigest() + ".npz")

    @lru_cache(maxsize=12)
    def chain(self, sequence):
        with np.load(self.path(sequence), allow_pickle=False) as data:
            residues, contacts = data["residues"], data["contacts"]
        n = len(sequence)
        if (
            residues.shape != (n, 1280)
            or contacts.shape != (n, n)
            or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in (residues, contacts))
        ):
            raise ValueError("Invalid full-chain protein features")
        return {"residues": residues, "contacts": contacts}


def proteins(records, folder, weights, device, tile_long_chains=False):
    store = ChainStore(folder)
    sequences = sorted({s for row in records for s in row["chains"].split(":")})
    pending = []
    for sequence in sequences:
        if store.path(sequence).is_file():
            store.chain(sequence)
        else:
            pending.append(sequence)
    if not pending:
        return
    for name, expected in ESM_FILES.items():
        if sha256_file(weights / name) != expected:
            raise ValueError(f"ESM-2 weight checksum mismatch for {name}")
    model, alphabet = load_model(weights)
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    for s in pending:
        if not s or set(s) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO"):
            raise ValueError("Invalid protein chain sequence")
        _, _, tokens = batch_converter([("chain", s)])
        extract = tiled_contacts if tile_long_chains and len(s) > 2500 else stream_contacts
        residues, contacts = extract(model, tokens.to(device))
        write_npz(store.path(s), residues=residues.cpu().numpy(), contacts=contacts.cpu().numpy())


@torch.inference_mode()
def teacher(records, output, device, weights=None):
    from transformers import AutoTokenizer, RobertaConfig, RobertaForMaskedLM
    from transformers.models.roberta.modeling_roberta import RobertaPooler

    if weights is None:
        from huggingface_hub import snapshot_download

        weights = Path(
            snapshot_download(
                TEACHER_REPO,
                revision=TEACHER_REVISION,
                allow_patterns=["*.json", "merges.txt", "pytorch_model.bin"],
            )
        )
    weight = weights / "pytorch_model.bin"
    if sha256_file(weight) != TEACHER_SHA:
        raise ValueError("Molecular teacher weight checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True, use_fast=True)
    config = RobertaConfig.from_pretrained(weights, local_files_only=True)
    config._attn_implementation = "eager"
    model = RobertaForMaskedLM(config)
    state = torch.load(weight, map_location="cpu", weights_only=True)
    if "roberta.pooler.dense.weight" in state:
        model.roberta.pooler = RobertaPooler(config)
    for field in ("position_ids", "token_type_ids"):
        key = f"roberta.embeddings.{field}"
        if key in state and key not in model.state_dict():
            expected = torch.arange(config.max_position_embeddings)[None, :]
            if field == "token_type_ids":
                expected = torch.zeros_like(expected)
            if not torch.equal(state.pop(key), expected):
                raise ValueError(f"Unexpected RoBERTa buffer {field}")
    model.load_state_dict(state, strict=True)
    del state
    model = model.roberta.eval().to(device)
    smiles = []
    candidates = sorted({r["smiles"] for r in records})
    for s in candidates:
        tokens = tokenizer(s, truncation=False, verbose=False)["input_ids"]
        if (
            tokenizer.unk_token_id in tokens
            or tokenizer.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            != s
        ):
            raise ValueError(f"SMILES cannot be recovered from its teacher tokens: {s}")
        if len(tokens) <= 510:
            smiles.append(s)
    if len(smiles) < 2:
        raise ValueError("Relation training requires at least two eligible molecules")
    vectors = []
    for start in range(0, len(smiles), 8):
        batch = tokenizer(
            smiles[start : start + 8],
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
            return_tensors="pt",
        ).to(device)
        special = batch.pop("special_tokens_mask").bool()
        mask = batch["attention_mask"].bool() & ~special
        states = model(**batch).last_hidden_state
        pooled = (states * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        vectors.append(pooled.cpu().numpy())
    matrix = np.concatenate(vectors)
    if not np.isfinite(matrix).all() or np.min(np.linalg.norm(matrix, axis=1)) <= 1e-6:
        raise ValueError("Invalid molecular teacher embeddings")
    write_npz(
        output,
        smiles=np.asarray(smiles),
        embedding=matrix,
        excluded=np.asarray(sorted(set(candidates) - set(smiles)), dtype=str),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("protein", "teacher"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tile-long-chains",
        action="store_true",
        help="Tile ESM attention for chains longer than 2,500 residues",
    )
    args = parser.parse_args()
    with args.input.open() as f:
        records = list(csv.DictReader(f, delimiter="\t"))
    if args.kind == "protein":
        if args.weights is None:
            parser.error("Protein extraction requires --weights with the two ESM-2 files")
        proteins(records, args.output, args.weights, args.device, args.tile_long_chains)
    else:
        teacher(records, args.output, args.device, args.weights)


if __name__ == "__main__":
    main()
