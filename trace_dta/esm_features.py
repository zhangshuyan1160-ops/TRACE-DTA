"""Frozen ESM-2 residue embeddings and layer-streamed contact probabilities."""

from argparse import Namespace
from pathlib import Path

import torch


def load_model(folder: Path):
    import esm

    name = "esm2_t33_650M_UR50D"
    with torch.serialization.safe_globals([Namespace]):
        checkpoint = torch.load(folder / (name + ".pt"), map_location="cpu", weights_only=True)
        regression = torch.load(
            folder / (name + "-contact-regression.pt"), map_location="cpu", weights_only=True
        )
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(name, checkpoint, regression)
    if (model.num_layers, model.embed_dim, model.attention_heads) != (33, 1280, 20):
        raise ValueError("Unexpected ESM architecture")
    model.eval().requires_grad_(False)
    return model, alphabet


@torch.inference_mode()
def stream_contacts(model, tokens):
    """One complete unpadded chain, returning all residue embeddings and contacts."""
    if model.training or tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("Frozen, single-chain inference is required")
    if not model.prepend_bos or not model.append_eos:
        raise ValueError("Unexpected ESM alphabet convention")
    if (tokens == model.padding_idx).any() or (tokens == model.mask_idx).any():
        raise ValueError("No padding, truncation or synthetic masks in extraction")
    if tokens[0, 0] != model.cls_idx or tokens[0, -1] != model.eos_idx:
        raise ValueError("Missing chain boundary tokens")
    x = model.embed_scale * model.embed_tokens(tokens)
    if model.token_dropout:
        x = x * (1 - 0.15 * 0.8)
    x = x.transpose(0, 1)
    length = tokens.shape[1] - 2
    weights = model.contact_head.regression.weight.reshape(model.num_layers, model.attention_heads)
    # Accumulating the small final logit matrix in double limits reordering error.
    logits = torch.full(
        (length, length),
        float(model.contact_head.regression.bias.item()),
        dtype=torch.float64,
        device=x.device,
    )
    for layer_index, layer in enumerate(model.layers):
        x, attention = layer(x, self_attn_padding_mask=None, need_head_weights=True)
        a = attention[:, 0, 1:-1, 1:-1]
        symmetric = a + a.transpose(-1, -2)
        row_sum = symmetric.sum(-1, keepdim=True)
        col_sum = symmetric.sum(-2, keepdim=True)
        total = symmetric.sum((-1, -2), keepdim=True)
        corrected = symmetric - (row_sum * col_sum).div_(total)
        contribution = torch.einsum("hij,h->ij", corrected, weights[layer_index])
        logits.add_(contribution.double())
        del attention, a, symmetric, row_sum, col_sum, total, corrected, contribution
    residues = model.emb_layer_norm_after(x).transpose(0, 1)[0, 1:-1]
    contacts = logits.sigmoid().float()
    if not torch.isfinite(residues).all() or not torch.isfinite(contacts).all():
        raise FloatingPointError("Nonfinite ESM feature")
    return residues, contacts


@torch.inference_mode()
def tiled_contacts(model, tokens, tile=128):
    """Compute complete-chain attention in query blocks to limit device memory."""
    from esm.modules import gelu

    assert not model.training and tokens.shape[0] == 1 and not (tokens == model.padding_idx).any()
    assert (
        not (tokens == model.mask_idx).any()
        and tokens[0, 0] == model.cls_idx
        and tokens[0, -1] == model.eos_idx
    )
    length = tokens.shape[1]
    n = length - 2
    heads = model.attention_heads
    x = model.embed_scale * model.embed_tokens(tokens)
    if model.token_dropout:
        x = x * (1 - 0.15 * 0.8)
    x = x.transpose(0, 1)
    weights = model.contact_head.regression.weight.reshape(model.num_layers, heads)
    logits = torch.full(
        (n, n), float(model.contact_head.regression.bias.item()), dtype=torch.float64, device="cpu"
    )
    for index, layer in enumerate(model.layers):
        residual = x
        z = layer.self_attn_layer_norm(x)
        attn = layer.self_attn
        assert attn.bias_k is None and not attn.add_zero_attn and not attn.training
        q = attn.q_proj(z) * attn.scaling
        k = attn.k_proj(z)
        v = attn.v_proj(z)
        q = q.contiguous().view(length, heads, attn.head_dim).transpose(0, 1)
        k = k.contiguous().view(length, heads, attn.head_dim).transpose(0, 1)
        v = v.contiguous().view(length, heads, attn.head_dim).transpose(0, 1)
        if attn.rot_emb:
            q, k = attn.rot_emb(q, k)
        outputs = []
        rows = torch.zeros((heads, n), device=x.device)
        columns = torch.zeros_like(rows)
        weighted = torch.empty((n, n), dtype=torch.float32, device="cpu")
        for start in range(0, length, tile):
            end = min(start + tile, length)
            probability = torch.softmax(
                torch.bmm(q[:, start:end], k.transpose(1, 2)), dim=-1, dtype=torch.float32
            )
            outputs.append(torch.bmm(probability, v))
            first = max(start, 1)
            last = min(end, length - 1)
            if first < last:
                a = probability[:, first - start : last - start, 1:-1]
                rows[:, first - 1 : last - 1] = a.sum(-1)
                columns.add_(a.sum(-2))
                weighted[first - 1 : last - 1] = torch.einsum("hij,h->ij", a, weights[index]).cpu()
        hidden = (
            torch.cat(outputs, dim=1).transpose(0, 1).contiguous().view(length, 1, model.embed_dim)
        )
        x = residual + attn.out_proj(hidden)
        residual = x
        x = residual + layer.fc2(gelu(layer.fc1(layer.final_layer_norm(x))))
        symmetric_sum = rows + columns
        total = symmetric_sum.sum(-1)
        assert (total > 0).all()
        scaled = symmetric_sum * (weights[index] / total)[:, None]
        # Weighted APC is a sum of head-specific outer products, evaluated in tiles.
        for start in range(0, n, tile):
            end = min(start + tile, n)
            correction = (symmetric_sum[:, start:end].T @ scaled).cpu()
            logits[start:end].add_(
                weighted[start:end].double()
                + weighted[:, start:end].T.double()
                - correction.double()
            )
        del (
            outputs,
            weighted,
            rows,
            columns,
            q,
            k,
            v,
            probability,
            hidden,
            z,
            scaled,
            symmetric_sum,
            correction,
        )
    residues = model.emb_layer_norm_after(x).transpose(0, 1)[0, 1:-1].cpu()
    contacts = logits.sigmoid().float()
    assert torch.isfinite(residues).all() and torch.isfinite(contacts).all()
    return residues, contacts
