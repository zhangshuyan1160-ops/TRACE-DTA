"""Bond, sequence and contact encoders."""

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .chemistry import (
    ATOM_FEATURE_NAMES,
    BOND_STEREO_VOCAB,
    BOND_TYPE_VOCAB,
    CHIRAL_CLASS_VOCAB,
    CIP_VOCAB,
    HYBRIDIZATION_VOCAB,
    ROLE_VOCAB,
)

RESIDUE_VOCAB = ("<PAD>", *tuple("ACDEFGHIKLMNPQRSTVWYBXZJUO"))


ATOM_NUMERIC_COLUMNS = (1, 2, 3, 4, 5, 7)
ATOM_NUMERIC_SCALES = (100.0, 4.0, 8.0, 8.0, 8.0, 4.0)
ATOM_CATEGORICAL_SIZES = {
    "atomic_number": 119,
    "aromatic": 2,
    "in_ring": 2,
    "hybridization": len(HYBRIDIZATION_VOCAB),
    "chiral_class": len(CHIRAL_CLASS_VOCAB),
    "cip_code": len(CIP_VOCAB),
}


class _GINELayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: Tensor, edges: Tensor, source: Tensor, target: Tensor) -> Tensor:
        aggregation = torch.zeros_like(nodes)
        if source.numel():
            messages = torch.relu(nodes[source] + edges)
            aggregation.index_add_(0, target, messages)
        update = self.mlp((1.0 + self.eps) * nodes + aggregation)
        return self.norm(nodes + update)


class DrugEncoder(nn.Module):
    """GINE encoding of atom and bond features."""

    def __init__(self, hidden_dim: int = 128, layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, hidden_dim) for name, size in ATOM_CATEGORICAL_SIZES.items()}
        )
        self.atom_numeric = nn.Linear(len(ATOM_NUMERIC_COLUMNS) + len(ROLE_VOCAB), hidden_dim)
        self.register_buffer("numeric_scales", torch.tensor(ATOM_NUMERIC_SCALES))
        self.bond_type_embedding = nn.Embedding(len(BOND_TYPE_VOCAB), hidden_dim)
        self.bond_stereo_embedding = nn.Embedding(len(BOND_STEREO_VOCAB), hidden_dim)
        self.bond_flags = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList(_GINELayer(hidden_dim) for _ in range(layers))

    def forward(self, features):
        """Encode the disconnected union of the input molecular graphs."""
        if not features:
            raise ValueError("Empty drug batch")
        ref = self.atom_numeric.weight
        sizes = [f.num_atoms for f in features]
        offsets = np.cumsum([0, *sizes[:-1]])
        atoms = torch.as_tensor(
            np.concatenate([f.atom_features for f in features]), device=ref.device
        ).long()
        bonds = torch.as_tensor(
            np.concatenate([f.bond_features for f in features]), device=ref.device
        ).long()
        index = torch.as_tensor(
            np.concatenate([f.bond_index + off for f, off in zip(features, offsets)], axis=1),
            device=ref.device,
        ).long()
        masks = torch.as_tensor(
            np.concatenate([f.role_bits for f in features]).astype(np.int64), device=ref.device
        )
        roles = torch.stack([(masks >> bit) & 1 for bit in range(len(ROLE_VOCAB))], dim=-1)
        numeric = atoms[:, ATOM_NUMERIC_COLUMNS].to(ref.dtype) / self.numeric_scales
        nodes = self.atom_numeric(torch.cat((numeric, roles.to(ref.dtype)), dim=-1))
        for name, embedding in self.atom_embeddings.items():
            nodes = nodes + embedding(atoms[:, ATOM_FEATURE_NAMES.index(name)])
        edges = (
            self.bond_type_embedding(bonds[:, 0])
            + self.bond_stereo_embedding(bonds[:, 4])
            + self.bond_flags(bonds[:, 1:4].to(ref.dtype))
        )
        source, target = torch.cat((index[0], index[1])), torch.cat((index[1], index[0]))
        edges = torch.cat((edges, edges))
        for layer in self.layers:
            nodes = layer(nodes, edges, source, target)
        return nodes.split(sizes)


class ContactBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.contact = nn.Linear(width, width, bias=False)
        self.sequence = nn.Conv1d(width, width, 3, padding=1)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden, adjacency):
        message = self.contact(adjacency @ hidden)
        ordered = self.sequence(hidden.T.unsqueeze(0)).squeeze(0).T
        return self.norm(hidden + torch.nn.functional.gelu(message + ordered))


class ProteinContext(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.projection = nn.Linear(1280, width)
        self.residue = nn.Embedding(len(RESIDUE_VOCAB), width)
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([ContactBlock(width) for _ in range(2)])
        self.pool = nn.Linear(2 * width, width)

    def encode_chain(self, sequence, arrays):
        reference = self.projection.weight
        h = torch.as_tensor(arrays["residues"], device=reference.device, dtype=reference.dtype)
        p = torch.as_tensor(
            arrays["contacts"], device=reference.device, dtype=reference.dtype
        ).clone()
        # The residual path carries each residue’s own state.
        p.fill_diagonal_(0)
        # The unit offset retains the scale of weak contact probabilities.
        p = p / (1 + p.sum(-1, keepdim=True))
        lookup = {c: i for i, c in enumerate(RESIDUE_VOCAB)}
        tokens = torch.tensor([lookup[c] for c in sequence], device=reference.device)
        h = self.input_norm(self.projection(h) + self.residue(tokens))
        for layer in self.layers:
            h = layer(h, p)
        return h

    def readout(self, chains):
        # Chain pooling preserves multiplicity.
        h = torch.cat(chains)
        return self.pool(torch.cat((h.mean(0), h.amax(0))))


class MolecularRelationEncoder(nn.Module):
    """Bond messages and a molecule-wide summary."""

    def __init__(self, width=128, hidden=64):
        super().__init__()
        self.atom = nn.Linear(width, hidden)
        self.edge = nn.Linear(width, hidden, bias=False)
        self.message = nn.Linear(hidden, hidden)
        self.global_summary = nn.Linear(2 * hidden, hidden)
        self.returned = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.pool = nn.Linear(2 * hidden, width)

    def forward(self, h, e, index):
        x = F.gelu(self.atom(h))
        source, target = torch.cat((index[0], index[1])), torch.cat((index[1], index[0]))
        edge = torch.cat((e, e))
        message = torch.tanh(x[source] + self.edge(edge))
        agg = torch.zeros_like(x).index_add(0, target, message)
        degree = torch.bincount(target, minlength=len(x)).to(x.dtype)
        x = self.norm1(x + F.gelu(self.message(agg / (degree[:, None] + 1))))
        summary = self.global_summary(torch.cat((x.mean(0), x.amax(0))))
        x = self.norm2(x + F.gelu(self.returned(x) + summary))
        return self.pool(torch.cat((x.mean(0), x.amax(0))))
