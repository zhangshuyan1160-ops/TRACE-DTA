"""TRACE-DTA affinity predictor."""

import torch
from torch import nn

from .chemistry import encode_drug
from .encoders import DrugEncoder, MolecularRelationEncoder, ProteinContext


class TraceDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_encoder = DrugEncoder()
        self.protein_context = ProteinContext()
        self.regression_head = nn.Sequential(nn.Linear(256, 2048), nn.ReLU(), nn.Linear(2048, 1))
        self.relation_encoder = MolecularRelationEncoder()
        self.target_gate = nn.Sequential(nn.Linear(128, 16), nn.GELU(), nn.Linear(16, 128))
        self.residual_projection = nn.Linear(128, 128, bias=False)
        nn.init.zeros_(self.residual_projection.weight)
        self.register_buffer("training_pkd_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("training_pkd_std", torch.ones((), dtype=torch.float64))
        self.set_phase("affinity")

    def set_phase(self, phase):
        if phase not in {"base", "relation", "affinity"}:
            raise ValueError(f"Unknown training phase {phase}")
        self.phase = phase
        for module in (self.drug_encoder, self.protein_context, self.regression_head):
            module.requires_grad_(phase == "base")
        self.relation_encoder.requires_grad_(phase == "relation")
        self.target_gate.requires_grad_(phase == "affinity")
        self.residual_projection.requires_grad_(phase == "affinity")
        self.clear_cache()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.clear_cache()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def clear_cache(self):
        self._graphs = {}
        self._ligands = {}
        self._targets = {}

    def _graph(self, smiles):
        if smiles not in self._graphs:
            graph = encode_drug(smiles)
            with torch.no_grad():
                atoms = self.drug_encoder([graph])[0]
                bonds = torch.as_tensor(graph.bond_features, device=atoms.device).long()
                edges = (
                    self.drug_encoder.bond_type_embedding(bonds[:, 0])
                    + self.drug_encoder.bond_stereo_embedding(bonds[:, 4])
                    + self.drug_encoder.bond_flags(bonds[:, 1:4].to(atoms.dtype))
                )
                index = torch.as_tensor(graph.bond_index, device=atoms.device).long()
            self._graphs[smiles] = tuple(t.cpu() for t in (atoms, edges, index))
        return tuple(t.to(self.residual_projection.weight.device) for t in self._graphs[smiles])

    def relation_vectors(self, smiles):
        return torch.stack([self.relation_encoder(*self._graph(s)) for s in smiles])

    def _target_vectors(self, targets, store):
        if self.phase == "base":
            encoded = {
                s: self.protein_context.encode_chain(s, store.chain(s))
                for s in dict.fromkeys(s for target in targets for s in target)
            }
            return torch.stack(
                [self.protein_context.readout([encoded[s] for s in target]) for target in targets]
            )
        for target in dict.fromkeys(targets):
            if target not in self._targets:
                with torch.no_grad():
                    chains = [self.protein_context.encode_chain(s, store.chain(s)) for s in target]
                    self._targets[target] = self.protein_context.readout(chains).cpu()
        return torch.stack([self._targets[target] for target in targets]).to(
            self.residual_projection.weight.device
        )

    def normalized_affinity(self, smiles, targets, store):
        if not smiles or len(smiles) != len(targets) or self.phase == "relation":
            raise ValueError(
                "Affinity batches require aligned molecule and target inputs in a supervised phase"
            )
        protein = self._target_vectors([tuple(t) for t in targets], store)
        if self.phase == "base":
            nodes = self.drug_encoder([encode_drug(s) for s in smiles])
            drug = torch.stack([h.amax(0) for h in nodes])
        else:
            for s in dict.fromkeys(smiles):
                if s not in self._ligands:
                    with torch.no_grad():
                        graph = self._graph(s)
                        self._ligands[s] = (
                            graph[0].amax(0).cpu(),
                            self.relation_encoder(*graph).cpu(),
                        )
            pooled, relation = zip(*(self._ligands[s] for s in smiles))
            drug = torch.stack(pooled).to(protein.device)
            context = torch.stack(relation).to(protein.device) * self.target_gate(protein).sigmoid()
            drug = drug + self.residual_projection(context)
        return self.regression_head(torch.cat((drug, protein), dim=-1)).squeeze(-1).double()

    def forward(self, smiles, targets, store):
        z = self.normalized_affinity(smiles, targets, store)
        return self.training_pkd_mean + self.training_pkd_std * z
