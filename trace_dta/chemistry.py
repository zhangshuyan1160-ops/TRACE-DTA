"""Molecular graphs and atom features for TRACE-DTA."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import ChemicalFeatures

RDKIT_VERSION = "2025.03.6"
BASE_FEATURES_SHA256 = "87ab54bd717af50a78a710d23e7737a2cafc96e5293bf6257ecdda86b6da3f65"
ROLE_VOCAB = (
    "Donor",
    "Acceptor",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
)
HYBRIDIZATION_VOCAB = (
    "UNSPECIFIED",
    "S",
    "SP",
    "SP2",
    "SP3",
    "SP2D",
    "SP3D",
    "SP3D2",
    "OTHER",
)
CIP_VOCAB = ("", "R", "S", "r", "s", "other")
CHIRAL_CLASS_VOCAB = (
    "unspecified",
    "tetrahedral",
    "other",
    "allene",
    "square_planar",
    "trigonal_bipyramidal",
    "octahedral",
)
BOND_TYPE_VOCAB = (
    "UNSPECIFIED",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "QUADRUPLE",
    "QUINTUPLE",
    "HEXTUPLE",
    "ONEANDAHALF",
    "TWOANDAHALF",
    "THREEANDAHALF",
    "FOURANDAHALF",
    "FIVEANDAHALF",
    "AROMATIC",
    "IONIC",
    "HYDROGEN",
    "THREECENTER",
    "DATIVEONE",
    "DATIVE",
    "DATIVEL",
    "DATIVER",
    "OTHER",
    "ZERO",
)
BOND_STEREO_VOCAB = (
    "STEREONONE",
    "STEREOANY",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
    "STEREOATROPCW",
    "STEREOATROPCCW",
)
ATOM_FEATURE_NAMES = (
    "atomic_number",
    "isotope",
    "formal_charge",
    "degree",
    "total_valence",
    "total_hydrogens",
    "explicit_hydrogens",
    "radical_electrons",
    "aromatic",
    "in_ring",
    "hybridization",
    "chiral_class",
    "cip_code",
)
BOND_FEATURE_NAMES = ("bond_type", "conjugated", "aromatic", "in_ring", "stereo")


@dataclass(frozen=True)
class DrugFeatures:
    smiles: str
    atom_features: np.ndarray
    bond_features: np.ndarray
    bond_index: np.ndarray
    role_bits: np.ndarray

    @property
    def num_atoms(self):
        return len(self.atom_features)


@lru_cache(maxsize=1)
def _feature_factory() -> object:
    if rdBase.rdkitVersion != RDKIT_VERSION:
        raise RuntimeError(
            f"Feature contract requires RDKit {RDKIT_VERSION}, got {rdBase.rdkitVersion}"
        )
    source = (Path(RDConfig.RDDataDir) / "BaseFeatures.fdef").read_bytes()
    if hashlib.sha256(source).hexdigest() != BASE_FEATURES_SHA256:
        raise RuntimeError("BaseFeatures.fdef differs from the feature definition")
    factory = ChemicalFeatures.BuildFeatureFactoryFromString(source.decode("utf-8"))
    if tuple(factory.GetFeatureFamilies()) != ROLE_VOCAB:
        raise RuntimeError("RDKit feature family order differs from the role vocabulary")
    return factory


def _chiral_class(tag: str) -> int:
    names = {
        "CHI_UNSPECIFIED": "unspecified",
        "CHI_TETRAHEDRAL_CW": "tetrahedral",
        "CHI_TETRAHEDRAL_CCW": "tetrahedral",
        "CHI_TETRAHEDRAL": "tetrahedral",
        "CHI_OTHER": "other",
        "CHI_ALLENE": "allene",
        "CHI_SQUAREPLANAR": "square_planar",
        "CHI_TRIGONALBIPYRAMIDAL": "trigonal_bipyramidal",
        "CHI_OCTAHEDRAL": "octahedral",
    }
    return CHIRAL_CLASS_VOCAB.index(names[tag])


@lru_cache(maxsize=16384)
def encode_drug(smiles: str) -> DrugFeatures:
    if not isinstance(smiles, str) or not smiles or any(c.isspace() for c in smiles):
        raise ValueError("SMILES must be a nonempty string without whitespace or trailing names")
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    parser.allowCXSMILES = False
    parser.parseName = False
    molecule = Chem.MolFromSmiles(smiles, parser)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("SMILES cannot be parsed and sanitized into a nonempty molecule")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    factory = _feature_factory()
    num_atoms = molecule.GetNumAtoms()
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()]
    roles = np.zeros(num_atoms, dtype=np.uint16)
    for feature in factory.GetFeaturesForMol(molecule):
        bit = np.uint16(1 << ROLE_VOCAB.index(feature.GetFamily()))
        for atom_id in feature.GetAtomIds():
            roles[atom_id] |= bit
    atom_rows = []
    for atom in molecule.GetAtoms():
        tag = str(atom.GetChiralTag())
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        atom_rows.append(
            (
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                atom.GetDegree(),
                atom.GetTotalValence(),
                atom.GetTotalNumHs(includeNeighbors=True),
                atom.GetNumExplicitHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                HYBRIDIZATION_VOCAB.index(str(atom.GetHybridization())),
                _chiral_class(tag),
                CIP_VOCAB.index(cip) if cip in CIP_VOCAB else CIP_VOCAB.index("other"),
            )
        )
    bond_rows = []
    for bond in molecule.GetBonds():
        bond_rows.append(
            (
                BOND_TYPE_VOCAB.index(str(bond.GetBondType())),
                int(bond.GetIsConjugated()),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
                BOND_STEREO_VOCAB.index(str(bond.GetStereo())),
            )
        )
    atoms = np.asarray(atom_rows, dtype=np.int32)
    edges = np.asarray(bond_rows, dtype=np.int32).reshape(-1, len(BOND_FEATURE_NAMES))
    if np.any(atoms[:, 11] > 1) or np.any(atoms[:, 12] == CIP_VOCAB.index("other")):
        raise ValueError("Unsupported atom stereochemistry")
    for bond_type, _, _, _, stereo in edges:
        if BOND_TYPE_VOCAB[bond_type] in {"DATIVEONE", "DATIVE", "DATIVEL", "DATIVER"}:
            raise ValueError("Directional dative bonds are unsupported")
        if BOND_STEREO_VOCAB[stereo] not in {"STEREONONE", "STEREOANY", "STEREOE", "STEREOZ"}:
            raise ValueError("Unsupported bond stereochemistry")
    return DrugFeatures(
        smiles, atoms, edges, np.asarray(bonds, dtype=np.int64).reshape(-1, 2).T, roles
    )
