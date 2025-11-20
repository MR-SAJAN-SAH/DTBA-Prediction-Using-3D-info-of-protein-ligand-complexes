#!/usr/bin/env python3
"""
preprocess_pdbbind.py

Preprocessing pipeline for PDBBind general/refined sets to generate
torch_geometric-ready .pt files containing:
 - ligand: Data(x, pos, edge_index, edge_attr)
 - protein: Data(x, pos, edge_index, edge_attr)
 - cross_edges: LongTensor [N_cross, 2] (ligand_idx, protein_idx)
 - cross_attr: FloatTensor [N_cross, rbf_dim]
 - y: FloatTensor [1] (pKd)

"""

import os
import re
import math
import json
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# RDKit & Bio
from rdkit import Chem
from Bio.PDB import PDBParser

# Try fast radius graph from torch_cluster if available
try:
    from torch_cluster import radius_graph
    _HAS_TORCH_CLUSTER = True
except Exception:
    _HAS_TORCH_CLUSTER = False

# ------------------------
# Reproducibility
# ------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------
# Constants & config
# ------------------------
AA_LIST = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

RBF_K = 16
RBF_MIN = 0.0
RBF_MAX = 12.0
RBF_CENTERS = np.linspace(RBF_MIN, RBF_MAX, RBF_K)
RBF_GAMMA = 5.0  # width param: higher -> narrower Gaussians

# Defaults
PROTEIN_CONTACT_CUTOFF = 10.0  # initial residue selection radius (Å) from ligand atoms
PROTEIN_EDGE_CUTOFF = 8.0      # residue-residue edge cutoff
CROSS_EDGE_CUTOFF = 6.0        # ligand atom <-> residue cutoff
LIGAND_NONBOND_CUTOFF = 5.0    # optional nonbonded edges inside ligand if desired

# ------------------------
# Utility functions
# ------------------------
def parse_binding_value(s: str) -> Optional[float]:
    """
    Parse Kd/Ki/IC50 entry and return pKd (-log10(M)).
    Returns None if parsing fails or the value contains inequalities.
    Accepts patterns like: Kd=49uM, Ki=0.43 uM, IC50=1.2nM, 150 nM, 1.0e-6M
    """
    if s is None:
        return None
    s = s.strip()
    # remove trailing semicolons/commas
    s = s.replace(';', ' ').replace(',', ' ')
    # reject inequalities for now (could be used as weak labels separately)
    if '>' in s or '<' in s:
        return None

    # common pattern: maybe has label like "Kd=" or just a number+unit
    m = re.search(r'([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)\s*(mM|uM|nM|pM|M)', s, flags=re.IGNORECASE)
    if not m:
        # try Ki/Kd/IC50 prefix forms
        m2 = re.search(r'(Kd|Ki|IC50)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(mM|uM|nM|pM|M)', s, flags=re.IGNORECASE)
        if not m2:
            return None
        value, unit = float(m2.group(2)), m2.group(3).lower()
    else:
        value, unit = float(m.group(1)), m.group(2).lower()

    unit_scale = {'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9, 'pm': 1e-12, 'm': 1.0}
    if unit.lower() not in unit_scale:
        return None
    mol = value * unit_scale[unit.lower()]
    if mol <= 0.0:
        return None
    pKd = -math.log10(mol)
    return pKd

def load_affinities(index_file: str) -> dict:
    """
    Parse a PDBBind index file (INDEX_general_PL.*.lst) into {pdbid: pKd}
    If the index file format differs, adapt parsing logic accordingly.
    """
    affinities = {}
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Attempt robust splitting; many INDEX files have: pdb_code  ligand_code  target  affinity ...
            parts = re.split(r'\s+', line)
            # First token is usually pdb id
            pdb = parts[0].lower()
            # affinity often at position 3 or last token; we scan tokens for number+unit
            pKd_val = None
            for tok in parts:
                parsed = parse_binding_value(tok)
                if parsed is not None:
                    pKd_val = parsed
                    break
            if pKd_val is not None:
                affinities[pdb] = pKd_val
    return affinities

# ------------------------
# RDKit helpers
# ------------------------
def atom_feature_vector(atom: Chem.rdchem.Atom) -> torch.Tensor:
    """
    Returns a float vector for an RDKit atom including:
    atomic_num, degree, formal_charge, total_hs, aromatic, hybridization (one-hot-ish), in_ring
    """
    an = float(atom.GetAtomicNum())
    deg = float(atom.GetTotalDegree())
    ch = float(atom.GetFormalCharge())
    hcount = float(atom.GetTotalNumHs())
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    inring = 1.0 if atom.IsInRing() else 0.0

    # Hybridization one-hot limited set
    hyb = atom.GetHybridization()
    hyb_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.S: 3,
        Chem.rdchem.HybridizationType.UNSPECIFIED: 4
    }
    hyb_vec = np.zeros(len(hyb_map), dtype=np.float32)
    hyb_vec[hyb_map.get(hyb, 4)] = 1.0

    vec = np.concatenate([[an, deg, ch, hcount, aromatic, inring], hyb_vec])
    return torch.tensor(vec, dtype=torch.float)

def bond_feature_vector(bond: Chem.rdchem.Bond) -> torch.Tensor:
    """
    Bond features: single/double/triple/aromatic, conjugation, ring
    """
    bt = bond.GetBondType()
    b_single = 1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0
    b_double = 1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0
    b_triple = 1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0
    b_aromatic = 1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0
    conj = 1.0 if bond.GetIsConjugated() else 0.0
    ring = 1.0 if bond.IsInRing() else 0.0
    return torch.tensor([b_single, b_double, b_triple, b_aromatic, conj, ring], dtype=torch.float)

# ------------------------
# Protein features
# ------------------------
def residue_one_hot(resname: str) -> torch.Tensor:
    r = resname.strip().upper()
    vec = np.zeros(len(AA_LIST), dtype=np.float32)
    if r in AA_TO_IDX:
        vec[AA_TO_IDX[r]] = 1.0
    return torch.tensor(vec, dtype=torch.float)

# ------------------------
# RBF encoding
# ------------------------
def rbf_encoding(dists: np.ndarray, centers: np.ndarray = RBF_CENTERS, gamma: float = RBF_GAMMA) -> np.ndarray:
    # dists: shape (N,) or broadcastable
    d = np.expand_dims(dists, -1)  # (...,1)
    c = centers.reshape((1,) * (d.ndim - 1) + (-1,))
    return np.exp(-gamma * (d - c) ** 2)

# ------------------------
# Graph construction helpers
# ------------------------
def radius_edge_index(pos: torch.Tensor, r: float, max_num_neighbors: int = 512) -> torch.Tensor:
    """
    Returns edge_index shape [2, E] for nodes within radius r (undirected edges listed both ways).
    Attempts to use torch_cluster.radius_graph if available for speed.
    """
    if _HAS_TORCH_CLUSTER:
        # torch_cluster expects (N,3) float tensor on cuda or cpu
        edge_index = radius_graph(pos, r, loop=False, max_num_neighbors=max_num_neighbors)
        return edge_index
    else:
        # Fallback (numpy): compute pairwise distances
        pos_np = pos.cpu().numpy()
        dmat = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=-1)
        rows, cols = np.where((dmat < r) & (dmat > 0.0))  # exclude self
        if len(rows) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
        return edge_index

# ------------------------
# Loaders
# ------------------------
def load_ligand_graph(mol2_path: str, remove_hs: bool = True) -> Data:
    """
    Load ligand from mol2 (or mol) using RDKit and construct Data object with edge_attr.
    remove_hs default True (reduces node count and usually fine).
    """
    # RDKit can read mol2; if fails try mol/sdf
    mol = Chem.MolFromMol2File(mol2_path, removeHs=remove_hs)
    if mol is None:
        # try MolFromMolFile generic
        mol = Chem.MolFromMolFile(mol2_path, removeHs=remove_hs)
    if mol is None:
        raise ValueError(f"Could not parse ligand file: {mol2_path}")

    # Ensure 3D coordinates exist
    if mol.GetNumConformers() == 0:
        raise ValueError(f"No conformer coordinates in ligand: {mol2_path}")

    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)  # (N_atoms, 3)
    xs = [atom_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.stack(xs, dim=0)

    # Build bond edge_index + edge_attr
    edge_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bf = bond_feature_vector(bond)
        # add both directions
        edge_list.append([a1, a2]); edge_attr_list.append(bf)
        edge_list.append([a2, a1]); edge_attr_list.append(bf)
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr_list, dim=0)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    return data

def load_protein_graph(pdb_path: str, ligand_coords: np.ndarray, residue_cutoff: float = PROTEIN_CONTACT_CUTOFF,
                       residue_edge_cutoff: float = PROTEIN_EDGE_CUTOFF) -> Tuple[Data, List[str]]:
    """
    Extract residues close to ligand atoms. Residue node position = CA coord.
    Returns (Data, residue_names_list)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)  # may raise error if corrupted
    # collect all residues with CA
    ca_coords = []
    resnames = []
    res_ids = []  # for debug/tracking (chain, resid)
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip hetero residues (ligands) and keep standard residues only
                # residue.get_resname() returns 3-letter code
                if 'CA' not in residue:
                    continue
                ca = residue['CA'].coord  # numpy array 3
                ca_coords.append(ca)
                resnames.append(residue.get_resname().upper())
                res_ids.append((chain.id, residue.id[1]))
    if len(ca_coords) == 0:
        raise ValueError("No CA coords found in protein PDB: " + pdb_path)
    ca_coords = np.stack(ca_coords, axis=0)  # (N_res, 3)

    # compute min distance from each residue CA to any ligand atom
    # ligand_coords is (N_lig, 3)
    dmat = np.linalg.norm(ca_coords[:, None, :] - ligand_coords[None, :, :], axis=-1)
    min_d = dmat.min(axis=1)  # (N_res,)
    mask = min_d < residue_cutoff
    if mask.sum() < 4:
        # try expanding cutoff a bit to avoid tiny pockets
        mask = min_d < (residue_cutoff + 4.0)
        if mask.sum() < 4:
            raise ValueError(f"Pocket too small: {mask.sum()} residues found (pdb={pdb_path})")

    selected_coords = ca_coords[mask]
    selected_resnames = [r for i, r in enumerate(resnames) if mask[i]]

    pos = torch.tensor(selected_coords, dtype=torch.float)
    xs = [residue_one_hot(r) for r in selected_resnames]
    x = torch.stack(xs, dim=0)

    # edges based on residue_edge_cutoff
    edge_index = radius_edge_index(pos, r=residue_edge_cutoff)

    # compute simple edge_attr = distance RBF between residues (optional)
    if edge_index.numel() == 0:
        edge_attr = torch.empty((0, RBF_K), dtype=torch.float)
    else:
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        dists = np.linalg.norm(selected_coords[src] - selected_coords[dst], axis=-1)
        edge_attr = torch.tensor(rbf_encoding(dists), dtype=torch.float)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    return data, selected_resnames

def compute_cross_edges_and_attrs(ligand: Data, protein: Data, cutoff: float = CROSS_EDGE_CUTOFF) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cross edges between ligand atoms (rows) and protein residues (cols).
    Returns:
      - edges: LongTensor [N_cross, 2] each row [ligand_idx, protein_idx]
      - attrs: FloatTensor [N_cross, RBF_K] RBF encoding of distances
    """
    lig_pos = ligand.pos.cpu().numpy()  # (n_lig, 3)
    prot_pos = protein.pos.cpu().numpy()  # (n_res, 3)
    if lig_pos.size == 0 or prot_pos.size == 0:
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0, RBF_K), dtype=torch.float)

    dmat = np.linalg.norm(lig_pos[:, None, :] - prot_pos[None, :, :], axis=-1)  # (n_lig, n_res)
    pairs = np.where(dmat < cutoff)
    if len(pairs[0]) == 0:
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0, RBF_K), dtype=torch.float)

    lig_idx = pairs[0]
    prot_idx = pairs[1]
    dists = dmat[lig_idx, prot_idx]
    edges = np.vstack([lig_idx, prot_idx]).T  # [N_cross, 2]
    attrs = rbf_encoding(dists)
    return torch.tensor(edges, dtype=torch.long), torch.tensor(attrs, dtype=torch.float)

# ------------------------
# Main preprocess pipeline
# ------------------------
def preprocess_pdbbind(root_dir: str, index_file: str, out_dir: str,
                       remove_hs: bool = True,
                       max_samples: Optional[int] = None):
    os.makedirs(out_dir, exist_ok=True)
    affinities = load_affinities(index_file)
    saved = []
    failed = []

    pdb_ids = sorted(affinities.keys())
    if max_samples:
        pdb_ids = pdb_ids[:max_samples]

    for pdb in tqdm(pdb_ids, desc="Preprocessing"):
        pKd = affinities[pdb]
        complex_dir = os.path.join(root_dir, pdb)
        ligand_path = os.path.join(complex_dir, f"{pdb}_ligand.mol2")
        protein_path = os.path.join(complex_dir, f"{pdb}_protein.pdb")

        if not (os.path.exists(ligand_path) and os.path.exists(protein_path)):
            failed.append((pdb, "missing files"))
            continue
        try:
            ligand = load_ligand_graph(ligand_path, remove_hs=remove_hs)
            ligand_coords = ligand.pos.cpu().numpy()

            # use ligand atoms to choose pocket residues
            protein, residue_names = load_protein_graph(protein_path, ligand_coords, residue_cutoff=PROTEIN_CONTACT_CUTOFF,
                                                       residue_edge_cutoff=PROTEIN_EDGE_CUTOFF)

            # compute cross edges + attributes
            cross_edges, cross_attr = compute_cross_edges_and_attrs(ligand, protein, cutoff=CROSS_EDGE_CUTOFF)

            # optionally: add nonbonded ligand atom edges within LIGAND_NONBOND_CUTOFF
            # (useful for capturing steric interactions not captured by covalent bonds)
            lig_nonbond_edge_index = torch.empty((2, 0), dtype=torch.long)
            lig_nonbond_edge_attr = torch.empty((0, RBF_K), dtype=torch.float)
            try:
                lig_nb = radius_edge_index(ligand.pos, r=LIGAND_NONBOND_CUTOFF)
                if lig_nb.numel() > 0:
                    # filter out edges that are actual covalent bonds (optional):
                    # we keep all and let model learn
                    lig_nonbond_edge_index = lig_nb
                    # compute distances
                    src, dst = lig_nb[0].numpy(), lig_nb[1].numpy()
                    dists = np.linalg.norm(ligand.pos.numpy()[src] - ligand.pos.numpy()[dst], axis=-1)
                    lig_nonbond_edge_attr = torch.tensor(rbf_encoding(dists), dtype=torch.float)
            except Exception:
                # ignore optional step on failure
                lig_nonbond_edge_index = torch.empty((2, 0), dtype=torch.long)
                lig_nonbond_edge_attr = torch.empty((0, RBF_K), dtype=torch.float)

            sample = {
                "ligand": ligand,
                "ligand_nonbond_edge_index": lig_nonbond_edge_index,
                "ligand_nonbond_edge_attr": lig_nonbond_edge_attr,
                "protein": protein,
                "residue_names": residue_names,
                "cross_edges": cross_edges,  
                "cross_attr": cross_attr,    
                "y": torch.tensor([pKd], dtype=torch.float)
            }

            out_path = os.path.join(out_dir, f"{pdb}.pt")
            torch.save(sample, out_path)
            saved.append(pdb)

        except Exception as e:
            failed.append((pdb, str(e)))

    torch.save(failed, os.path.join(out_dir, "failed.pt"))
    print(f"✅ Completed preprocessing. Saved: {len(saved)}, Failed: {len(failed)}")
    if len(saved) == 0:
        print("⚠️ Warning: No .pt files saved. Check paths or file naming pattern.")
    else:
        print(f"Example saved file count: {len(saved)}")
        if len(failed) > 0:
            print(f"Some failures (first 10): {failed[:10]}")

if __name__ == "__main__":
    ROOT = r"C:\\Users\\sajan\\Desktop\\res_dl\\3D DUAL GRAPH\\pdbbind\\data"  
    INDEX = r"C:\\Users\\sajan\\Desktop\\res_dl\\3D DUAL GRAPH\\pdbbind\\index\\INDEX_general_PL.2020R1.lst"
    OUT = r"C:\\Users\\sajan\\Desktop\\res_dl\\3D DUAL GRAPH\\pdbbind\\processed_general"

    MAX_SAMPLES = None 

    preprocess_pdbbind(ROOT, INDEX, OUT, remove_hs=True, max_samples=MAX_SAMPLES)
