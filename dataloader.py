# dataloader.py
"""
Data loader and collate utilities for the Dual 3D Graph Transformer.
Assumes each sample is saved as a torch .pt file containing a dict with keys:
 - 'ligand' : torch_geometric.data.Data (x, pos, edge_index, edge_attr optional)
 - 'protein': torch_geometric.data.Data (x, pos, edge_index, edge_attr optional)
 - 'cross_edges': LongTensor [N_cross, 2] (ligand_idx, protein_idx)
 - 'cross_attr' : FloatTensor [N_cross, RBF_dim]
 - 'y' : FloatTensor([pKd])
 Optionally:
 - 'residue_names' : list of strings
 - 'pdb' : pdb id (or we will infer from filename)
"""

import os
import glob
import random
import math
import json
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GeomData, Batch
from tqdm import tqdm
import numpy as np

# deterministic seeds
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)


class PDBBindPTDataset(Dataset):
    def __init__(self, folder: str, file_list: Optional[List[str]] = None, max_samples: Optional[int] = None):
        """
        folder: directory containing *.pt files from preprocessing
        file_list: optional list of filenames (basename sans .pt) to include
        """
        self.folder = folder
        all_files = sorted(glob.glob(os.path.join(folder, "*.pt")))
        if file_list:
            # file_list expected as pdb ids without .pt
            all_files = [os.path.join(folder, f"{pdb}.pt") for pdb in file_list if os.path.exists(os.path.join(folder, f"{pdb}.pt"))]
        self.files = all_files[:max_samples] if max_samples else all_files
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        sample = torch.load(path, weights_only=False)
        # ensure 'pdb' present
        pdb = sample.get("pdb", None)
        if pdb is None:
            pdb = os.path.splitext(os.path.basename(path))[0]
            sample["pdb"] = pdb
        # standardize keys if needed
        return sample


def collate_dual_graph(samples: List[Dict]):
    """
    Collate function to create a batch for the dual graph model.
    Produces:
      - batched_ligand: torch_geometric.data.Batch
      - batched_protein: torch_geometric.data.Batch
      - cross_edges: LongTensor [N_cross, 2] where indices are (global_lig_idx, global_prot_idx)
      - cross_attr: FloatTensor [N_cross, rbf_dim]
      - y: FloatTensor [B]
      - meta: list of pdb ids
    Important: we offset protein indices to global protein index space, but we also
    need to offset when batching; we will represent cross_edges as (lig_idx + lig_offset, prot_idx + prot_offset)
    where lig_offset accumulates across samples.
    """
    lig_list = []
    prot_list = []
    cross_edges_list = []
    cross_attr_list = []
    ys = []
    pdbs = []
    lig_offset = 0
    prot_offset = 0

    for s in samples:
        ligand: GeomData = s["ligand"]
        protein: GeomData = s["protein"]
        cross = s.get("cross_edges", None)
        cross_attr = s.get("cross_attr", None)
        y = s["y"]

        # ensure important fields exist
        if not hasattr(ligand, "pos") or not hasattr(protein, "pos"):
            raise ValueError("Ligand or protein missing pos field")

        # append ligand and protein data (Batch will handle node offsets internally but we need global cross edges)
        lig_list.append(ligand)
        prot_list.append(protein)

        n_lig_nodes = ligand.pos.shape[0]
        n_prot_nodes = protein.pos.shape[0]

        if cross is None or cross.shape[0] == 0:
            # no cross edges: skip but maintain alignment
            # represent as empty tensor
            pass
        else:
            # cross is Nx2 with ligand_idx, prot_idx (local to sample)
            # convert to global ids by adding current offsets
            cross_np = cross.cpu().numpy()
            global_lig = cross_np[:, 0] + lig_offset
            global_prot = cross_np[:, 1] + prot_offset
            global_edges = np.vstack([global_lig, global_prot]).T  # [N_cross,2]
            cross_edges_list.append(torch.tensor(global_edges, dtype=torch.long))
            if cross_attr is not None:
                cross_attr_list.append(cross_attr)

        ys.append(y.view(-1))
        pdbs.append(s.get("pdb", None))
        lig_offset += n_lig_nodes
        prot_offset += n_prot_nodes

    # Create batched Geometric data
    batched_ligand = Batch.from_data_list(lig_list)
    batched_protein = Batch.from_data_list(prot_list)

    # concatenate cross edges and attrs if any
    if len(cross_edges_list) > 0:
        cross_edges = torch.cat(cross_edges_list, dim=0)
        cross_attr = torch.cat(cross_attr_list, dim=0) if len(cross_attr_list) > 0 else None
    else:
        cross_edges = torch.empty((0, 2), dtype=torch.long)
        cross_attr = torch.empty((0, 1), dtype=torch.float)

    ys = torch.cat(ys, dim=0)
    return {
        "ligand": batched_ligand,
        "protein": batched_protein,
        "cross_edges": cross_edges,
        "cross_attr": cross_attr,
        "y": ys,
        "pdbs": pdbs
    }


def deterministic_split(file_paths: List[str], ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = RNG_SEED):
    """
    Deterministic split of file basenames into train/val/test. Returns (train_ids, val_ids, test_ids)
    This uses a hash of the filename to avoid accidental time-based splits.
    Note: For publication, replace this with scaffold and protein-family splits.
    """
    basenames = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]
    # use stable hash sorting
    hashed = [(b, (hash(b) & 0xffffffff)) for b in basenames]
    hashed.sort(key=lambda x: x[1])
    ordered = [h[0] for h in hashed]

    n = len(ordered)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = ordered[:n_train]
    val = ordered[n_train:n_train + n_val]
    test = ordered[n_train + n_val:]
    return train, val, test


def get_dataloaders(processed_folder: str,
                    batch_size: int = 4,
                    num_workers: int = 4,
                    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                    seed: int = RNG_SEED):
    all_files = sorted(glob.glob(os.path.join(processed_folder, "*.pt")))
    train_ids, val_ids, test_ids = deterministic_split(all_files, ratios=ratios, seed=seed)
    train_ds = PDBBindPTDataset(processed_folder, file_list=train_ids)
    val_ds = PDBBindPTDataset(processed_folder, file_list=val_ids)
    test_ds = PDBBindPTDataset(processed_folder, file_list=test_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_dual_graph)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_dual_graph)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_dual_graph)

    return train_loader, val_loader, test_loader