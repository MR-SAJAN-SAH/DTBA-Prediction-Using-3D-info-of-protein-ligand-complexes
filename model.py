# model.py
"""
Dual 3D Graph Transformer model. Practical, high-performing implementation that:
 - uses E(n)-equivariant blocks (EGNN-style) for intra-graph encoding (ligand + protein)
 - uses a cross-attention module with RBF distance bias + directional encodings
 - stacks iterative blocks: [ligand-EGNN, protein-EGNN, cross-attention] x N
 - pooling and MLP head outputting mean pKd and log-variance (uncertainty)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = torch.zeros((dim_size, src.size(1)), device=src.device)
    count = torch.zeros(dim_size, device=src.device)
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    count = count.clamp(min=1).unsqueeze(-1)
    return out / count

def scatter_add(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = torch.zeros((dim_size, src.size(1)), device=src.device)
    out.index_add_(0, index, src)
    return out


# Basic EGNN layer (E(n)-equivariant graph neural network) - compact variant
class EGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate input dimension for edge MLP
        edge_mlp_input_dim = 2 * in_channels + edge_attr_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_input_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        # for coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Tanh()
        )

    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        x: [N, F]
        pos: [N, 3]
        edge_index: [2, E] (src, dst)
        edge_attr: [E, D] optional
        returns: x', pos'
        """
        src, dst = edge_index[0], edge_index[1]
        
        x_src = x[src]
        x_dst = x[dst]
        
        if edge_attr is None:
            edge_in = torch.cat([x_src, x_dst], dim=-1)
        else:
            edge_in = torch.cat([x_src, x_dst, edge_attr], dim=-1)
            
        e = self.edge_mlp(edge_in)  # [E, out]
        
        # message aggregation
        m = scatter_mean(e, dst, dim=0, dim_size=x.shape[0])  # mean messages to dst
        x_in = torch.cat([x, m], dim=-1)
        x_out = self.node_mlp(x_in)
        
        # coordinate update (equivariant-like): compute per-edge scalar and vector shifts
        vec = pos[src] - pos[dst]  # vector from dst to src
        # scalar per-edge
        coord_coeff = self.coord_mlp(e)  # [E,1]
        # weighted sum of vectors to each dst
        vec_msg = vec * coord_coeff  # [E,3]
        pos_update = scatter_mean(vec_msg, dst, dim=0, dim_size=pos.shape[0])
        pos_out = pos + pos_update
        
        return x_out, pos_out


# Simple RBF encoding helper
def rbf_tensor(distances: torch.Tensor, centers: torch.Tensor, gamma: float = 5.0):
    # distances: [N] or broadcastable
    # centers: [K]
    diff = distances.unsqueeze(-1) - centers.view(1, -1)
    return torch.exp(-gamma * (diff ** 2))


class CrossAttentionModule(nn.Module):
    """
    Cross Attention: ligand nodes (Q) attend to protein nodes (K,V) and vice versa.
    Uses dot-product attention on scalar embeddings with additive bias from RBF encodings of distances.
    This module operates on global node indices (batched).
    """
    def __init__(self, dim: int, heads: int = 8, rbf_k: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        # map RBF embedding to scalar bias per head
        self.rbf_proj = nn.Linear(rbf_k, heads)

    def forward(self,
                ligand_x, ligand_idx_range,  # ligand_x: [N_lig, D]
                protein_x, protein_idx_range,  # protein_x: [N_prot, D]
                cross_edges, cross_attr  # cross_edges: [N_cross,2] (global indices), cross_attr: [N_cross, rbf_k]
                ):
        """
        ligand_idx_range: tuple (lig_start, lig_end) the global index range where ligand nodes reside
        protein_idx_range: tuple (prot_start, prot_end)
        cross_edges: LongTensor [N_cross, 2] in global indexing
        cross_attr: FloatTensor [N_cross, rbf_k]
        Returns:
          ligand_out (same shape as ligand_x),
          protein_out (same shape as protein_x)
        """
        N_lig = ligand_x.shape[0]
        N_prot = protein_x.shape[0]
        device = ligand_x.device

        # Linear projections
        q_l = self.to_q(ligand_x)  # [N_lig, D]
        k_p = self.to_k(protein_x)  # [N_prot, D]
        v_p = self.to_v(protein_x)

        q_p = self.to_q(protein_x)
        k_l = self.to_k(ligand_x)
        v_l = self.to_v(ligand_x)

        # reshape for heads
        def reshape_heads(t):
            bsz = t.shape[0]
            return t.view(bsz, self.heads, -1)  # (N, heads, head_dim)
        
        q_lh = reshape_heads(q_l)
        k_ph = reshape_heads(k_p)
        v_ph = reshape_heads(v_p)
        q_ph = reshape_heads(q_p)
        k_lh = reshape_heads(k_l)
        v_lh = reshape_heads(v_l)

        # Prepare sparse attention along cross_edges
        if cross_edges.numel() == 0:
            # no cross edges: return identity
            return ligand_x, protein_x

        # Convert global indices into local indices
        lig_start, lig_end = ligand_idx_range
        prot_start, prot_end = protein_idx_range
        lig_global = cross_edges[:, 0]
        prot_global = cross_edges[:, 1]
        lig_local = lig_global - lig_start
        prot_local = prot_global - prot_start

        # gather relevant vectors
        q_pairs = q_lh[lig_local]  # [Nc, heads, head_dim]
        k_pairs = k_ph[prot_local]
        v_pairs = v_ph[prot_local]

        # compute per-edge attention logits per head: dot product q·k
        logits = (q_pairs * k_pairs).sum(dim=-1) * self.scale  # [Nc, heads]

        # add RBF bias projected to heads
        if cross_attr is not None and cross_attr.numel() > 0:
            rbf_bias = self.rbf_proj(cross_attr)  # [Nc, heads]
            logits = logits + rbf_bias.to(device)

        # Ligand attention - manual softmax without torch_scatter
        group_idx = lig_local
        unique_groups = torch.unique(group_idx)
        
        # Compute max per group manually
        max_logits = torch.full((N_lig, self.heads), -float('inf'), device=device)
        for group in unique_groups:
            mask = (group_idx == group)
            if mask.any():
                max_logits[group] = torch.max(logits[mask], dim=0)[0]
        
        max_per_edge = max_logits[group_idx]
        exp_logits = torch.exp(logits - max_per_edge)
        
        # Compute sum per group manually
        sum_exp = torch.zeros((N_lig, self.heads), device=device)
        for group in unique_groups:
            mask = (group_idx == group)
            if mask.any():
                sum_exp[group] = torch.sum(exp_logits[mask], dim=0)
        
        sum_per_edge = sum_exp[group_idx] + 1e-8
        attn_weights = (exp_logits / sum_per_edge).unsqueeze(-1)  # [Nc, heads, 1]

        # Aggregate ligand outputs manually
        head_dim = v_pairs.shape[-1]
        lig_agg = torch.zeros((N_lig, self.heads * head_dim), device=device)
        
        for group in unique_groups:
            mask = (group_idx == group)
            if mask.any():
                for h in range(self.heads):
                    w = attn_weights[mask, h, 0].unsqueeze(-1)  # [group_size, 1]
                    v_h = v_pairs[mask, h, :]  # [group_size, head_dim]
                    weighted_h = w * v_h  # [group_size, head_dim]
                    agg_h = torch.sum(weighted_h, dim=0)  # [head_dim]
                    lig_agg[group, h*head_dim:(h+1)*head_dim] = agg_h
        
        ligand_out = self.out(lig_agg)
        ligand_out = self.dropout(ligand_out)

        # Protein attention (symmetric) - manual implementation
        q_pairs_p = q_ph[prot_local]
        k_pairs_l = k_lh[lig_local]
        v_pairs_l = v_lh[lig_local]
        logits_p = (q_pairs_p * k_pairs_l).sum(dim=-1) * self.scale
        
        if cross_attr is not None and cross_attr.numel() > 0:
            logits_p = logits_p + self.rbf_proj(cross_attr).to(device)
            
        group_idx_p = prot_local
        unique_groups_p = torch.unique(group_idx_p)
        
        # Compute max per group manually for protein
        max_p = torch.full((N_prot, self.heads), -float('inf'), device=device)
        for group in unique_groups_p:
            mask = (group_idx_p == group)
            if mask.any():
                max_p[group] = torch.max(logits_p[mask], dim=0)[0]
        
        max_per_edge_p = max_p[group_idx_p]
        exp_p = torch.exp(logits_p - max_per_edge_p)
        
        # Compute sum per group manually for protein
        sum_exp_p = torch.zeros((N_prot, self.heads), device=device)
        for group in unique_groups_p:
            mask = (group_idx_p == group)
            if mask.any():
                sum_exp_p[group] = torch.sum(exp_p[mask], dim=0)
        
        sum_per_edge_p = sum_exp_p[group_idx_p] + 1e-8
        attn_p = (exp_p / sum_per_edge_p).unsqueeze(-1)
        
        # Aggregate protein outputs manually
        prot_agg = torch.zeros((N_prot, self.heads * head_dim), device=device)
        
        for group in unique_groups_p:
            mask = (group_idx_p == group)
            if mask.any():
                for h in range(self.heads):
                    w = attn_p[mask, h, 0].unsqueeze(-1)
                    v_h = v_pairs_l[mask, h, :]
                    weighted_h = w * v_h
                    agg_h = torch.sum(weighted_h, dim=0)
                    prot_agg[group, h*head_dim:(h+1)*head_dim] = agg_h
        
        protein_out = self.out(prot_agg)
        protein_out = self.dropout(protein_out)

        return ligand_out, protein_out


class DualGraphTransformer(nn.Module):
    def __init__(self,
                 node_dim: int = 64,
                 hidden_dim: int = 128,
                 n_layers: int = 4,
                 rbf_k: int = 16,
                 heads: int = 8,
                 dropout: float = 0.1,
                 lig_edge_attr_dim: int = 6,   # Fixed dimensions from your data
                 prot_edge_attr_dim: int = 16): # Fixed dimensions from your data
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.lig_edge_attr_dim = lig_edge_attr_dim
        self.prot_edge_attr_dim = prot_edge_attr_dim

        # initial projections - match your data dimensions
        self.lig_proj = nn.Linear(11, hidden_dim)  # ligand features: 11
        self.prot_proj = nn.Linear(20, hidden_dim)  # protein features: 20

        # EGNN stacks for intra-graph processing - initialize with fixed dimensions
        self.lig_egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, edge_attr_dim=lig_edge_attr_dim)
            for _ in range(n_layers)
        ])
        self.pro_egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, edge_attr_dim=prot_edge_attr_dim)
            for _ in range(n_layers)
        ])

        # cross-attention module
        self.cross_attn = CrossAttentionModule(
            dim=hidden_dim, heads=heads, rbf_k=rbf_k, dropout=dropout
        )

        self.T = n_layers

        # readout heads
        self.lig_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pro_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # final pooling and MLP
        self.pool_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # regression head outputs mean and log variance
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.logvar = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # rbf centers for cross-attn bias
        self.register_buffer("rbf_centers", torch.linspace(0.0, 12.0, rbf_k))

    def forward(self, ligand, protein, cross_edges, cross_attr):
        """
        ligand: Batched torch_geometric Batch object for ligands
        protein: Batched torch_geometric Batch object for proteins  
        cross_edges: [N_cross, 2] global indices
        cross_attr: [N_cross, rbf_k]
        Returns:
            mu: [B] predicted pKd mean
            logvar: [B] predicted log variance
            aux: dict for interpretability
        """
        device = ligand.x.device

        # initial projections
        lig_x = self.lig_proj(ligand.x)  # [N_lig_total, hidden]
        prot_x = self.prot_proj(protein.x)  # [N_prot_total, hidden]
        lig_pos = ligand.pos
        prot_pos = protein.pos

        # compute per-sample index ranges
        lig_batch = ligand.batch
        prot_batch = protein.batch
        num_lig_samples = int(lig_batch.max().item()) + 1
        num_prot_samples = int(prot_batch.max().item()) + 1
        
        # compute counts and offsets
        lig_counts = torch.bincount(lig_batch)
        prot_counts = torch.bincount(prot_batch)
        lig_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), lig_counts.cumsum(dim=0)[:-1]])
        prot_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), prot_counts.cumsum(dim=0)[:-1]])

        # Get edge attributes - use fixed dimensions
        lig_edge_attr = getattr(ligand, 'edge_attr', None)
        prot_edge_attr = getattr(protein, 'edge_attr', None)
        
        # Handle cases where edge attributes might have wrong dimensions
        if lig_edge_attr is not None and lig_edge_attr.shape[-1] != self.lig_edge_attr_dim:
            # If dimensions don't match, use only the first self.lig_edge_attr_dim features
            lig_edge_attr = lig_edge_attr[:, :self.lig_edge_attr_dim]
        
        if prot_edge_attr is not None and prot_edge_attr.shape[-1] != self.prot_edge_attr_dim:
            # If dimensions don't match, use only the first self.prot_edge_attr_dim features
            prot_edge_attr = prot_edge_attr[:, :self.prot_edge_attr_dim]

        # Iterate over blocks
        for t in range(self.T):
            # intra-ligand EGNN
            lig_x, lig_pos = self._apply_egnn_layer(
                self.lig_egnn_layers[t], lig_x, lig_pos, 
                ligand.edge_index, 
                lig_edge_attr
            )
            
            # intra-protein EGNN  
            prot_x, prot_pos = self._apply_egnn_layer(
                self.pro_egnn_layers[t], prot_x, prot_pos,
                protein.edge_index,
                prot_edge_attr
            )

            # cross-attention per sample
            lig_out_chunks = []
            prot_out_chunks = []
            
            for sample_idx in range(num_lig_samples):
                lig_start = lig_offsets[sample_idx].item()
                lig_end = lig_start + lig_counts[sample_idx].item()
                prot_start = prot_offsets[sample_idx].item() 
                prot_end = prot_start + prot_counts[sample_idx].item()

                lig_slice_x = lig_x[lig_start:lig_end]
                prot_slice_x = prot_x[prot_start:prot_end]

                if cross_edges.numel() == 0:
                    lig_att_out, prot_att_out = lig_slice_x, prot_slice_x
                else:
                    mask = (cross_edges[:, 0] >= lig_start) & (cross_edges[:, 0] < lig_end) & \
                           (cross_edges[:, 1] >= prot_start) & (cross_edges[:, 1] < prot_end)
                    
                    if mask.sum() == 0:
                        lig_att_out, prot_att_out = lig_slice_x, prot_slice_x
                    else:
                        sample_cross = cross_edges[mask].clone()
                        sample_attr = cross_attr[mask].clone() if cross_attr is not None else None
                        
                        sample_cross[:, 0] = sample_cross[:, 0] - lig_start
                        sample_cross[:, 1] = sample_cross[:, 1] - prot_start
                        
                        lig_att_out, prot_att_out = self.cross_attn(
                            lig_slice_x, (0, lig_slice_x.shape[0]),
                            prot_slice_x, (0, prot_slice_x.shape[0]),
                            sample_cross, sample_attr
                        )
                
                lig_out_chunks.append(lig_att_out)
                prot_out_chunks.append(prot_att_out)

            lig_x = torch.cat(lig_out_chunks, dim=0)
            prot_x = torch.cat(prot_out_chunks, dim=0)

        # Readout and pooling
        lig_emb = self.lig_readout(lig_x)
        prot_emb = self.pro_readout(prot_x)

        lig_pool = scatter_mean(lig_emb, lig_batch, dim=0)
        prot_pool = scatter_mean(prot_emb, prot_batch, dim=0)

        global_emb = torch.cat([lig_pool, prot_pool], dim=-1)
        pooled = self.pool_mlp(global_emb)

        mu = self.regressor(pooled).squeeze(-1)
        logvar = self.logvar(pooled).squeeze(-1)

        return mu, logvar, {"lig_pool": lig_pool, "prot_pool": prot_pool}

    def _apply_egnn_layer(self, layer, x, pos, edge_index, edge_attr):
        """Apply EGNN layer with residual connections"""
        if edge_index is None or edge_index.numel() == 0:
            return x, pos
            
        x_out, pos_out = layer(x, pos, edge_index, edge_attr)
        
        x = x + x_out
        pos = pos + 0.1 * (pos_out - pos) 
        
        return x, pos
