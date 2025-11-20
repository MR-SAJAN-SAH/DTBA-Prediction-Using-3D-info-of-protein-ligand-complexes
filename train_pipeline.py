# train_pipeline.py
"""
Enhanced training pipeline with advanced scheduling and training options.
Supports: cosine annealing with warm restarts, ranking loss, mixed precision, and comprehensive checkpointing.
"""

import os
import math
import argparse
import time
from copy import deepcopy
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score

from dataloader import get_dataloaders
from model import DualGraphTransformer

def safe_load_checkpoint(filename, device):
    """Safely load checkpoint with compatibility for PyTorch 2.6+ weights_only=True"""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        try:
            # First try with weights_only=True (safe mode)
            checkpoint = torch.load(filename, map_location=device, weights_only=True)
            return checkpoint
        except Exception as e:
            print(f"Safe loading failed: {e}")
            print("Attempting to load with weights_only=False (ensure you trust this checkpoint)")
            try:
                # Fallback to weights_only=False with safe globals
                import numpy
                torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
                checkpoint = torch.load(filename, map_location=device, weights_only=False)
                return checkpoint
            except:
                # Final fallback - direct load
                print("Using direct load without weights_only restriction")
                checkpoint = torch.load(filename, map_location=device)
                return checkpoint
    else:
        print(f"No checkpoint found at '{filename}'")
        return None



def concordance_index(y_true, y_pred):
    """
    Concordance index (Harrell's C-index) for regression ranking.
    Returns value in [0,1].
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = 0.0
    n_concordant = 0.0
    for i in range(len(y_true)):
        for j in range(i+1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            n += 1.0
            if (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0:
                n_concordant += 1.0
            elif (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) == 0:
                n_concordant += 0.5
    return n_concordant / n if n > 0 else 0.5

def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))

def pearson_r(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0,1])

# -----------------------------
# Losses
# -----------------------------
class HeteroscedasticLoss(nn.Module):
    """
    Negative log-likelihood for Gaussian with predicted mean mu and log variance logvar.
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar, y):
        # y: [B], mu: [B], logvar: [B]
        prec = torch.exp(-logvar)
        mse = (mu - y) ** 2
        loss = 0.5 * (mse * prec + logvar)
        return loss.mean()


class PairwiseRankingLoss(nn.Module):
    """
    Hinge ranking loss applied to pairs where y_i > y_j.
    We will mine pairs within each batch for samples sharing the same protein if pdb ids present.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, pdbs: list):
        # preds: [B], targets: [B]
        # build pairs where targets[i] > targets[j]
        loss = preds.new_zeros(1)
        count = 0
        B = preds.shape[0]
        for i in range(B):
            for j in range(B):
                if i == j:
                    continue
                # only consider pairs with same protein pdb (more meaningful ranking)
                if pdbs[i] != pdbs[j]:
                    continue
                if targets[i] > targets[j] + 1e-6:
                    margin_term = torch.clamp(self.margin - (preds[i] - preds[j]), min=0.0)
                    loss += margin_term
                    count += 1
        if count == 0:
            return preds.new_zeros(1).mean()
        return loss / count


# -----------------------------
# Scheduler Factory
# -----------------------------
def get_scheduler(optimizer, args):
    """Create learning rate scheduler based on type"""
    if args.scheduler_type == "cosine_warm":
        # Cosine annealing with warm restarts
        return CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=args.T_0, 
            T_mult=args.T_mult, 
            eta_min=args.min_lr
        )
    elif args.scheduler_type == "cosine":
        # Standard cosine annealing
        return CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )
    elif args.scheduler_type == "step":
        # Step decay
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.step_size, 
            gamma=args.gamma
        )
    else:
        # Constant LR
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)


# -----------------------------
# Training / Evaluation loops
# -----------------------------
def train_epoch(model, device, dataloader, optimizer, scheduler, scaler, mse_loss_fn, rank_loss_fn, epoch, args):
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_rank_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        ligand = batch["ligand"].to(device)
        protein = batch["protein"].to(device)
        cross_edges = batch["cross_edges"].to(device)
        cross_attr = batch["cross_attr"].to(device) if batch["cross_attr"] is not None else torch.empty((0, args.rbf_k), device=device)
        y = batch["y"].to(device).view(-1)

        # Mixed precision context
        if args.amp and device.type == 'cuda':
            context = torch.cuda.amp.autocast()
        else:
            context = torch.cuda.amp.autocast(enabled=False)  # Disable if no CUDA
        
        with context:
            mu, logvar, aux = model(ligand, protein, cross_edges, cross_attr)
            
            # Calculate losses
            mse_l = mse_loss_fn(mu, y)
            hetero_l = 0.0
            
            if args.use_heteroscedastic:
                het_fn = HeteroscedasticLoss().to(device)
                hetero_l = het_fn(mu, logvar, y)
                base_loss = hetero_l
            else:
                base_loss = mse_l
                
            rank_l = rank_loss_fn(mu, y, batch["pdbs"])
            loss = args.lambda_mse * base_loss + args.lambda_rank * rank_l

        if args.amp and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Update scheduler if step-based
        if args.scheduler_type == "cosine_warm" and batch_idx % args.scheduler_step_interval == 0:
            scheduler.step(epoch + batch_idx / len(dataloader))

        total_loss += float(loss.detach().cpu().numpy())
        total_mse_loss += float(base_loss.detach().cpu().numpy())
        total_rank_loss += float(rank_l.detach().cpu().numpy())
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'mse': total_mse_loss / (batch_idx + 1),
            'rank': total_rank_loss / (batch_idx + 1),
            'lr': f'{current_lr:.2e}'
        })
    
    # Update scheduler if epoch-based
    if args.scheduler_type != "cosine_warm":
        scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse_loss / len(dataloader)
    avg_rank = total_rank_loss / len(dataloader)
    
    return avg_loss, avg_mse, avg_rank


def evaluate(model, device, dataloader, args):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Eval", leave=False)
        for batch in pbar:
            ligand = batch["ligand"].to(device)
            protein = batch["protein"].to(device)
            cross_edges = batch["cross_edges"].to(device)
            cross_attr = batch["cross_attr"].to(device) if batch["cross_attr"] is not None else torch.empty((0, args.rbf_k), device=device)
            y = batch["y"].to(device).view(-1)
            mu, logvar, aux = model(ligand, protein, cross_edges, cross_attr)
            preds.append(mu.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    ci = concordance_index(targets, preds)
    r = pearson_r(targets, preds)
    _rmse = rmse(targets, preds)
    r2 = r2_score(targets, preds)
    return {"ci": ci, "pearson": r, "rmse": _rmse, "r2": r2, "y_true": targets, "y_pred": preds}


def save_checkpoint(state, filename):
    """Save training checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, device):
    """Load training checkpoint with PyTorch 2.6+ compatibility"""
    return safe_load_checkpoint(filename, device)


# -----------------------------
# Preset Configurations
# -----------------------------
def apply_preset(args, preset_name):
    """Apply preset configurations"""
    if preset_name == "phase2":
        # Phase 2 training with larger model and advanced scheduling
        args.hidden_dim = 320
        args.n_layers = 7
        args.heads = 8
        args.batch_size = 8
        args.lr = 2e-4
        args.min_lr = 2e-5
        args.ranking_margin = 0.2
        args.lambda_rank = 1.2
        args.scheduler_type = "cosine_warm"
        args.T_0 = 10
        args.T_mult = 2
        args.epochs = 120
        args.checkpoint_interval = 1
        args.amp = True
        print("Applied Phase 2 preset configuration")
    
    elif preset_name == "phase1":
        # Phase 1 training with smaller model
        args.hidden_dim = 128
        args.n_layers = 4
        args.heads = 4
        args.batch_size = 16
        args.lr = 1e-4
        args.min_lr = 1e-6
        args.ranking_margin = 0.5
        args.lambda_rank = 0.5
        args.scheduler_type = "cosine"
        args.epochs = 60
        args.checkpoint_interval = 10
        args.amp = True


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    
    # Data & basic training
    parser.add_argument("--processed_folder", type=str, required=True, help="folder with processed .pt files")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--node_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--rbf_k", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    
    # Loss weights
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=0.5)
    parser.add_argument("--ranking_margin", type=float, default=0.5, help="Margin for ranking loss")
    parser.add_argument("--use_heteroscedastic", action="store_true")
    
    # Scheduling
    parser.add_argument("--scheduler_type", type=str, default="cosine", 
                        choices=["cosine", "cosine_warm", "step", "constant"])
    parser.add_argument("--min_lr", type=float, default=2e-5, help="Minimum learning rate")
    parser.add_argument("--T_0", type=int, default=10, help="Number of epochs for first restart (cosine_warm)")
    parser.add_argument("--T_mult", type=int, default=2, help="Multiplication factor for T_0 after restart")
    parser.add_argument("--step_size", type=int, default=30, help="Step size for step scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for step scheduler")
    parser.add_argument("--scheduler_step_interval", type=int, default=100, help="Steps between LR updates (for cosine_warm)")
    
    # Training options
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="save checkpoint every N epochs")
    parser.add_argument("--preset", type=str, help="Apply preset configuration", choices=["phase1", "phase2"])
    
    args = parser.parse_args()

    # Apply preset if specified
    if args.preset:
        apply_preset(args, args.preset)

    # deterministic seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Save training arguments
    with open(os.path.join(args.save_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        args.processed_folder, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    # model
    model = DualGraphTransformer(
        node_dim=args.node_dim, 
        hidden_dim=args.hidden_dim, 
        n_layers=args.n_layers,
        rbf_k=args.rbf_k, 
        heads=args.heads,
        lig_edge_attr_dim=6,
        prot_edge_attr_dim=16
    ).to(device)

    # optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args)
    
    # Fix GradScaler for CPU
    if args.amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
        if args.amp:
            print("Warning: AMP requested but CUDA not available. Disabling AMP.")
    
    mse_loss_fn = nn.MSELoss()
    rank_loss_fn = PairwiseRankingLoss(margin=args.ranking_margin)

    # Training state
    start_epoch = 1
    best_ci = -1.0
    best_epoch = 0
    train_history = {
        'train_loss': [],
        'train_mse': [],
        'train_rank': [],
        'val_ci': [],
        'val_pearson': [],
        'val_rmse': [],
        'val_r2': [],
        'learning_rates': [],
        'epochs': []
    }

    # Resume from checkpoint if requested
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            if scaler and 'scaler_state' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_ci = checkpoint.get('best_ci', -1.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            train_history = checkpoint.get('train_history', train_history)
            print(f"Resumed training from epoch {start_epoch}")
    
    # Load best model if exists (for evaluation) - but don't fail if it doesn't exist
    best_checkpoint = None
    if os.path.exists(best_model_path):
        best_checkpoint = load_checkpoint(best_model_path, device)
        if best_checkpoint:
            best_ci = best_checkpoint.get('best_ci', best_ci)
            best_epoch = best_checkpoint.get('best_epoch', best_epoch)

    print(f"Starting training from epoch {start_epoch}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training with scheduler: {args.scheduler_type}")
    print(f"Using device: {device}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_mse, train_rank = train_epoch(
            model, device, train_loader, optimizer, scheduler, scaler, 
            mse_loss_fn, rank_loss_fn, epoch, args
        )
        t1 = time.time()
        
        val_metrics = evaluate(model, device, val_loader, args)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update training history
        train_history['train_loss'].append(train_loss)
        train_history['train_mse'].append(train_mse)
        train_history['train_rank'].append(train_rank)
        train_history['val_ci'].append(val_metrics['ci'])
        train_history['val_pearson'].append(val_metrics['pearson'])
        train_history['val_rmse'].append(val_metrics['rmse'])
        train_history['val_r2'].append(val_metrics['r2'])
        train_history['learning_rates'].append(current_lr)
        train_history['epochs'].append(epoch)
        
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"TrainLoss: {train_loss:.4f} (MSE: {train_mse:.4f}, Rank: {train_rank:.4f}) | "
              f"Val CI: {val_metrics['ci']:.4f} | Pearson: {val_metrics['pearson']:.4f} | "
              f"RMSE: {val_metrics['rmse']:.4f} | LR: {current_lr:.2e} | Time: {t1-t0:.1f}s")
        
        # Save best model
        if val_metrics["ci"] > best_ci:
            best_ci = val_metrics["ci"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            torch.save({
                "model_state": best_state,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "best_ci": best_ci,
                "best_epoch": best_epoch,
                "train_history": train_history,
                "args": vars(args)
            }, best_model_path)
            print(f"New best model saved with CI: {best_ci:.4f}")

        # Save checkpoint regularly
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_ci': best_ci,
                'best_epoch': best_epoch,
                'train_history': train_history,
                'args': vars(args)
            }
            if scaler:
                checkpoint['scaler_state'] = scaler.state_dict()
            save_checkpoint(checkpoint, checkpoint_path)

    # Final evaluation on test set
    print("Loading best model for test evaluation...")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    
    test_metrics = evaluate(model, device, test_loader, args)
    print("=== Final Test Metrics ===")
    print(f"CI: {test_metrics['ci']:.4f} | Pearson: {test_metrics['pearson']:.4f} | "
          f"RMSE: {test_metrics['rmse']:.4f} | R²: {test_metrics['r2']:.4f}")
    
    # Save final predictions and training history
    np.savez(os.path.join(args.save_dir, "test_preds.npz"), 
             y_true=test_metrics["y_true"], 
             y_pred=test_metrics["y_pred"])
    
    # Save training history
    np.savez(os.path.join(args.save_dir, "training_history.npz"), **train_history)
    
    print(f"Training completed. Best model CI: {best_ci:.4f} at epoch {best_epoch}")
    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()