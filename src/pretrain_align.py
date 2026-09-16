"""
Stage 1: SCENT Multimodal Contrastive Alignment (MS <-> Open-POM)
Aligns Mass Spectrometry Transformer with Frozen Pretrained Open-POM GNN representations
using paired MS-SMILES data from GS-LF/NIST datasets.

Reference:
- SCENT (arXiv:2605.27009): "Aligning Mass Spectra with Molecular Structure for Olfactory Perception"
- Lee et al., Science 2023: "A principal odor map unifies diverse tasks in olfactory perception"
- Formula (Eq. 1 in SCENT):
    L_align = - 1/(2B) * sum_i [ log( exp(s_i^T m_i / tau) / sum_j exp(s_i^T m_j / tau) )
                               + log( exp(s_i^T m_i / tau) / sum_j exp(s_j^T m_i / tau) ) ]
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src and root are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset import build_datasets, ODOR_CLASSES_138
from model import MSOdorTransformer
from loss import MultimodalInfoNCELoss
#from pom_encoder import get_pom_encoder

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def train_align_epoch(ms_encoder, dataloader, infonce_criterion, optimizer, scaler, device, use_amp):
    ms_encoder.train()
    total_loss = 0.0
    total_acc = 0.0
    valid_batches = 0

    pbar = tqdm(dataloader, desc="Align Training", leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        # Filter samples with valid Open-POM embeddings (CPU indexing)
        valid_mask = (batch['has_pom'].squeeze(-1) > 0.5)
        if valid_mask.sum() < 2:
            continue

        mzs = batch['mzs'][valid_mask].to(device, non_blocking=True)
        intensities = batch['intensities'][valid_mask].to(device, non_blocking=True)
        neutral_losses = batch['neutral_losses'][valid_mask].to(device, non_blocking=True)
        precursor_mz = batch['precursor_mz'][valid_mask].to(device, non_blocking=True)
        has_precursor = batch['has_precursor'][valid_mask].to(device, non_blocking=True)
        ce = batch['ce'][valid_mask].to(device, non_blocking=True)
        has_ce = batch['has_ce'][valid_mask].to(device, non_blocking=True)
        instrument_mode = batch['instrument_mode'][valid_mask].to(device, non_blocking=True)
        padding_mask = batch['padding_mask'][valid_mask].to(device, non_blocking=True)
        pom_embs = batch['pom_embedding'][valid_mask].to(device, non_blocking=True) # (B_valid, 256)

        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                m_i = ms_encoder.get_aligned_embedding(
                    mzs, intensities, neutral_losses,
                    precursor_mz, has_precursor,
                    ce, has_ce, instrument_mode,
                    padding_mask=padding_mask,
                    normalize=False
                )
                loss_align, mean_acc, tau_val = infonce_criterion(m_i, pom_embs.detach())
                
            scaler.scale(loss_align).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ms_encoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            m_i = ms_encoder.get_aligned_embedding(
                mzs, intensities, neutral_losses,
                precursor_mz, has_precursor,
                ce, has_ce, instrument_mode,
                padding_mask=padding_mask,
                normalize=False
            )
            loss_align, mean_acc, tau_val = infonce_criterion(m_i, pom_embs.detach())
            loss_align.backward()
            torch.nn.utils.clip_grad_norm_(ms_encoder.parameters(), max_norm=1.0)
            optimizer.step()

        acc_float = mean_acc.item() if isinstance(mean_acc, torch.Tensor) else float(mean_acc)
        loss_float = loss_align.item()
        
        total_loss += loss_float
        total_acc += acc_float
        valid_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss_float:.4f}",
            'acc': f"{acc_float*100:.1f}%",
            'tau': f"{tau_val:.3f}"
        })

        if device.type == 'mps' and batch_idx % 100 == 0:
            torch.mps.empty_cache()

    avg_loss = total_loss / max(1, valid_batches)
    avg_acc = total_acc / max(1, valid_batches)
    return avg_loss, avg_acc

def evaluate_alignment(ms_encoder, dataloader, infonce_criterion, device):
    ms_encoder.eval()
    total_loss = 0.0
    total_acc = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            valid_mask = (batch['has_pom'].squeeze(-1) > 0.5)
            if valid_mask.sum() < 2:
                continue

            mzs = batch['mzs'][valid_mask].to(device)
            intensities = batch['intensities'][valid_mask].to(device)
            neutral_losses = batch['neutral_losses'][valid_mask].to(device)
            precursor_mz = batch['precursor_mz'][valid_mask].to(device)
            has_precursor = batch['has_precursor'][valid_mask].to(device)
            ce = batch['ce'][valid_mask].to(device)
            has_ce = batch['has_ce'][valid_mask].to(device)
            instrument_mode = batch['instrument_mode'][valid_mask].to(device)
            padding_mask = batch['padding_mask'][valid_mask].to(device)
            pom_embs = batch['pom_embedding'][valid_mask].to(device)

            m_i = ms_encoder.get_aligned_embedding(
                mzs, intensities, neutral_losses,
                precursor_mz, has_precursor,
                ce, has_ce, instrument_mode,
                padding_mask=padding_mask,
                normalize=False
            )
            loss_align, mean_acc, _ = infonce_criterion(m_i, pom_embs.detach())

            acc_float = mean_acc.item() if isinstance(mean_acc, torch.Tensor) else float(mean_acc)
            total_loss += loss_align.item()
            total_acc += acc_float
            valid_batches += 1

    avg_loss = total_loss / max(1, valid_batches)
    avg_acc = total_acc / max(1, valid_batches)
    return avg_loss, avg_acc

def main():
    parser = argparse.ArgumentParser(description="SCENT Stage 1: Modality Alignment Pretraining")
    parser.add_argument('--data_dir', type=str, default=os.path.join(ROOT_DIR, 'GS_LF_MS_DATA'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4, help="Encoder learning rate (SCENT Table C1: 1e-4)")
    parser.add_argument('--tau', type=float, default=0.03, help="CLIP temperature (SCENT Table C1: 0.03)")
    parser.add_argument('--learnable_tau', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default=os.path.join(ROOT_DIR, 'checkpoints'))
    parser.add_argument('--pom_cache', type=str, default=os.path.join(ROOT_DIR, 'pom_cache.pt'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)
    args = parser.parse_args()

    device = get_device()
    print("====================================================================")
    print("   SCENT Stage 1: Modality Alignment Pre-training (MS <-> Open-POM) ")
    print(f"   Device: {device} | Batch Size: {args.batch_size} | Base LR: {args.lr}")
    print(f"   Temperature tau: {args.tau} (Learnable: {args.learnable_tau})")
    print("====================================================================")

    # 1. Load Data
    csv_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found in {args.data_dir}")

    print(f"Loading {len(csv_paths)} CSV datasets with strict 0% molecular overlap split...")
    train_dataset, val_dataset, test_dataset, _, _, pom_cache = build_datasets(
        csv_paths,
        split_by='molecule',
        seed=args.seed,
        build_pom_cache=True,
        pom_cache_path=args.pom_cache
    )

    num_workers = 0 if os.name == 'nt' or sys.platform == 'darwin' else 4
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )

    # 2. Initialize Models
    ms_encoder = MSOdorTransformer(d_model=256, latent_dim=256, num_layers=4, nhead=8).to(device)
    infonce_criterion = MultimodalInfoNCELoss(init_tau=args.tau, learnable_tau=args.learnable_tau).to(device)

    optimizer = torch.optim.AdamW(
        ms_encoder.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp and device.type == 'cuda'))

    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, 'scent_aligned_ms_encoder.pth')

    best_val_loss = float('inf')
    best_val_acc = 0.0

    print("\nStarting Stage 1 Alignment Training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_align_epoch(
            ms_encoder, train_loader, infonce_criterion, optimizer, scaler, device, args.use_amp
        )
        val_loss, val_acc = evaluate_alignment(
            ms_encoder, val_loader, infonce_criterion, device
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Top-1 Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f}, Top-1 Acc: {val_acc*100:.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': ms_encoder.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args)
            }, best_ckpt_path)
            print(f"  --> Saved new best aligned checkpoint to {best_ckpt_path} (Val Loss: {val_loss:.4f}, Top-1: {val_acc*100:.1f}%)")

    print("\n====================================================================")
    print(f" Stage 1 Pretraining Complete! Best Val Loss: {best_val_loss:.4f}, Top-1: {best_val_acc*100:.1f}%")
    print(f" Aligned weights ready at: {best_ckpt_path}")
    print(" Next: Run downstream training with train.py --pretrained_align_ckpt checkpoints/scent_aligned_ms_encoder.pth")
    print("====================================================================")

if __name__ == '__main__':
    main()
