import os
import sys
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# Local module imports
from dataset import build_datasets, ODOR_CLASSES_138
from model import MSOdorTransformer, OdorAutoencoder
from loss import AsymmetricLoss, MultiLabelFocalLoss, MultimodalInfoNCELoss

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def train_one_epoch(transformer, autoencoder, dataloader, criterion, optimizer, scaler,
                    lambda_pred, lambda_recon, lambda_latent, lambda_align,
                    infonce_criterion, use_pom_align, use_direct_head, device, use_amp):
    transformer.train()
    if autoencoder is not None:
        autoencoder.train()
    
    total_loss = 0.0
    total_l_pred = 0.0
    total_l_align = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        mzs = batch['mzs'].to(device, non_blocking=True)
        intensities = batch['intensities'].to(device, non_blocking=True)
        neutral_losses = batch['neutral_losses'].to(device, non_blocking=True)
        precursor_mz = batch['precursor_mz'].to(device, non_blocking=True)
        has_precursor = batch['has_precursor'].to(device, non_blocking=True)
        ce = batch['ce'].to(device, non_blocking=True)
        has_ce = batch['has_ce'].to(device, non_blocking=True)
        instrument_mode = batch['instrument_mode'].to(device, non_blocking=True)
        padding_mask = batch['padding_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        amp_enabled = use_amp and device.type == 'cuda'
        with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
            # 1. Forward MS Transformer
            res = transformer(
                mzs, intensities, neutral_losses,
                precursor_mz, has_precursor,
                ce, has_ce, instrument_mode,
                padding_mask=padding_mask,
                return_all=True
            )
            
            if use_direct_head:
                logits_pred = res['logits']
            else:
                logits_pred = autoencoder.decoder(res['pom_pred'])
                
            loss_pred = criterion(logits_pred, labels)
            loss = lambda_pred * loss_pred
            
            # Optional Autoencoder auxiliary loss
            if autoencoder is not None and not use_direct_head:
                logits_recon, latent_true = autoencoder(labels)
                loss_recon = criterion(logits_recon, labels)
                loss_latent = F.mse_loss(res['pom_pred'], latent_true.detach())
                loss = loss + lambda_recon * loss_recon + lambda_latent * loss_latent

            # 2. SCENT Joint Open-POM Alignment Regularization
            loss_align = torch.tensor(0.0, device=device)
            if use_pom_align and infonce_criterion is not None:
                has_pom = batch['has_pom'].squeeze(-1) > 0.5
                if has_pom.sum() >= 2:
                    m_align = res['align_pred'][has_pom.to(device)]
                    pom_targets = batch['pom_embedding'][has_pom].to(device).detach()
                    l_a, _, _ = infonce_criterion(m_align, pom_targets)
                    loss_align = l_a
                    loss = loss + lambda_align * loss_align

        if use_amp and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            if autoencoder is not None:
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            if autoencoder is not None:
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            
        total_loss += loss.item()
        total_l_pred += loss_pred.item()
        total_l_align += loss_align.item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'pred': f"{loss_pred.item():.4f}"})
        if device.type == 'mps' and batch_idx % 100 == 0:
            torch.mps.empty_cache()
        
    n_batches = len(dataloader)
    return (total_loss / n_batches, total_l_pred / n_batches, total_l_align / n_batches)

def evaluate_predictions(transformer, autoencoder, dataloader, criterion, use_direct_head, device, use_amp):
    transformer.eval()
    if autoencoder is not None:
        autoencoder.eval()
    
    total_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            mzs = batch['mzs'].to(device, non_blocking=True)
            intensities = batch['intensities'].to(device, non_blocking=True)
            neutral_losses = batch['neutral_losses'].to(device, non_blocking=True)
            precursor_mz = batch['precursor_mz'].to(device, non_blocking=True)
            has_precursor = batch['has_precursor'].to(device, non_blocking=True)
            ce = batch['ce'].to(device, non_blocking=True)
            has_ce = batch['has_ce'].to(device, non_blocking=True)
            instrument_mode = batch['instrument_mode'].to(device, non_blocking=True)
            padding_mask = batch['padding_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            amp_enabled = use_amp and device.type == 'cuda'
            with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
                res = transformer(
                    mzs, intensities, neutral_losses,
                    precursor_mz, has_precursor,
                    ce, has_ce, instrument_mode,
                    padding_mask=padding_mask,
                    return_all=True
                )
                if use_direct_head:
                    logits_pred = res['logits']
                else:
                    logits_pred = autoencoder.decoder(res['pom_pred'])
                    
                loss = criterion(logits_pred, labels)
                
            total_loss += loss.item()
            probs = torch.sigmoid(logits_pred)
            
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    mean_loss = total_loss / len(dataloader)
    
    return mean_loss, all_targets, all_probs

def compute_metrics(all_targets, all_probs, thresholds=None):
    """
    Computes comprehensive evaluation metrics matching SCENT (arXiv:2605.27009 Table 1):
    - Micro-AUROC & Weighted-AUROC & Macro-AUROC
    - Micro-AUPRC & Macro-AUPRC
    - Micro-F1 & Macro-F1
    """
    num_classes = all_targets.shape[1]
    
    # 1. ROC-AUC Metrics (SCENT Table 1 primary metrics)
    try:
        micro_auc = roc_auc_score(all_targets, all_probs, average='micro')
    except Exception:
        micro_auc = 0.0
        
    try:
        weighted_auc = roc_auc_score(all_targets, all_probs, average='weighted')
    except Exception:
        weighted_auc = 0.0

    try:
        macro_auc = roc_auc_score(all_targets, all_probs, average='macro')
    except Exception:
        macro_auc = 0.0

    # 2. Precision-Recall Metrics
    try:
        micro_auprc = average_precision_score(all_targets, all_probs, average='micro')
    except Exception:
        micro_auprc = 0.0
        
    class_aps = []
    for c in range(num_classes):
        if all_targets[:, c].sum() > 0:
            ap = average_precision_score(all_targets[:, c], all_probs[:, c])
            class_aps.append(ap)
        else:
            class_aps.append(0.0)
    macro_auprc = float(np.mean(class_aps))
    
    # 3. Threshold determination & F1 scores
    if thresholds is None:
        best_thresholds = np.zeros(num_classes, dtype=np.float32)
        search_space = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        
        for c in range(num_classes):
            best_f1_c = 0.0
            best_th_c = 0.20
            y_true_c = all_targets[:, c]
            if y_true_c.sum() == 0:
                best_thresholds[c] = 0.50
                continue
            for th in search_space:
                y_pred_c = (all_probs[:, c] >= th).astype(float)
                f1_c = f1_score(y_true_c, y_pred_c, zero_division=0)
                if f1_c > best_f1_c:
                    best_f1_c = f1_c
                    best_th_c = th
            best_thresholds[c] = best_th_c
        thresholds = best_thresholds
    
    preds = (all_probs >= thresholds).astype(float)
    macro_f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_targets, preds, average='micro', zero_division=0)
    
    return {
        'micro_auc': micro_auc,
        'weighted_auc': weighted_auc,
        'macro_auc': macro_auc,
        'macro_auprc': macro_auprc,
        'micro_auprc': micro_auprc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'thresholds': thresholds,
        'class_aps': class_aps
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Tandem Mass Spec End-to-End Odor Prediction with SCENT POM Alignment")
    parser.add_argument('--data_dir', type=str, default='GS_LF_MS_DATA', help='Directory containing merged CSV files')
    parser.add_argument('--split_by', type=str, default='molecule', choices=['molecule', 'spectrum'],
                        help="'molecule': strict SMILES isolation (0% leakage); 'spectrum': stratified spectrum baseline")
    parser.add_argument('--loss_type', type=str, default='asl', choices=['asl', 'focal'],
                        help="Loss function type: 'asl' (Asymmetric Loss) or 'focal' (Focal Loss)")
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--d_model', type=int, default=256, help='Transformer embedding dimension')
    parser.add_argument('--latent_dim', type=int, default=256, help='POM perceptual manifold dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer encoder layers')
    parser.add_argument('--nhead', type=int, default=8, help='Transformer attention heads')
    parser.add_argument('--max_seq_len', type=int, default=200, help='Max peaks per spectrum')
    
    # SCENT Alignment & Transfer Parameters
    parser.add_argument('--pretrained_align_ckpt', type=str, default='',
                        help="Path to Stage 1 contrastive alignment checkpoint (e.g. checkpoints/scent_aligned_ms_encoder.pth)")
    parser.add_argument('--freeze_ms_encoder', action='store_true', default=False,
                        help="Freeze MS encoder backbone and train only downstream MLP classification head (linear probe)")
    parser.add_argument('--use_pom_align', action='store_true', default=False,
                        help="Enable continuous SCENT InfoNCE joint alignment with Open-POM during downstream training")
    parser.add_argument('--lambda_align', type=float, default=0.2, help="Weight of Open-POM InfoNCE alignment loss")
    parser.add_argument('--tau', type=float, default=0.03, help="CLIP temperature for InfoNCE loss")
    parser.add_argument('--use_direct_head', action='store_true', default=True,
                        help="Use direct MLP classification head from Transformer (default True, matching SCENT)")
    
    # Legacy auxiliary weights
    parser.add_argument('--lambda_pred', type=float, default=1.0, help='Prediction loss weight')
    parser.add_argument('--lambda_recon', type=float, default=0.5, help='Autoencoder reconstruction loss weight')
    parser.add_argument('--lambda_latent', type=float, default=0.2, help='Latent alignment loss weight')
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0, help='ASL negative focusing gamma')
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0, help='ASL positive focusing gamma')
    parser.add_argument('--asl_clip', type=float, default=0.05, help='ASL negative margin clip')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save best model checkpoint')
    parser.add_argument('--log_dir', type=str, default='runs/odor_scent_experiment', help='TensorBoard log directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = get_device()
    use_amp = (not args.no_amp) and (device.type == 'cuda')
    
    num_workers = args.num_workers
    if sys.platform != 'linux':
        num_workers = 0
        
    print(f"================================================================")
    print(f"  Tandem Mass Spec Odor Prediction Pipeline with SCENT Open-POM")
    print(f"  Device: {device} | AMP: {use_amp} | Batch Size: {args.batch_size}")
    print(f"  Split Mode: {args.split_by.upper()} | Loss Function: {args.loss_type.upper()}")
    if args.pretrained_align_ckpt:
        print(f"  Stage 1 Pretrained Checkpoint: {args.pretrained_align_ckpt}")
        print(f"  Backbone Frozen: {args.freeze_ms_encoder}")
    if args.use_pom_align:
        print(f"  Joint POM Alignment Regularization: Enabled (lambda_align={args.lambda_align})")
    print(f"================================================================")
    
    # 1. Dataset Preparation
    csv_files = [
        os.path.join(args.data_dir, 'GS_LF_MassSpecGym_merged.csv'),
        os.path.join(args.data_dir, 'GS_LF_MoNA_GC-MS_merged.csv'),
        os.path.join(args.data_dir, 'GS_LF_NIST20_hrmsms_merged.csv'),
        os.path.join(args.data_dir, 'GS_LF_NIST20_lrmsms_merged.csv'),
        os.path.join(args.data_dir, 'GS_LF_NIST20_mainlib_merged.csv')
    ]
    
    train_dataset, val_dataset, test_dataset, class_weights, prior_probs, pom_cache = build_datasets(
        csv_files, split_by=args.split_by, max_seq_len=args.max_seq_len, seed=args.seed,
        build_pom_cache=True, pom_cache_path='pom_cache.pt'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == 'cuda'), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == 'cuda'))
                             
    # 2. Model Initialization
    transformer = MSOdorTransformer(
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=0.1
    ).to(device)
    
    # Load Stage 1 Pretrained Alignment Weights if provided
    if args.pretrained_align_ckpt and os.path.exists(args.pretrained_align_ckpt):
        print(f"Loading Stage 1 SCENT alignment weights from: {args.pretrained_align_ckpt}")
        ckpt = torch.load(args.pretrained_align_ckpt, map_location=device, weights_only=False)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, unexpected = transformer.load_state_dict(sd, strict=False)
        print(f"Loaded MS Encoder weights! Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
    if args.freeze_ms_encoder:
        print("Freezing MS Transformer encoder backbone for linear/MLP classification probing...")
        for name, param in transformer.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                
    autoencoder = None
    if not args.use_direct_head:
        autoencoder = OdorAutoencoder(
            pom_dim=args.latent_dim,
            num_classes=len(ODOR_CLASSES_138),
            hidden_dim=args.d_model,
            prior_prob=prior_probs
        ).to(device)
    
    # 3. Loss & Optimizer
    if args.loss_type == 'asl':
        criterion = AsymmetricLoss(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip,
            pos_weight=class_weights
        ).to(device)
    else:
        criterion = MultiLabelFocalLoss(
            alpha=class_weights,
            gamma=2.0
        ).to(device)
        
    infonce_criterion = MultimodalInfoNCELoss(init_tau=args.tau, learnable_tau=False).to(device) if args.use_pom_align else None
    
    params = [p for p in transformer.parameters() if p.requires_grad]
    if autoencoder is not None:
        params += [p for p in autoencoder.parameters() if p.requires_grad]
        
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    warmup_epochs = 5
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 4. Training & Validation Loop
    best_val_score = 0.0
    best_val_metrics = None
    best_thresholds = None
    patience = 25
    epochs_no_improve = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss, l_pred, l_align = train_one_epoch(
            transformer, autoencoder, train_loader, criterion, optimizer, scaler,
            args.lambda_pred, args.lambda_recon, args.lambda_latent, args.lambda_align,
            infonce_criterion, args.use_pom_align, args.use_direct_head, device, use_amp
        )
        scheduler.step()
        
        # Validation Evaluation
        val_loss, val_targets, val_probs = evaluate_predictions(
            transformer, autoencoder, val_loader, criterion, args.use_direct_head, device, use_amp
        )
        val_metrics = compute_metrics(val_targets, val_probs, thresholds=None)
        
        # Primary selection metric: Micro-AUROC + Macro-AUPRC
        val_score = val_metrics['micro_auc']
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/Train_Total', train_loss, epoch)
        writer.add_scalar('Loss/Train_Pred', l_pred, epoch)
        writer.add_scalar('Loss/Train_Align', l_align, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/Val_Micro_AUC', val_metrics['micro_auc'], epoch)
        writer.add_scalar('Metrics/Val_Weighted_AUC', val_metrics['weighted_auc'], epoch)
        writer.add_scalar('Metrics/Val_Macro_AUPRC', val_metrics['macro_auprc'], epoch)
        writer.add_scalar('Metrics/Val_Micro_F1', val_metrics['micro_f1'], epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)
        
        print(f"Epoch [{epoch+1:03d}/{args.epochs:03d}] "
              f"TrLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
              f"Val Micro-AUC: {val_metrics['micro_auc']*100:.2f}% | "
              f"Val W-AUC: {val_metrics['weighted_auc']*100:.2f}% | "
              f"Val Macro-AUPRC: {val_metrics['macro_auprc']:.4f}")
              
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_metrics = val_metrics
            best_thresholds = val_metrics['thresholds']
            epochs_no_improve = 0
            
            torch.save({
                'epoch': epoch + 1,
                'transformer': transformer.state_dict(),
                'autoencoder': autoencoder.state_dict() if autoencoder is not None else None,
                'val_metrics': val_metrics,
                'thresholds': best_thresholds,
                'args': vars(args)
            }, args.save_path)
            print(f"  >>> Checkpoint saved! New best Val Micro-AUC: {val_score*100:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and epoch >= 30:
                print(f"\n[Early Stopping] No improvement for {patience} consecutive epochs. Stopping.")
                break
                
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    writer.close()
    
    # 5. Independent Blind Test Set Evaluation
    print(f"\n================================================================")
    print(f"  Independent Blind Test Evaluation (Zero-Leakage Final Check)")
    print(f"================================================================")
    if os.path.exists(args.save_path):
        checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
        transformer.load_state_dict(checkpoint['transformer'])
        if autoencoder is not None and checkpoint['autoencoder'] is not None:
            autoencoder.load_state_dict(checkpoint['autoencoder'])
        best_thresholds = checkpoint['thresholds']
        print(f"Loaded best checkpoint from Epoch {checkpoint['epoch']}")
        
    test_loss, test_targets, test_probs = evaluate_predictions(
        transformer, autoencoder, test_loader, criterion, args.use_direct_head, device, use_amp
    )
    test_metrics = compute_metrics(test_targets, test_probs, thresholds=best_thresholds)
    
    print(f"Test Set Overall Metrics (SCENT Table 1 Aligned):")
    print(f"  - Test Loss:         {test_loss:.4f}")
    print(f"  - Test Micro-AUROC:  {test_metrics['micro_auc']*100:.2f}%")
    print(f"  - Test Weighted-AUC: {test_metrics['weighted_auc']*100:.2f}%")
    print(f"  - Test Macro-AUROC:  {test_metrics['macro_auc']*100:.2f}%")
    print(f"  - Test Macro-AUPRC:  {test_metrics['macro_auprc']:.4f}")
    print(f"  - Test Micro-AUPRC:  {test_metrics['micro_auprc']:.4f}")
    print(f"  - Test Micro-F1:     {test_metrics['micro_f1']:.4f}")
    print(f"================================================================")

if __name__ == '__main__':
    main()
