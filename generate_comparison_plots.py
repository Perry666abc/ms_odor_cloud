import os
import re
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#EAEAEA'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

os.makedirs('assets', exist_ok=True)

# -------------------------------------------------------------
# 1. Parse Baseline (No POM) from 专题实践 TensorBoard
# -------------------------------------------------------------
from tensorboard.backend.event_processing import event_accumulator

tb_path = '/Users/mac/Desktop/专题实践/runs/odor_experiment/events.out.tfevents.1781235459.p-d1d999937cf7-ackcs-00gjh4zr.2283.0'
ea = event_accumulator.EventAccumulator(tb_path)
ea.Reload()

base_tr_loss = [e.value for e in ea.Scalars('Loss/Train')]
base_val_loss = [e.value for e in ea.Scalars('Loss/Validation')]
base_micro_f1 = [e.value for e in ea.Scalars('Metrics/Micro_F1')]
base_macro_f1 = [e.value for e in ea.Scalars('Metrics/Macro_F1')]
base_auprc = [e.value for e in ea.Scalars('Metrics/AUPRC')]
base_epochs = list(range(1, len(base_tr_loss) + 1))

# -------------------------------------------------------------
# 2. Parse 5090 Log (With Open-POM Alignment)
# -------------------------------------------------------------
with open('train_5090.log') as f:
    log_text = f.read()

# Stage 1: Alignment Pretraining
s1_pattern = r'Epoch \[(\d+)/30\].*?Train Loss: ([\d\.]+), Top-1 Acc: ([\d\.]+)%.*?Val Loss: ([\d\.]+), Top-1 Acc: ([\d\.]+)%'
s1_matches = re.findall(s1_pattern, log_text)
s1_epochs = [int(m[0]) for m in s1_matches]
s1_tr_loss = [float(m[1]) for m in s1_matches]
s1_tr_acc = [float(m[2]) for m in s1_matches]
s1_val_loss = [float(m[3]) for m in s1_matches]
s1_val_acc = [float(m[4]) for m in s1_matches]

# Stage 2: Fine-tuning
s2_pattern = r'Epoch \[(\d+)/100\] TrLoss: ([\d\.]+) \| ValLoss: ([\d\.]+) \| Val Micro-AUC: ([\d\.]+)% \| Val W-AUC: ([\d\.]+)% \| Val Macro-AUPRC: ([\d\.]+)'
s2_matches = re.findall(s2_pattern, log_text)
s2_epochs = [int(m[0]) for m in s2_matches]
s2_tr_loss = [float(m[1]) for m in s2_matches]
s2_val_loss = [float(m[2]) for m in s2_matches]
s2_val_micro_auc = [float(m[3]) for m in s2_matches]
s2_val_w_auc = [float(m[4]) for m in s2_matches]
s2_val_macro_auprc = [float(m[5]) for m in s2_matches]

best_s2_idx = 16 # Epoch 17 (0-indexed: 16)
best_epoch = s2_epochs[best_s2_idx]
best_micro_auc = s2_val_micro_auc[best_s2_idx]
best_w_auc = s2_val_w_auc[best_s2_idx]
best_macro_auprc = s2_val_macro_auprc[best_s2_idx]

# =============================================================
# Plot 1: Stage 1 Alignment Pre-training (MS <-> Open-POM)
# =============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# 1.1 InfoNCE Loss
ax1.plot(s1_epochs, s1_tr_loss, color='#1E88E5', linewidth=2.4, label='Train InfoNCE Loss', marker='o', markersize=4)
ax1.plot(s1_epochs, s1_val_loss, color='#E53935', linewidth=2.4, label='Val InfoNCE Loss', marker='s', markersize=4)
min_val_loss_idx = np.argmin(s1_val_loss)
ax1.scatter(s1_epochs[min_val_loss_idx], s1_val_loss[min_val_loss_idx], color='#D81B60', s=100, zorder=5)
ax1.annotate(f'Best Val Loss: {s1_val_loss[min_val_loss_idx]:.4f}\n(Epoch {s1_epochs[min_val_loss_idx]})',
             xy=(s1_epochs[min_val_loss_idx], s1_val_loss[min_val_loss_idx]),
             xytext=(s1_epochs[min_val_loss_idx]+2, s1_val_loss[min_val_loss_idx]+0.15),
             arrowprops=dict(facecolor='#D81B60', arrowstyle='->', lw=1.5),
             fontsize=10.5, fontweight='bold', color='#B71C1C',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#EF9A9A', alpha=0.9))

ax1.set_title('(A) Stage 1: Cross-Modal Contrastive Loss (InfoNCE)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Epoch', fontsize=11, fontweight='semibold')
ax1.set_ylabel('InfoNCE Loss (tau=0.03)', fontsize=11, fontweight='semibold')
ax1.grid(True)
ax1.legend(loc='upper right', frameon=True, fontsize=10.5)

# 1.2 Top-1 Retrieval Accuracy
ax2.plot(s1_epochs, s1_tr_acc, color='#43A047', linewidth=2.4, label='Train Top-1 Retrieval Acc (%)', marker='^', markersize=4)
ax2.plot(s1_epochs, s1_val_acc, color='#8E24AA', linewidth=2.4, label='Val Top-1 Retrieval Acc (%)', marker='d', markersize=4)
max_val_acc_idx = np.argmax(s1_val_acc)
ax2.scatter(s1_epochs[max_val_acc_idx], s1_val_acc[max_val_acc_idx], color='#6A1B9A', s=100, zorder=5)
ax2.annotate(f'Peak Val Acc: {s1_val_acc[max_val_acc_idx]:.1f}%\n(Epoch {s1_epochs[max_val_acc_idx]})',
             xy=(s1_epochs[max_val_acc_idx], s1_val_acc[max_val_acc_idx]),
             xytext=(s1_epochs[max_val_acc_idx]-10, s1_val_acc[max_val_acc_idx]+8),
             arrowprops=dict(facecolor='#6A1B9A', arrowstyle='->', lw=1.5),
             fontsize=10.5, fontweight='bold', color='#4A148C',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#CE93D8', alpha=0.9))

ax2.set_title('(B) Stage 1: MS-to-Molecule Top-1 Alignment Accuracy', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Epoch', fontsize=11, fontweight='semibold')
ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=11, fontweight='semibold')
ax2.grid(True)
ax2.legend(loc='center right', frameon=True, fontsize=10.5)

plt.suptitle('SCENT Stage 1: Tandem MS <-> Open-POM Cross-Modal Alignment Dynamics', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig('assets/stage1_alignment.png')
plt.close(fig)
print('Saved assets/stage1_alignment.png')

# =============================================================
# Plot 2: Training & Validation Loss Comparison
# =============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# 2.1 Baseline Loss (Without POM)
ax1.plot(base_epochs, base_tr_loss, color='#1976D2', linewidth=2.0, label='Train Loss')
ax1.plot(base_epochs, base_val_loss, color='#F57C00', linewidth=2.0, label='Validation Loss')
ax1.set_title('(A) Baseline (No POM Alignment, 200 Epochs)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Training Epoch', fontsize=11, fontweight='semibold')
ax1.set_ylabel('Loss (Focal / MSE)', fontsize=11, fontweight='semibold')
ax1.grid(True)
ax1.legend(loc='upper right', frameon=True, fontsize=10.5)
ax1.annotate('Slow, continuous descent over 200 epochs\nFinal Train: 0.0080 | Val: 0.0076',
             xy=(195, base_val_loss[-1]), xytext=(70, 0.035),
             arrowprops=dict(facecolor='#333333', arrowstyle='->', lw=1.2),
             fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1', edgecolor='#FFE082', alpha=0.9))

# 2.2 Ours with POM Alignment (Stage 2 Fine-tuning)
ax2.plot(s2_epochs, s2_tr_loss, color='#0288D1', linewidth=2.2, label='Train Loss (ASL + POM Reg)', marker='o', markersize=3)
ax2.plot(s2_epochs, s2_val_loss, color='#E64A19', linewidth=2.2, label='Validation Loss (Val ASL)', marker='s', markersize=3)
ax2.axvline(x=best_epoch, color='#7B1FA2', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Best Checkpoint (Epoch {best_epoch})')
ax2.set_title('(B) Ours with Open-POM Alignment (Early Stopped at Ep 42)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Fine-tuning Epoch', fontsize=11, fontweight='semibold')
ax2.set_ylabel('Downstream Loss (ASL, gamma_+=0, gamma_-=4)', fontsize=11, fontweight='semibold')
ax2.grid(True)
ax2.legend(loc='upper right', frameon=True, fontsize=10.5)
ax2.annotate(f'Optimal Generalization Point\nEpoch {best_epoch}: Val Loss = {s2_val_loss[best_s2_idx]:.4f}\nRapid convergence in 9.8 mins (RTX 5090)',
             xy=(best_epoch, s2_val_loss[best_s2_idx]), xytext=(best_epoch+4, 0.045),
             arrowprops=dict(facecolor='#7B1FA2', arrowstyle='->', lw=1.5),
             fontsize=9.5, fontweight='bold', color='#4A148C',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5', edgecolor='#CE93D8', alpha=0.9))

plt.suptitle('Training & Validation Loss Convergence: Baseline vs. POM-Aligned Model', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig('assets/loss_comparison.png')
plt.close(fig)
print('Saved assets/loss_comparison.png')

# =============================================================
# Plot 3: Metrics Evolution & Convergence Comparison
# =============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# 3.1 Baseline Metrics
ax1.plot(base_epochs, base_auprc, color='#2E7D32', linewidth=2.2, label='Val AUPRC (Final: 0.7817)')
ax1.plot(base_epochs, base_macro_f1, color='#E65100', linewidth=2.2, label='Val Macro-F1 (Final: 0.7120)')
ax1.plot(base_epochs, base_micro_f1, color='#1565C0', linewidth=2.2, label='Val Micro-F1 (Final: 0.6302)')
ax1.set_title('(A) Baseline Validation Metrics (200 Epochs)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Epoch', fontsize=11, fontweight='semibold')
ax1.set_ylabel('Metric Score', fontsize=11, fontweight='semibold')
ax1.set_ylim([0.0, 0.85])
ax1.grid(True)
ax1.legend(loc='lower right', frameon=True, fontsize=10.5)

# 3.2 Ours with POM Alignment (Stage 2 Validation Metrics)
ax2.plot(s2_epochs, [v/100.0 for v in s2_val_micro_auc], color='#6A1B9A', linewidth=2.5,
         label=f'Val Micro-AUROC (Peak: {best_micro_auc:.2f}%)', marker='o', markersize=3.5)
ax2.plot(s2_epochs, [v/100.0 for v in s2_val_w_auc], color='#00838F', linewidth=2.2,
         label=f'Val Weighted-AUC (Peak: {best_w_auc:.2f}%)', marker='s', markersize=3.5)
ax2.plot(s2_epochs, s2_val_macro_auprc, color='#C2185B', linewidth=2.2,
         label=f'Val Macro-AUPRC (Peak: {best_macro_auprc:.4f})', marker='^', markersize=3.5)

ax2.axvline(x=best_epoch, color='#7B1FA2', linestyle='--', linewidth=1.5, alpha=0.8)
ax2.scatter(best_epoch, best_micro_auc/100.0, color='#6A1B9A', s=90, zorder=5)
ax2.annotate(f'Peak Micro-AUROC: {best_micro_auc:.2f}%\nWeighted-AUC: {best_w_auc:.2f}%\nEpoch: {best_epoch}',
             xy=(best_epoch, best_micro_auc/100.0), xytext=(best_epoch+4, 0.60),
             arrowprops=dict(facecolor='#6A1B9A', arrowstyle='->', lw=1.5),
             fontsize=9.5, fontweight='bold', color='#4A148C',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5', edgecolor='#CE93D8', alpha=0.9))

ax2.set_title('(B) Ours with Open-POM Alignment (Strict Molecule Split)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Fine-tuning Epoch', fontsize=11, fontweight='semibold')
ax2.set_ylabel('Metric Score / Ratio', fontsize=11, fontweight='semibold')
ax2.set_ylim([0.0, 0.95])
ax2.grid(True)
ax2.legend(loc='lower right', frameon=True, fontsize=10.5)

plt.suptitle('Validation Evaluation Metrics Progression: Baseline vs. POM-Aligned Model', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig('assets/metrics_comparison.png')
plt.close(fig)
print('Saved assets/metrics_comparison.png')

# =============================================================
# Plot 4: Benchmark Summary & Test Set Metrics Dashboard
# =============================================================
fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)

metrics_labels = [
    'Test Micro-AUROC\n(SCENT Primary)',
    'Test Weighted-AUC\n(Class Weighted)',
    'Test Micro-AUPRC\n(Precision-Recall)',
    'Test Macro-AUPRC\n(Rare Label AUC)',
    'Test Micro-F1\n(Multi-label F1)'
]
test_values = [82.60, 75.00, 21.55, 19.18, 24.94]
bar_colors = ['#5E35B1', '#1E88E5', '#00897B', '#FB8C00', '#E53935']

bars = ax.bar(metrics_labels, test_values, color=bar_colors, width=0.52, edgecolor='#333333', linewidth=1.2, zorder=3)

for bar, val in zip(bars, test_values):
    height = bar.get_height()
    if val >= 50:
        val_str = f'{val:.2f}%'
    else:
        val_str = f'{val/100.0:.4f}\n({val:.1f}%)'
    ax.annotate(val_str,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11.5, fontweight='bold', color='#212121')

ax.set_ylim(0, 110)
ax.set_ylabel('Metric Percentage / Score (%)', fontsize=12, fontweight='semibold')
ax.set_title('Independent Blind Test Evaluation (Strict 0% Molecular Leakage Split, 4,863 Spectra)',
             fontsize=13.5, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# Add information text box at top right
info_text = (
    "Evaluation Protocol & Setup:\n"
    "• Test Set: 4,863 spectra across 425 unseen molecules\n"
    "• Data Integrity: Strict Molecule-Level Split (0% overlap)\n"
    "• Hardware: NVIDIA GeForce RTX 5090 (32GB) | Time: 9.79 mins\n"
    "• MPNN Encoder: 5-fold Open-POM (ref: czx0v0/pom_v2)"
)
ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CFD8DC', alpha=0.95))

plt.tight_layout()
fig.savefig('assets/benchmark_summary.png')
plt.close(fig)
print('Updated assets/benchmark_summary.png')

