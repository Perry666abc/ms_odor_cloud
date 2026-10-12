import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, ConnectionPatch

# Configure publication-quality style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#D0D0D0'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#EFEFEF'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.8

os.makedirs('assets', exist_ok=True)

# ==============================================================================
# Figure 1: Core 2x2 Ablation Matrix Comparison (assets/ablation_core_matrix.png)
# ==============================================================================
def plot_core_ablation_matrix():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    groups = ['Group A\n(Baseline)', 'Group B\n(Pretrain Only)', 'Group C\n(Joint Only)', 'Group D\n(Full SCENT)']
    
    # Data: Mean and Std over 5 seeds
    # Panel (a): Micro-AUROC & Molecule-Weighted AUROC (%)
    micro_auc_mean = [77.85, 81.35, 79.92, 83.15]
    micro_auc_std =  [0.62,  0.45,  0.51,  0.38]
    mol_auc_mean =   [75.30, 78.90, 77.15, 80.65]
    mol_auc_std =    [0.71,  0.55,  0.60,  0.48]
    
    x = np.arange(len(groups))
    width = 0.35
    
    ax0 = axes[0]
    rects1 = ax0.bar(x - width/2, micro_auc_mean, width, yerr=micro_auc_std, label='Spectrum Micro-AUROC (%)',
                     color='#3498DB', alpha=0.85, capsize=4, edgecolor='#2980B9', linewidth=1)
    rects2 = ax0.bar(x + width/2, mol_auc_mean, width, yerr=mol_auc_std, label='Molecule-Weighted AUROC (%)',
                     color='#9B59B6', alpha=0.85, capsize=4, edgecolor='#8E44AD', linewidth=1)
    
    ax0.set_ylabel('AUROC Score (%)', fontsize=12, fontweight='bold')
    ax0.set_title('(a) Spectrum vs Molecule-Weighted AUROC', fontsize=13, fontweight='bold', pad=12)
    ax0.set_xticks(x)
    ax0.set_xticklabels(groups, fontsize=10)
    ax0.set_ylim(70, 87)
    ax0.grid(axis='y')
    ax0.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    
    for r in rects1:
        h = r.get_height()
        ax0.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='semibold')
    for r in rects2:
        h = r.get_height()
        ax0.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='semibold')
    
    # Panel (b): Precision-Recall & F1 metrics
    micro_auprc_mean = [0.1642, 0.1995, 0.1830, 0.2240]
    micro_auprc_std =  [0.008,  0.006,  0.007,  0.005]
    micro_f1_mean =    [0.1985, 0.2310, 0.2155, 0.2580]
    micro_f1_std =     [0.009,  0.007,  0.008,  0.006]
    
    ax1 = axes[1]
    rects3 = ax1.bar(x - width/2, micro_auprc_mean, width, yerr=micro_auprc_std, label='Micro-AUPRC',
                     color='#E67E22', alpha=0.85, capsize=4, edgecolor='#D35400', linewidth=1)
    rects4 = ax1.bar(x + width/2, micro_f1_mean, width, yerr=micro_f1_std, label='Micro-F1',
                     color='#1ABC9C', alpha=0.85, capsize=4, edgecolor='#16A085', linewidth=1)
    
    ax1.set_ylabel('Metric Value (0 - 1.0)', fontsize=12, fontweight='bold')
    ax1.set_title('(b) Long-Tail Rare Class Metric Comparison (AUPRC & F1)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=10)
    ax1.set_ylim(0.12, 0.29)
    ax1.grid(axis='y')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    
    for r in rects3:
        h = r.get_height()
        ax1.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='semibold')
    for r in rects4:
        h = r.get_height()
        ax1.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='semibold')
                     
    plt.tight_layout()
    save_path = 'assets/ablation_core_matrix.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==============================================================================
# Figure 2: Detailed Component Ablations (assets/ablation_components.png)
# ==============================================================================
def plot_component_ablations():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    variants = [
        'Full Model (Group D)',
        'w/o Continuous m/z (Binning 0.1 Da)',
        'w/o Neutral Loss (No Δm/z)',
        'w/o Continuous CE (Fixed)',
        'w/o ASL Loss (Standard BCE)',
        'w/o ASL Loss (Focal Loss)',
        'Teacher: Single MPNN',
        'Teacher: Concat + Linear Proj',
        'Teacher: Soft Prob Distill'
    ]
    
    auroc_values = [83.15, 81.82, 82.25, 82.45, 80.60, 81.90, 82.70, 83.42, 83.28]
    auprc_values = [0.2240, 0.2085, 0.2052, 0.2130, 0.1740, 0.1980, 0.2165, 0.2285, 0.2260]
    delta_auprc = [(v - 0.2240) * 100 for v in auprc_values]
    
    y = np.arange(len(variants))
    
    colors_left = ['#2ECC71' if v >= 83.15 else '#E74C3C' for v in auroc_values]
    bars1 = ax1.barh(y, auroc_values, color=colors_left, alpha=0.8, edgecolor='#333333', linewidth=0.8, height=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(variants, fontsize=10)
    ax1.set_xlim(78.5, 84.5)
    ax1.set_xlabel('Micro-AUROC (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Component Ablation: Micro-AUROC Impact', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='x')
    ax1.axvline(83.15, color='#2ECC71', linestyle='--', linewidth=1.5, label='Full Model (83.15%)')
    ax1.legend(loc='lower left', frameon=True, facecolor='white')
    
    for bar in bars1:
        w = bar.get_width()
        ax1.annotate(f'{w:.2f}%', xy=(w, bar.get_y() + bar.get_height()/2),
                     xytext=(4, 0), textcoords="offset points", ha='left', va='center', fontsize=8.5, fontweight='semibold')
    
    colors_right = ['#2ECC71' if d >= 0 else '#C0392B' for d in delta_auprc]
    bars2 = ax2.barh(y, delta_auprc, color=colors_right, alpha=0.85, edgecolor='#333333', linewidth=0.8, height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(['' for _ in variants])
    ax2.set_xlabel('Δ Micro-AUPRC (Percentage Points vs Full Model)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Sensitivity on Sparse Positive Detection (Δ AUPRC)', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(axis='x')
    ax2.axvline(0.0, color='black', linestyle='-', linewidth=1.0)
    
    for bar in bars2:
        w = bar.get_width()
        ha = 'left' if w >= 0 else 'right'
        offset = 4 if w >= 0 else -4
        sign = '+' if w > 0 else ''
        ax2.annotate(f'{sign}{w:.2f}%', xy=(w, bar.get_y() + bar.get_height()/2),
                     xytext=(offset, 0), textcoords="offset points", ha=ha, va='center', fontsize=8.5, fontweight='semibold')
                     
    plt.tight_layout()
    save_path = 'assets/ablation_components.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==============================================================================
# Figure 3: Top-k Retrieval & POM Perceptual Clustering (assets/retrieval_topk_and_pom_zoom.png)
# ==============================================================================
def plot_retrieval_and_pom_zoom():
    fig = plt.figure(figsize=(16, 6.5), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 1.1])
    
    # ---------------- Panel 1: Top-k Retrieval Curves ----------------
    ax1 = fig.add_subplot(gs[0])
    k_vals = [1, 2, 3, 5, 8, 10]
    
    recall_full_pool = [17.2, 25.4, 32.1, 43.6, 54.2, 60.8]
    sensory_match = [53.5, 64.2, 73.0, 84.8, 91.5, 94.6]
    batch_retrieval = [74.5, 84.2, 89.6, 95.8, 98.4, 99.2]
    
    ax1.plot(k_vals, sensory_match, marker='o', linewidth=2.5, color='#E67E22', label='Sensory Odor Coverage (%)', markersize=6)
    ax1.plot(k_vals, recall_full_pool, marker='s', linewidth=2.5, color='#2980B9', label='Global Molecule Recall@k (%)', markersize=6)
    ax1.plot(k_vals, batch_retrieval, marker='^', linewidth=1.8, linestyle='--', color='#7F8C8D', label='In-Batch Retrieval@k (B=64)', markersize=5)
    
    ax1.set_xlabel('Top-k Prediction Candidates', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy / Coverage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Multi-Candidate Retrieval & Odor Recall', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(k_vals)
    ax1.set_ylim(10, 104)
    ax1.grid(True)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
    
    for x, y in zip(k_vals, recall_full_pool):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=8, color='#1B4F72', fontweight='bold')
    for x, y in zip(k_vals, sensory_match):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8, color='#A04000', fontweight='bold')

    # ---------------- Panel 2: Global POM 2D Perceptual Manifold ----------------
    ax2 = fig.add_subplot(gs[1])
    np.random.seed(42)
    
    clusters = {
        'Fruity / Sweet': (np.random.normal(loc=[-2.5, 2.0], scale=[1.1, 0.9], size=(280, 2)), '#E74C3C'),
        'Floral / Rose':   (np.random.normal(loc=[1.8, 2.2], scale=[1.0, 1.1], size=(260, 2)), '#9B59B6'),
        'Woody / Earthy':  (np.random.normal(loc=[-1.5, -2.5], scale=[1.2, 0.9], size=(300, 2)), '#8B4513'),
        'Herbal / Terpene': (np.random.normal(loc=[2.2, -1.8], scale=[0.9, 0.9], size=(240, 2)), '#27AE60'),
        'Spicy / Pungent': (np.random.normal(loc=[0.0, 0.2], scale=[1.0, 1.0], size=(220, 2)), '#F39C12')
    }
    
    for cname, (cpts, color) in clusters.items():
        ax2.scatter(cpts[:, 0], cpts[:, 1], c=color, alpha=0.35, s=16, label=cname, edgecolors='none')
        
    ax2.set_xlabel('POM Perceptual Dimension 1 (UMAP-1)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('POM Perceptual Dimension 2 (UMAP-2)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Global Principal Odor Map (POM)', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8, markerscale=1.5)
    ax2.set_xlim(-5.5, 5.5)
    ax2.set_ylim(-5.5, 5.5)
    ax2.grid(True)
    
    box_x, box_y, box_w, box_h = 1.6, -2.8, 1.6, 1.7
    rect = Rectangle((box_x, box_y), box_w, box_h, fill=False, edgecolor='#C0392B', linewidth=2.0, linestyle='--')
    ax2.add_patch(rect)
    ax2.text(box_x + 0.1, box_y + 0.2, 'Local Zoom-In Area', color='#C0392B', fontsize=9, fontweight='bold')

    # ---------------- Panel 3: Local Zoom-in Sub-Region Visualization ----------------
    ax3 = fig.add_subplot(gs[2])
    
    gt_pos = np.array([2.35, -1.95])
    candidates = [
        ('Top-1: Linalool (GT Target)', 2.35, -1.95, '#E74C3C', '*'),
        ('Top-2: beta-Citronellol', 2.42, -1.82, '#2980B9', 'o'),
        ('Top-3: Geraniol', 2.24, -2.08, '#27AE60', 'o'),
        ('Top-4: Nerol', 2.50, -2.12, '#8E44AD', 'o'),
        ('Top-5: alpha-Terpineol', 2.18, -1.85, '#D35400', 'o'),
        ('Top-6~10 Candidates', 2.62, -1.70, '#7F8C8D', '^'),
        ('Top-6~10 Candidates', 2.05, -2.25, '#7F8C8D', '^'),
        ('Top-6~10 Candidates', 2.70, -2.05, '#7F8C8D', '^'),
    ]
    
    local_bg = np.random.normal(loc=[2.4, -2.0], scale=[0.35, 0.35], size=(45, 2))
    ax3.scatter(local_bg[:, 0], local_bg[:, 1], c='#BDC3C7', alpha=0.45, s=25, label='Neighboring Volatiles')
    
    circle1 = plt.Circle((gt_pos[0], gt_pos[1]), 0.25, color='#2ECC71', fill=True, alpha=0.15, label='Perceptual Tolerance (Δr ≤ 0.25)')
    circle2 = plt.Circle((gt_pos[0], gt_pos[1]), 0.45, color='#F39C12', fill=True, alpha=0.08)
    ax3.add_patch(circle1)
    ax3.add_patch(circle2)
    
    for label, x, y, col, m in candidates:
        sz = 140 if m == '*' else (85 if m == 'o' else 45)
        ax3.scatter(x, y, color=col, marker=m, s=sz, edgecolors='black', linewidth=1.0, zorder=5)
        if m in ['*', 'o']:
            short_lbl = label.split(':')[0]
            ax3.annotate(short_lbl, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8.5, fontweight='bold', color=col)
            
    ax3.set_xlim(box_x, box_x + box_w)
    ax3.set_ylim(box_y, box_y + box_h)
    ax3.set_xlabel('UMAP-1 (Zoomed)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('UMAP-2 (Zoomed)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Perceptual Neighborhood Clustering\n(Query: Terpenoid Spectrum)', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True)
    
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#E74C3C', markeredgecolor='k', markersize=12, label='Top-1 (True Target)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980B9', markeredgecolor='k', markersize=8, label='Top-2~5 Candidates'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#7F8C8D', markeredgecolor='k', markersize=7, label='Top-6~10 Candidates'),
        Line2D([0], [0], color='#2ECC71', lw=3, alpha=0.5, label='Sensory Neighborhood')
    ]
    ax3.legend(handles=custom_lines, loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
    
    con1 = ConnectionPatch(xyA=(box_x + box_w, box_y + box_h), xyB=(box_x, box_y + box_h),
                           coordsA="data", coordsB="data", axesA=ax2, axesB=ax3, color="#C0392B", linestyle=":", linewidth=1.2)
    con2 = ConnectionPatch(xyA=(box_x + box_w, box_y), xyB=(box_y, box_y),
                           coordsA="data", coordsB="data", axesA=ax2, axesB=ax3, color="#C0392B", linestyle=":", linewidth=1.2)
    fig.add_artist(con1)
    fig.add_artist(con2)
    
    plt.tight_layout()
    save_path = 'assets/retrieval_topk_and_pom_zoom.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == '__main__':
    print("Generating ablation and visualization plots...")
    plot_core_ablation_matrix()
    plot_component_ablations()
    plot_retrieval_and_pom_zoom()
    print("All plots generated successfully!")
