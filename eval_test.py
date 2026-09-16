import os
import sys
import torch
from torch.utils.data import DataLoader

# 将 src 目录和根目录加入 Python 搜索路径，兼容模块导入
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dataset import build_datasets
from model import MSOdorTransformer
from loss import AsymmetricLoss
from train import evaluate_predictions, compute_metrics, get_device

def main():
    device = get_device()
    use_amp = (device.type == 'cuda')
    
    # 查找保存的最优权重文件
    ckpt_path = 'best_model.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join('checkpoints', 'scent_stage2_best.pth')
        
    if not os.path.exists(ckpt_path):
        print(f"❌ 错误: 未找到权重文件 {ckpt_path}，请确认当前工作目录。")
        return
        
    print(f"📦 正在加载最优模型权重: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch_saved = checkpoint.get('epoch', 'Unknown')
    print(f"✅ 成功加载在第 {epoch_saved} 轮保存的最优检查点！")
    
    # 数据集准备（严格 0% 分子重叠测试集）
    data_dir = 'GS_LF_MS_DATA'
    csv_files = [
        os.path.join(data_dir, 'GS_LF_MassSpecGym_merged.csv'),
        os.path.join(data_dir, 'GS_LF_MoNA_GC-MS_merged.csv'),
        os.path.join(data_dir, 'GS_LF_NIST20_hrmsms_merged.csv'),
        os.path.join(data_dir, 'GS_LF_NIST20_lrmsms_merged.csv'),
        os.path.join(data_dir, 'GS_LF_NIST20_mainlib_merged.csv')
    ]
    
    print("\n📊 正在装配零泄漏盲测集 (Strict Molecule-Level Split Test Set)...")
    _, _, test_dataset, _, _, _ = build_datasets(
        csv_files, split_by='molecule', max_seq_len=200, seed=42,
        build_pom_cache=True, pom_cache_path='pom_cache.pt'
    )
    
    num_workers = 4 if sys.platform == 'linux' else 0
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    
    # 实例化模型并载入权重
    transformer = MSOdorTransformer(
        latent_dim=256,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    transformer.load_state_dict(checkpoint['transformer'])
    
    best_thresholds = checkpoint.get('thresholds')
    criterion = AsymmetricLoss()
    
    print("🚀 正在对独立测试集进行盲测推理与指标计算...")
    test_loss, test_targets, test_probs = evaluate_predictions(
        transformer, None, test_loader, criterion, True, device, use_amp
    )
    test_metrics = compute_metrics(test_targets, test_probs, thresholds=best_thresholds)
    
    print("\n" + "="*66)
    print("  🎉 独立盲测测试集最终评测报告 (SCENT Table 1 对齐标准):")
    print("="*66)
    print(f"  - Test Loss:         {test_loss:.4f}")
    print(f"  - Test Micro-AUROC:  {test_metrics['micro_auc']*100:.2f}%")
    print(f"  - Test Weighted-AUC: {test_metrics['weighted_auc']*100:.2f}%")
    print(f"  - Test Macro-AUROC:  {test_metrics['macro_auc']*100:.2f}%")
    print(f"  - Test Macro-AUPRC:  {test_metrics['macro_auprc']:.4f}")
    print(f"  - Test Micro-AUPRC:  {test_metrics['micro_auprc']:.4f}")
    print(f"  - Test Micro-F1:     {test_metrics['micro_f1']:.4f}")
    print("="*66 + "\n")

if __name__ == '__main__':
    main()
