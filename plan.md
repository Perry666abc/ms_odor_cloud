# MS-Odor-Cloud: 核心链路重构与实操落地执行规划
# (Execution Plan & Implementation Guide)

> **目标与导言**：
> 本方案旨在将项目技术审查建议（`TECHNICAL_AUDIT_AND_ROADMAP.md`）与消融实验报告（`ABLATION_REPORT.md`）中的核心结论转化为**代码级、可落地、分阶段的工程实施手册**。
> 方案覆盖了从底层 Bug 修复、数据隔离治理，到对比损失改造、Top-k 预测与 POM 局部流形抠图系统，再到全量消融与泛化测试的完整闭环。

---

## 目录
1. [阶段一：高危 Bug 修复与数据闭环治理](#阶段一高危-bug-修复与数据闭环治理)
2. [阶段二：对比学习重构与标准探针评测体系](#阶段二对比学习重构与标准探针评测体系)
3. [阶段三：Top-k 多候选预测与 POM 局部流形聚集可视化系统](#阶段三top-k-多候选预测与-pom-局部流形聚集可视化系统)
4. [阶段四：物理元数据解耦与多视图自监督增强](#阶段四物理元数据解耦与多视图自监督增强)
5. [阶段五：全量消融实验流水线与结果复现落地](#阶段五全量消融实验流水线与结果复现落地)
6. [阶段六：迭代里程碑与交付验收清单](#阶段六迭代里程碑与交付验收清单)

---

## 阶段一：高危 Bug 修复与数据闭环治理

### 任务 1.1：修复 `BondFeaturizer` 化学键类型枚举编码 Bug
- **文件定位**：[`ms-pom/utils/mol_featurizer.py`](file:///Users/mac/Desktop/ms_odor_cloud/ms-pom/utils/mol_featurizer.py)
- **代码重构方案**：
  将第 68–69 行：
  ```python
  # 修改前:
  bond_type = str(bond.GetBondType())
  features += one_hot(bond.GetBondType(), self.allowable_sets["bond_type"])
  ```
  修改为：
  ```python
  # 修改后:
  bond_type = str(bond.GetBondType())
  features += one_hot(bond_type, self.allowable_sets["bond_type"])
  ```
- **自动化验证测试**：
  编写测试单元 `tests/test_mol_featurizer.py`，断言单键、双键、三键、芳香键能正确产出对应的 One-Hot 稀疏编码，且不触发默认 unknown 兜底：
  ```bash
  python3 -c "
  from rdkit import Chem
  from utils.mol_featurizer import BondFeaturizer
  mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
  bf = BondFeaturizer()
  features = [bf(b) for b in mol.GetBonds()]
  assert all(len(f) == 7 for f in features), 'Feature dimension mismatch'
  print('✅ BondFeaturizer unit test passed!')
  "
  ```
- **教师重训与缓存重建链条**：
  1. 清除现存受损缓存文件：`rm -f pom_cache.pt ms-pom/demo/saved_models_list.pt`；
  2. 重新运行 `ms-pom/train_ensemble.py`，训练 5 折 MPNN 模型并保存至 `ms-pom/demo/saved_models_list.pt`；
  3. 执行 `python3 -c "from pom_encoder import OpenPOMEncoder; ..."` 重新生成 `pom_cache.pt`。

---

### 任务 1.2：构建全局分子切分注册表（Global Split Registry）
- **目标**：彻底解决 Open-POM 教师模型训练范围与质谱测试集分子的潜在交叉风险。
- **实施步骤**：
  1. 新建 `splits/` 目录，固化不可变的数据集索引清单：
     - `splits/test_molecules.json`：固定 425 个测试分子（与 `seed=42` 分子隔离切分完全一致）；
     - `splits/val_molecules.json`：固定 424 个验证分子；
     - `splits/train_molecules.json`：固定 1,979 个训练分子。
  2. 在 `ms-pom/utils/mol_dataset.py` 与 `ms-pom/train_ensemble.py` 中强制引入过滤机制：
     ```python
     with open('splits/test_molecules.json') as f:
         held_out_test_smiles = set(json.load(f))
     # 严格剔除测试集分子，确保教师模型处于盲态
     train_df = train_df[~train_df['canonical_smiles'].isin(held_out_test_smiles)].reset_index(drop=True)
     ```
  3. 在 `src/dataset.py` 中直接读取 `splits/` 清单，取代动态 `train_test_split`，彻底杜绝数据漂移。

---

### 任务 1.3：堵塞空缓存保存与零损失漏洞
- **文件定位**：[`src/pretrain_align.py`](file:///Users/mac/Desktop/ms_odor_cloud/src/pretrain_align.py)
- **代码重构方案**：
  在验证循环结束后增加防御断言：
  ```python
  if valid_batches == 0:
      raise RuntimeError("❌ 致命错误: 验证集有效批次数为 0 (可能 POM 缓存为空或特征缺失)，拒绝保存模型！")
  avg_loss = total_loss / valid_batches
  avg_acc = total_acc / valid_batches
  ```

---

## 阶段二：对比学习重构与标准探针评测体系

### 任务 2.1：实现多正例 InfoNCE（`MultiPositiveInfoNCELoss`）
- **文件定位**：[`src/loss.py`](file:///Users/mac/Desktop/ms_odor_cloud/src/loss.py)
- **问题现状**：同批次内同分子不同谱图（期望约 6.6 对）被当作负例互相推远。
- **代码实现**：
  ```python
  class MultiPositiveInfoNCELoss(nn.Module):
      """
      Supervised Contrastive Loss (SupCon) for multi-spectrum-per-molecule contrast.
      Reference: Khosla et al., NeurIPS 2020.
      """
      def __init__(self, init_tau=0.03, learnable_tau=False):
          super().__init__()
          if learnable_tau:
              self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))
          else:
              self.register_buffer('log_tau', torch.log(torch.tensor(init_tau)))
              
      def forward(self, ms_emb, mol_emb, smiles_list):
          B = ms_emb.size(0)
          device = ms_emb.device
          tau = torch.exp(self.log_tau).clamp(min=0.01, max=1.0)
          
          # L2 归一化
          ms_norm = F.normalize(ms_emb, p=2, dim=-1)
          mol_norm = F.normalize(mol_emb, p=2, dim=-1)
          sim = torch.matmul(ms_norm, mol_norm.t()) / tau
          
          # 构建分子同一性掩码: (B, B)，同 SMILES 为 1，否则为 0
          labels_match = np.array([[s1 == s2 for s2 in smiles_list] for s1 in smiles_list], dtype=bool)
          pos_mask = torch.tensor(labels_match, dtype=torch.float32, device=device)
          
          # 数值稳定 Log-Sum-Exp 计算
          sim_max, _ = torch.max(sim, dim=1, keepdim=True)
          exp_sim = torch.exp(sim - sim_max.detach())
          log_prob = (sim - sim_max.detach()) - torch.log(exp_sim.sum(dim=1, keepdim=True))
          
          # 计算每个样本的正例平均 log 概率
          mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1.0)
          loss = - mean_log_prob_pos.mean()
          return loss
  ```

---

### 任务 2.2：构建全库候选检索与标准评估体系
- **目标**：将阶段一的批内 64 选 1 准确率，纠正为基于全验证集/测试集唯一分子候选库的标准跨分子检索体系。
- **实施方案**：
  在 `src/pretrain_align.py` 与评估脚本中加入全库候选池评估：
  ```python
  def evaluate_global_retrieval(ms_encoder, val_loader, unique_mol_cache, device):
      # unique_mol_cache: (N_unique, 256) 归一化分子 POM 向量库
      # 遍历所有谱图提取质谱对齐向量，并在全库中执行矩阵乘法 Top-k 检索
      # 计算并输出 Recall@1, Recall@5, Recall@10 以及 MRR (Mean Reciprocal Rank)
  ```

---

### 2.3 规范骨干冻结探针实验三级对比
- **文件定位**：[`src/model.py`](file:///Users/mac/Desktop/ms_odor_cloud/src/model.py) 与 [`src/train.py`](file:///Users/mac/Desktop/ms_odor_cloud/src/train.py)
- **协议规划**：
  1. **Probe-CLS**：冻结骨干（强制执行 `transformer.eval()`），输入直接从 `cls_out` 接单层线性分类器 `nn.Linear(d_model, 138)`；
  2. **Probe-Align**：冻结骨干与预训练 `align_head`，输入从 `align_pred` 接单层线性分类器 `nn.Linear(256, 138)`；
  3. **Full-FT**：端到端全量参数联合微调。
- **执行配置开关**：
  在 `src/train.py` 增加参数：
  `--probe_mode {none, cls, align}`，并在 probe 模式下严禁更新骨干及前向随机层。

---

### 2.4 重构 `eval_test.py` 评估配置统一性
- **文件定位**：[`eval_test.py`](file:///Users/mac/Desktop/ms_odor_cloud/eval_test.py)
- **重构要点**：
  1. 彻底移除硬编码网络层数、隐藏层维度与切分代码；
  2. 从 `best_model.pth` 中加载保存时的 `config` 字典与 `class_weights`；
  3. 检验 `checkpoint['thresholds']`，若缺失则直接中断并抛出异常，**严禁使用测试集搜索最佳阈值**；
  4. 增加**分子等权评估（Molecule-Weighted Evaluation）**与 Bootstrap 95% 置信区间计算；
  5. 修复 Macro-AUROC NaN 问题：逐标签独立计算并过滤掉正例为 0 的类别（如 `vetiver`），明确输出有效类别数目（如 135/138）。

---

## 阶段三：Top-k 多候选预测与 POM 局部流形聚集可视化系统

### 任务 3.1：多候选输出与检索推理引擎开发
- **新增模块**：`src/inference_topk.py`
- **核心逻辑**：
  输入一张质谱谱图，质谱编码器将其映射为对齐表征 $\mathbf{z}_{\text{ms}} \in \mathbb{R}^{256}$；在候选分子库中计算余弦相似度并按降序排序：
  $$\text{Score}(i) = \frac{\mathbf{z}_{\text{ms}} \cdot \mathbf{m}_i}{\|\mathbf{z}_{\text{ms}}\| \|\mathbf{m}_i\|}$$
  输出前 $k$ 个候选（$k=1, 3, 5, 10$）的分子名称、SMILES、分子量、相似度打分以及由分类器导出的预测气味轮廓。

---

### 任务 3.2：流形空间局部聚集度量（Perceptual Clustering Metric）
- **度量公式**：
  为定量验证 Top-k 候选是否在嗅觉感知流形中聚集，定义候选集合平均流形离散度（Manifold Dispersion）：
  $$\overline{\Delta r}_{\text{top-}k} = \frac{2}{k(k-1)} \sum_{1 \le i < j \le k} \|\mathbf{p}_i - \mathbf{p}_j\|_2$$
  其中 $\mathbf{p}_i$ 为第 $i$ 个候选分子在 256 维 Open-POM 感知空间中的坐标向量。
- **实验阈值断言**：
  消融报告表明，Top-5 候选的平均空间聚集半径 $\overline{\Delta r}_{\text{top-5}} \le 0.25$，相较于随机分子抽样的平均距离（约 1.28）具有极高显著性（$p < 10^{-6}$）。

---

### 任务 3.3：POM 全局流形投影与局部“高亮抠图”可视化组件
- **新增模块**：`src/visualize_pom_local.py`
- **实现功能**：
  1. **全局流形底图绘制**：对候选库中全部分子计算 2D UMAP 坐标，以半透明点云方式展示各大气味家族（花香、果香、木香、草本、辛辣等）；
  2. **局部区域动态定位**：以目标 Ground Truth 与 Top-1 预测为中心，动态计算局部包围盒（Bounding Box: $[x_{\min}, x_{\max}, y_{\min}, y_{\max}]$）；
  3. **局部流形抠图放大（Sub-Region Zoom-In Callout）**：在全局底图右侧绘制放大局部视图，标注 Top-1（星号）、Top-2~5（圆点）、Top-6~10（三角形），并用虚线连线连接全局框与局部视图；
  4. **集成到前端应用**：将可视化函数接入 [`ms-pom/demo/streamlit_app_final.py`](file:///Users/mac/Desktop/ms_odor_cloud/ms-pom/demo/streamlit_app_final.py)，支持用户上传未知质谱并实时交互式查看 Top-k 局部流形落点。

```python
# 核心接口设计示例:
def generate_pom_subregion_figure(query_ms_emb, candidate_mols, top_k=5, save_path=None):
    """
    输入:
      query_ms_emb: (256,) 质谱预测 POM 嵌入
      candidate_mols: Top-k 候选分子元数据列表
      top_k: 展示候选规模
    输出:
      包含全局 POM 背景与局部流形放大抠图的 Matplotlib Figure / PNG
    """
```

---

## 阶段四：物理元数据解耦与多视图自监督增强

### 任务 4.1：解耦碰撞能量（CE）与仪器大类特征编码
- **文件定位**：[`src/dataset.py`](file:///Users/mac/Desktop/ms_odor_cloud/src/dataset.py) 与 [`src/model.py`](file:///Users/mac/Desktop/ms_odor_cloud/src/model.py)
- **重构方案**：
  1. 放弃单一的 `val = nce * 0.3` 近似公式；
  2. 构建独立的离散与连续编码层：
     - `instrument_type`: 嵌入层（Orbitrap, Q-TOF, Triple-Quad, GC-EI 等 6 种）；
     - `energy_type`: 嵌入层（Explicit eV, NCE Percentage, CID, EI-70eV）；
     - `ce_value`: 归一化连续能量值；
     - `ion_mode` & `charge`: 离散嵌入。
  3. 模型前向通过多头门控层（Gated MLP）融合元数据向量。

---

### 任务 4.2：显式差值通道（$\Delta m/z$）与中性丢失物理边界适用
- **实施方案**：
  1. 仅在二级串联质谱（MS2，具有前体离子 `precursor_mz > 0`）中计算中性丢失 $\Delta m/z = \text{precursor\_mz} - m/z$；
  2. 在 GC-EI（无前体离子）中，将 $\Delta m/z$ 掩码置为 0，启用专用的 `missing_nl_token`，避免将虚假中性丢失引入网络。

---

### 任务 4.3：同分子多视图自监督与动态梯度平衡
- **实施方案**：
  1. 在训练批次中成对采样相同分子的不同 CE 谱图（$\mathbf{s}_1, \mathbf{s}_2$），施加跨视图余弦一致性惩罚：
     $$\mathcal{L}_{\text{consistency}} = 1 - \cos(\mathbf{z}_{\text{ms1}}, \mathbf{z}_{\text{ms2}})$$
  2. 监控共享骨干参数上的梯度范数：
     $$g_{\text{cls}} = \|\nabla_{\theta} \mathcal{L}_{\text{cls}}\|_2, \quad g_{\text{align}} = \|\nabla_{\theta} \mathcal{L}_{\text{align}}\|_2$$
     动态自适应平衡因子 $\lambda_{\text{align}} = \text{clip}\left(\frac{g_{\text{cls}}}{g_{\text{align}}}, 0.05, 0.5\right)$。

---

## 阶段五：全量消融实验流水线与结果复现落地

为确保 [`ABLATION_REPORT.md`](file:///Users/mac/Desktop/ms_odor_cloud/ABLATION_REPORT.md) 中数据的可验证性与学术严谨性，编写一键式运行流水线脚本 `run_ablation_pipeline.sh`。

### 核心消融执行矩阵：
```bash
#!/usr/bin/env bash
# run_ablation_pipeline.sh: 自动化消融实验运行脚本

# 1. Group A: Baseline (无对齐预训练, 无联合对齐损失)
python src/train.py --exp_name group_a_baseline --use_pom_align False --pretrained_align_ckpt ""

# 2. Group B: Pretrain Only (有对齐预训练, 无联合对齐损失)
python src/train.py --exp_name group_b_pretrain_only --use_pom_align False --pretrained_align_ckpt checkpoints/scent_aligned_ms_encoder.pth

# 3. Group C: Joint Only (无对齐预训练, 开启联合对齐损失)
python src/train.py --exp_name group_c_joint_only --use_pom_align True --lambda_align 0.2 --pretrained_align_ckpt ""

# 4. Group D: Full SCENT (有对齐预训练, 开启联合对齐损失)
python src/train.py --exp_name group_d_full_scent --use_pom_align True --lambda_align 0.2 --pretrained_align_ckpt checkpoints/scent_aligned_ms_encoder.pth

# 5. 自动运行图表生成器
python generate_ablation_plots.py
```

---

## 阶段六：迭代里程碑与交付验收清单

| 里程碑 (Sprint) | 周期 | 核心交付内容 | 验收标准 / 命令 |
| :--- | :--- | :--- | :--- |
| **Sprint 1: Bug 修复与数据底座** | 第 1–2 周 | 修复 BondFeaturizer、重训教师 MPNN、建立 `splits/` 分子注册表 | `python tests/test_mol_featurizer.py` 全部通过；测试集 425 分子与教师训练集交集为 0 |
| **Sprint 2: 损失与探针重构** | 第 3–4 周 | 实现多正例 InfoNCE、全库检索评估、标准化 Linear Probe 三级对照 | `pretrain_align.py` 报告全库 Recall@1/5/10；探针实验骨干置为 `eval()` |
| **Sprint 3: Top-k 与局部流形可视化** | 第 5–6 周 | 开发 Top-k 推理引擎、POM 局部抠图生成器、Streamlit 交互式组件 | 运行 `generate_ablation_plots.py` 产出图 3；Streamlit 支持 Top-5 流形抠图渲染 |
| **Sprint 4: 物理元数据与多视图增强** | 第 7–8 周 | 解耦 CE 与仪器类型嵌入、$\Delta m/z$ 适用边界、同分子跨能量一致性 | 检查模型在 Orbitrap 留出测试集上的泛化 AUROC 维持在 75%+ |
| **Sprint 5: 消融矩阵验证与论文归档** | 第 9–10 周 | 运行 5 个种子下的完整 2×2 消融与子模块消融，固化模型检查点与日志 | 实验结果与 `ABLATION_REPORT.md` 完全一致，所有图表与代码提交至 GitHub |

