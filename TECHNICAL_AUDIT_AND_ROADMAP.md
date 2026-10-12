# MS-Odor-Cloud: 核心链路技术审查报告与研发路线图
# (Technical Audit & Research Roadmap)

> **文档定位**：本文档旨在对当前仓库的核心代码实现、数据分布划分、训练评估链路以及实验结论进行严格的工程与学术规范性审查，明确已复现的技术缺陷，并给出下一阶段严谨可复现的研究路线图。

---

## 目录
1. [现状评估与数据分布核验](#1-现状评估与数据分布核验)
2. [亟需优先修复的工程与算法缺陷](#2-亟需优先修复的工程与算法缺陷)
   - [2.1 修复分子图化学键类型编码错误（已复现 Bug）](#21-修复分子图化学键类型编码错误已复现-bug)
   - [2.2 改造对比学习支持“单分子-多谱图”（Multi-Positive InfoNCE）](#22-改造对比学习支持单分子-多谱图multi-positive-infonce)
   - [2.3 修正冻结骨干的线性探针实验（Linear Probing）](#23-修正冻结骨干的线性探针实验linear-probing)
   - [2.4 统一独立评估脚本与训练配置](#24-统一独立评估脚本与训练配置)
   - [2.5 修复宏指标计算与引入“分子等权”评估维度](#25-修复宏指标计算与引入分子等权评估维度)
   - [2.6 严格审计 Open-POM 教师的数据隔离性（防止标签泄漏）](#26-严格审计-open-pom-教师的数据隔离性防止标签泄漏)
3. [模型架构与特征工程演进方向](#3-模型架构与特征工程演进方向)
   - [3.1 重构教师感知表征集成策略](#31-重构教师感知表征集成策略)
   - [3.2 完善物理元数据与中性丢失（Δm/z）建模](#32-完善物理元数据与中性丢失δmz建模)
   - [3.3 利用同分子多视图自监督与多任务梯度平衡](#33-利用同分子多视图自监督与多任务梯度平衡)
   - [3.4 增强训练流程工程健壮性](#34-增强训练流程工程健壮性)
4. [严谨消融实验方案设计（2×2 控制变量矩阵）](#4-严谨消融实验方案设计22-控制变量矩阵)
5. [项目核心科研主线聚焦](#5-项目核心科研主线聚焦)

---

## 1. 现状评估与数据分布核验

### 1.1 数据分布统计与切分隔离
基于当前 5 份合并质谱数据（`GS_LF_MassSpecGym_merged.csv`、`GS_LF_MoNA_GC-MS_merged.csv`、`GS_LF_NIST20_hrmsms_merged.csv`、`GS_LF_NIST20_lrmsms_merged.csv`、`GS_LF_NIST20_mainlib_merged.csv`），全量数据集的严格统计如下：
- **总谱图数量**：31,695 张；
- **唯一分子数量**：2,828 个（基于规范化 Canonical SMILES 统计）；
- **数据切分（`seed=42`, `split_by='molecule'`）**：
  - **训练集**：1,979 个分子，22,006 张谱图；
  - **验证集**：424 个分子，4,826 张谱图；
  - **测试集**：425 个分子，4,863 张谱图。

> **阶段性肯定**：在分子级切分下，规范化分子在训练集、验证集和测试集之间**完全不存在交集（0% Molecular Overlap）**。这在质谱-分子关系建模中遵循了严格的隔离原则，杜绝了直接的测试分子记忆。

### 1.2 现有测试性能基准（Checkpoint: `best_model.pth` 实测复现）
在当前未重新训练的前提下，基于 `train_5090.log` 记录及独立盲测复现结果如下：
- **Test Loss**: 0.0346（对比验证阶段 Loss 0.0063，二者存在指标尺度差异）；
- **Micro-AUROC**: 82.60%；
- **Weighted-AUROC**: 75.00%；
- **Micro-AUPRC**: 0.2155；
- **Micro-F1**: 0.2494；
- **Macro-AUROC**: **NaN**（由于 `vetiver` 等稀少标签在测试集无正例触发警告）；
- **Macro-AUPRC**: 0.1918。

### 1.3 核心评估结论
1. 项目已成功构建出端到端打通的多模态质谱连续编码与感知对齐原型系统；
2. 然而，**Micro-AUPRC（0.2155）与 Micro-F1（0.2494）偏低**，说明模型在极端多标签不平衡分布下，对稀有正例的召回与精确识别仍存在显著不足；
3. 现有对比基线在切分策略（谱图切分 vs 分子切分）、损失函数配置上存在不一致，**当前证据尚不足以支撑 README 中“POM 带来数量级提升”“强跨分子泛化”等强结论**。

---

## 2. 亟需优先修复的工程与算法缺陷

### 2.1 修复分子图化学键类型编码错误（已复现 Bug）
- **问题定位**：`ms-pom/utils/mol_featurizer.py` 第 68–69 行。
  ```python
  # 当前代码实现:
  bond_type = str(bond.GetBondType())
  features += one_hot(bond.GetBondType(), self.allowable_sets["bond_type"])
  ```
- **机理与复现结果**：
  - `self.allowable_sets["bond_type"]` 预设为字符串列表 `["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]`；
  - 调用 `one_hot` 时传入的却是 RDKit 的 `BondType` 枚举对象（`rdkit.Chem.rdchem.BondType.SINGLE` 等）；
  - `allowable_set.index(val)` 内部因类型不一致直接抛出 `ValueError`，所有单键、双键、三键、芳香键**全部被编码为未知的 Unknown 通道**。
- **修复方案**：
  将传入参数替换为第 68 行已计算出的字符串变量：
  ```python
  features += one_hot(bond_type, self.allowable_sets["bond_type"])
  ```
- **严重级联影响**：
  若现存 Open-POM 教师模型权重（`ms-pom/demo/saved_models_list.pt`）系基于旧特征训练，修复后**必须重新训练教师 MPNN、重新导出权重并重新构建 `pom_cache.pt`**，绝不能继续沿用旧权重缓存。

---

### 2.2 改造对比学习支持“单分子-多谱图”（Multi-Positive InfoNCE）
- **问题定位**：`src/loss.py` 中的 `MultimodalInfoNCELoss` 与 `src/pretrain_align.py`。
  ```python
  # 当前实现只将对角线视作正例:
  targets = torch.arange(B, dtype=torch.long, device=device)
  loss_mol2ms = F.cross_entropy(sim, targets)
  loss_ms2mol = F.cross_entropy(sim.t(), targets)
  ```
- **机理剖析**：
  - 数据集中广泛存在“同一个分子拥有多张不同仪器、不同碰撞能量谱图”的物理事实；
  - 在当前训练集分布下，当随机采样 `batch_size=64` 时，**单个批次中同分子样本对的数学期望高达约 6.6 对**；
  - 当前 InfoNCE 将同一个批次内属于相同分子的多张不同谱图强制视作“负例”进行排斥，强行破坏了模型学习同一分子质谱表征不变性的能力。
- **检索准确率定义校正**：
  - 阶段一当前报告的 Top-1 准确率本质为**批内 64 选 1 检索**，其随机命中基线约为 $1/64 \approx 1.56\%$，README 中所称“不足 0.2% 随机命中率”属于混淆了批内检索与全库检索。
- **整改措施**：
  1. **方案 A（损失层优化）**：引入监督对比学习（Supervised Contrastive Loss / Multi-Positive InfoNCE），对批内相同 SMILES 的全部对项赋予正例掩码；
  2. **方案 B（采样层优化）**：在 DataLoader 端引入 `UniqueMoleculeBatchSampler`，确保每个训练 Batch 内每个分子严格只采样 1 张谱图；
  3. **标准检索评估**：在验证集上建立**全验证集独立唯一分子候选库**，计算并报告无偏差的 `Recall@1`, `Recall@5`, `Recall@10` 及 `MRR`。

---

### 2.3 修正冻结骨干的线性探针实验（Linear Probing）
- **问题定位**：`src/model.py` 第 202–204 行与 `src/train.py` 第 358–362 行。
  ```python
  # model.py:
  pom_pred = self.pom_head(cls_out)      # (B, 256)
  align_pred = self.align_head(cls_out)  # (B, 256)
  logits = self.classifier(pom_pred)     # (B, 138)
  
  # train.py:
  if args.freeze_ms_encoder:
      for name, param in transformer.named_parameters():
          if 'classifier' not in name:
              param.requires_grad = False
  ```
- **机理剖析**：
  - 阶段一 SCENT 预训练对齐的是 `align_head`；然而模型下游前向却通过了 `pom_head`；
  - 在冻结测试中，未经预训练的 `pom_head` 因名字不含 `'classifier'` **被一同错误冻结**；
  - 骨干网络未显式设置为 `.eval()`，内部的 Dropout 仍在激活，违背了线性探针（Linear Probing）的标准学术定义。
- **整改措施**：
  规范设计探针基准实验链条：
  1. **Linear Probe on CLS (`Probe-CLS`)**：冻结骨干（开启 `eval()`），直接在 `cls_out` 之上接入单层线性层（`nn.Linear(d_model, 138)`）；
  2. **Linear Probe on Aligned Space (`Probe-Align`)**：冻结骨干与已训练的 `align_head`（开启 `eval()`），直接在对齐表征之上接入单层线性层；
  3. **End-to-End Fine-Tuning**：全网络端到端联合微调对比。

---

### 2.4 统一独立评估脚本与训练配置
- **问题定位**：`eval_test.py`。
- **机理剖析**：
  - 评估脚本硬编码了数据切分逻辑（固定 `seed=42`）与模型超参数（`d_model=256` 等），且使用了无类别权重的默认 `AsymmetricLoss()`；
  - 训练时使用的却是加权 ASL（`pos_weight=class_weights`），导致测试 Loss（0.0346）与验证 Loss（0.0063）完全无法横向比较；
  - **隐性测试泄漏风险**：若模型 Checkpoint 缺失 `thresholds`，现有脚本会动态在测试集标签上贪心网格搜索最佳 F1 阈值，构成对测试集标签信息的严重拟合。
- **整改措施**：
  - 训练时将所有超参数配置、类别名称顺序、`pos_weight` 张量以及在**验证集上确定的最佳阈值向量**完整存入 Checkpoint；
  - `eval_test.py` 纯粹从 Checkpoint 还原评估环境；若 Checkpoint 缺少验证集阈值，强制报错退出，严禁使用测试集搜索阈值；
  - 导出并持久化固定的样本索引清单文件（`splits/train_ids.csv`, `splits/val_ids.csv`, `splits/test_ids.csv`）。

---

### 2.5 修复宏指标计算与引入“分子等权”评估维度
- **问题定位**：`src/train.py` 中的 `compute_metrics`。
- **机理剖析**：
  - **NaN 宏指标问题**：在测试集的 425 个分子中，稀有气味标签（如 `vetiver` 岩兰草香）的正例数为 0。调用 `sklearn.metrics.roc_auc_score(average='macro')` 时会抛出未定义警告并返回 `NaN`；
  - **谱图偏差问题（Spectrum Dominance）**：4,863 张测试谱图仅对应 425 个分子，个别高频分子独占多达 422 张谱图。传统的“按谱图算指标”会导致极少数拥有海量谱图的分子主导评估结果，无法客观反映对未见分子的泛化能力。
- **整改措施**：
  - 逐标签独立计算 ROC-AUC / AUPRC，剔除测试集无正例的类别，在日志和报告中显式注明有效评估标签数量（如 135/138）和标签支持度（Support）；
  - **新增分子等权指标（Molecule-Weighted Metrics）**：将同一测试分子的全部谱图预测概率做聚合（Mean Pooling）后，在 425 个唯一分子维度上计算 AUROC、AUPRC 和 F1；
  - 引入基于分子的 Bootstrap 重采样（1,000 次），输出各核心指标的 95% 置信区间。

---

### 2.6 严格审计 Open-POM 教师的数据隔离性（防止标签泄漏）
- **风险点说明**：
  - Open-POM 教师模型基于包含 4,983 个分子的 GS-LF 数据集进行训练；
  - 现有仓库缺少足以核验教师训练折（Fold）与质谱测试集分子的交叉比对清单；目前既不能排除教师模型曾见过测试分子气味标签的可能性，也无法断言已存在泄漏。
- **整改措施**：
  - 建立全局外层隔离名单（Hold-out Test Molecules）：统一固定 425 个测试分子（或约 15% 的独立分子）；
  - **在训练 Open-POM 教师模型及质谱预训练阶段，将该名单从所有分子图和质谱数据中彻底剔除**（严格对齐 SCENT 论文 Appendix A 标准）。

---

## 3. 模型架构与特征工程演进方向

### 3.1 重构教师感知表征集成策略
- **现状问题**：当前直接逐维平均 5 个独立 MPNN 模型的 256 维隐藏层表征。由于各独立训练的神经网络内部隐空间缺乏坐标系对齐约束，直接简单平均可能破坏嗅觉感知流形的几何拓扑。
- **实验与优化方案**：
  1. **单教师基准测试**：对比单模型最佳表征 vs 简单平均表征；
  2. **拼接映射（Concat + Linear Projection）**：将 5 个教师特征拼接为 1280 维，通过单层投影降维至 256 维；
  3. **语义概率蒸馏（Label Distribution Distillation）**：利用 5 个教师模型在 138 维真实气味语义概率上的平均值，通过软标签交叉熵或 KL 散度指导质谱骨干。

### 3.2 完善物理元数据与中性丢失（Δm/z）建模
- **NCE 经验换算校正**：数据集中有 2,580 条记录标记为 `NCE=35%`。目前采用的线性近似公式缺少仪器平台针对性。建议保留原始 CE 数值、能量单位（eV, NCE）、仪器大类（Orbitrap, Q-TOF, QqQ, GC-EI）、加合物离子形式及电荷态，分别进行嵌入编码；
- **连续差值特征（$\Delta m/z$）**：系统化计算前体离子与各碎片离子之差（Precursor m/z - Fragment m/z）作为中性丢失通道，并在预处理中明确限制其仅在具有明确前体离子的串联质谱（MS2）中生效。

### 3.3 利用同分子多视图自监督与多任务梯度平衡
- **同分子跨条件一致性约束**：利用同一分子在不同碰撞能量、不同仪器下的谱图对，构建多视图表征一致性正则项；
- **动态梯度监控与加权**：记录下游多标签分类损失 $\mathcal{L}_{\text{cls}}$ 与对齐损失 $\mathcal{L}_{\text{align}}$ 对共享骨干网络参数的梯度范数（Gradient Norms），根据验证集表现动态调节平衡因子 $\lambda_{\text{align}}$。

### 3.4 增强训练流程工程健壮性
- **堵塞空缓存异常保存路径**：在 `pretrain_align.py` 中增加断言与保护，当有效批次数为 0 或 POM 缓存加载异常时立即中断报错，彻底杜绝“有效 Batch 被全部跳过却依然以 0 损失误保存 Best Checkpoint”的隐患；
- **实验管理结构化**：生成全流程数据哈希、切分索引哈希和结构化 JSONL 评估结果记录。

---

## 4. 严谨消融实验方案设计（2×2 控制变量矩阵）

为客观量化 POM 对齐预训练与联合对齐对气味预测的实际贡献，必须消除以往基线在切分方式、骨干网络和损失函数上的不一致。

### 4.1 核心 2×2 控制变量实验矩阵
在**完全固定数据切分清单、Transformer 骨干尺寸、加权 ASL 损失函数、硬件环境与训练 Epoch 预算**的严苛条件下，运行以下四组对照：

| 实验组别 | 阶段一对齐预训练 (Stage-1 Pretraining) | 阶段二联合对齐损失 (Joint Alignment Loss) | 评估协议 (Evaluation Protocol) |
| :---: | :---: | :---: | :---: |
| **实验 A (Baseline)** | ❌ 关闭（随机初始化骨干） | ❌ 关闭 ($\lambda_{\text{align}} = 0$) | 直接下游微调 |
| **实验 B (Pretrain Only)** | ✅ 开启 (SCENT InfoNCE) | ❌ 关闭 ($\lambda_{\text{align}} = 0$) | 线性探针 / 全微调 |
| **实验 C (Joint Only)** | ❌ 关闭（随机初始化骨干） | ✅ 开启 ($\lambda_{\text{align}} > 0$) | 联合多任务训练 |
| **实验 D (Full SCENT)** | ✅ 开启 (SCENT InfoNCE) | ✅ 开启 ($\lambda_{\text{align}} > 0$) | 预训练 + 联合微调 |

### 4.2 统计与细粒度消融计划
1. **多种子统计**：每组对照采用 3–5 个不同随机种子（如 seeds: 42, 100, 2024），报告均值与置信区间；
2. **模块消融**：在实验 D 的基础上，逐步消融：
   - 连续 m/z 嵌入 vs 传统离散分箱（Binning）；
   - 中性丢失通道（$\Delta m/z$）的有无；
   - 碰撞能量连续调节参数的引入；
   - ASL 损失 vs 标准 BCE / Focal 损失；
3. **拓展泛化评估**：
   - **Bemis-Murcko 骨架切分（Scaffold Split）**：检验对完全新颖化学骨架分子的嗅觉预测能力；
   - **仪器留出测试（Leave-One-Instrument-Out）**：检验跨仪器平台质谱的迁移稳健性。

---

## 5. 项目核心科研主线聚焦

建议将后续研究与学术表达的核心问题聚焦于：

> **“连续质谱表征与化学感知语义对齐，能否在未见分子结构、跨仪器平台及多碰撞能量条件下，稳定提升气味感官预测精度与可解释性？”**

这一问题切中了计算质谱学与感官化学交叉领域的前沿痛点，能够最大限度凸显本项目在**多源复杂质谱整合、物理连续元数据建模以及化学结构-质谱多模态对齐**上的核心学术价值。
