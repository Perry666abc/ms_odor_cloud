import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.samplers import TPESampler
from utils.mol_dataset import MolDataset
from models.mpnn_model_final_v1 import MPNNModel
import random
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time
from utils.optim_utils import Lion
from torch_geometric.data import Data, Batch
from datetime import datetime
from utils.plot_global_metrics import plot_global_metrics
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from omegaconf import OmegaConf
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
import pandas as pd
from torch.utils.data import Subset

from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score
from omegaconf import OmegaConf
from sklearn.metrics import precision_recall_curve, auc
import gc


def create_MPNN_model(config):
    """
    创建 MPNN 模型实例
    args:
        config: 配置字典
    returns:
        MPNNModel 实例
    """
    # 在运行代码时，发现一个之前代码中的错误，即config中的参数没在之前的写法中没有被正确地传入，而是直接传入了整个字典，之前误以为第一个元素是字典，之后的元素是被解构，但实际上后面的元素一直都是默认值，没有改变，所以重新运行实验。
    model = MPNNModel(
        mpnn_node_input_dim=config["mpnn_node_input_dim"],
        mpnn_node_hidden_dim=config["mpnn_node_hidden_dim"],
        mpnn_edge_input_dim=config["mpnn_edge_input_dim"],
        mpnn_edge_hidden_dim=config["mpnn_edge_hidden_dim"],
        readout_type=config["readout_type"],
        readout_dim=config["readout_dim"],
        dropout_size=config["dropout_size"],
        bn_layers=config["bn_layers"],
        fnn_num_layers=config["fnn_num_layers"],
        fnn_hidden_dim=config["fnn_hidden_dim"],
        num_tasks=config["num_tasks"],
        pom_dim=config["pom_dim"],
        mpnn_proj_type=config["mpnn_proj_type"],
        use_atten=config["use_atten"],
        use_fp=config["use_fp"],
        fp_dim=config["fp_dim"],
    )
    return model


def args_parser():
    parser = argparse.ArgumentParser(description="MPNN Model Training")
    parser.add_argument(
        "--config_dir",
        default="./configs/config_new_1.yaml",
        type=str,
        help="Path to the configuration file",
    )
    return parser.parse_args()


def stratified_split(dataset, labels, test_size=0.2, random_state=None):
    """分层多标签划分训练集和测试集"""
    # 检查数据一致性
    if len(dataset) != len(labels):
        raise ValueError(
            f"Dataset and labels have inconsistent lengths: {len(dataset)} vs {len(labels)}"
        )
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(dataset, labels))
    return train_idx, test_idx


def create_data_loaders(dataset, indices, batch_size, num_workers, shuffle=False):
    """创建数据加载器"""
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=shuffle,
        # collate_fn=collate_fn,
        # 当设置bn层且batchsize较小时开启
        drop_last=True if batch_size < 32 else False,
    )


def run_k_fold_experiment(
    labels, config, dataset, train_idx, log_dir, label_mapping, pos_weight, kfold_k=5
):
    """
    运行 K-Fold 交叉验证实验
    args:
        labels: 标签数组
        config: 配置字典
        dataset: 数据集对象
        train_idx: 训练集索引
        log_dir: 日志目录
        label_mapping: 标签映射字典
        pos_weight: 正样本权重
    returns:
        fold_results: 每折的验证指标列表
    """
    # 创建日志和TF目录
    exp_log_dir = os.path.join(log_dir, "k_fold_experiment")
    os.makedirs(exp_log_dir, exist_ok=True)

    kf = MultilabelStratifiedKFold(
        n_splits=kfold_k,
        shuffle=True,
        random_state=config["train"]["params"]["random_seed"],
    )

    train_labels = labels[train_idx]
    splits = [
        (train_idx[sub_train_idx], train_idx[sub_test_idx])
        for sub_train_idx, sub_test_idx in kf.split(
            np.zeros(len(train_labels)), train_labels
        )
    ]
    fold_results = []
    fold_models = []

    for fold, (train_fold_idx, val_fold_idx) in enumerate(splits):
        print(f"Fold {fold + 1}/{len(splits)}")
        logging.info(f"Fold {fold + 1}/{len(splits)}")
        fold_log_dir = f"{exp_log_dir}/fold_{fold + 1}"
        kfold_writer = SummaryWriter(f"{fold_log_dir}/tf")
        os.makedirs(fold_log_dir, exist_ok=True)
        print(f"训练集大小: {len(train_fold_idx)}, 验证集大小: {len(val_fold_idx)}")
        logging.info(
            f"训练集大小: {len(train_fold_idx)}, 验证集大小: {len(val_fold_idx)}"
        )
        kfold_model = create_MPNN_model(config["model"]["params"])
        # 创建数据加载器
        train_loader = create_data_loaders(
            dataset,
            train_fold_idx,
            config["train"]["params"]["batch_size"],
            num_workers=config["train"]["params"]["num_workers"],
            shuffle=True,
        )
        val_loader = create_data_loaders(
            dataset,
            val_fold_idx,
            batch_size=config["train"]["params"]["batch_size"],
            num_workers=config["train"]["params"]["num_workers"],
            shuffle=False,
        )
        if config["hpo"]["enable"]:
            # 运行 HPO
            best_params = run_hpo(
                config, train_loader, val_loader, exp_log_dir, label_mapping, pos_weight
            )
            config["model"]["params"].update(best_params["model_params"])
            config["train"]["params"].update(best_params["train_params"])
        if config["train"]["params"]["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                kfold_model.parameters(),
                lr=config["train"]["params"]["learning_rate"],
                weight_decay=config["train"]["params"]["weight_decay"],
            )
        elif config["train"]["params"]["optimizer"] == "lion":
            optimizer = Lion(
                kfold_model.parameters(),
                lr=config["train"]["params"]["learning_rate"],
                weight_decay=config["train"]["params"]["weight_decay"],
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["train"]["params"]["t_0"],
            T_mult=config["train"]["params"]["t_mult"],
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 应用权重损失
        embedding_dir = os.path.join(fold_log_dir, "embeddings")
        last_checkpoint_path = os.path.join(fold_log_dir, "last_checkpoint.pth")
        best_checkpoint_path = os.path.join(fold_log_dir, "best_checkpoint.pth")
        plot_path = os.path.join(fold_log_dir, "plots")
        os.makedirs(embedding_dir, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)

        # 训练模型
        trainer = Trainer(
            kfold_model,
            train_loader,
            val_loader,
            last_checkpoint_path,
            best_checkpoint_path,
            plot_path,
            pos_weight,
            optimizer,
            criterion,
            scheduler,
            config["train"]["params"]["kfold_epochs"],
            config["train"]["params"]["batch_size"],
            config["train"]["params"]["learning_rate"],
            kfold_writer,
            patience=config["train"]["params"]["patience"],
            embedding_dir=embedding_dir,
            label_mapping=label_mapping,
            sigmoid_threshold=config["train"]["params"]["sigmoid_threshold"],
        )
        trainer.train(epochs=config["train"]["params"]["kfold_epochs"])
        # 评估验证集
        kfold_score = evaluate(
            kfold_model,
            test_loader=val_loader,
            embedding_path=embedding_dir,
            plot_path=plot_path,
            writer=kfold_writer,
            label_mapping=label_mapping,
            sigmoid_threshold=config["train"]["params"]["sigmoid_threshold"],
        )
        fold_results.append(kfold_score)
        fold_models.append(kfold_model)
        kfold_writer.close()

    return fold_results, fold_models


def run_hpo(
    config,
    train_loader,
    val_loader,
    exp_log_dir,
    label_mapping,
    pos_weight,
):
    """
    运行超参数搜索
    args:
        config: 配置字典
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        exp_log_dir: 实验日志目录
        label_mapping: 标签映射字典
        pos_weight: 正样本权重
    returns:
        best_params: 最佳超参数字典
        {'model_params': best_model_params, 'train_params': best_train_params}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_hpo_log_dir = os.path.join(exp_log_dir, f"hpo_{timestamp}")
    os.makedirs(all_hpo_log_dir, exist_ok=True)
    # 从 trial 中获取超参数
    model_param_ranges = {
        "mpnn_num_layers": config["hpo"]["params"]["mpnn_num_layers"],
        "bn_layers": config["hpo"]["params"]["bn_layers"],
        "dropout_size": config["hpo"]["params"]["dropout_size"],
        "readout_type": config["hpo"]["params"]["readout_type"],
        "mpnn_proj_type": config["hpo"]["params"]["mpnn_proj_type"],
    }
    train_param_ranges = {
        "learning_rate": config["hpo"]["params"]["learning_rate"],
        "weight_decay": config["hpo"]["params"]["weight_decay"],
        "t_0": config["hpo"]["params"]["t_0"],
        "t_mult": config["hpo"]["params"]["t_mult"],
    }
    hpo_epochs = config["hpo"]["epochs"]

    def objective(trial):
        """Optuna 目标函数"""
        print("开始超参数优化")
        logging.info("开始超参数优化")
        # 输出轮次
        print(f"Trial {trial.number} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(
            f"Trial {trial.number} started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        # 单次训练文件夹
        hpo_log_dir = os.path.join(all_hpo_log_dir, f"trial_{trial.number}")
        model_params = {
            k: (
                trial.suggest_int(k, *v)
                if isinstance(v, list) and k not in ["bn_layers", "readout_type"]
                else trial.suggest_categorical(k, v)
            )
            for k, v in model_param_ranges.items()
        }
        # 动态生成训练参数
        train_params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", *train_param_ranges["learning_rate"], log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *train_param_ranges["weight_decay"]
            ),
            "t_0": trial.suggest_categorical("t_0", train_param_ranges["t_0"]),
            "t_mult": trial.suggest_categorical("t_mult", train_param_ranges["t_mult"]),
        }
        hpo_writer = SummaryWriter(f"{hpo_log_dir}/tf")
        embedding_dir = os.path.join(hpo_log_dir, "embeddings")
        last_checkpoint_path = os.path.join(hpo_log_dir, "last_checkpoint.pth")
        best_checkpoint_path = os.path.join(hpo_log_dir, "best_checkpoint.pth")
        plot_path = os.path.join(hpo_log_dir, "plots")
        os.makedirs(embedding_dir, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)

        # 合并静态模型参数和动态模型参数
        merged_model_params = {**config["model"]["params"], **model_params}
        merged_train_params = {
            **config["train"]["params"],
            **train_params,
        }
        # 创建模型和优化器
        hpo_model = create_MPNN_model(merged_model_params)
        if config["train"]["params"]["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                hpo_model.parameters(),
                lr=merged_train_params["learning_rate"],
                weight_decay=merged_train_params["weight_decay"],
            )
        elif config["train"]["params"]["optimizer"] == "lion":
            optimizer = Lion(
                hpo_model.parameters(),
                lr=merged_train_params["learning_rate"],
                weight_decay=merged_train_params["weight_decay"],
            )

        # 创建调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=merged_train_params["t_0"],
            T_mult=merged_train_params["t_mult"],
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 应用权重损失

        # 记录当前model_params
        hpo_config_trial = {
            "model_params": merged_model_params,
            "train_params": merged_train_params,
        }
        hpo_config_trial_file = os.path.join(hpo_log_dir, "hpo_config.yaml")
        OmegaConf.save(OmegaConf.create(hpo_config_trial), hpo_config_trial_file)
        print(f"模型已保存至{hpo_config_trial_file}")
        logging.info(f"模型已保存至{hpo_config_trial_file}")

        # 创建训练器
        trainer = Trainer(
            hpo_model,
            train_loader,
            val_loader,
            last_checkpoint_path,
            best_checkpoint_path,
            plot_path,
            pos_weight,
            optimizer,
            criterion,
            scheduler,
            hpo_epochs,
            config["train"]["params"]["batch_size"],
            merged_train_params["learning_rate"],
            writer=hpo_writer,
            patience=config["train"]["params"]["patience"],
            embedding_dir=embedding_dir,
            label_mapping=label_mapping,
            sigmoid_threshold=config["train"]["params"]["sigmoid_threshold"],
        )
        # 训练模型
        hpo_score = trainer.train(epochs=hpo_epochs)
        # 评估验证集
        # val_map = evaluate(
        #     hpo_model,
        #     val_loader,
        #     embedding_path=embedding_dir,
        #     plot_path=plot_path,
        #     writer=hpo_writer,
        #     label_mapping=label_mapping,
        #     sigmoid_threshold=config["train"]["params"]["sigmoid_threshold"],
        # )
        hpo_writer.close()
        return hpo_score

    # Optimize
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=config["train"]["params"]["random_seed"]),
    )
    study.optimize(objective, n_trials=config["hpo"]["n_trials"])

    # 返回拆分后的最佳参数
    best_params = study.best_params
    best_model_params = {k: best_params[k] for k in model_param_ranges.keys()}
    best_train_params = {k: best_params[k] for k in train_param_ranges.keys()}
    # 保存hpo模型参数config文件
    hpo_config = {
        "model_params": best_model_params,
        "train_params": best_train_params,
    }
    hpo_config_path = os.path.join(all_hpo_log_dir, "hpo_config.yaml")
    OmegaConf.save(OmegaConf.create(hpo_config), hpo_config_path)
    return {"model_params": best_model_params, "train_params": best_train_params}


def main(args):
    # 加载参数和数据数据集
    config = OmegaConf.load(args.config_dir)
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("kfold_logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        encoding="utf-8",  # 显式指定编码
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    set_seed(42)  # 固定随机种子
    print("配置加载成功")
    logging.info("配置加载成功")
    print(OmegaConf.to_yaml(config))
    logging.info(OmegaConf.to_yaml(config))

    dataset = MolDataset(config["train"]["params"]["data_path"])
    df = pd.read_csv(config["train"]["params"]["data_path"])
    labels = df.iloc[:, 2:].values

    train_idx, test_idx = stratified_split(
        dataset,
        labels,
        test_size=config["train"]["params"]["test_size"],
        random_state=config["train"]["params"]["random_seed"],
    )

    logging.info(
        "加载数据集: %s, 数据集大小: %d",
        config["train"]["params"]["data_path"],
        len(dataset),
    )
    label_mapping = load_label_mapping(config["train"]["params"]["data_path"])
    print("标签加载成功")

    # 根据样本数目设置损失权重, 计算正样本权重 log(1+ratio)）
    pos_counts = df.iloc[:, 2:140].sum(axis=0).values  # 各标签正样本数 shape=(138,)
    neg_counts = len(df) - pos_counts  # 各标签负样本数 shape=(138,)
    imbalance_ratio = neg_counts / (pos_counts + 1e-8)  # 防止除零
    pos_weight = np.log1p(imbalance_ratio)  # log(1+ratio) shape=(138,)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    # 运行 K-Fold 实验
    print(f"正样本权重: {pos_weight}")
    logging.info(f"正样本权重: {pos_weight}")

    if config["train"]["params"]["enable_kfold"] == True:
        fold_models = []
        print("开始训练K-Fold实验，时间：", timestamp)
        logging.info(f"开始训练K-Fold实验，时间：{timestamp}")
        fold_results, fold_models = run_k_fold_experiment(
            labels,
            config,
            dataset,
            train_idx,
            log_dir,
            label_mapping,
            pos_weight,
            kfold_k=config["train"]["params"]["kfold_k"],
        )
        print(f"K-Fold 平均验证指标: {np.mean(fold_results):.6f}")
        logging.info(f"K-Fold 平均验证指标: {np.mean(fold_results):.6f}")

    # 训练最终模型
    final_train_loader = create_data_loaders(
        dataset,
        train_idx,
        config["train"]["params"]["batch_size"],
        num_workers=config["train"]["params"]["num_workers"],
        shuffle=True,
    )
    final_test_loader = create_data_loaders(
        dataset,
        test_idx,
        config["train"]["params"]["batch_size"],
        num_workers=config["train"]["params"]["num_workers"],
        shuffle=False,
    )

    # 训练最终模型（全训练集）
    final_log_dir = os.path.join(log_dir, "final_model")
    os.makedirs(final_log_dir, exist_ok=True)

    last_checkpoint_path = os.path.join(final_log_dir, "last_checkpoint.pth")
    best_checkpoint_path = os.path.join(final_log_dir, "best_checkpoint.pth")
    embedding_path = os.path.join(final_log_dir, "embeddings")
    plot_path = os.path.join(final_log_dir, "plots")
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(embedding_path, exist_ok=True)

    final_writer = SummaryWriter(f"{final_log_dir}/tf")
    # 加载标签映射
    # 打印参数
    print("Final Model Configuration:")
    print(OmegaConf.to_yaml(config))
    logging.info("Final Model Configuration:")
    logging.info(OmegaConf.to_yaml(config))
    final_train_model = create_MPNN_model(config["model"]["params"])

    if config["train"]["params"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            final_train_model.parameters(),
            lr=config["train"]["params"]["learning_rate"],
            weight_decay=config["train"]["params"]["weight_decay"],
        )
    elif config["train"]["params"]["optimizer"] == "lion":
        optimizer = Lion(
            final_train_model.parameters(),
            lr=config["train"]["params"]["learning_rate"],
            weight_decay=config["train"]["params"]["weight_decay"],
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config["train"]["params"]["t_0"],
        T_mult=config["train"]["params"]["t_mult"],
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 应用权重损失
    # 创建训练器
    print("开始训练最终模型")
    logging.info("开始训练最终模型")
    final_trainer = Trainer(
        final_train_model,
        final_train_loader,
        final_test_loader,
        last_checkpoint_path,
        best_checkpoint_path,
        plot_path,
        pos_weight,
        optimizer,
        criterion,
        scheduler,
        epochs=config["train"]["params"]["epochs"],
        batch_size=config["train"]["params"]["batch_size"],
        learning_rate=config["train"]["params"]["learning_rate"],
        writer=final_writer,
        patience=config["train"]["params"]["patience"],
        embedding_dir=embedding_path,
        label_mapping=label_mapping,
        sigmoid_threshold=config["train"]["params"]["sigmoid_threshold"],
    )
    final_trainer.train(epochs=config["train"]["params"]["epochs"])
    print("最终模型训练完成")
    logging.info("最终模型训练完成")
    # 评估测试集
    test_map = evaluate(
        final_train_model,
        test_loader=final_test_loader,
        embedding_path=embedding_path,
        plot_path=plot_path,
        writer=final_writer,
        label_mapping=label_mapping,
        sigmoid_threshold=config["train"]["params"]["sigmoid_threshold"],
    )
    # 输出和保存最佳参数
    # 保存完整config文件
    config_save_path = os.path.join(final_log_dir, "final_config.yaml")
    OmegaConf.save(config, config_save_path)
    print(f"测试集map: {test_map:.6f}")
    logging.info(f"测试集map: {test_map:.6f}")
    final_writer.close()

    # 新增集成评估
    if config["train"]["params"].get("enable_kfold", True) and config["train"][
        "params"
    ].get("enable_ensemble", True):
        print("开始集成评估")
        logging.info("开始集成评估")
        ensemble_dir = os.path.join(log_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        ensemble_models_file = os.path.join(ensemble_dir, "saved_models_list.pt")
        ensemble_models_file_demo = os.path.join("./demo", "saved_models_list.pt")
        # fold_models.append(final_train_model)
        torch.save(fold_models, ensemble_models_file)
        torch.save(fold_models, ensemble_models_file_demo)
        embedding_path = os.path.join(ensemble_dir, "embeddings")
        plot_path = os.path.join(ensemble_dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        os.makedirs(embedding_path, exist_ok=True)
        ensemble_writer = SummaryWriter(f"{ensemble_dir}/tf")
        ensemble_metrics = ensemble_evaluate(
            models=fold_models,
            test_loader=final_test_loader,
            embedding_path=embedding_path,
            plot_path=plot_path,
            writer=ensemble_writer,
            label_mapping=label_mapping,
            sigmoid_threshold=config["train"]["params"]["sigmoid_threshold"],
        )
        print(f"集成训练结果：{ensemble_metrics}")
        logging.info(f"集成训练结果：{ensemble_metrics}")
        ensemble_writer.close()

    print("训练完成，日志保存在:", log_dir)


def set_seed(seed=42):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        last_checkpoint_path,
        best_checkpoint_path,
        plot_path,
        pos_weight,
        optimizer,
        criterion,
        scheduler,
        epochs,
        batch_size,
        learning_rate,
        writer,
        patience,
        embedding_dir,
        label_mapping,
        sigmoid_threshold,
    ):
        """
        初始化训练器
        Args:
            model: MPNN模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            last_checkpoint_path: 上次训练的检查点路径
            best_checkpoint_path: 最佳模型保存路径
            pos_weight: 正样本权重
            optimizer: 优化器
            criterion: 损失函数
            scheduler: 学习率调度器
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            writer: TensorBoard记录器
            patience: 早停耐心值
            embedding_dir: 嵌入向量保存目录
            label_mapping: 标签映射字典
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_loader.dataset
        self.val_dataset = val_loader.dataset
        self.last_checkpoint_path = last_checkpoint_path
        self.best_checkpoint_path = best_checkpoint_path
        self.pos_weight = pos_weight.to(device)  # 将权重移动到设备上
        self.has_last_checkpoint = False
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.plot_path = plot_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.writer = writer
        self.patience = patience
        self.embedding_dir = embedding_dir
        self.label_mapping = label_mapping
        self.sigmoid_threshold = sigmoid_threshold
        # 检测是否存在记录的checkpoints
        if os.path.exists(self.last_checkpoint_path):
            print(f"加载上次训练的模型: {self.last_checkpoint_path}")
            self.model.load_state_dict(
                torch.load(self.last_checkpoint_path, weights_only=True)
            )
            self.has_last_checkpoint = True
            print("上次训练的模型加载成功。")
        else:
            print("没有找到上次训练的模型，开始新的训练。")

        # 设置训练记录
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_precision": [],
            "val_precision": [],
            "train_map": [],
            "val_map": [],
            "lr": [],
        }

        # 训练参数
        params = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "model_params": sum(p.numel() for p in model.parameters()),
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "patience": self.patience,
        }

        logging.info(
            "param: \nepochs:%d \nbatch_size:%d \nlearning_rate:%.6f \nmodel_params:%.6f \noptimizer:%s \ncriterion:%s\n sigmoid_threshold:%.2f",
            self.epochs,
            self.batch_size,
            self.learning_rate,
            sum(p.numel() for p in model.parameters()),
            self.optimizer,
            self.criterion,
            self.sigmoid_threshold,
        )

    def _compute_metrics(self, preds, targets):
        """
        计算多标签分类的评估指标
        Args:
            preds: 模型预测结果 (batch_size, num_classes)
            targets: 真实标签 (batch_size, num_classes)
        """
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        # 通过阈值（如0.5）将概率转换为二进制预测
        binary_preds = (preds > self.sigmoid_threshold).astype(int)
        # 使用 micro平均
        precision = precision_score(
            targets, binary_preds, average="micro", zero_division=0
        )
        recall = recall_score(targets, binary_preds, average="micro")
        f1 = f1_score(targets, binary_preds, average="micro")
        return precision, recall, f1

    def train_epoch(self, epoch):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []
        # tqdm进度条
        pbar = tqdm(self.train_loader, desc="Training")
        for data in pbar:
            # 将数据移动到设备上
            data = data.to(device)
            self.optimizer.zero_grad()

            # 前向传播
            out = self.model(data)
            loss = self.criterion(out, data.y)

            # l1正则化(稀疏化)
            l1_reg = torch.tensor(0.0, device=device)
            for param in self.model.parameters():
                l1_reg += torch.norm(param, p=1)
            loss += 1e-5 * l1_reg  # 调整系数

            # # L2正则化（已经在optimizer中设置weight_decay）
            # l2_reg = torch.tensor(0.0, device=device)
            # for param in self.model.parameters():
            #     l2_reg += torch.norm(param, p=2)  # L2正则化
            # loss += 1e-4 * l2_reg  # 调整系数

            # 反向传播
            # print(loss)
            loss.backward()
            self.optimizer.step()

            # 记录
            total_loss += loss.item()

            # 保存概率用于评估
            all_preds.append(torch.sigmoid(out).detach().cpu())
            all_targets.append(data.y.detach())
            pbar.set_postfix({"loss": loss.item()})

        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        precision, recall, f1 = self._compute_metrics(preds, targets)
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        map_score = average_precision_score(targets, preds, average="macro")
        self.history["train_map"].append(map_score)
        self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

        self.history["train_loss"].append(avg_loss)
        self.history["train_precision"].append(precision)

        # 记录到TensorBoard
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        self.writer.add_scalar("Precision/train", precision, epoch)
        self.writer.add_scalar("Recall/train", recall, epoch)
        self.writer.add_scalar("F1/train", f1, epoch)
        self.writer.add_scalar("mAP/train", map_score, epoch)
        self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
        # 记录梯度范数
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"grad/{name}", param.grad, epoch)

        # 记录权重分布
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param, epoch)
        return avg_loss, map_score

    def validate(self, epoch):
        """验证过程"""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(device)
                out = self.model(data)
                loss = self.criterion(out, data.y)
                total_loss += loss.item()

                # 保存概率用于评估
                all_preds.append(torch.sigmoid(out).detach().cpu())
                all_targets.append(data.y.detach())

        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        precision, recall, f1 = self._compute_metrics(preds, targets)
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        map_score = average_precision_score(targets, preds, average="macro")
        val_auroc = roc_auc_score(targets, preds, average="macro")
        binary_preds = (preds > self.sigmoid_threshold).astype(int)
        macro_pr_auc = cal_prauc_macro(targets, binary_preds)

        # 更新历史记录
        self.history["val_loss"].append(avg_loss)
        self.history["val_precision"].append(precision)
        self.history["val_map"].append(map_score)
        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        self.writer.add_scalar("Precision/val", precision, epoch)
        self.writer.add_scalar("Recall/val", recall, epoch)
        self.writer.add_scalar("F1/val", f1, epoch)
        self.writer.add_scalar("mAP/val", map_score, epoch)
        return avg_loss, map_score, val_auroc, macro_pr_auc

    def train(self, epochs=50):
        # best_map = 0.0
        best_auroc = 0.0
        early_stop_counter = 0
        """完整训练流程"""
        best_loss = float("inf")
        for epoch in range(epochs):
            train_loss, train_map = self.train_epoch(epoch)

            self.scheduler.step(train_loss)
            torch.save(self.model.state_dict(), self.last_checkpoint_path)
            # 保存最佳模型
            # if val_map > best_map:
            # best_map = val_map
            # early_stop_counter = 0
            # torch.save(self.model.state_dict(), self.best_checkpoint_path)
            if epoch % 5 == 0:
                # 每五次计算一次验证集
                val_loss, val_map, val_auroc, val_macro_pr_auc = self.validate(epoch)
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    early_stop_counter = 0
                    torch.save(self.model.state_dict(), self.best_checkpoint_path)
                else:
                    early_stop_counter += 1
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                print(
                    f"Train Precision: {self.history['train_precision'][-1]:.6f} | Val Precision: {self.history['val_precision'][-1]:.6f}"
                )
                # 打印日志
                print(f"Train mAP: {train_map:.6f} | Val mAP: {val_map:.6f}")
                print(f"Learning Rate:{self.history['lr'][-1]:.6f}")
                print(f"Val auroc:{val_auroc:.6f}")
                print(f"Val macro prauc:{val_macro_pr_auc:.6f}")
                logging.info(f"Epoch {epoch+1}/{epochs}")
                logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                logging.info(
                    f"Train Precision: {self.history['train_precision'][-1]:.6f} | Val Precision: {self.history['val_precision'][-1]:.6f}"
                )
                logging.info(f"Train mAP: {train_map:.6f} | Val mAP: {val_map:.6f}")
                logging.info(f"Learning Rate:{self.history['lr'][-1]:.6f}")
                logging.info(f"Val auroc:{val_auroc:.6f}")
                logging.info(f"Val macro prauc:{val_macro_pr_auc:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Train Precision: {self.history['train_precision'][-1]:.6f}")
                # 打印日志
                print(f"Train mAP: {train_map:.6f}")
                print(f"Learning Rate:{self.history['lr'][-1]:.6f}")
                logging.info(f"Epoch {epoch+1}/{epochs}")
                logging.info(f"Train Loss: {train_loss:.4f}")
                logging.info(
                    f"Train Precision: {self.history['train_precision'][-1]:.6f}"
                )
                logging.info(f"Train mAP: {train_map:.6f}")
                logging.info(f"Learning Rate:{self.history['lr'][-1]:.6f}")

            # 记录Embedding
            if epoch % 100 == 0:
                self.record_embeddings(epoch, "train")
                self.record_embeddings(epoch, "val")
            # 早停检查
            if early_stop_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            gc.collect()
            torch.cuda.empty_cache()

        self.plot_training_curve()
        # 返回验证集最终指标
        # return val_map
        return val_auroc

    def plot_training_curve(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))
        try:
            plt.subplot(1, 2, 1)
            plt.plot(self.history["train_loss"], label="Train")
            plt.plot(self.history["val_loss"], label="Validation")
            plt.title("Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(self.history["train_precision"], label="Train")
            plt.plot(self.history["val_precision"], label="Validation")
            plt.title("Precision Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Precision")
            plt.legend()

            plt.tight_layout()
            save_pth = os.path.join(self.plot_path, "training_curve.png")
            plt.savefig(save_pth)
        finally:
            plt.close()

    def record_embeddings(self, epoch, mode="train"):
        """记录嵌入向量"""
        self.model.eval()
        all_embeddings = []
        all_targets = []  # 原始标签
        all_predsIdx = []  # 预测idx
        all_smiles = []  # smiles
        all_descriptors = []  # 描述符

        # dataset = self.train_dataset if mode == "train" else self.val_dataset
        # loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        loader = self.train_loader if mode == "train" else self.val_loader

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = self.model(data)
                probabilities = (
                    torch.sigmoid(out).detach().cpu().numpy()
                )  # Convert to numpy array after sigmoid
                batch_size = probabilities.shape[0]
                # 处理每个样本的预测
                for i in range(batch_size):
                    single_probs = probabilities[i]
                    indexed_prob = list(enumerate(single_probs))
                    pred = [
                        (idx, val)
                        for idx, val in indexed_prob
                        if val > self.sigmoid_threshold
                    ]
                    pred_sorted = sorted(pred, key=lambda x: x[1], reverse=True)
                    all_predsIdx.append(pred_sorted)  # 存储每个样本的预测
                # 处理整个batch的嵌入向量
                embeddings = self.model.get_embedding(data)
                all_embeddings.append(embeddings.cpu())
                all_targets.append(data.y.cpu())
                all_smiles.extend(data.smiles)
                all_descriptors.extend(data.descriptors)

        # 拼接结果
        embeddings_tensor = torch.cat(all_embeddings)
        targets_tensor = torch.cat(all_targets)

        # 确保所有列表长度一致
        total_samples = len(all_smiles)

        # 验证长度一致性
        assert len(all_smiles) == total_samples
        assert len(all_descriptors) == total_samples
        assert (
            len(all_predsIdx) == total_samples
        ), f"all_predsIdx has {len(all_predsIdx)} items, expected {total_samples}"
        assert (
            targets_tensor.shape[0] == total_samples
        ), f"targets_tensor has {targets_tensor.shape[0]} items, expected {total_samples}"

        # 创建metadata
        label_idx_list = []
        for i in range(total_samples):
            ts = targets_tensor[i]
            idx = torch.argmax(ts).item()
            label_idx_list.append(idx)

        label_name_list = []
        for i in range(total_samples):
            pred = all_predsIdx[i]  # 获取当前样本的预测结果
            mol_pred_res = ""
            for idx, val in pred:
                label_name = self.label_mapping.get(str(idx), f"Label {idx}")
                mol_pred_res += f"L{idx}:{label_name}({val:.2f}) "  # 添加概率值便于分析
            label_name_list.append(mol_pred_res.strip())

        # 创建metadata字典 - 确保所有列表长度相同
        metadata = {
            "Smiles": all_smiles,
            "TrueLabelIdx": label_idx_list,
            "Descriptors": all_descriptors,
            "PredLabelNames": label_name_list,
        }
        # 转换为DataFrame并保存为TSV
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(
            os.path.join(self.embedding_dir, "molecule_metadata.tsv"),
            sep="\t",
            index=False,
        )
        # 保存为带分子ID的CSV
        smiles = np.array(all_smiles)
        descriptors = np.array(all_descriptors)
        # embeddings = np.concatenate(all_embeddings, axis=0)
        embeddings = torch.cat(all_embeddings).numpy()
        embedding_df = pd.DataFrame(
            embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])]
        )
        embedding_df.insert(0, "descriptors", descriptors)  # 第一列为分子ID
        embedding_df.insert(0, "smiles", smiles)  # 第一列为分子ID
        embedding_df.to_csv(
            os.path.join(self.embedding_dir, f"graph_embeddings_{mode}.csv"),
            index=False,
        )

        self.writer.add_embedding(
            mat=embeddings_tensor,
            metadata=metadata_df.values.tolist(),  # 所有元数据
            metadata_header=metadata_df.columns.tolist(),  # 所有列名
            tag=f"{mode}_molecule_embeddings",
            global_step=epoch,
        )

        # 添加标签颜色映射（可选，在TensorBoard中也可设置）
        self.writer.add_text(
            "label_color_mapping",
            "0=Red, 1=Blue, 2=Green, 3=Yellow, 4=Purple",
        )

        print(f"Saved {mode} embeddings for epoch {epoch+1}")


# 评估模块
def evaluate(
    model,
    test_loader,
    embedding_path,
    plot_path,
    writer,
    label_mapping,
    sigmoid_threshold,
):
    """模型评估
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        embedding_path: 嵌入向量保存路径
        plot_path: 可视化结果保存路径
        writer: TensorBoard记录器
        label_mapping: 标签映射字典
    """
    model.eval()
    all_preds, all_targets = [], []
    all_embeddings = []
    all_smiles = []
    all_descriptors = []
    all_predsIdx = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            probabilities = torch.sigmoid(out).cpu().numpy()
            batch_size = probabilities.shape[0]
            # 处理每个样本的预测
            for i in range(batch_size):
                single_probs = probabilities[i]
                indexed_prob = list(enumerate(single_probs))
                pred = [
                    (idx, val) for idx, val in indexed_prob if val > sigmoid_threshold
                ]
                pred_sorted = sorted(pred, key=lambda x: x[1], reverse=True)
                all_predsIdx.append(pred_sorted)  # 存储每个样本的预测
            embedding = model.get_embedding(data)  # 获取图嵌入
            all_embeddings.append(embedding.cpu())
            all_smiles.extend(data.smiles)
            all_descriptors.extend(data.descriptors)
            # 保存概率

            all_preds.append(torch.sigmoid(out).detach().cpu())  # 概率
            all_targets.append(data.y.cpu())

    smiles = np.array(all_smiles)
    descriptors = np.array(all_descriptors)
    # embeddings = np.concatenate(all_embeddings, axis=0)
    embeddings = torch.cat(all_embeddings).numpy()
    assert len(smiles) == embeddings.shape[0], "分子数与嵌入数量不匹配"
    assert len(smiles) == len(descriptors), "分子数与标签数不匹配"

    # 保存为带分子ID的CSV
    embedding_df = pd.DataFrame(
        embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])]
    )
    embedding_df.insert(0, "descriptors", descriptors)  # 第一列为分子ID
    embedding_df.insert(0, "smiles", smiles)  # 第一列为分子ID
    embedding_df.to_csv(
        os.path.join(embedding_path, "graph_embeddings.csv"), index=False
    )
    embeddings_tensor = torch.cat(all_embeddings)
    targets_tensor = torch.cat(all_targets)
    # 确保所有列表长度一致
    total_samples = len(all_smiles)

    # 验证长度一致性
    assert len(all_smiles) == total_samples
    assert len(all_descriptors) == total_samples
    assert (
        len(all_predsIdx) == total_samples
    ), f"all_predsIdx has {len(all_predsIdx)} items, expected {total_samples}"
    assert (
        targets_tensor.shape[0] == total_samples
    ), f"targets_tensor has {targets_tensor.shape[0]} items, expected {total_samples}"

    # 创建metadata
    label_idx_list = []
    for i in range(total_samples):
        ts = targets_tensor[i]
        idx = torch.argmax(ts).item()
        label_idx_list.append(idx)

    label_name_list = []
    for i in range(total_samples):
        pred = all_predsIdx[i]  # 获取当前样本的预测结果
        mol_pred_res = ""
        for idx, val in pred:
            label_name = label_mapping.get(str(idx), f"Label {idx}")
            mol_pred_res += f"L{idx}:{label_name}({val:.2f}) "  # 添加概率值便于分析
        label_name_list.append(mol_pred_res.strip())

    # 创建metadata字典 - 确保所有列表长度相同
    metadata = {
        "Smiles": all_smiles,
        "TrueLabelIdx": label_idx_list,
        "Descriptors": all_descriptors,
        "PredLabelNames": label_name_list,
    }
    # 转换为DataFrame并保存为TSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(
        os.path.join(embedding_path, "molecule_metadata.tsv"),
        sep="\t",
        index=False,
    )

    writer.add_embedding(
        mat=embeddings_tensor,
        metadata=metadata_df.values.tolist(),  # 所有元数据
        metadata_header=metadata_df.columns.tolist(),  # 所有列名
        tag=f"test_molecule_embeddings",
    )

    # 添加标签颜色映射（可选，在TensorBoard中也可设置）
    writer.add_text(
        "label_color_mapping",
        "0=Red, 1=Blue, 2=Green, 3=Yellow, 4=Purple",
    )

    # 计算指标
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    binary_preds = (preds > sigmoid_threshold).astype(int)
    precision = precision_score(targets, binary_preds, average="micro", zero_division=0)
    recall = recall_score(targets, binary_preds, average="micro")
    f1 = f1_score(targets, binary_preds, average="micro")
    auroc = roc_auc_score(targets, preds, average="macro")
    map_score = average_precision_score(targets, preds, average="macro")

    writer.add_scalar("Precision/test", precision)
    writer.add_scalar("Recall/test", recall)
    writer.add_scalar("F1/test", f1)
    writer.add_scalar("mAP/test", map_score)

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"mAP:{map_score:.4f}")
    macro_pr_auc = cal_prauc_macro(targets, binary_preds)
    print(f"PRAUC:{macro_pr_auc:.4f}")
    logging.info(
        "\nEvaluation Results: \n Precision: %.6f \n Recall: %.6f \n F1 Score: %.6f \n AUROC: %.6f \n mAP: %.6f \n PRAUC: %.6f",
        precision,
        recall,
        f1,
        auroc,
        map_score,
        macro_pr_auc,
    )
    # 可视化
    print(f"label_mapping.length{len(label_mapping)}")
    logging.info(f"label_mapping.length{len(label_mapping)}")
    print(f"label_mapping{label_mapping}")
    logging.info(f"label_mapping{label_mapping}")
    plot_global_metrics(targets, binary_preds, plot_path, label_mapping)

    # 返回指标
    # return map_score
    return auroc


def ensemble_evaluate(
    models,
    test_loader,
    embedding_path,
    plot_path,
    writer,
    label_mapping,
    sigmoid_threshold,
):
    """集成评估"""
    model = models[0]  # 嵌入生成
    device = next(model.parameters()).device

    # --- 数据收集阶段 ---
    all_embeddings, all_smiles, all_descriptors = [], [], []
    all_targets, all_probs = [], []  # 收集概率值

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # 收集嵌入（使用第一个模型）
            all_embeddings.append(model.get_embedding(data).cpu())
            all_smiles.extend(data.smiles)
            all_descriptors.extend(data.descriptors)
            all_targets.append(data.y.cpu())

            # 集成
            ensemble_probs = torch.stack([torch.sigmoid(m(data)) for m in models]).mean(
                dim=0
            )
            all_probs.append(ensemble_probs.cpu())

    embeddings = torch.cat(all_embeddings).numpy()
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    binary_preds = (probs > sigmoid_threshold).astype(int)  # 基于阈值生成预测

    embedding_df = pd.DataFrame(
        embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])]
    )
    embedding_df.insert(0, "smiles", all_smiles)
    embedding_df.insert(1, "descriptors", all_descriptors)
    embedding_df.to_csv(
        os.path.join(embedding_path, "ensemble_embeddings.csv"), index=False
    )
    embedding_df.to_csv(os.path.join("./demo", "ensemble_embeddings.csv"), index=False)
    all_predsIdx = [
        sorted([(i, p) for i, p in enumerate(sample)], key=lambda x: -x[1])
        for sample in probs
    ]
    # 创建metadata

    metadata = {
        "Smiles": all_smiles,
        "Descriptors": all_descriptors,
        "PredLabelNames": [
            " ".join(
                f"L{idx}:{label_mapping.get(str(idx), f'Label {idx}')}({prob:.2f})"
                for idx, prob in pred
                if prob >= 0.5
            )
            for pred in all_predsIdx
        ],
    }
    # 转换为DataFrame并保存为TSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(
        os.path.join(embedding_path, "molecule_metadata.tsv"),
        sep="\t",
        index=False,
    )

    writer.add_embedding(
        mat=torch.cat(all_embeddings),
        metadata=metadata_df.values.tolist(),
        metadata_header=metadata_df.columns.tolist(),
        tag="ensemble_embeddings",
    )

    # 添加标签颜色映射（可选，在TensorBoard中也可设置）
    writer.add_text(
        "label_color_mapping",
        "0=Red, 1=Blue, 2=Green, 3=Yellow, 4=Purple",
    )

    precision = precision_score(targets, binary_preds, average="micro", zero_division=0)
    recall = recall_score(targets, binary_preds, average="micro")
    f1 = f1_score(targets, binary_preds, average="micro")
    auroc = roc_auc_score(targets, probs, average="macro")
    map_score = average_precision_score(targets, probs, average="macro")

    writer.add_scalar("Precision/test", precision)
    writer.add_scalar("Recall/test", recall)
    writer.add_scalar("F1/test", f1)
    writer.add_scalar("mAP/test", map_score)

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"mAP:{map_score:.4f}")
    macro_pr_auc = cal_prauc_macro(targets, binary_preds)
    print(f"PRAUC:{macro_pr_auc:.4f}")
    logging.info(
        "\n Ensemble Evaluation Results: \n Precision: %.6f \n Recall: %.6f \n F1 Score: %.6f \n AUROC: %.6f \n mAP: %.6f \n PRAUC: %.6f",
        precision,
        recall,
        f1,
        auroc,
        map_score,
        macro_pr_auc,
    )
    # 可视化
    print(f"label_mapping.length{len(label_mapping)}")
    logging.info(f"label_mapping.length{len(label_mapping)}")
    print(f"label_mapping{label_mapping}")
    logging.info(f"label_mapping{label_mapping}")
    plot_global_metrics(targets, binary_preds, plot_path, label_mapping)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "map": map_score,
        "embedding_path": os.path.join(embedding_path, "ensemble_embeddings.csv"),
    }


# 加载标签映射
def load_label_mapping(csv_file):
    """从CSV文件加载标签映射"""
    try:
        df = pd.read_csv(csv_file, header=None, nrows=1)
        labels = df.iloc[0].tolist()[2:]
        return {str(i): label for i, label in enumerate(labels)}
    except Exception as e:
        print(f"无法加载标签映射: {str(e)}")
        return {str(i): f"Label {i}" for i in range(138)}


def cal_prauc_macro(targets, binary_preds):
    n_classes = targets.shape[1]
    pr_aucs = []

    # 对每个类别计算 PR AUC
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(targets[:, i], binary_preds[:, i])
        pr_aucs.append(auc(recall, precision))

    macro_pr_auc = np.mean(pr_aucs)
    return macro_pr_auc


# 自定义批处理函数
def collate_fn(batch):
    # batch = Batch.from_data_list(batch)
    # # 手动处理指纹拼接
    # batch.fingerprint = torch.stack([d.fingerprint for d in batch.to_data_list()])
    # return batch
    pyg_batch = Batch.from_data_list(batch)

    # 收集所有指纹并重构为2D张量
    fingerprints = []
    for data in batch:
        # 确保指纹是1D张量
        fp = data.fingerprint
        if fp.dim() == 0:  # 标量
            fp = fp.unsqueeze(0)
        elif fp.dim() > 1:  # 多维，展平
            fp = fp.flatten()

        # 检查维度并修正
        if len(fp) < 1024:
            # 填充0
            fp = torch.cat([fp, torch.zeros(1024 - len(fp))])
        elif len(fp) > 1024:
            # 截取前1024个
            fp = fp[:1024]

        fingerprints.append(fp)
    print(
        f"重构指纹: 原始形状 {[f.fingerprint.shape for f in batch]} -> 批处理形状 {pyg_batch.fingerprint.shape}"
    )
    # 堆叠为 [batch_size, 1024]
    pyg_batch.fingerprint = torch.stack(fingerprints)

    return pyg_batch


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args_parser())
