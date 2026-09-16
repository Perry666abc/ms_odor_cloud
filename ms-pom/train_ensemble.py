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
import sys
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score
from omegaconf import OmegaConf
from sklearn.metrics import precision_recall_curve, auc
import gc
from pathlib import Path
from train_mpnn_final import (
    ensemble_evaluate,
    load_label_mapping,
    cal_prauc_macro,
    stratified_split,
    create_data_loaders,
)


def args_parser():
    parser = argparse.ArgumentParser(description="MPNN Model Training")
    parser.add_argument(
        "--config_dir",
        default="./configs/config_new_2.yaml",
        type=str,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--file_path",
        default="./demo/saved_models_list.pt",
        type=str,
        help="Path to the configuration file",
    )
    return parser.parse_args()


def set_seed(seed=42):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    # 加载参数和数据数据集
    config = OmegaConf.load(args.config_dir)
    file_path = args.file_path
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_ensemble")
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
    print(f"filePath:{file_path}")
    logging.info(f"filePath:{file_path}")

    dataset = MolDataset(config["train"]["params"]["data_path"])
    df = pd.read_csv(config["train"]["params"]["data_path"])
    labels = df.iloc[:, 2:].values

    train_idx, test_idx = stratified_split(
        dataset,
        labels,
        test_size=config["train"]["params"]["test_size"],
        random_state=config["train"]["params"]["random_seed"],
    )
    # final_train_loader = create_data_loaders(
    #     dataset,
    #     train_idx,
    #     config["train"]["params"]["batch_size"],
    #     num_workers=config["train"]["params"]["num_workers"],
    #     shuffle=True,
    # )
    # final_test_loader = create_data_loaders(
    #     dataset,
    #     test_idx,
    #     config["train"]["params"]["batch_size"],
    #     num_workers=config["train"]["params"]["num_workers"],
    #     shuffle=False,
    # )
    final_test_loader = DataLoader(
        dataset,
        batch_size=config["train"]["params"]["batch_size"],
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        # collate_fn=collate_fn,
        # 当设置bn层且batchsize较小时开启
        drop_last=True if config["train"]["params"]["batch_size"] < 32 else False,
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

    # 新增集成评估
    if config["train"]["params"].get("enable_kfold", True) and config["train"][
        "params"
    ].get("enable_ensemble", True):
        print("开始集成评估")
        logging.info("开始集成评估")
        ensemble_dir = os.path.join(log_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        embedding_path = os.path.join(ensemble_dir, "embeddings")
        plot_path = os.path.join(ensemble_dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        os.makedirs(embedding_path, exist_ok=True)
        ensemble_writer = SummaryWriter(f"{ensemble_dir}/tf")
        fold_models = torch.load(file_path, weights_only=False)
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


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args_parser())
