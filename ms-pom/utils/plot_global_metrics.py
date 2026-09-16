import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.patches import Patch


def calculate_metrics(y_true, y_pred, num_tasks=138):
    """计算多标签分类的指标"""
    metrics = []
    for task in range(num_tasks):
        y_true_task = y_true[:, task]
        y_pred_task = y_pred[:, task]

        cm = confusion_matrix(y_true_task, y_pred_task).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_task, y_pred_task, average="binary", zero_division=0
        )

        metrics.append(
            {
                "Task": task,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "TP": cm[1, 1],
                "count": y_true_task.sum(),
            }
        )
    return pd.DataFrame(metrics).fillna(0)


def plot_class_report(metrics_df, plot_path):
    """绘制分类报告指标"""
    plt.figure(figsize=(15, 20))
    try:
        for i, metric in enumerate(["Precision", "Recall", "F1"], 1):
            plt.subplot(3, 1, i)
            sns.barplot(x=metric, y="Task", data=metrics_df, orient="h")
            plt.title(f"{metric} per Task")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "class_report.png"), dpi=300)
    finally:
        plt.close()


def plot_metrics_distribution(metrics_df, plot_path, label_mapping):
    """绘制指标分布"""
    plt.figure(figsize=(20, 15))

    try:
        top_k = 30
        bottom_k = 20
        top_count_classes = metrics_df.nlargest(top_k, "count")
        bottom_count_classes = metrics_df.nsmallest(bottom_k, "count")[::-1]
        selected_classes_with_counts = pd.concat(
            [top_count_classes, bottom_count_classes]
        )
        # 正例数目图
        plt.subplot(2, 2, 1)
        x_pos_count = np.arange(len(selected_classes_with_counts))
        plt.bar(
            x_pos_count,
            selected_classes_with_counts["count"],
            alpha=0.7,
            color="plum",
            label="Number of Positive Samples",
        )
        display_class_names = [
            label_mapping.get(str(idx), f"Label {idx}")
            for idx in selected_classes_with_counts.index
        ]
        plt.xticks(x_pos_count, display_class_names, rotation=45, ha="right")
        plt.ylabel("Number of Positive Samples")
        plt.title(
            f"Positive Sample Counts for Top {top_k} and Bottom {bottom_k} Classes"
        )
        # 添加分隔线区分Top和Bottom
        sep_line = top_k - 0.5
        plt.legend()
        plt.axvline(x=sep_line, color="gray", linestyle="--", alpha=0.7)
        plt.text(
            sep_line + 0.5,
            plt.ylim()[1] * 0.95,
            "Bottom Classes",
            ha="left",
            color="gray",
        )
        plt.text(
            sep_line - 0.5,
            plt.ylim()[1] * 0.95,
            "Top Classes",
            ha="right",
            color="gray",
        )
        # 添加数值标签
        for i, count in enumerate(selected_classes_with_counts["count"]):
            plt.text(
                i,
                count + metrics_df["count"].max() * 0.01,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=45,
                alpha=0.7,
            )

        # 小提琴图+箱线图
        plt.subplot(2, 2, 2)
        colors = ["skyblue", "lightcoral", "bisque"]
        violin_parts = sns.violinplot(
            data=metrics_df[["Precision", "Recall", "F1"]],
            palette=colors,
            inner=None,
            alpha=0.3,
        )
        sns.boxplot(
            data=metrics_df[["Precision", "Recall", "F1"]],
            palette=colors,
            width=0.2,
            boxprops={"alpha": 0.7},
            notch=True,
        )
        plt.title("Classification Metrics Distribution Across 138 Tasks", fontsize=14)
        legend_elements = [
            Patch(facecolor=colors[0], alpha=0.7, label="Precision"),
            Patch(facecolor=colors[1], alpha=0.7, label="Recall"),
            Patch(facecolor=colors[2], alpha=0.7, label="F1"),
        ]
        plt.legend(handles=legend_elements)

        # 预测分布直方图
        # plt.rcParams['axes.edgecolor'] = 'black'
        # plt.rcParams['axes.linewidth'] = 0.8
        plt.subplot(2, 2, 3)
        plt.grid(False)
        plt.tick_params(axis="both", colors="black")
        top_classes = metrics_df.nlargest(top_k, "Precision")
        bottom_classes = metrics_df.nsmallest(bottom_k, "Precision")[::-1]
        selected_classes = pd.concat([top_classes, bottom_classes])
        # 为每一个类别创建索引
        x_pos = np.arange(len(selected_classes))
        bar_width = 0.35
        plt.bar(
            x_pos - bar_width / 2,
            selected_classes["Precision"],
            bar_width,
            alpha=0.7,
            color=colors[0],
            label="Precision",
        )
        plt.bar(
            x_pos + bar_width / 2,
            selected_classes["Recall"],
            bar_width,
            alpha=0.7,
            color=colors[1],
            label="Recall",
        )
        display_names = [
            label_mapping.get(str(idx), f"Label {idx}")
            for idx in selected_classes.index
        ]
        print(f"display_names:{display_names}")
        plt.xticks(x_pos, display_names, rotation=45, ha="right")
        plt.ylabel("Score")
        plt.title(
            f"Precision and Recall for Top {top_k} and Bottom {bottom_k} classes(by Precision)"
        )
        plt.legend()
        sep_line = top_k - 0.5
        plt.axvline(x=sep_line, color="gray", linestyle="--", alpha=0.5)
        plt.text(
            sep_line + 0.5,
            plt.ylim()[1] * 0.95,
            "Bottom Classes",
            ha="left",
            color="gray",
        )
        plt.text(
            sep_line - 0.5,
            plt.ylim()[1] * 0.95,
            "Top Classes",
            ha="right",
            color="gray",
        )
        # 添加数值标签
        for i, (idx, row) in enumerate(selected_classes.iterrows()):
            plt.text(
                i - bar_width / 2,
                row["Precision"] + 0.01,
                f"{row['Precision']:.1f}",
                ha="center",
                va="bottom",
                fontsize=5,
                rotation=45,
                alpha=0.7,
            )
            plt.text(
                i + bar_width / 2,
                row["Recall"] + 0.01,
                f"{row['Recall']:.1f}",
                ha="center",
                va="bottom",
                fontsize=5,
                rotation=45,
                alpha=0.7,
            )

        # 任务指标散点矩阵
        plt.subplot(2, 2, 4)
        plt.grid(False)
        plt.tick_params(axis="both", colors="black")
        sns.scatterplot(
            data=metrics_df,
            x="Precision",
            y="Recall",
            hue="F1",
            palette="YlOrBr",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        plt.title("Precision-Recall Tradeoff")

        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "global_metrics.png"), dpi=300)
    finally:
        plt.close()


def plot_global_metrics(targets, preds, plot_path, label_mapping):
    """全局指标计算和可视化"""
    try:
        metrics_df = calculate_metrics(targets, preds)
        plot_class_report(metrics_df, plot_path)
        plot_metrics_distribution(metrics_df, plot_path, label_mapping)
    except:
        print("未绘制")
