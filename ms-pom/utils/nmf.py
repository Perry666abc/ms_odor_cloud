from sklearn.decomposition import NMF
import matplotlib as plt
import os
import pandas as pd


def nmf_rgb_colors_visualization(odor_df, coordinates, plot_path):
    """
    计算基于NMF的RGB颜色坐标
    params:
    odor_df: m×n的DataFrame, m个化合物, n个气味描述符，用于分解RGB
    coordinates:降维后的embeddings坐标
    plot_path:绘图保存位置
    return:
    rgb_colors: m×3数组, 每个化合物的RGB颜色
    """
    # NMF分解
    model = NMF(n_components=3, init="random", random_state=42, max_iter=200)
    W = model.fit_transform(odor_df.values)  # 化合物坐标矩阵

    # 将W矩阵归一化到[0,1]范围用于RGB
    rgb_colors = (W - W.min(axis=0)) / (W.max(axis=0) - W.min(axis=0))

    # 绘图
    ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=rgb_colors, s=30)

    ax.set_xlabel("NMF Dimension 1 (R)")
    ax.set_ylabel("NMF Dimension 2 (G)")
    ax.set_title("Odor Space Visualization\n(RGB from NMF Coordinates)")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "nmf_pom.png"), dpi=300)
    plt.close()
