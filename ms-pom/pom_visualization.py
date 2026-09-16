from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
import umap
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 根据embeddings文件和原始数据，绘制POM


class OdorVisualizationPipeline:
    """可视化"""

    def __init__(self, data_dir="./pom_plots/"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def normalize_coordinates(self, coordinates, method="minmax", feature_range=(0, 1)):
        """
        归一化坐标

        参数:
        coordinates: 降维后的坐标数组 (n_samples, n_dims)
        method: 归一化方法 ['minmax', 'standard', 'robust', 'unit_norm']
        feature_range: 归一化范围 (仅对minmax有效)

        返回:
        normalized_coords: 归一化后的坐标
        """

        if method == "minmax":
            # 最常用：归一化到 [0, 1] 或指定范围
            scaler = MinMaxScaler(feature_range=feature_range)
            normalized_coords = scaler.fit_transform(coordinates)

        elif method == "standard":
            # 标准化：均值为0，标准差为1
            scaler = StandardScaler()
            normalized_coords = scaler.fit_transform(coordinates)

        elif method == "robust":
            # 鲁棒归一化：使用中位数和四分位数，对异常值不敏感
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            normalized_coords = scaler.fit_transform(coordinates)

        elif method == "unit_norm":
            # 单位范数：将每个样本归一化为单位范数
            from sklearn.preprocessing import Normalizer

            scaler = Normalizer(norm="l2")
            normalized_coords = scaler.fit_transform(coordinates)

        elif method == "manual_minmax":
            # 手动实现minmax归一化
            coords_min = coordinates.min(axis=0)
            coords_max = coordinates.max(axis=0)
            normalized_coords = (coordinates - coords_min) / (coords_max - coords_min)
            # 缩放到指定范围
            range_min, range_max = feature_range
            normalized_coords = normalized_coords * (range_max - range_min) + range_min

        else:
            raise ValueError(f"未知的归一化方法: {method}")

        return normalized_coords

    def compute_nmf_coordinates(
        self,
        input_file="./datasets/curated_GS_LF_merged_4983.csv",
        output_file="nmf_coordinates.csv",
        start_col=2,
        end_col=140,
        n_components=3,
    ):
        """
        计算NMF坐标
        参数:
        input_file: 输入CSV文件
        output_file: 输出CSV文件
        start_col: 开始列索引 (0-based)
        end_col: 结束列索引
        n_components: NMF维度
        """
        print("正在计算NMF坐标...")

        # 读取数据 (自动处理表头)
        dtype_dict = {"smiles": str, "descriptors": str}
        for i in range(start_col, end_col):
            dtype_dict[f"col_{i}"] = float

        metadata_df = pd.read_csv(input_file, header=0, dtype=dtype_dict)
        metadata_df["nonStereoSMILES"] = metadata_df["nonStereoSMILES"].astype(str)
        metadata_df["descriptors"] = metadata_df["descriptors"].astype(str)

        # 获取标签名称（从第3列开始）
        label_names = metadata_df.columns[2:].tolist()
        print(metadata_df.head(5))
        # 提取特征列
        feature_columns = metadata_df.iloc[:, start_col:end_col]
        # 检查数据
        print(f"数据形状: {feature_columns.shape}")
        print(f"特征范围: 列 {start_col} 到 {end_col}")

        # NMF分解
        nmf = NMF(n_components=n_components, random_state=42, max_iter=250)
        nmf_coordinates = nmf.fit_transform(feature_columns)
        # normalized_coords = self.normalize_coordinates(nmf_coordinates, method="minmax")
        # 创建输出DataFrame
        output_df = pd.DataFrame(
            nmf_coordinates, columns=[f"NMF_{i}" for i in range(n_components)]
        )

        # 添加标识列 (如果有的话)
        if metadata_df.shape[1] > 0:
            output_df["nonStereoSMILES"] = metadata_df.iloc[:, 0]  # 假设第一列是标识符

        # 保存结果
        output_df.to_csv(os.path.join(self.data_dir, output_file), index=False)
        print(f"NMF坐标已保存至: {os.path.join(self.data_dir, output_file)}")

        return output_df

    def compute_embeddings(
        self,
        input_file="./kfold_logs/20250930_125615_ensemble/ensemble/embeddings/ensemble_embeddings.csv",
        methods=["umap", "pca", "tsne"],
        n_components=2,
    ):
        """
        计算嵌入降维

        参数:
        input_file: 输入CSV文件
        methods: 降维方法列表
        n_components: 目标维度
        """
        print("正在计算嵌入降维...")
        dtype_dict = {"smiles": str, "descriptors": str}
        for i in range(2, 258):
            dtype_dict[f"emb_{i-2}"] = float
        embeddings_df = pd.read_csv(input_file, header=0, dtype=dtype_dict)
        embeddings_df["smiles"] = embeddings_df["smiles"].astype(str)
        embeddings_df["descriptors"] = embeddings_df["descriptors"].astype(str)
        embeddings = embeddings_df.iloc[:, 2:].values.astype("float32")
        print(f"嵌入数据形状: {embeddings.shape}")
        results = {}

        for method in methods:
            print(f"正在计算 {method.upper()}...")

            if method.lower() == "umap":
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_epochs=300,
                    spread=1.0,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric="euclidean",
                    n_jobs=1,
                    repulsion_strength=1.0,  # 排斥强度
                    learning_rate=1.0,  # 学习率
                    densmap=False,  # 是否保持密度
                    verbose=False,
                )
                coordinates = reducer.fit_transform(embeddings)
                output_file = "emb_umap.csv"

            elif method.lower() == "pca":
                reducer = PCA(
                    n_components=n_components,
                    random_state=42,
                    svd_solver="auto",
                    whiten=False,
                    tol=0.0,
                    iterated_power="auto",
                )
                coordinates = reducer.fit_transform(embeddings)
                output_file = "emb_pca.csv"

            elif method.lower() == "tsne":
                reducer = TSNE(
                    n_components=n_components,
                    random_state=42,
                    verbose=0,
                    max_iter=1200,
                    learning_rate="auto",
                    perplexity=30,  # 困惑度（5-50）
                    early_exaggeration=12.0,  # 早期放大
                    n_iter_without_progress=300,  # 提前停止
                    min_grad_norm=1e-7,  # 梯度阈值
                    metric="euclidean",  # 距离度量
                    init="random",  # 初始化方式
                )
                coordinates = reducer.fit_transform(embeddings)
                output_file = "emb_tsne.csv"

            else:
                print(f"未知方法: {method}")
                continue
            normalized_coords = self.normalize_coordinates(coordinates, method="minmax")
            # 创建输出DataFrame
            coord_df = pd.DataFrame(
                normalized_coords,
                columns=[f"{method.upper()}_{i}" for i in range(n_components)],
            )

            # 添加标识符
            if embeddings_df.shape[1] > embeddings.shape[1]:  # 如果有额外的标识列
                coord_df["nonStereoSMILES"] = embeddings_df.iloc[:, 0]

            # 保存结果
            coord_df.to_csv(os.path.join(self.data_dir, output_file), index=False)
            results[method] = coord_df
            print(
                f"{method.upper()}坐标已保存至: {os.path.join(self.data_dir, output_file)}"
            )

        return results

    def load_coordinates(
        self,
        nmf_file="nmf_coordinates.csv",
        pca_file="emb_pca.csv",
        umap_file="emb_umap.csv",
        tsne_file="emb_tsne.csv",
    ):
        """加载所有坐标文件"""
        print("正在加载坐标文件...")

        coordinates = {}

        files = {"nmf": nmf_file, "pca": pca_file, "umap": umap_file, "tsne": tsne_file}

        for name, file in files.items():
            try:
                df = pd.read_csv(os.path.join(self.data_dir, file))
                # 提取坐标列 (排除标识符列)
                coord_cols = [
                    col for col in df.columns if "identifier" not in col.lower()
                ]
                coordinates[name] = df[coord_cols].values
                print(f"已加载 {name}: {coordinates[name].shape}")
            except FileNotFoundError:
                print(f"警告: 未找到文件 {file}")
                coordinates[name] = None

        return coordinates

    def calculate_rgb_from_nmf(self, nmf_coordinates):
        """从NMF坐标计算RGB颜色"""
        if nmf_coordinates is None or nmf_coordinates.shape[1] < 3:
            raise ValueError("NMF坐标需要至少3个维度")

        # 使用前3个维度
        rgb_coords = nmf_coordinates[:, :3]

        # 归一化到[0,1]
        rgb_colors = (rgb_coords - rgb_coords.min(axis=0)) / (
            rgb_coords.max(axis=0) - rgb_coords.min(axis=0)
        )

        return rgb_colors

    def draw_scatter_plots(self, coordinates_dict, figsize=(18, 6), alpha=0.3, size=5):
        """
        创建散点图

        参数:
        coordinates_dict: 包含各种坐标的字典
        """
        if coordinates_dict.get("nmf") is None:
            raise ValueError("需要NMF坐标来计算颜色")

        # 计算RGB颜色
        rgb_colors = self.calculate_rgb_from_nmf(coordinates_dict["nmf"])

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        plot_configs = [
            ("pca", "PCA", axes[0]),
            ("umap", "UMAP", axes[1]),
            ("tsne", "t-SNE", axes[2]),
        ]

        for coord_type, title, ax in plot_configs:
            if coordinates_dict.get(coord_type) is None:
                ax.text(
                    0.5,
                    0.5,
                    f"{title}坐标未找到",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            coords = coordinates_dict[coord_type]

            # 所有点都用彩色
            ax.scatter(coords[:, 0], coords[:, 1], c=rgb_colors, s=size, alpha=alpha)

            ax.set_xlabel(f"{title} Dimension 1")
            ax.set_ylabel(f"{title} Dimension 2")
            ax.set_title(f"Odor Space - {title}\n(Colors from NMF)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.data_dir, "visualization.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def run_full_pipeline(
        self,
        embedding_input_file="./kfold_logs/20250930_125615_ensemble/ensemble/embeddings/ensemble_embeddings.csv",
    ):
        """运行完整管道"""
        print("=" * 50)
        print("开始运行完整可视化管道")
        print("=" * 50)

        # 1. 计算NMF坐标
        nmf_df = self.compute_nmf_coordinates()

        # 2. 计算嵌入降维
        embedding_results = self.compute_embeddings(
            embedding_input_file, methods=["umap", "pca", "tsne"], n_components=2
        )

        # 3. 加载所有坐标
        coordinates = self.load_coordinates()

        # 4. 创建可视化
        self.draw_scatter_plots(coordinates)

        print("管道运行完成!")


def main():
    # 初始化管道
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("pom_plots", timestamp)
    pipeline = OdorVisualizationPipeline(data_dir=log_dir)

    # 方法1: 运行完整管道
    pipeline.run_full_pipeline(
        embedding_input_file="./kfold_logs/20250930_152224_ensemble/ensemble/embeddings/ensemble_embeddings.csv"
    )


if __name__ == "__main__":
    main()
