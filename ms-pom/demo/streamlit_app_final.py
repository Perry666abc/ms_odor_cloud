import umap
import faiss
from io import StringIO
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
import requests
import os

METADATA_FILE = "./datasets/curated_GS_LF_merged_4983.csv"  # 元数据文件
EMBEDDING_FILE = "./demo/ensemble_embeddings.csv"  # 嵌入文件cd pom


def get_molecule_name(smiles):
    """简单的分子名称识别"""
    common_molecules = {
        "CCO": "乙醇",
        "C1=CC=CC=C1": "苯",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": "咖啡因",
        "CC(=O)OC1=CC=CC=C1C(=O)O": "阿司匹林",
    }
    return common_molecules.get(smiles, "未知分子")


def smiles_to_mol(smiles):
    """将SMILES转换为RDKit分子对象"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的SMILES: {smiles}")
    return mol


def mol_to_image(mol, size=(300, 300)):
    """将分子转换为PIL图像"""
    return Draw.MolToImage(mol, size=size)


def predict_from_api(smiles):
    """调用FastAPI后端获取预测结果"""
    response = requests.post(
        "http://localhost:8000/predict", json={"smiles": smiles}, timeout=10
    )
    if response.status_code != 200:
        raise RuntimeError(f"API错误: {response.text}")
    return response.json()


def load_label_mapping(csv_file=METADATA_FILE):
    """从CSV文件加载标签映射"""
    try:
        df = pd.read_csv(csv_file, header=None, nrows=1)
        labels = df.iloc[0].tolist()
        return {str(i): label for i, label in enumerate(labels)}
    except Exception as e:
        print(f"无法加载标签映射: {str(e)}")
        return {str(i): f"Label {i}" for i in range(138)}


# 配置常量
LABEL_MAPPING = load_label_mapping()


# 页面配置
st.set_page_config(layout="wide", page_title="分子气味探索系统")


# 初始化会话状态
def init_session_state():
    """确保所有必要的会话状态变量都已初始化"""
    # 设置默认值
    defaults = {
        "metadata": None,
        "embeddings": None,
        "faiss_index": None,
        "umap_coords": None,
        "umap_reducer": None,
        "current_smiles": "CCO",
        "current_descriptors": "sweet",
        "predictions": [],
        "initialized": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# 加载和准备数据
def prepare_data():
    """加载并准备元数据和嵌入数据"""
    try:
        dtype_dict = {"smiles": str, "descriptors": str}
        for i in range(2, 140):
            dtype_dict[f"col_{i}"] = float

        metadata_df = pd.read_csv(METADATA_FILE, header=0, dtype=dtype_dict)
        metadata_df["nonStereoSMILES"] = metadata_df["nonStereoSMILES"].astype(str)
        metadata_df["descriptors"] = metadata_df["descriptors"].astype(str)

        # 获取标签名称（从第3列开始）
        label_names = metadata_df.columns[2:].tolist()

        st.session_state.metadata = {
            "df": metadata_df,
            "label_names": label_names,
            "smiles": metadata_df["nonStereoSMILES"].tolist(),
            "descriptors": metadata_df["descriptors"].tolist(),
        }
        st.success(f"加载元数据: {len(metadata_df)} 个分子")
    except Exception as e:
        st.error(f"处理元数据失败: {str(e)}")
        return False

    try:
        dtype_dict = {"smiles": str, "descriptors": str}
        for i in range(2, 258):
            dtype_dict[f"emb_{i-2}"] = float

        embeddings_df = pd.read_csv(EMBEDDING_FILE, header=0, dtype=dtype_dict)
        embeddings_df["smiles"] = embeddings_df["smiles"].astype(str)
        embeddings_df["descriptors"] = embeddings_df["descriptors"].astype(str)
        embeddings = embeddings_df.iloc[:, 2:].values.astype("float32")

        # 验证嵌入维度
        if embeddings.shape[1] != 256:
            st.warning(f"嵌入维度不匹配: 预期256, 实际{embeddings.shape[1]}")
            if embeddings.shape[1] > 256:
                embeddings = embeddings[:, :256]
            else:
                padding = np.zeros((embeddings.shape[0], 256 - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])

        st.session_state.embeddings = {
            "df": embeddings_df,
            "array": embeddings,
            "smiles": embeddings_df["smiles"].tolist(),
            "descriptors": embeddings_df["descriptors"].tolist(),
        }
        st.success(f"加载嵌入数据: {len(embeddings_df)} 个嵌入向量")
    except Exception as e:
        st.error(f"处理嵌入数据失败: {str(e)}")
        return False

    return True


# 创建Faiss索引
def create_faiss_index():
    """创建Faiss相似度搜索索引"""
    try:
        if st.session_state.embeddings is None:
            st.error("嵌入数据未加载，无法创建Faiss索引")
            return False

        embeddings = st.session_state.embeddings["array"]
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        st.session_state.faiss_index = index
        st.success("Faiss索引创建成功")
        return True
    except Exception as e:
        st.error(f"创建Faiss索引失败: {str(e)}")
        return False


# 计算UMAP坐标
def compute_umap_coords():
    """计算UMAP降维坐标"""
    try:
        if st.session_state.embeddings is None:
            st.error("嵌入数据未加载，无法计算UMAP坐标")
            return False

        embeddings = st.session_state.embeddings["array"]

        with st.spinner("正在计算分子地图，这可能需要几分钟..."):
            # 使用UMAP降维
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="euclidean",
                random_state=42,
                n_epochs=300,
            )
            coords = reducer.fit_transform(embeddings)

            # 保存到会话状态
            st.session_state.umap_coords = coords
            st.session_state.umap_reducer = reducer  # 保存reducer用于新分子

            # 保存到文件避免重复计算
            np.save(os.path.join("./demo", "umap_coords.npy"), coords)
            st.success("UMAP坐标计算完成")
        return True
    except Exception as e:
        st.error(f"UMAP计算失败: {str(e)}")
        return False


# 初始化应用
def initialize_application():
    """初始化应用程序"""
    init_session_state()
    if st.session_state.initialized:
        return

    # 显示初始化状态
    status_placeholder = st.empty()
    status_placeholder.info("正在初始化应用程序，请稍候...")

    # 加载数据
    if not prepare_data():
        status_placeholder.error("数据加载失败，请检查文件路径和格式")
        return

    # 创建Faiss索引
    if not create_faiss_index():
        status_placeholder.warning("Faiss索引创建失败，相似分子搜索不可用")

    # 计算UMAP坐标
    if not compute_umap_coords():
        status_placeholder.warning("UMAP计算失败，分子地图不可用")

    # 标记为已初始化
    st.session_state.initialized = True
    status_placeholder.success("应用程序初始化完成!")


# 页面标题
st.title("分子气味预测")
# st.markdown("使用图神经网络预测分子的气味特性，并探索分子气味空间")

# 初始化应用
initialize_application()

# 显示状态信息
if st.session_state.metadata:
    st.sidebar.info(f"分子数据库: {len(st.session_state.metadata['df'])} 个分子")
if st.session_state.embeddings:
    st.sidebar.info(
        f"嵌入数据: {len(st.session_state.embeddings['smiles'])} 个嵌入向量"
    )
if st.session_state.faiss_index:
    st.sidebar.success("Faiss索引就绪")
if st.session_state.umap_coords is not None:
    st.sidebar.success("分子地图就绪")
else:
    st.sidebar.warning("分子地图未初始化")
if "current_descriptors" not in st.session_state:
    st.session_state.current_descriptors = ""

# 输入部分
with st.sidebar:
    st.header("分子输入")

    input_method = st.radio(
        "选择输入方式",
        ["SMILES输入", "手绘分子"],
        index=0,
        help="通过SMILES字符串或手绘输入分子结构",
    )

    if input_method == "SMILES输入":
        smiles_input = st.text_input(
            "输入SMILES字符串", "CCO", help="例如: CCO (乙醇), C1=CC=CC=C1 (苯)"
        )
        submit_btn = st.button("分析分子", type="primary")
    else:
        st.info("使用下方画板绘制分子结构")
        try:
            from streamlit_ketcher import st_ketcher

            smile_code = st_ketcher(height=300)
            submit_btn = st.button("分析手绘分子", type="primary")
        except ImportError:
            st.warning("请安装streamlit-ketcher: pip install streamlit-ketcher")
            smile_code = ""
            submit_btn = False

# 处理提交
if submit_btn:
    try:
        # 获取SMILES
        if input_method == "SMILES输入":
            smiles = smiles_input
        else:
            smiles = smile_code

        # 验证SMILES
        mol = smiles_to_mol(smiles)
        st.session_state.current_smiles = smiles

        # 显示加载状态
        with st.spinner("正在分析分子..."):
            # 调用API获取预测结果
            result = predict_from_api(smiles)

            if "error" in result:
                st.error(f"API错误: {result['error']}")
                print(f"API错误: {result['error']}")
                st.session_state.predictions = []
            else:
                # 处理预测结果
                # logits = np.array(result.get("logits", []))
                # probabilities = sigmoid(logits)
                probs = np.array(result.get("probs", []))
                embedding = np.array(result.get("embeddings", [])).reshape(1, -1)

                # 获取预测标签
                preds = [(idx, val) for idx, val in enumerate(probs) if val > 0.5]
                top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:10]

                # 转换为可读标签
                predictions = []
                predict_descriptors = ""
                for idx, val in top_preds:
                    label_name = st.session_state.metadata["label_names"][idx]
                    predictions.append(
                        {"标签": label_name, "置信度": float(val), "原始标签索引": idx}
                    )
                    predict_descriptors += label_name + ";"

                # 存储结果
                st.session_state.predictions = predictions
                st.session_state.current_descriptors = predict_descriptors

                # 查找相似分子
                if st.session_state.faiss_index is not None:
                    try:
                        distances, indices = st.session_state.faiss_index.search(
                            embedding.astype("float32"), 10
                        )

                        # 获取相似分子的SMILES
                        similar_smiles = [
                            st.session_state.embeddings["smiles"][i] for i in indices[0]
                        ]
                        similar_descriptors = [
                            st.session_state.embeddings["descriptors"][i]
                            for i in indices[0]
                        ]

                        # 存储相似分子信息
                        st.session_state.similar_mols = similar_smiles
                        st.session_state.similar_descriptors = similar_descriptors
                    except Exception as e:
                        st.error(f"相似分子搜索失败: {str(e)}")

                # 计算当前分子在UMAP空间的位置
                if st.session_state.umap_reducer is not None:
                    try:
                        current_coord = st.session_state.umap_reducer.transform(
                            embedding
                        )
                        st.session_state.current_coord = current_coord[0]
                    except Exception as e:
                        st.error(f"分子地图定位失败: {str(e)}")

                # 成功提示
                st.success("分析完成!")

    except Exception as e:
        st.error(f"处理错误: {str(e)}")

# 主界面
if st.session_state.current_smiles:
    st.subheader("分子分析结果")

    col1, col2 = st.columns([1, 2])

    with col1:
        # 显示分子结构
        try:
            mol = smiles_to_mol(st.session_state.current_smiles)
            img = mol_to_image(mol)
            st.image(img, caption=st.session_state.current_smiles)
            st.caption(
                f"分子名称: {get_molecule_name(st.session_state.current_smiles)}"
            )
        except Exception as e:
            st.error(f"无法显示分子结构: {str(e)}")

    with col2:
        # 显示预测结果
        st.subheader("气味预测")
        if st.session_state.predictions:
            # 显示为表格
            df = pd.DataFrame(st.session_state.predictions)
            df["置信度"] = df["置信度"].apply(lambda x: f"{x:.2%}")
            st.dataframe(
                df[["标签", "置信度"]], hide_index=True, use_container_width=True
            )

            # 显示为柱状图
            if len(df) > 0:
                chart_data = df.head(5).copy()
                chart_data["置信度"] = (
                    chart_data["置信度"].str.rstrip("%").astype("float") / 100
                )
                chart = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x="置信度:Q",
                        y=alt.Y("标签:N", sort="-x"),
                        color=alt.Color(
                            "置信度:Q", scale=alt.Scale(scheme="redyellowgreen")
                        ),
                    )
                )
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("没有预测结果或置信度低于阈值")

    # 显示相似分子
    if hasattr(st.session_state, "similar_mols") and st.session_state.similar_mols:
        st.subheader("相似分子")
        cols = st.columns(5)
        for i, smiles in enumerate(st.session_state.similar_mols[:10]):
            with cols[i % 5]:
                try:
                    mol = smiles_to_mol(smiles)
                    img = mol_to_image(mol, size=(200, 200))
                    st.image(img, caption=smiles)
                    # st.caption(f"分子名称: {get_molecule_name(smiles)}")
                    # st.caption(f"{smiles}")
                    st.text(f"{st.session_state.similar_descriptors[i]}")
                except Exception:
                    st.warning(f"无法显示分子: {smiles}")

# 显示分子地图
if st.session_state.umap_coords is not None:
    st.subheader("分子气味空间地图")
    # 创建基础数据框（添加descriptors信息）
    plot_data = pd.DataFrame(
        {
            "x": st.session_state.umap_coords[:, 0],
            "y": st.session_state.umap_coords[:, 1],
            "smiles": st.session_state.embeddings["smiles"],
            "descriptors": st.session_state.embeddings["descriptors"],
            "type": "数据库分子",
        }
    )

    # 添加当前分子（如果存在）
    if hasattr(st.session_state, "current_coord") and st.session_state.current_smiles:
        current_df = pd.DataFrame(
            {
                "x": [st.session_state.current_coord[0]],
                "y": [st.session_state.current_coord[1]],
                "smiles": [st.session_state.current_smiles],
                "descriptors": [st.session_state.current_descriptors],
                "type": "当前分子",
            }
        )
        plot_data = pd.concat([plot_data, current_df])

    # 添加相似分子（如果存在）
    if hasattr(st.session_state, "similar_mols") and st.session_state.similar_mols:
        similar_indices = [
            st.session_state.embeddings["smiles"].index(smiles)
            for smiles in st.session_state.similar_mols
            if smiles in st.session_state.embeddings["smiles"]
        ]

        similar_df = pd.DataFrame(
            {
                "x": st.session_state.umap_coords[similar_indices, 0],
                "y": st.session_state.umap_coords[similar_indices, 1],
                "smiles": [
                    st.session_state.embeddings["smiles"][i] for i in similar_indices
                ],
                "descriptors": [
                    st.session_state.embeddings["descriptors"][i]
                    for i in similar_indices
                ],
                "type": "相似分子",
            }
        )
        plot_data = pd.concat([plot_data, similar_df])

    # 创建Altair图表（添加更多tooltip信息）
    chart = (
        alt.Chart(plot_data)
        .mark_point(filled=True, size=60)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color(
                "type:N",
                scale=alt.Scale(
                    domain=["数据库分子", "当前分子", "相似分子"],
                    range=["#CCCCCC", "#FF0000", "#0000FF"],
                ),
            ),
            tooltip=["smiles:N", "descriptors:N", "type:N"],  # 显示更多信息
        )
        .properties(width=800, height=500)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

else:
    st.warning("分子地图未初始化，无法显示")

# 页脚
st.divider()
