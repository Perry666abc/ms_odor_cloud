# POM复现与应用优化研究_v2

## 项目简介

本项目实现基于Message Passing Neural Network (MPNN)的图神经网络模型，用于图结构数据的建模与预测。

## 文件结构

```
.
├── configs/              # 配置文件
├── datasets/             # 数据集
├── demo/                 # 演示Demo
├── exp_logs/             # 旧的实验日志
├── kfold_logs/           # 新的实验日志
├── models/               # 模型
├── old/                  # 旧版本代码和记录存档
├── utils/                # 工具函数，包括数据处理、优化器、绘图等
├── __init__.py           # Python包初始化
├── readme.md             # 项目说明
├── requirements.txt      # Python依赖
└── train_mpnn_final.py   # 主训练脚本
```

- `configs/`: 配置文件
- `datasets/`: 数据集
- `demo/`: 演示Demo
- `exp_logs/`: 旧的实验日志
- `kfold_logs/`: 新的实验日志
- `models/ `: 模型
- `old/`: 旧版本代码和记录存档
- `utils/`: 工具函数，包括数据处理、优化器、绘图等
  └── train_mpnn_final.py   # 主训练脚本

## 环境配置指南

1. 新建conda环境并激活:

```
conda create -n mpnn_pom python=3.9.16
conda activate mpnn_pom
```

2. 安装pytorch（以2.4.0为例）:
   前往pytorch官网[https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)获取cuda对应版本安装方式。

```
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

3. 安装版本匹配的torch_geometry:
   安装方式是前往官网[https://data.pyg.org/whl/](https://data.pyg.org/whl/)下载对应版本5个whl文件，放到同一个文件夹中，并cd进入放whl文件的目录安装，以pt24cu124为例。

```
pip install pyg_lib-0.4.0+pt24cu124-cp39-cp39-win_amd64.whl
pip install torch_scatter-2.1.2+pt24cu124-cp39-cp39-win_amd64.whl
pip install torch_cluster-1.6.3+pt24cu124-cp39-cp39-win_amd64.whl
pip install torch_spline_conv-1.2.2+pt24cu124-cp39-cp39-win_amd64.whl 
pip install torch_sparse-0.6.18+pt24cu124-cp39-cp39-win_amd64.whl 
```

上面的安装完成后，安装torch_geometry。

```
pip install torch-geometric 
```

4. 对照requirements.txt安装rdkit、matplotlib等其他需要的库（如果只运行train文件可以先不安装streamlit、faiss等）。

可以用conda：

```
conda install rdkit -c conda-forge
```

或者用pip（选择一种方式即可）：

```
pip install rdkit
```

Tips：

- faiss安装可能会出现错误提示，可以尝试如下语句。
  ```
  conda install -c pytorch faiss-cpu
  ```
- umap库是umap-learn不是umap。
  ```
  pip install umap-learn
  ```

## 使用指南

### 文件夹和环境

1. 在命令行进入code文件夹根目录。
2. 编译器选择激活的conda环境的编译器。

### 训练和预测

1. 默认的config文件是 `config_mini.yaml`，见 `train_mpnn_final.py`的传入的config_dir参数。可以先在较小批次和epoch下运行，检测没有报错后切换其他的设置如 `/configs/ablation_config_9_1.yaml`。
2. **在code根目录下**，运行 `train_mpnn_final.py`，可以通过 `--config_dir /configs/config_mini.yaml`，传入需要的参数（config_mini.yaml替换为需要的配置文件名字）。
3. 新的实验结果保存在 `kfold_log`s中。

### Demo部署

1. 训练完成后，`ensemble_embeddings.csv`和 `saved_models_list.pt`应当已自动保存到demo路径中。如果没有，手动将实验日志目录中 `ensemble`目录中的 `saved_models_list.pt`文件，以及实验日志目录中 `ensemble`目录中的 `embeddings`目录中的 `ensemble_embeddings.csv`文件这两个文件复制到 `demo`文件夹（与 `app_final.py`和 `streamlit_final_app.py`中的常量文件地址匹配即可）。
2. **在code根路径下**，python运行 `/demo/app.py`。
3. 新建终端，**在code根目录下**,运行 `streamlit run /demo/streamlit_app_final.py`命令。

### TensorBoard查看

1. tf文件也在对应实验目录中，将 `<directory_name>`替换为对应路径查看。

```
pip install tensorboard
tensorboard --logdir=<directory_name>
```

1. 可以在右上角选择Embedding Projector中查看效果。
2. 如果需要对比多次试验，可以新建 `tf_logs`文件夹，将多个实验文件夹放在新建的文件夹下。

## 其他

1. 在训练过程中会记录很多日志文件，这可能会导致项目占用大量的空间，在运行前需要确保具有足够的存储空间，可以在运行结束后删除不需要的文件。
2. 设置在第一轮epoch训练以及每100次训练结束后，会记录Embedding，这会花费一段时间，如果希望节省时间和空间可以将 `train_mpnn_final.py`中第903至905行注释掉。
3. 在数据处理阶段，如果出现 `DEPRECATION WARNING: please use MorganGenerator`提示可能与rdkit版本有关，可以忽略。

## 对新的分子测试

1. 使用demo
2. 修改 `datasets\curated_GS_LF_eval_template.csv`，并运行 `eval_ensemble_new.py`。
