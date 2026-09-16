import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data, Dataset
from utils.mol_featurizer import AtomFeaturizer, BondFeaturizer
from rdkit.Chem import rdmolops



class MolDataset(Dataset):
    """分子图数据集处理类"""

    def __init__(self, csv_path):
        """
        参数：
            csv_path: 数据文件路径
        """
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer()
        self.data_list = self._preprocess()

    def _preprocess(self):
        """预处理生成图数据"""
        data_list = []
        for idx in range(len(self.df)):
            smiles = self.df.iloc[idx, 0]  # 第1列为SMILES
            descriptors = self.df.iloc[idx, 1]  # 第二列是所有的标签的描述符
            labels = self.df.iloc[idx, 2:140].values.astype(float)  # 3-140列为标签

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            # 添加氢原子并计算属性（可选，会导致训练过程变慢）
            # mol = Chem.AddHs(mol)

            # 生成原子特征
            atom_features = [self.atom_featurizer(atom) for atom in mol.GetAtoms()]
            x = torch.tensor(atom_features, dtype=torch.float)

            dist_matrix = rdmolops.GetDistanceMatrix(mol)  # 整数矩阵

            # 生成边信息
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                features = self.bond_featurizer(bond)

                d = dist_matrix[i, j]

                edge_index.extend([(i, j), (j, i)])  # 无向图双向边
                edge_attr.extend([features, features])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(labels).unsqueeze(0),  # 添加维度
                smiles=str(smiles),  # 添加分子SMILES
                descriptors=str(descriptors),  # 描述符
            )  # 添加维度 (1, 138)
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_dataset(file_dir):
    print(" getting dataset: ", file_dir)
    df = pd.read_csv(file_dir, index_col=[0])
    print("completed")
    return df
