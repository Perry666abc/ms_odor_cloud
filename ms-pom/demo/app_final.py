from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torch_geometric.data import Data
from pathlib import Path
import sys
from omegaconf import OmegaConf
import torch.serialization

print("Updated sys.path:", sys.path)
sys.path.append(str(Path(__file__).parent.parent))
# 添加项目根目录到模块搜索路径
import models.mpnn_model_final_v1

torch.serialization.add_safe_globals([models.mpnn_model_final_v1.MPNNModel])
from rdkit import Chem
from utils.mol_featurizer import (
    AtomFeaturizer,
    BondFeaturizer,
)  # 使用您现有的特征器
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors

MODEL_PATH = "./demo/saved_models_list.pt"
CONFIG_PATH = "./configs/config_new_1.yaml"

app = FastAPI()
# 初始化特征提取器（与训练时一致）
atom_featurizer = AtomFeaturizer()
bond_featurizer = BondFeaturizer()
config = OmegaConf.load(CONFIG_PATH)
# 加载模型
models = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
model = models[0]
device = next(model.parameters()).device


class PredictionRequest(BaseModel):
    smiles: str


def smiles_to_graph(smiles):
    """与MolDataset._preprocess()保持一致的预处理"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES")

    # 原子特征
    x = torch.tensor(
        [atom_featurizer(atom) for atom in mol.GetAtoms()], dtype=torch.float
    )

    # 边特征
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_featurizer(bond)
        edge_index.extend([(i, j), (j, i)])
        edge_attr.extend([features, features])
    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        smiles=smiles,
    )


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        data = smiles_to_graph(request.smiles)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        with torch.no_grad():
            # 确保输入数据在 CPU 上
            data = data.to(device)
            # 集成
            ensemble_probs = torch.stack([torch.sigmoid(m(data)) for m in models]).mean(
                dim=0
            )
            embeddings = model.get_embedding(data).cpu().numpy()
            # probs = ensemble_probs.cpu().numpy()
            probs = ensemble_probs.flatten().cpu().tolist()

        return {
            "smiles": request.smiles,
            "probs": probs,
            "embeddings": embeddings.tolist(),
            "status": "success",
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
