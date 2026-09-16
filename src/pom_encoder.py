import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Ensure ms-pom is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MS_POM_DIR = PROJECT_ROOT / "ms-pom"
if str(MS_POM_DIR) not in sys.path:
    sys.path.insert(0, str(MS_POM_DIR))

import models.mpnn_model_final_v1
try:
    torch.serialization.add_safe_globals([models.mpnn_model_final_v1.MPNNModel])
except Exception:
    pass

from rdkit import Chem
from utils.mol_featurizer import AtomFeaturizer, BondFeaturizer
from torch_geometric.data import Data, Batch

class OpenPOMEncoder(nn.Module):
    """
    Open-POM Chemical Structure Encoder based on Lee et al. (Science 2023) / Barsainyan et al. (2023).
    Extracts 256-dimensional Principal Odor Map (POM) perceptual manifold embeddings from SMILES.
    
    Loads pretrained MPNN ensemble weights from ms-pom/demo/saved_models_list.pt.
    Kept frozen during Stage 1 SCENT alignment and Stage 2 downstream fine-tuning.
    """
    def __init__(self, model_path=None, ensemble=True, device=None):
        super().__init__()
        if model_path is None:
            model_path = MS_POM_DIR / "demo" / "saved_models_list.pt"
        self.model_path = Path(model_path)
        self.ensemble = ensemble
        self.target_device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer()
        self.pom_dim = 256
        
        # Load models
        if not self.model_path.exists():
            raise FileNotFoundError(f"Open-POM model checkpoint not found at: {self.model_path}")
            
        loaded = torch.load(self.model_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, list):
            self.models = nn.ModuleList(loaded if ensemble else [loaded[0]])
        else:
            self.models = nn.ModuleList([loaded])
            
        # Freeze all parameters
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
                
        self.to(self.target_device)
        
    def smiles_to_graph(self, smiles):
        """Converts SMILES string to PyTorch Geometric Data object."""
        if not smiles or not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        x = torch.tensor(
            [self.atom_featurizer(atom) for atom in mol.GetAtoms()],
            dtype=torch.float
        )
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            f = self.bond_featurizer(bond)
            edge_index.extend([(i, j), (j, i)])
            edge_attr.extend([f, f])
            
        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 7), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

    @torch.no_grad()
    def encode_graphs(self, graph_list):
        """Encodes a list of PyG Data graphs to (B, 256) POM embeddings."""
        valid_graphs = []
        valid_indices = []
        for idx, g in enumerate(graph_list):
            if g is not None and g.x.size(0) > 0:
                valid_graphs.append(g)
                valid_indices.append(idx)
                
        total_graphs = len(graph_list)
        results = torch.zeros(total_graphs, self.pom_dim, device=self.target_device)
        
        if not valid_graphs:
            return results
            
        batch = Batch.from_data_list(valid_graphs).to(self.target_device)
        
        # Ensemble average of get_embedding
        embs = []
        for m in self.models:
            m.eval()
            emb = m.get_embedding(batch)
            embs.append(emb)
        avg_emb = torch.stack(embs, dim=0).mean(dim=0)  # (N_valid, 256)
        
        for v_idx, orig_idx in enumerate(valid_indices):
            results[orig_idx] = avg_emb[v_idx]
            
        return results

    @torch.no_grad()
    def encode_smiles(self, smiles_input):
        """
        Takes a single SMILES string or a list of SMILES, returns (B, 256) tensor.
        """
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
            single = True
        else:
            smiles_list = list(smiles_input)
            single = False
            
        graphs = [self.smiles_to_graph(s) for s in smiles_list]
        embs = self.encode_graphs(graphs)
        return embs[0] if single else embs

    def build_pom_cache(self, unique_smiles_list, cache_file=None, batch_size=128):
        """
        Pre-computes and caches 256-dim POM representations for a list of SMILES.
        Saves disk file and returns dict: smiles -> torch.Tensor(256).
        """
        if cache_file and os.path.exists(cache_file):
            print(f"Loading existing POM cache from: {cache_file}")
            cache_dict = torch.load(cache_file, map_location='cpu')
            missing = [s for s in unique_smiles_list if s not in cache_dict and isinstance(s, str) and s]
            if not missing:
                return cache_dict
            print(f"Found {len(missing)} new SMILES not in cache. Computing...")
        else:
            cache_dict = {}
            missing = [s for s in unique_smiles_list if isinstance(s, str) and s]

        print(f"Computing Open-POM embeddings for {len(missing)} unique molecules...")
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            embs = self.encode_smiles(chunk).cpu()
            for s, e in zip(chunk, embs):
                cache_dict[s] = e

        if cache_file:
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
            torch.save(cache_dict, cache_file)
            print(f"Saved POM cache ({len(cache_dict)} entries) to {cache_file}")

        return cache_dict

def get_pom_encoder(device=None):
    """Factory helper to obtain a singleton or new OpenPOMEncoder instance."""
    return OpenPOMEncoder(device=device)

if __name__ == '__main__':
    print("Testing OpenPOMEncoder...")
    encoder = OpenPOMEncoder()
    test_smiles = ['O=Cc1ccc(O)c(OC)c1', 'CCO', 'c1ccccc1']
    embs = encoder.encode_smiles(test_smiles)
    print(f"Test SMILES encoded successfully! Shape: {embs.shape}")
    print(f"Vanillin first 5 POM coords: {embs[0, :5].cpu().numpy()}")
