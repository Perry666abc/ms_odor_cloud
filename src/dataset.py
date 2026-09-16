import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ODOR_CLASSES_138 = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic', 'apple', 'apricot', 'aromatic', 
    'balsamic', 'banana', 'beefy', 'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'buttery', 
    'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 
    'citrus', 'clean', 'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling', 'cortex', 
    'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty', 'fermented', 'fishy', 
    'floral', 'fresh', 'fruit skin', 'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy', 
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth', 'jasmin', 'juicy', 'ketonic', 
    'lactonic', 'lavender', 'leafy', 'leathery', 'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 
    'metallic', 'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty', 'odorless', 
    'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone', 'peach', 'pear', 'phenolic', 'pine', 
    'pineapple', 'plum', 'popcorn', 'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted', 
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy', 'solvent', 'sour', 'spicy', 
    'strawberry', 'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 
    'vegetable', 'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]

def parse_fast_float_array(val):
    """High performance parser for string representation of float array."""
    if val is None or pd.isna(val) or val == '':
        return np.array([0.0], dtype=np.float32)
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=np.float32)
        return arr if len(arr) > 0 else np.array([0.0], dtype=np.float32)
    if isinstance(val, str):
        v = val.strip('[] \t\n\r')
        if not v:
            return np.array([0.0], dtype=np.float32)
        try:
            return np.fromstring(v, sep=',', dtype=np.float32)
        except Exception:
            return np.array([0.0], dtype=np.float32)
    return np.array([0.0], dtype=np.float32)

def parse_collision_energy(ce_raw, instrument_mode=3, precursor_mz=None):
    """
    Physically grounded Collision Energy (CE) parser.
    Handles:
    - Explicit eV strings: 'NCE=45% 13eV' -> 13.0, '40 eV' -> 40.0
    - Pure numeric values: 20.0, '15' -> 20.0, 15.0
    - NCE percentages: 'NCE=35%' -> calibrated approx eV
    - GC-EI instruments: standard 70 eV electron ionization is an ion source parameter,
      not CID collision cell energy, so it returns (0.0, False) to trigger missing_ce_token.
    """
    if ce_raw is None or pd.isna(ce_raw) or ce_raw == '':
        return 0.0, False

    # GC-EI has no collision cell
    if instrument_mode == 0:
        return 0.0, False

    ce_str = str(ce_raw).strip()

    # Pattern 1: Explicit eV (e.g. 'NCE=45% 13eV', '40 eV', '15.5eV')
    ev_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*ev', ce_str, re.IGNORECASE)
    if ev_match:
        val = float(ev_match.group(1))
        if 0.5 <= val <= 250.0:
            return val, True

    # Pattern 2: NCE percentage (e.g. 'NCE=35%')
    nce_match = re.search(r'nce\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%', ce_str, re.IGNORECASE)
    if nce_match:
        nce_val = float(nce_match.group(1))
        if precursor_mz is not None and precursor_mz > 20.0:
            val = nce_val * (precursor_mz / 500.0) * 0.3
        else:
            val = nce_val * 0.3
        val = max(5.0, min(150.0, val))
        return val, True

    # Pattern 3: Pure numeric or float string
    num_match = re.search(r'^[+-]?([0-9]+(?:\.[0-9]+)?)$', ce_str)
    if num_match:
        val = float(num_match.group(1))
        if 0.5 <= val <= 250.0:
            return val, True

    return 0.0, False

class MSOdorDataset(Dataset):
    """
    Tandem Mass Spectrometry to Odor Dataset with continuous physics-informed representations.
    
    Preserves:
    - Continuous multi-scale m/z and neutral loss (Delta m/z)
    - Square-root abundance scaling (p = 0.5, matching SCENT / EIMS2Vec)
    - Decoupled metadata (PMZ, CE, Instrument Mode)
    - Open-POM 256-dim chemical structure semantics
    """
    def __init__(self, df, max_seq_len=200, sqrt_intensity=True, pom_cache=None, smiles_col=None):
        self.df = df.reset_index(drop=True)
        self.max_seq_len = max_seq_len
        self.sqrt_intensity = sqrt_intensity
        self.odor_classes = ODOR_CLASSES_138
        self.pom_cache = pom_cache or {}
        
        # Determine SMILES column
        if smiles_col and smiles_col in self.df.columns:
            self.smiles_col = smiles_col
        else:
            candidates = ['nonStereoSMILES', 'standardized_SMILES', 'smiles', 'SMILES']
            self.smiles_col = None
            for c in candidates:
                if c in self.df.columns:
                    self.smiles_col = c
                    break

        if self.smiles_col:
            self.smiles_list = self.df[self.smiles_col].fillna('').astype(str).values
        else:
            self.smiles_list = [''] * len(self.df)

        # Ensure all odor columns exist
        for col in self.odor_classes:
            if col not in self.df.columns:
                self.df[col] = 0.0
                
        self.labels = self.df[self.odor_classes].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = self.smiles_list[idx]
        
        # 1. Parse M/Z and Intensities (Continuous)
        mzs = parse_fast_float_array(row.get('mzs', ''))
        intensities = parse_fast_float_array(row.get('intensities', ''))
        
        if len(mzs) == 0 or len(intensities) == 0:
            mzs = np.array([0.0], dtype=np.float32)
            intensities = np.array([0.0], dtype=np.float32)
            
        # Ensure equal length
        min_len = min(len(mzs), len(intensities))
        mzs = mzs[:min_len]
        intensities = intensities[:min_len]
        
        # Sort by m/z
        sort_idx = np.argsort(mzs)
        mzs = mzs[sort_idx]
        intensities = intensities[sort_idx]
        
        # 2. Intensity Normalization & Nonlinear Power Scaling (p=0.5)
        max_int = intensities.max()
        if max_int > 0:
            intensities = intensities / max_int
            if self.sqrt_intensity:
                # Square-root intensity scaling boosts diagnostic low-abundance fragments
                intensities = np.sqrt(intensities)
                
        # 3. Peak Selection / Truncation (Keep top max_seq_len by intensity, preserve m/z ordering)
        if len(mzs) > self.max_seq_len:
            top_idx = np.argsort(intensities)[-self.max_seq_len:]
            top_idx = np.sort(top_idx)  # Re-sort by m/z to preserve mass order
            mzs = mzs[top_idx]
            intensities = intensities[top_idx]
            
        seq_len = len(mzs)
        pad_len = self.max_seq_len - seq_len
        
        # 4. Instrument & Ion Mode
        # 0: GC-EI, 1: LC-ESI-Pos, 2: LC-ESI-Neg, 3: Other/Unknown
        instrument_mode = 3
        src_file = str(row.get('_source_file', '')).lower()
        inst_type = str(row.get('instrument_type', '')).lower()
        ion_mode_raw = str(row.get('ion_mode', '')).strip().upper()
        adduct_raw = str(row.get('adduct', '')).strip()
        
        if 'gc-ms' in src_file or 'mainlib' in src_file or 'ei' in inst_type:
            instrument_mode = 0  # GC-EI
        elif ion_mode_raw in ['P', 'POSITIVE'] or '+' in adduct_raw:
            instrument_mode = 1  # LC-ESI-Pos
        elif ion_mode_raw in ['N', 'NEGATIVE'] or '-' in adduct_raw:
            instrument_mode = 2  # LC-ESI-Neg

        # 5. Precursor m/z (Continuous)
        pmz_raw = row.get('precursor_mz', np.nan)
        has_pmz = False
        precursor_mz = 0.0
        if not pd.isna(pmz_raw):
            try:
                if isinstance(pmz_raw, str):
                    pmz_val = float(pmz_raw.replace('[', '').replace(']', '').strip())
                else:
                    pmz_val = float(pmz_raw)
                if pmz_val > 10.0:  # Physically valid precursor mass
                    precursor_mz = pmz_val
                    has_pmz = True
            except Exception:
                has_pmz = False

        # 6. Collision Energy (CE) Parsing
        ce_raw = row.get('collision_energy', np.nan)
        ce, has_ce = parse_collision_energy(ce_raw, instrument_mode=instrument_mode, precursor_mz=precursor_mz if has_pmz else None)

        # 7. Neutral Loss (Delta m/z, Continuous)
        # Delta m/z = precursor_mz - fragment_mz (positive values represent neutral losses)
        if has_pmz:
            neutral_losses = np.maximum(precursor_mz - mzs, 0.0)
        else:
            neutral_losses = np.zeros_like(mzs)
            
        # 8. Tensor Padding
        pad_mzs = np.pad(mzs, (0, pad_len), 'constant', constant_values=0.0)
        pad_ints = np.pad(intensities, (0, pad_len), 'constant', constant_values=0.0)
        pad_nl = np.pad(neutral_losses, (0, pad_len), 'constant', constant_values=0.0)
        padding_mask = np.pad(np.zeros(seq_len, dtype=bool), (0, pad_len), 'constant', constant_values=True)
        
        # 9. Multi-label Target
        label = self.labels[idx]

        # 10. Open-POM Chemical Semantics Target
        if smiles and smiles in self.pom_cache:
            pom_embedding = self.pom_cache[smiles]
            if isinstance(pom_embedding, np.ndarray):
                pom_embedding = torch.from_numpy(pom_embedding).float()
            has_pom = 1.0
        else:
            pom_embedding = torch.zeros(256, dtype=torch.float32)
            has_pom = 0.0
        
        return {
            'mzs': torch.tensor(pad_mzs, dtype=torch.float32),
            'intensities': torch.tensor(pad_ints, dtype=torch.float32),
            'neutral_losses': torch.tensor(pad_nl, dtype=torch.float32),
            'precursor_mz': torch.tensor([precursor_mz], dtype=torch.float32),
            'has_precursor': torch.tensor([1.0 if has_pmz else 0.0], dtype=torch.float32),
            'ce': torch.tensor([ce], dtype=torch.float32),
            'has_ce': torch.tensor([1.0 if has_ce else 0.0], dtype=torch.float32),
            'instrument_mode': torch.tensor([instrument_mode], dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.float32),
            'smiles': smiles,
            'pom_embedding': pom_embedding,
            'has_pom': torch.tensor([has_pom], dtype=torch.float32)
        }

def compute_class_statistics(train_df, odor_classes=ODOR_CLASSES_138):
    """
    Compute positive class weights and prior probabilities strictly on the training set.
    Prevents any data leakage into validation or test sets.
    """
    labels = train_df[odor_classes].values.astype(np.float32)
    freqs = labels.mean(axis=0)
    freqs = np.clip(freqs, 1e-5, 1.0 - 1e-5)
    
    # Class weights for focal loss (inverse frequency normalized)
    inv_freqs = 1.0 / freqs
    alpha = inv_freqs / inv_freqs.mean()
    
    return torch.tensor(alpha, dtype=torch.float32), torch.tensor(freqs, dtype=torch.float32)

def build_datasets(csv_paths, split_by='molecule', max_seq_len=200, seed=42, sqrt_intensity=True,
                   build_pom_cache=True, pom_cache_path=None):
    """
    Loads all merged CSVs and partitions data into Train, Val, Test sets.
    
    split_by:
        - 'molecule': Group by nonStereoSMILES. Guarantees 0% molecular overlap between Train, Val, Test.
                      Essential for evaluating zero-shot/novel chemical generalization.
        - 'spectrum': Stratified spectrum split across all samples. Suitable for compound retrieval baseline.
    """
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df['_source_file'] = os.path.basename(path)
        dfs.append(df)
        
    if not dfs:
        raise ValueError("No valid CSV files loaded!")
        
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full_df)} total spectra from {len(dfs)} files.")
    
    # Ensure SMILES column exists for molecular grouping
    smiles_col = None
    for candidate in ['nonStereoSMILES', 'standardized_SMILES', 'smiles', 'InChI']:
        if candidate in full_df.columns:
            smiles_col = candidate
            break
            
    if smiles_col is None:
        print("Warning: No SMILES column detected, falling back to spectrum split.")
        split_by = 'spectrum'
        
    np.random.seed(seed)
    n_total = len(full_df)
    
    if split_by == 'molecule':
        print("Executing strict Molecule-Level Split (0% molecular overlap)...")
        # Extract unique molecules
        unique_mols = full_df[smiles_col].dropna().unique()
        np.random.shuffle(unique_mols)
        
        n_mols = len(unique_mols)
        n_train_mols = int(0.70 * n_mols)
        n_val_mols = int(0.15 * n_mols)
        
        train_mols = set(unique_mols[:n_train_mols])
        val_mols = set(unique_mols[n_train_mols:n_train_mols + n_val_mols])
        test_mols = set(unique_mols[n_train_mols + n_val_mols:])
        
        train_mask = full_df[smiles_col].isin(train_mols)
        val_mask = full_df[smiles_col].isin(val_mols)
        test_mask = full_df[smiles_col].isin(test_mols)
        
        # Spectra with missing SMILES default to train set
        missing_mask = full_df[smiles_col].isna()
        if missing_mask.sum() > 0:
            train_mask = train_mask | missing_mask
            
        train_df = full_df[train_mask].reset_index(drop=True)
        val_df = full_df[val_mask].reset_index(drop=True)
        test_df = full_df[test_mask].reset_index(drop=True)
        
        print(f"Unique Molecules -> Train: {len(train_mols)}, Val: {len(val_mols)}, Test: {len(test_mols)}")
        print(f"Spectra Count    -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    else:  # 'spectrum'
        print("Executing Spectrum-Level Split (70/15/15)...")
        indices = np.random.permutation(n_total)
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        test_df = full_df.iloc[test_idx].reset_index(drop=True)
        
        print(f"Spectra Count -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Pre-build Open-POM 256-dim cache if requested
    pom_cache = {}
    if build_pom_cache and smiles_col:
        unique_all_smiles = full_df[smiles_col].dropna().unique().tolist()
        if pom_cache_path is None:
            pom_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pom_cache.pt')

        # 1. 优先直接读取已有缓存文件，完全绕开 torch_scatter 依赖
        if os.path.exists(pom_cache_path):
            try:
                pom_cache = torch.load(pom_cache_path, map_location='cpu', weights_only=False)
                print(f"Loaded Open-POM embedding cache directly from {pom_cache_path} ({len(pom_cache)} entries).")
            except Exception as e:
                print(f"Warning: Failed to load {pom_cache_path}: {e}")
                pom_cache = {}

        # 2. 如果文件不存在，才尝试实时推理生成
        if not pom_cache:
            try:
                from pom_encoder import get_pom_encoder
                encoder = get_pom_encoder()
                pom_cache = encoder.build_pom_cache(unique_all_smiles, cache_file=pom_cache_path)
                print(f"Open-POM embedding cache ready ({len(pom_cache)} entries).")
            except Exception as e:
                print(f"Warning: Could not build POM cache ({e}). Training will proceed without POM cache.")
                pom_cache = {}

    # Compute statistics strictly on train set
    class_weights, prior_probs = compute_class_statistics(train_df, ODOR_CLASSES_138)
    
    train_dataset = MSOdorDataset(train_df, max_seq_len=max_seq_len, sqrt_intensity=sqrt_intensity,
                                  pom_cache=pom_cache, smiles_col=smiles_col)
    val_dataset = MSOdorDataset(val_df, max_seq_len=max_seq_len, sqrt_intensity=sqrt_intensity,
                                pom_cache=pom_cache, smiles_col=smiles_col)
    test_dataset = MSOdorDataset(test_df, max_seq_len=max_seq_len, sqrt_intensity=sqrt_intensity,
                                 pom_cache=pom_cache, smiles_col=smiles_col)
    
    return train_dataset, val_dataset, test_dataset, class_weights, prior_probs, pom_cache
