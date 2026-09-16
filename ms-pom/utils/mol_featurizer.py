def one_hot(val, allowable_set):
    """基于允许集合的one-hot编码（自动包含未知维度）"""
    vec = [0] * (len(allowable_set) + 1)
    try:
        idx = list(allowable_set).index(val)
        vec[idx] = 1
    except ValueError:
        vec[-1] = 1  # 未知类型
    return vec


class AtomFeaturizer:
    """修正后的原子特征编码器"""

    def __init__(self):
        # 定义允许集合（保持顺序）
        self.allowable_sets = {
            "atomicNum": list(range(1, 101)),  # 1-100
            "valence": [0, 1, 2, 3, 4, 5, 6],
            "degree": [0, 1, 2, 3, 4, 5],
            "hydrogensCount": [0, 1, 2, 3, 4],
            "formalCharge": [-3, -2, -1, 0, 1, 2, 3],
            "hybridization": ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"],
        }
        # 计算总维度：每个特征维度+1（未知）
        self.dim = sum([len(v) + 1 for v in self.allowable_sets.values()])

    def __call__(self, atom):
        features = []
        # 原子序数 (101D)
        features += one_hot(atom.GetAtomicNum(), self.allowable_sets["atomicNum"])
        # 化合价 (8D)
        features += one_hot(atom.GetTotalValence(), self.allowable_sets["valence"])
        # 连接数 (7D)
        features += one_hot(atom.GetDegree(), self.allowable_sets["degree"])
        # 氢数 (6D)
        features += one_hot(atom.GetTotalNumHs(), self.allowable_sets["hydrogensCount"])
        # 形式电荷 (8D)
        features += one_hot(atom.GetFormalCharge(), self.allowable_sets["formalCharge"])
        # 杂化方式 (7D)
        hybrid = str(atom.GetHybridization())
        features += one_hot(hybrid, self.allowable_sets["hybridization"])

        return features


class BondFeaturizer:
    def __init__(self):
        self.allowable_sets = {
            "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
            "isInRing": [True, False],
        }

    def one_hot_isRing(self, val, allowable_set):
        """基于允许集合的one-hot编码（不包含未知维度）"""
        vec = [0] * (len(allowable_set))
        try:
            idx = list(allowable_set).index(val)
            vec[idx] = 1
        except ValueError:
            vec[-1] = 1  # 未知类型
        return vec

    def __call__(self, bond):
        features = []

        # 1. 键类型 (5D)
        bond_type = str(bond.GetBondType())
        features += one_hot(bond.GetBondType(), self.allowable_sets["bond_type"])

        # 2. 是否在环中 (2D)
        is_ring = bond.IsInRing()
        features += self.one_hot_isRing(
            bond.IsInRing(), self.allowable_sets["isInRing"]
        )

        return features
