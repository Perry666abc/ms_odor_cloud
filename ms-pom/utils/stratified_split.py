from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from collections import Counter
from sklearn.model_selection import ShuffleSplit


def stratified_split_train_val_test(
    labels, test_size=0.2, val_size=0.2, random_state=42
):
    """
    多标签多类别的分层抽样分割（训练集、验证集、测试集）。

    Args:
        labels (np.ndarray): 多标签矩阵，形状为 [n_samples, n_classes]。
        test_size (float): 测试集比例。
        val_size (float): 验证集比例（相对于剩余的训练集）。
        random_state (int): 随机种子。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (训练索引, 验证索引, 测试索引)。
    """
    # 第一次分割：
    # 第一次分割：训练+验证 vs 测试
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

    # 第二次分割：
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    train_idx, val_idx = next(
        sss.split(np.zeros(len(train_val_idx)), labels[train_val_idx])
    )

    # 映射回原始索引
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    return train_idx, val_idx, test_idx


def stratified_split_train_test(labels, test_size=0.2, random_state=42):
    """
    分层划分训练集和测试集，如果类别样本不足则回退到普通划分。

    Args:
        labels (np.ndarray): 样本标签（多标签）。
        test_size (float): 测试集比例。
        random_state (int): 随机种子。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (训练索引,  测试索引)。
    """
    # 第一次分割：训练+验证 vs 测试
    labels_hashable = [
        tuple(label) if isinstance(label, (list, np.ndarray)) else label
        for label in labels
    ]
    label_counts = Counter(labels_hashable)
    min_samples = min(label_counts.values())

    if min_samples < 2:
        print(
            "Warning: Some classes have fewer than 2 samples. Using non-stratified split."
        )
        sss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    else:
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

    return train_idx, test_idx
