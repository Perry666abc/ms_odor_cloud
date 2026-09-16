import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class CustomCELoss(nn.Module):
    """
    参考openpom，自定义损失函数,多分类任务，累加138种分类任务的CrossEntropyLoss
    args:
        class__imbalance_ratio:类别不平衡比例
    """

    def __init__(self, imbalance_ratio=None):
        super().__init__()
        if imbalance_ratio is not None:
            self.balancing_factors = torch.log(1 + torch.tensor(imbalance_ratio))
        else:
            self.balancing_factors = None

    def forward(self, outputs, targets):
        """
        args:
            outputs: torch.Tensor, shape (batch, n_classes) - 每个任务的 logits
            targets: torch.Tensor, shape (batch, n_classes) - 每个任务的真实标签（0或1）
        """
        # 计算二分类概率
        if len(output.shape) == 2:
            output = output.unsqueeze(-1)  # Shape: (batch_size, tasks, 1)

        probabilities = torch.sigmoid(output)  # Apply sigmoid if not already done
        complement_probabilities = 1 - probabilities
        binary_output = torch.stack(
            [complement_probabilities, probabilities], dim=1
        )  # 堆叠为Shape: (batch, 2, n_classes)

        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        ce_loss = ce_loss_fn(binary_output, targets.long())

        # 类别权重
        if self.balancing_factors is not None:
            self.balancing_factors = self.balancing_factors.to(outputs.device)
            balanced_loss = torch.mul(ce_loss, self.balancing_factors)
        else:
            balanced_loss = ce_loss

        loss = balanced_loss.sum(dim=1)

        return loss.unsqueeze(-1).repeat(1, output.shape[-1])


def cal_imbalance_ratio(data_list):
    """
    根据数据集计算类别不平衡比例
    params：
        data_list: List[Any]
    return:
        List[float]
    """
    labels = np.concatenate([data.y for data in data_list], axis=0)
    df = pd.DataFrame(labels)
    class_counts = df.sum().to_numpy()  # np.ndarray
    max_count = max(class_counts)
    class_imbalance_ratio = (max_count / class_counts).tolist()

    return class_imbalance_ratio
