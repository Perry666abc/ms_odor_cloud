import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add


class EdgeConditionedConv(MessagePassing):
    """
    MPNN层：边缘条件矩阵乘法和GRU更新
    """

    def __init__(self, in_channels, out_channels, edge_dim, dropout_size):
        super(EdgeConditionedConv, self).__init__(aggr="add")  # 使用加法聚合
        # 用于从边特征生成权重矩阵的神经网络
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, in_channels * out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_size),
            nn.Linear(in_channels * out_channels, in_channels * out_channels),
        )
        self.gru = nn.GRUCell(out_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_encoder[0].weight)
        nn.init.xavier_uniform_(self.edge_encoder[3].weight)
        nn.init.zeros_(self.edge_encoder[0].bias)
        nn.init.zeros_(self.edge_encoder[3].bias)

    def forward(self, x, edge_index, edge_attr):
        # 准备消息传递
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # 根据边特征生成权重矩阵
        weight = self.edge_encoder(edge_attr)  # [E, in_channels * out_channels]
        weight = weight.view(
            -1, self.in_channels, self.out_channels
        )  # [E, in_channels, out_channels]

        # 边缘条件矩阵乘法
        x_j = x_j.unsqueeze(1)  # [E, 1, in_channels]
        message = torch.bmm(x_j, weight).squeeze(1)  # [E, out_channels]

        return message

    def update(self, aggr_out, x):
        # GRU更新节点特征
        new_x = self.gru(aggr_out, x)
        return new_x


class BondFusionReadout(nn.Module):
    def __init__(self, node_dim, edge_dim, fusion_dim, dropout_size, readout_hidden_dim=256):
        super(BondFusionReadout, self).__init__()
        self.fusion_dim = fusion_dim

        # 将节点特征和相邻边特征融合为152维嵌入
        self.fusion_net = nn.Sequential(
            nn.Linear(node_dim + edge_dim, readout_hidden_dim),
            nn.BatchNorm1d(readout_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_size),
            nn.Linear(readout_hidden_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_attr, edge_index, batch):
        row, col = edge_index  # row是目标节点，col是源节点
        device = x.device
        row = row.to(device)
        # 将边特征分配到目标节点
        edge_feat_per_node = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
        # 将节点特征与平均边特征拼接
        combined = torch.cat(
            [x, edge_feat_per_node], dim=1
        )  # [N, node_dim + edge_dim]
        # 融合得到152维嵌入
        folded_embedding = self.fusion_net(combined)  # [N, 152]
        # 求和得分子嵌入
        molecule_embedding = scatter_add(folded_embedding, batch, dim=0)  # [B, 152]
        return molecule_embedding


class Predictor(nn.Module):
    """4层全连接，POM为256维，输出138维"""

    def __init__(
        self,
        input_dim=152,
        pom_dim=256,
        num_tasks=138,
        fnn_num_layers=4,
        fnn_hidden_dim=[1024, 512, 384, 256],
        bn_layers=[1, 0, 0, 1],
        dropout_size=0.1,
    ):
        super().__init__()
        self.fc = nn.Sequential()
        for i in range(fnn_num_layers):
            self.fc.add_module(
                f"fc_{i}",
                nn.Sequential(
                    nn.Linear(
                        input_dim if i == 0 else fnn_hidden_dim[i - 1],
                        fnn_hidden_dim[i],
                    ),  # (batch,152)→(batch,1024)
                    (
                        nn.BatchNorm1d(fnn_hidden_dim[i])
                        if bn_layers[i]
                        else nn.Identity()
                    ),
                    (
                        nn.ReLU()
                        if i != fnn_num_layers - 1
                        else nn.LeakyReLU(negative_slope=0.01)
                    ),
                    (
                        nn.Dropout(dropout_size)
                        if i != fnn_num_layers - 1
                        else nn.Identity()
                    ),
                ),
            )
        self.out = nn.Linear(pom_dim, num_tasks)  # (batch,256)→(batch,138)
        self.embedding_cache = None

    def forward(self, x):
        x = self.fc(x)  # (batch,256)
        self.embedding_cache = x  # POM
        x = self.out(x)  # (batch,138)
        return x  # 直接返回logits，训练时用BCEWithLogitsLoss


class MPNNModel(nn.Module):
    """完整模型"""

    def __init__(
        self,
        mpnn_node_input_dim=137,
        mpnn_node_hidden_dim=128,
        mpnn_edge_input_dim=7,
        mpnn_edge_hidden_dim=64,
        mpnn_num_layers=2,
        readout_type="set2set",
        readout_dim=152,
        dropout_size=0.1,
        bn_layers=[1, 0, 0, 1],
        fnn_num_layers=4,
        fnn_hidden_dim=[1024, 512, 392, 256],
        num_tasks=138,
        pom_dim=256,
        mpnn_proj_type="gru",
        use_atten=True,
        use_fp=True,
        fp_dim=1024,
    ):
        """
        args:
            mpnn_node_input_dim:int,MPNN节点输入维度，默认137
            mpnn_node_hidden_dim:int, MPNN节点隐藏维度，默认128
            mpnn_edge_input_dim:int, MPNN边输入维度，默认7
            mpnn_edge_hidden_dim:int, MPNN边隐藏维度，默认64
            mpnn_num_layers: int, MPNN层数，默认2
            readout_type: str, 读出方式，默认'global_add_pool',可选set2set, global_mean_pool, global_max_pool
            readout_dim: int, 读出层输出维度，默认152
            dropout_size: float, dropout率，默认0.1
            bn_layers: list, 全连接层BatchNorm层
            fnn_num_layers: int, 全连接层数，默认4
            fnn_hidden_dim: list, 全连接层隐藏维度，默认[1024,512,384,256]
            num_tasks: int, 任务数，默认138
            mpnn_proj_type:str,投影类型,默认"gru",可选"gru,gin"
            use_atten: bool，是否加入注意力层，默认True
            use_fp: 是否加入指纹信息，默认True
            fp_dim: int，分子指纹维度
        """
        # 解决传入config参数问题
        super().__init__()
        print(mpnn_node_input_dim, mpnn_node_hidden_dim)
        print(mpnn_edge_input_dim, mpnn_edge_hidden_dim)
        self.mpnn_node_input_dim = mpnn_node_input_dim
        self.mpnn_node_hidden_dim = mpnn_node_hidden_dim
        self.mpnn_edge_input_dim = mpnn_edge_input_dim
        self.mpnn_edge_hidden_dim = mpnn_edge_hidden_dim
        self.mpnn_num_layers = mpnn_num_layers
        self.readout_type = readout_type
        self.readout_dim = readout_dim
        self.dropout_size = dropout_size
        self.bn_layers = bn_layers
        self.fnn_num_layers = fnn_num_layers
        self.fnn_hidden_dim = fnn_hidden_dim
        self.num_tasks = num_tasks
        self.pom_dim = pom_dim
        self.mpnn_proj_type = mpnn_proj_type
        self.use_atten = use_atten
        self.use_fp = use_fp
        self.fp_dim = fp_dim
        self.readout_hidden_dim = 256
        # 将node_dim映射到hidden_dim
        self.initial_encoder = nn.Sequential(
            nn.Linear(mpnn_node_input_dim, mpnn_node_hidden_dim),
            nn.BatchNorm1d(mpnn_node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_size),
        )
        # MPNN Layer
        self.convs = nn.ModuleList()
        for _ in range(mpnn_num_layers):
            self.convs.append(
                EdgeConditionedConv(
                    mpnn_node_hidden_dim, mpnn_node_hidden_dim, mpnn_edge_input_dim,dropout_size
                )
            )
        # Readout
        self.readout = BondFusionReadout(
            mpnn_node_hidden_dim, mpnn_edge_input_dim, readout_dim,dropout_size,self.readout_hidden_dim
        )
        # Fully Connected Layer
        self.predictor = Predictor(
            input_dim=readout_dim,
            pom_dim=pom_dim,
            num_tasks=num_tasks,
            fnn_num_layers=fnn_num_layers,
            fnn_hidden_dim=fnn_hidden_dim,
            bn_layers=bn_layers,
            dropout_size=dropout_size,
        )
        # 初始化参数
        self.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        # 初始编码
        x = self.initial_encoder(x)

        # 消息传递层
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))

        # 读出层：生成分子嵌入
        mol_embedding = self.readout(x, edge_attr, edge_index, batch)
        out = self.predictor(mol_embedding)
        return out

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        nn.init.xavier_uniform_(self.initial_encoder[0].weight)
        nn.init.zeros_(self.initial_encoder[0].bias)

    def get_embedding(self, data):
        """获取POM（pom_dim,默认256维）"""
        with torch.no_grad():
            self.forward(data)
            return self.predictor.embedding_cache  # (batch,256)
