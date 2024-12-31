import torch.nn
from torch.nn import Module
import torch.nn.functional as F
import torch_geometric.utils as gutils
from torch_geometric.nn import SAGEConv, MeanAggregation
from copy import deepcopy

class GraphSAGE(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, device='cpu'):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout
        self.device = device


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGE_Basic(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, device):
        super(GraphSAGE_Basic, self).__init__()
        self.conv1 = MeanAggregation()
        self.conv2 = MeanAggregation()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.device = device


    def forward(self, data_ori):
        data = deepcopy(data_ori)

        data.edge_index, _ = gutils.add_self_loops(data.edge_index, num_nodes=data.x.shape[0])
        all_latest_index_1 = data.edge_index[1]

        h = self.conv1(data.x[all_latest_index_1], data.edge_index[0])

        x = F.relu(self.l1(h))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.l2(self.conv2(x[all_latest_index_1], data.edge_index[0]))
        return F.log_softmax(x, dim=1)