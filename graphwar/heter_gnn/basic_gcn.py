from torch.nn import Linear
import torch
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.utils import add_self_loops, contains_self_loops
from .utils import remove_edges, gcn_norm, edge_index_to_sparse_tensor_adj
import torch_geometric.utils as gutils

import torch.nn.functional as F
from sklearn.neighbors._unsupervised import NearestNeighbors
from copy import deepcopy


# GCN!!!!!!!!!!!!!!!!!!!!!!!!!!
# high-order neighbors, potential neighbors, ego-neighbor separation, inter-layer combination

class GCN0000(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN0000, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden)
        self.gcn2 = GCNConv(num_hidden, num_classes)

        self.lin = Linear(num_classes, num_classes)
        print('currently is 0000')



    def forward(self, data):

        h1 = self.gcn1(data.x, data.edge_index)

        h1 = F.relu(F.dropout(h1, p=self.dropout, training=self.training))

        h2 = self.gcn2(h1, data.edge_index)


        # R2 = F.relu(F.dropout(h2, p=self.dropout, training=self.training))
        R2 = F.dropout(h2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN1000(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN1000, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden)
        self.gcn2 = GCNConv(2 * num_hidden, num_classes)

        self.lin = Linear(num_classes * 2, num_classes)

        print('currently is 1000')

    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index, data.x.shape[0])


        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.k_hop_edge_index[0])
        R1 = torch.cat([h1, h1_2], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.k_hop_edge_index[0])

        R2 = torch.cat([h2, h2_2], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN0100(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN0100, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden)
        self.gcn2 = GCNConv(2 * num_hidden, num_classes)

        self.lin = Linear(num_classes * 2, num_classes)

        number_knn_neighbor = 5

        knn = NearestNeighbors(n_neighbors=number_knn_neighbor, metric='cosine', algorithm='brute', n_jobs=1)
        knn.fit(deepcopy(data.x).cpu())

        knn_neighbors = torch.tensor(knn.kneighbors(return_distance=False))
        edge_index_knn = torch.empty((2, number_knn_neighbor*data.x.shape[0]), dtype=torch.int32)



        edge_index_knn[0, :] = torch.arange(edge_index_knn.shape[1])/number_knn_neighbor
        edge_index_knn[1, :] = knn_neighbors.flatten()
        self.edge_index_knn = edge_index_knn.long()
        self.edge_index_knn = self.edge_index_knn.to(device)
        print('currently is 0100')



    def forward(self, data):




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.edge_index_knn)
        R1 = torch.cat([h1, h1_2], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.edge_index_knn)

        R2 = torch.cat([h2, h2_2], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN0010(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN0010, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(2*num_hidden, num_classes, add_self_loops=False)

        self.lin = Linear(num_classes * 2, num_classes)
        print('currently is 0010')



    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)

        temp_edge_index, _ = gutils.add_self_loops(data.edge_index)
        identify_edge_index = remove_edges(temp_edge_index, data.edge_index).to(self.device)

        data.edge_index, _ = gutils.remove_self_loops(data.edge_index)


        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, identify_edge_index)

        R1 = torch.cat([h1, h1_2], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, identify_edge_index)


        R2 = torch.cat([h2, h2_2], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN0001(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN0001, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden)
        self.gcn2 = GCNConv(num_hidden, num_classes)

        self.lin = Linear(num_hidden+num_classes, num_classes)
        print('currently is 0001')



    def forward(self, data):

        h1 = self.gcn1(data.x, data.edge_index)

        R1 = h1

        h1 = F.relu(F.dropout(h1, p=self.dropout, training=self.training))

        h2 = self.gcn2(h1, data.edge_index)


        R2 = torch.cat([R1, h2], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN1100(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN1100, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden)
        self.gcn2 = GCNConv(3 * num_hidden, num_classes)

        self.lin = Linear(num_classes * 3, num_classes)

        number_knn_neighbor = 5

        knn = NearestNeighbors(n_neighbors=number_knn_neighbor, metric='cosine', algorithm='brute', n_jobs=1)
        knn.fit(deepcopy(data.x).cpu())

        knn_neighbors = torch.tensor(knn.kneighbors(return_distance=False))
        edge_index_knn = torch.empty((2, number_knn_neighbor*data.x.shape[0]), dtype=torch.int32)



        edge_index_knn[0, :] = torch.arange(edge_index_knn.shape[1])/number_knn_neighbor
        edge_index_knn[1, :] = knn_neighbors.flatten()
        self.edge_index_knn = edge_index_knn.long()
        self.edge_index_knn = self.edge_index_knn.to(device)
        print('currently is 1100')



    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index, data.x.shape[0])


        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.k_hop_edge_index[0])
        h1_3 = self.gcn1(data.x, self.edge_index_knn)
        R1 = torch.cat([h1, h1_2, h1_3], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.k_hop_edge_index[0])
        h2_3 = self.gcn2(R1, self.edge_index_knn)

        R2 = torch.cat([h2, h2_2, h2_3], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN1010(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN1010, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(3 * num_hidden, num_classes, add_self_loops=False)

        self.lin = Linear(num_classes * 3, num_classes)
        print('currently is 1010')



    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        temp_edge_index, _ = gutils.add_self_loops(data.edge_index)
        identify_edge_index = remove_edges(temp_edge_index, data.edge_index).to(self.device)
        data.edge_index, _ = gutils.remove_self_loops(data.edge_index)


        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index, data.x.shape[0])





        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.k_hop_edge_index[0])
        h1_3 = self.gcn1(data.x, identify_edge_index)

        R1 = torch.cat([h1, h1_2, h1_3], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.k_hop_edge_index[0])
        h2_3 = self.gcn2(R1, identify_edge_index)

        R2 = torch.cat([h2, h2_2, h2_3], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN1001(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN1001, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden)
        self.gcn2 = GCNConv(2 * num_hidden, num_classes)

        self.lin = Linear(2 * num_hidden + num_classes * 2, num_classes)
        print('currently is 1001')




    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index, data.x.shape[0])


        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.k_hop_edge_index[0])
        R1 = torch.cat([h1, h1_2], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.k_hop_edge_index[0])

        R2 = torch.cat([R1, h2, h2_2], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)


        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN0110(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN0110, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(3 * num_hidden, num_classes, add_self_loops=False)

        self.lin = Linear(num_classes * 3, num_classes)

        number_knn_neighbor = 5

        knn = NearestNeighbors(n_neighbors=number_knn_neighbor, metric='cosine', algorithm='brute', n_jobs=1)
        knn.fit(deepcopy(data.x).cpu())

        knn_neighbors = torch.tensor(knn.kneighbors(return_distance=False))
        edge_index_knn = torch.empty((2, number_knn_neighbor*data.x.shape[0]), dtype=torch.int32)



        edge_index_knn[0, :] = torch.arange(edge_index_knn.shape[1])/number_knn_neighbor
        edge_index_knn[1, :] = knn_neighbors.flatten()
        self.edge_index_knn = edge_index_knn.long()
        self.edge_index_knn = self.edge_index_knn.to(device)
        print('currently is 0110')



    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)

        temp_edge_index, _ = gutils.add_self_loops(data.edge_index)
        identify_edge_index = remove_edges(temp_edge_index, data.edge_index).to(self.device)

        data.edge_index, _ = gutils.remove_self_loops(data.edge_index)


        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.edge_index_knn)
        h1_3 = self.gcn1(data.x, identify_edge_index)
        R1 = torch.cat([h1, h1_2, h1_3], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.edge_index_knn)
        h2_3 = self.gcn2(R1, identify_edge_index)

        R2 = torch.cat([h2, h2_2, h2_3], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN0101(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN0101, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden)
        self.gcn2 = GCNConv(2 * num_hidden, num_classes)

        self.lin = Linear(2 * num_hidden + num_classes * 2, num_classes)

        number_knn_neighbor = 5

        knn = NearestNeighbors(n_neighbors=number_knn_neighbor, metric='cosine', algorithm='brute', n_jobs=1)
        knn.fit(deepcopy(data.x).cpu())

        knn_neighbors = torch.tensor(knn.kneighbors(return_distance=False))
        edge_index_knn = torch.empty((2, number_knn_neighbor*data.x.shape[0]), dtype=torch.int32)



        edge_index_knn[0, :] = torch.arange(edge_index_knn.shape[1])/number_knn_neighbor
        edge_index_knn[1, :] = knn_neighbors.flatten()
        self.edge_index_knn = edge_index_knn.long()
        self.edge_index_knn = self.edge_index_knn.to(device)
        print('currently is 0101')



    def forward(self, data):




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.edge_index_knn)
        R1 = torch.cat([h1, h1_2], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.edge_index_knn)

        R2 = torch.cat([R1, h2, h2_2], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN0011(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN0011, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(2*num_hidden, num_classes, add_self_loops=False)

        self.lin = Linear(2*num_hidden + num_classes * 2, num_classes)
        print('currently is 0011')




    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)

        temp_edge_index, _ = gutils.add_self_loops(data.edge_index)
        identify_edge_index = remove_edges(temp_edge_index, data.edge_index).to(self.device)

        data.edge_index, _ = gutils.remove_self_loops(data.edge_index)


        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, identify_edge_index)

        R1 = torch.cat([h1, h1_2], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, identify_edge_index)


        R2 = torch.cat([R1, h2, h2_2], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN1110(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN1110, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(4 * num_hidden, num_classes, add_self_loops=False)

        self.lin = Linear(num_classes * 4, num_classes)

        number_knn_neighbor = 5

        knn = NearestNeighbors(n_neighbors=number_knn_neighbor, metric='cosine', algorithm='brute', n_jobs=1)
        knn.fit(deepcopy(data.x).cpu())

        knn_neighbors = torch.tensor(knn.kneighbors(return_distance=False))
        edge_index_knn = torch.empty((2, number_knn_neighbor*data.x.shape[0]), dtype=torch.int32)



        edge_index_knn[0, :] = torch.arange(edge_index_knn.shape[1])/number_knn_neighbor
        edge_index_knn[1, :] = knn_neighbors.flatten()
        self.edge_index_knn = edge_index_knn.long()
        self.edge_index_knn = self.edge_index_knn.to(device)
        print('currently is 1110')



    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)

        temp_edge_index, _ = gutils.add_self_loops(data.edge_index)
        identify_edge_index = remove_edges(temp_edge_index, data.edge_index).to(self.device)

        data.edge_index, _ = gutils.remove_self_loops(data.edge_index)




        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index, data.x.shape[0])


        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.k_hop_edge_index[0])
        h1_3 = self.gcn1(data.x, self.edge_index_knn)
        h1_4 = self.gcn1(data.x, identify_edge_index)
        R1 = torch.cat([h1, h1_2, h1_3, h1_4], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.k_hop_edge_index[0])
        h2_3 = self.gcn2(R1, self.edge_index_knn)
        h2_4 = self.gcn2(R1, identify_edge_index)

        R2 = torch.cat([h2, h2_2, h2_3, h2_4], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN1101(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN1101, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden)
        self.gcn2 = GCNConv(3 * num_hidden, num_classes)

        self.lin = Linear(3 * num_hidden + num_classes * 3, num_classes)

        number_knn_neighbor = 5

        knn = NearestNeighbors(n_neighbors=number_knn_neighbor, metric='cosine', algorithm='brute', n_jobs=1)
        knn.fit(deepcopy(data.x).cpu())

        knn_neighbors = torch.tensor(knn.kneighbors(return_distance=False))
        edge_index_knn = torch.empty((2, number_knn_neighbor*data.x.shape[0]), dtype=torch.int32)



        edge_index_knn[0, :] = torch.arange(edge_index_knn.shape[1])/number_knn_neighbor
        edge_index_knn[1, :] = knn_neighbors.flatten()
        self.edge_index_knn = edge_index_knn.long()
        self.edge_index_knn = self.edge_index_knn.to(device)
        print('currently is 1101')



    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index, data.x.shape[0])


        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.k_hop_edge_index[0])
        h1_3 = self.gcn1(data.x, self.edge_index_knn)
        R1 = torch.cat([h1, h1_2, h1_3], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.k_hop_edge_index[0])
        h2_3 = self.gcn2(R1, self.edge_index_knn)

        R2 = torch.cat([R1, h2, h2_2, h2_3], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN1011(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN1011, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(3 * num_hidden, num_classes, add_self_loops=False)

        self.lin = Linear(3 * num_hidden + num_classes * 3, num_classes)
        print('currently is 1011')



    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        temp_edge_index, _ = gutils.add_self_loops(data.edge_index)
        identify_edge_index = remove_edges(temp_edge_index, data.edge_index).to(self.device)
        data.edge_index, _ = gutils.remove_self_loops(data.edge_index)


        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index, data.x.shape[0])





        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.k_hop_edge_index[0])
        h1_3 = self.gcn1(data.x, identify_edge_index)

        R1 = torch.cat([h1, h1_2, h1_3], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.k_hop_edge_index[0])
        h2_3 = self.gcn2(R1, identify_edge_index)

        R2 = torch.cat([R1, h2, h2_2, h2_3], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN0111(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN0111, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(3 * num_hidden, num_classes, add_self_loops=False)

        self.lin = Linear(3 * num_hidden + num_classes * 3, num_classes)

        number_knn_neighbor = 5

        knn = NearestNeighbors(n_neighbors=number_knn_neighbor, metric='cosine', algorithm='brute', n_jobs=1)
        knn.fit(deepcopy(data.x).cpu())

        knn_neighbors = torch.tensor(knn.kneighbors(return_distance=False))
        edge_index_knn = torch.empty((2, number_knn_neighbor*data.x.shape[0]), dtype=torch.int32)



        edge_index_knn[0, :] = torch.arange(edge_index_knn.shape[1])/number_knn_neighbor
        edge_index_knn[1, :] = knn_neighbors.flatten()
        self.edge_index_knn = edge_index_knn.long()
        self.edge_index_knn = self.edge_index_knn.to(device)
        print('currently is 0111')



    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)

        temp_edge_index, _ = gutils.add_self_loops(data.edge_index)
        identify_edge_index = remove_edges(temp_edge_index, data.edge_index).to(self.device)

        data.edge_index, _ = gutils.remove_self_loops(data.edge_index)


        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.edge_index_knn)
        h1_3 = self.gcn1(data.x, identify_edge_index)
        R1 = torch.cat([h1, h1_2, h1_3], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.edge_index_knn)
        h2_3 = self.gcn2(R1, identify_edge_index)

        R2 = torch.cat([R1, h2, h2_2, h2_3], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)

class GCN1111(torch.nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, device='cpu'):
        super(GCN1111, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gcn1 = GCNConv(num_features, num_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(4 * num_hidden, num_classes, add_self_loops=False)

        self.lin = Linear(4 * num_hidden + num_classes * 4, num_classes)

        number_knn_neighbor = 5

        knn = NearestNeighbors(n_neighbors=number_knn_neighbor, metric='cosine', algorithm='brute', n_jobs=1)
        knn.fit(deepcopy(data.x).cpu())

        knn_neighbors = torch.tensor(knn.kneighbors(return_distance=False))
        edge_index_knn = torch.empty((2, number_knn_neighbor*data.x.shape[0]), dtype=torch.int32)



        edge_index_knn[0, :] = torch.arange(edge_index_knn.shape[1])/number_knn_neighbor
        edge_index_knn[1, :] = knn_neighbors.flatten()
        self.edge_index_knn = edge_index_knn.long()
        self.edge_index_knn = self.edge_index_knn.to(device)
        print('currently is 1111')



    def forward(self, data):

        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)

        temp_edge_index, _ = gutils.add_self_loops(data.edge_index)
        identify_edge_index = remove_edges(temp_edge_index, data.edge_index).to(self.device)

        data.edge_index, _ = gutils.remove_self_loops(data.edge_index)




        temp_loop_edge_index, _ = gutils.add_self_loops(data.edge_index)
        sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index, data.x.shape[0])


        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index



        for i in range(1):
            self.k_hop_edge_index[i], _ = gutils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = remove_edges(self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)




        h1 = self.gcn1(data.x, data.edge_index)
        h1_2 = self.gcn1(data.x, self.k_hop_edge_index[0])
        h1_3 = self.gcn1(data.x, self.edge_index_knn)
        h1_4 = self.gcn1(data.x, identify_edge_index)
        R1 = torch.cat([h1, h1_2, h1_3, h1_4], dim=1)


        R1 = F.relu(F.dropout(R1, p=self.dropout, training=self.training))

        h2 = self.gcn2(R1, data.edge_index)
        h2_2 = self.gcn2(R1, self.k_hop_edge_index[0])
        h2_3 = self.gcn2(R1, self.edge_index_knn)
        h2_4 = self.gcn2(R1, identify_edge_index)

        R2 = torch.cat([R1, h2, h2_2, h2_3, h2_4], dim=1)
        # R2 = F.relu(F.dropout(R2, p=self.dropout, training=self.training))
        R2 = F.dropout(R2, p=self.dropout, training=self.training)

        final_h = self.lin(R2)

        return torch.nn.functional.log_softmax(final_h, 1)
