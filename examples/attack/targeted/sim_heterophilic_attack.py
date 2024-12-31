import torch
import numpy as np
import torch_geometric.transforms as T
import os
import torch_geometric
import random

from graphwar.dataset import GraphWarDataset
from graphwar import set_seed
from graphwar.nn.models import GCN, SGC
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import split_nodes
from graphwar.attack.targeted import RandomAttack, Nettack, FGAttack, SGAttack
from scipy import io
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_dense_adj, sort_edge_index, from_scipy_sparse_matrix, \
    to_scipy_sparse_matrix
from copy import deepcopy
import scipy.sparse as sp

import torch.nn.functional as F

from graphwar.heter_gnn.model import FAGCN
from graphwar.heter_gnn.utils import accuracy, predicted_label_func
from graphwar.heter_gnn.mlp import MLP, GCNMLP
from graphwar.heter_gnn.basic_gcn import *
from graphwar.heter_gnn.h2gcn import H2GNN
from graphwar.heter_gnn.nlgnn import NLGNN

homo_datasets = ['cora', 'citeseer', 'cora_ml', 'pubmed']
heter_datasets = ['chameleon', 'wisconsin', 'snap', 'film']
victim_gnn_list = ['h2gcn', 'nlgnn', 'basic_gcn']
attack_method_list = ['nettack']

global_before_acc = []
gloabl_after_eva_acc = []
gloabl_after_poi_acc = []

for model_id in range(16):

    if model_id == 0:
        gcn_config = "GCN" + "0000"
    elif model_id == 1:
        gcn_config = "GCN" + "0001"
    elif model_id == 2:
        gcn_config = "GCN" + "0010"
    elif model_id == 3:
        gcn_config = "GCN" + "0011"
    elif model_id == 4:
        gcn_config = "GCN" + "0100"
    elif model_id == 5:
        gcn_config = "GCN" + "0101"
    elif model_id == 6:
        gcn_config = "GCN" + "0110"
    elif model_id == 7:
        gcn_config = "GCN" + "0111"
    elif model_id == 8:
        gcn_config = "GCN" + "1000"
    elif model_id == 9:
        gcn_config = "GCN" + "1001"
    elif model_id == 10:
        gcn_config = "GCN" + "1010"
    elif model_id == 11:
        gcn_config = "GCN" + "1011"
    elif model_id == 12:
        gcn_config = "GCN" + "1100"
    elif model_id == 13:
        gcn_config = "GCN" + "1101"
    elif model_id == 14:
        gcn_config = "GCN" + "1110"
    elif model_id == 15:
        gcn_config = "GCN" + "1111"
    else:
        os._exit()


    attack_node_num = 5
    true_target_node_flag = False


    dataset_name = homo_datasets[0]
    # dataset_name = heter_datasets[0]


    victim_gnn = victim_gnn_list[0]
    attack_method = attack_method_list[0]

    set_seed(123)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if dataset_name in ['cora', 'citeseer', 'pubmed', 'cora_ml']:
        dataset = GraphWarDataset(root='~/data/pygdata', name=dataset_name,
                                  transform=T.LargestConnectedComponents())
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes

    elif dataset_name in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        DATAPATH = '/home/jyfang/code/GraphWar-heter-attack/heterophily_datasets_matlab'
        fulldata = io.loadmat(f'{DATAPATH}/{dataset_name}.mat')
        edge_index = fulldata['edge_index']
        node_feat = fulldata['node_feat']
        label = np.array(fulldata['label'], dtype=np.int32).flatten()
        num_features = node_feat.shape[1]
        num_classes = np.max(label) + 1
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(node_feat)

        y = torch.tensor(label, dtype=torch.long)
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
        data = Data(x=x, edge_index=edge_index, y=y)
    elif dataset_name in ['fb', 'snap']:
        print(dataset_name)
        DATAPATH = '/home/jyfang/code/GraphWar-heter-attack/heterophily_datasets_matlab'

        ori_data = np.load(f'{DATAPATH}/{dataset_name}.npz')

        adj_data = ori_data[ori_data.files[0]]
        adj_indices = ori_data[ori_data.files[1]]
        adj_indptr = ori_data[ori_data.files[2]]
        adj_shape = ori_data[ori_data.files[3]]

        f_data = ori_data[ori_data.files[4]]
        f_indices = ori_data[ori_data.files[5]]
        f_indptr = ori_data[ori_data.files[6]]
        f_shape = ori_data[ori_data.files[7]]

        label = ori_data[ori_data.files[8]]

        adj = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape).tocoo()
        feature = sp.csr_matrix((f_data, f_indices, f_indptr), shape=f_shape).toarray()

        edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)

        x = torch.tensor(feature)
        y = torch.tensor(label, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)

        num_features = feature.shape[1]
        num_classes = np.max(label) + 1



    label_max = torch.max(data.y) + 1
    enough_flag = True
    for i in range(label_max):
        mask = (data.y == i)
        if torch.sum(mask) <= 1:
            enough_flag = False
            break

    print(f"enough flag: {enough_flag}")
    splits = split_nodes(data.y, random_state=18, is_stratify=enough_flag)



    def edge_index_to_sparse_tensor_adj(edge_index):
        sparse_adj_adj = to_scipy_sparse_matrix(edge_index)
        values = sparse_adj_adj.data
        indices = np.vstack((sparse_adj_adj.row, sparse_adj_adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = sparse_adj_adj.shape
        sparse_adj_adj_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        return sparse_adj_adj_tensor


    def gcn_norm(edge_index, num_nodes):
        a1 = edge_index_to_sparse_tensor_adj(edge_index)
        d1_adj = torch.diag(degree(edge_index[0], num_nodes=num_nodes)).to_sparse()
        d1_adj = torch.pow(d1_adj, -0.5)
        return torch.sparse.mm(torch.sparse.mm(d1_adj, a1), d1_adj)


    # ================================================================== #
    #                     Attack Setting                                 #
    # ================================================================== #
    # target = 9  # target node to attack
    # target_label = data.y[target].item()
    width = 5
    data = data.to(device)

    basic_model = locals()[gcn_config](data, data.x.shape[1], 16, int(torch.max(data.y) + 1), dropout=0, device=device)


    def train_basic_gcn(net):

        net = net.to(device)
        if dataset_name in ['cora', 'citeseer']:
            mlp_optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-3)
        elif dataset_name in ['chameleon', 'squirrel', 'film']:
            mlp_optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-5)
        elif dataset_name in ['pubmed']:
            mlp_optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
        else:  # coraml
            mlp_optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

        idx_train, idx_val, idx_test = splits.train_nodes, splits.val_nodes, splits.test_nodes

        counter = 0
        min_loss = 100.0
        max_acc = 0.0
        patience = 30

        for epoch in range(200):
            net.train()

            logp = net(data)

            cla_loss = F.nll_loss(logp[idx_train], data.y[idx_train])
            loss = cla_loss
            train_acc = accuracy(logp[idx_train], data.y[idx_train])

            mlp_optimizer.zero_grad()
            loss.backward()
            mlp_optimizer.step()

            net.eval()
            logp = net(data)
            test_acc = accuracy(logp[idx_test], data.y[idx_test])
            loss_val = F.nll_loss(logp[idx_val], data.y[idx_val]).item()
            val_acc = accuracy(logp[idx_val], data.y[idx_val])

            if loss_val < min_loss and max_acc < val_acc:
                min_loss = loss_val
                max_acc = val_acc
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print('early stop')
                break

            # if epoch % 10 == 0:
            #     print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(
            #         epoch, loss_val, train_acc, val_acc, test_acc))

        print(f"basic_gcn test acc {test_acc}")
        return net


    basic_gcn = train_basic_gcn(basic_model)

    # ================================================================== #
    #                      Before Attack                                 #
    # ================================================================== #
    if victim_gnn == 'basic_gcn':
        before_model = basic_gcn
        before_model.eval()
        pre = predicted_label_func(before_model(data))
        print('Before test acc: {}'.format(
            (torch.sum(pre[splits.test_nodes] == data.y[splits.test_nodes]) / len(splits.test_nodes)).item()))
    else:
        trainer_before = Trainer(GCN(num_features, num_classes), device=device)
        ckp = ModelCheckpoint('model_before2.pth', monitor='val_acc')
        trainer_before.fit({'data': data, 'mask': splits.train_nodes},
                           {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])

        all_out = trainer_before.predict({'data': data, 'mask': None})

        print('Before test acc: {}'.format((torch.sum(
            all_out[splits.test_nodes].max(1)[1] == data.y[splits.test_nodes]) / len(splits.test_nodes)).item()))
    # print(torch.where(all_out[splits.test_nodes].max(1)[1] == data.y[splits.test_nodes]))

    if true_target_node_flag:
        if victim_gnn in ['basic_gcn']:
            target_list = splits.test_nodes[torch.where(pre[splits.test_nodes] == data.y[splits.test_nodes])]
        else:
            target_list = splits.test_nodes[
                torch.where(all_out[splits.test_nodes].max(1)[1] == data.y[splits.test_nodes])]
    else:
        target_list = splits.test_nodes

    final_attack_list = random.sample(list(target_list.numpy()), attack_node_num)
    final_attack_list = torch.tensor(final_attack_list)

    print(final_attack_list)

    if victim_gnn == 'basic_gcn':
        before_model.eval()
        final_attack_list_acc_before = accuracy(before_model(data)[final_attack_list], data.y[final_attack_list])
    else:
        final_attack_list_acc_before = \
        torch.where(all_out[final_attack_list].max(1)[1] == data.y[final_attack_list])[0].shape[0] / len(
            final_attack_list)

    print(f'final_attack_list_acc_before: {final_attack_list_acc_before}')

    global_before_acc.append(final_attack_list_acc_before)

    if attack_method in ['nettack']:
        surrogate_trainer = Trainer(SGC(num_features, num_classes), device=device, lr=0.1, weight_decay=1e-5)
        ckp = ModelCheckpoint('surrogate_trainer2.pth', monitor='val_acc')
        surrogate_trainer.fit({'data': data, 'mask': splits.train_nodes},
                              {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
        surrogate_trainer.cache_clear()

    eva_cnt = torch.zeros(attack_node_num)
    poi_cnt = torch.zeros(attack_node_num)


    def add_edges(edge_index, edges_to_add):
        """add edges to the graph `edge_index`.

        Parameters
        ----------
        edge_index : Tensor
            the graph instance where edges will be removed from.
        edges_to_add : torch.Tensor
            shape [2, M], the edges to be added into the graph.
        symmetric : bool
            whether the graph is symmetric, if True,
            it would add the edges into the graph by:
            `edges_to_add = torch.cat([edges_to_add, edges_to_add.flip(0)], dim=1)`

        Returns
        -------
        Tensor
            the graph instance `edge_index` with edges added.
        """
        edges_to_add = torch.cat([edges_to_add, edges_to_add.flip(0)], dim=1)

        edges_to_add = edges_to_add.to(edge_index)
        edge_index = torch.cat([edge_index, edges_to_add], dim=1)
        edge_index = sort_edge_index(edge_index)
        return edge_index


    def remove_edges(edge_index, edges_to_remove):
        """remove edges from the graph `edge_index`.

        Parameters
        ----------
        edge_index : Tensor
            the graph instance where edges will be removed from.
        edges_to_remove : torch.Tensor
            shape [2, M], the edges to be removed in the graph.
        symmetric : bool
            whether the graph is symmetric, if True,
            it would remove the edges from the graph by:
            `edges_to_remove = torch.cat([edges_to_remove, edges_to_remove.flip(0)], dim=1)`

        Returns
        -------
        Tensor
            the graph instance `edge_index` with edges removed.
        """
        edges_to_remove = torch.cat(
            [edges_to_remove, edges_to_remove.flip(0)], dim=1)
        edges_to_remove = edges_to_remove.to(edge_index)

        # it's not intuitive to remove edges from a graph represented as `edge_index`
        edge_weight_remove = torch.zeros(edges_to_remove.size(1)) - 1e5
        edge_weight = torch.cat(
            [torch.ones(edge_index.size(1)), edge_weight_remove], dim=0)
        edge_index = torch.cat([edge_index, edges_to_remove], dim=1).cpu().numpy()
        adj_matrix = sp.csr_matrix(
            (edge_weight.cpu().numpy(), (edge_index[0], edge_index[1])))
        adj_matrix.data[adj_matrix.data < 0] = 0.
        adj_matrix.eliminate_zeros()
        edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
        return edge_index.to(device)


    for i in range(attack_node_num):

        target = final_attack_list[i]

        print(f"\nCurrent model id, node no. and id are {model_id}, {i} and {target}.")

        target_label = data.y[target].item()

        if victim_gnn == 'basic_gcn':
            before_model.eval()
            logits = before_model(data)
            torch.set_printoptions(sci_mode=False)
            print(
                f"Before attack (target_label={target_label})\n predicted logits={torch.exp(logits)[target].cpu().detach().numpy()}")
        else:
            output = trainer_before.predict({'data': data, 'mask': target})
            print(f"Before attack (target_label={target_label})\n {np.round(output.tolist(), 2)}")
            print('-' * target_label * width + '----👆' + '-' * max(num_classes - target_label - 1, 0) * width)

        # ================================================================== #
        #                      Attacking                                     #
        # ================================================================== #

        if attack_method == 'nettack':
            attacker = Nettack(data, device=device)
            attacker.setup_surrogate(surrogate_trainer.model)
            attacker.reset()
            attacker.attack(target)
        else:
            print('Undefined attack method')
            os._exit()

        copy_data = attacker.data()
        attack_list = []
        nettack_predicted_label = torch.argmax(surrogate_trainer.predict({'data': data}), dim=1).cpu()
        print(f"target node's predicted label: {nettack_predicted_label[target]}, label: {copy_data.y[target]}")

        if attacker.added_edges() != None:
            for k in range(attacker.added_edges().shape[1]):
                attack_list.append((attacker.added_edges()[0, k].cpu(), attacker.added_edges()[1, k].cpu()))
                print(
                    f"add node {attack_list[-1][1]}, predicted label: {nettack_predicted_label[attack_list[-1][1]]}, label: {copy_data.y[attack_list[-1][1]]}")
        if attacker.removed_edges() != None:
            for k in range(attacker.removed_edges().shape[1]):
                attack_list.append((attacker.removed_edges()[0, k].cpu(), attacker.removed_edges()[1, k].cpu()))
                print(
                    f"remove node {attack_list[-1][1]}, predicted label: {nettack_predicted_label[attack_list[-1][1]]}, label: {copy_data.y[attack_list[-1][1]]}")

        attack_list = np.array(attack_list, dtype=np.int16)
        np.savetxt('../perturbations/' + dataset_name + '_' + attack_method + '_' + 'attack_list.txt', attack_list)

        # ================================================================== #
        #                      After evasion Attack                          #
        # ================================================================== #
        if victim_gnn == 'basic_gcn':
            before_model.eval()
            logits = before_model(copy_data)
            pre = predicted_label_func(logits)
            torch.set_printoptions(sci_mode=False)
            print(
                f"After evasion attack (target_label={target_label})\n predicted logits={torch.exp(logits)[target].cpu().detach().numpy()}")

            if target_label == pre[target]:
                eva_cnt[i] = 1
                print(f"eva cnt: {eva_cnt[i]}")
        else:
            trainer_before.cache_clear()
            output = trainer_before.predict({'data': copy_data, 'mask': target})
            torch.set_printoptions(sci_mode=False)

            print(f"After evasion attack (target_label={target_label})\n {np.round(output.tolist(), 2)}")
            print('-' * target_label * width + '----👆' + '-' * max(num_classes - target_label - 1, 0) * width)

            if output.argmax() == target_label:
                eva_cnt[i] = 1
                print(f"eva cnt: {eva_cnt[i]}")

        # ================================================================== #
        #                      After poisoning Attack                        #
        # ================================================================== #
        if victim_gnn == 'basic_gcn':
            after_model = locals()[gcn_config](copy_data, data.x.shape[1], 16, int(torch.max(data.y) + 1), dropout=0,
                                               device=device)

            net = after_model
            net = net.to(device)

            if dataset_name in ['cora', 'citeseer']:
                mlp_optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-3)
            elif dataset_name in ['chameleon', 'squirrel', 'film']:
                mlp_optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-5)
            elif dataset_name in ['pubmed']:
                mlp_optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
            else:
                mlp_optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

            idx_train, idx_val, idx_test = splits.train_nodes, splits.val_nodes, splits.test_nodes

            counter = 0
            min_loss = 100.0
            max_acc = 0.0
            patience = 30

            for epoch in range(200):
                net.train()

                logp = net(data)

                cla_loss = F.nll_loss(logp[idx_train], data.y[idx_train])
                loss = cla_loss
                train_acc = accuracy(logp[idx_train], data.y[idx_train])

                mlp_optimizer.zero_grad()
                loss.backward()
                mlp_optimizer.step()

                net.eval()
                logp = net(data)
                test_acc = accuracy(logp[idx_test], data.y[idx_test])
                loss_val = F.nll_loss(logp[idx_val], data.y[idx_val]).item()
                val_acc = accuracy(logp[idx_val], data.y[idx_val])

                if loss_val < min_loss and max_acc < val_acc:
                    min_loss = loss_val
                    max_acc = val_acc
                    counter = 0
                else:
                    counter += 1

                if counter >= patience:
                    print('early stop')
                    break

                # print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(
                #     epoch, loss_val, train_acc, val_acc, test_acc))
            poi_model = net

            poi_model.eval()
            logits = poi_model(copy_data)
            torch.set_printoptions(sci_mode=False)

            print(
                f"After poisoning attack (target_label={target_label})\n predicted logits={torch.exp(logits)[target].cpu().detach().numpy()}")
            pre = predicted_label_func(logits)

            if target_label == pre[target]:
                poi_cnt[i] = 1
                print(f"poi cnt: {poi_cnt[i]}")

        else:
            trainer_after = Trainer(GCN(num_features, num_classes), device=device)
            ckp = ModelCheckpoint('model_after2.pth', monitor='val_acc')
            trainer_after.fit({'data': copy_data, 'mask': splits.train_nodes},
                              {'data': copy_data, 'mask': splits.val_nodes}, callbacks=[ckp])
            output = trainer_after.predict({'data': copy_data, 'mask': target})
            torch.set_printoptions(sci_mode=False)

            print(f"After poisoning attack (target_label={target_label})\n {np.round(output.tolist(), 2)}")
            print('-' * target_label * width + '----👆' + '-' * max(num_classes - target_label - 1, 0) * width)

            if output.argmax() == target_label:
                poi_cnt[i] = 1
                print(f"poi cnt: {poi_cnt[i]}")

    print(eva_cnt)
    print(poi_cnt)
    print(attack_node_num)

    gloabl_after_eva_acc.append(torch.sum(eva_cnt).numpy() / attack_node_num)
    gloabl_after_poi_acc.append(torch.sum(poi_cnt).numpy() / attack_node_num)

    print(f'final_attack_list_acc_before: {final_attack_list_acc_before}')
    print('eva acc: {}'.format(torch.sum(eva_cnt) / attack_node_num))
    print('poi acc: {}\n\n\n'.format(torch.sum(poi_cnt) / attack_node_num))

print(global_before_acc)
print(gloabl_after_eva_acc)
print(gloabl_after_poi_acc)