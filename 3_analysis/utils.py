import scipy.sparse as sp
import numpy as np
import torch
import networkx as nx
import pickle


class RandomGraphDataset(torch.utils.data.Dataset):
    def __init__(self, nr_nodes=None, adj_norm_factor=1):
        self.nr_nodes = nr_nodes
        self.adj_norm_factor = adj_norm_factor

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        if self.nr_nodes is None:
            self.nr_nodes = np.random.choice([10, 15, 20])
        feats = np.ones((self.nr_nodes, 1))
        label = np.random.rand() * 0.8 + 0.1
        random_graph = nx.generators.random_graphs.fast_gnp_random_graph(self.nr_nodes, label, directed=True)
        # adjacency = (
        #   np.array(nx.adjacency_matrix(random_graph).todense()) + np.identity(self.nr_nodes) * self.adj_norm_factor
        # )
        adjacency = sparse_mx_to_torch_sparse_tensor(preprocess_adj(nx.adjacency_matrix(random_graph)))
        return adjacency, feats, label


class MobilityGraphDataset(torch.utils.data.Dataset):
    def __init__(self, path, node_importance=25):
        AG_dict = pickle.load(open(path, "rb"))
        users = []
        self.nx_graphs = []
        self.adjacency = []
        for user_id, ag in AG_dict.items():
            users.append(user_id)
            # TODO: rewrite k importance nodes such that it is filtered by the fraction of occurence, not the abs number
            important_nodes = ag.get_k_importance_nodes(50)
            ag_sub = ag.G.subgraph(important_nodes)
            self.nx_graphs.append(ag_sub)
            adjacency = sparse_mx_to_torch_sparse_tensor(preprocess_adj(nx.adjacency_matrix(ag_sub)))
            # adjacency = np.array(nx.adjacency_matrix(ag_sub).todense())
            self.adjacency.append(adjacency)
        # TODO: node features
        self.nr_graphs = len(self.adjacency)

    def preprocess(self, use_log=False, norm_max=None, self_loops=False):
        # manual preprocessing
        for i, adj in enumerate(self.adjacency):
            new_adj = adj.copy()
            if norm_max is not None:
                new_adj = new_adj / norm_max
            if use_log:
                new_adj[new_adj > 0] = np.log(new_adj[new_adj > 0])
            if self_loops:
                new_adj = new_adj + np.identity(len(new_adj))
            self.adjacency[i] = new_adj

    def __len__(self):
        return self.nr_graphs

    def __getitem__(self, idx):
        return self.adjacency[idx]


def normalize_adj(adj):
    """From tkipf: Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """From tkipf: Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """From tkipf: Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
