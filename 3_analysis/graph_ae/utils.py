import scipy.sparse as sp
import numpy as np
import torch
import networkx as nx
import pickle
import os
import json
import psycopg2
from future_trackintel.utils import read_graphs_from_postgresql


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


def load_graphs_postgis(study):
    DBLOGIN_FILE = os.path.join("./dblogin.json")
    with open(DBLOGIN_FILE) as json_file:
        LOGIN_DATA = json.load(json_file)

    con = psycopg2.connect(
        dbname=LOGIN_DATA["database"],
        user=LOGIN_DATA["user"],
        password=LOGIN_DATA["password"],
        host=LOGIN_DATA["host"],
        port=LOGIN_DATA["port"],
    )
    graph_dict = read_graphs_from_postgresql(graph_table_name=study, psycopg_con=con, file_name="graph_data")
    return graph_dict


class MobilityGraphDataset(torch.utils.data.Dataset):
    def __init__(self, study, nr_nodes=25, load_from_pickle=None):
        """[summary]

        Parameters
        ----------
        study : str
            study name
        nr_nodes : int, optional
            n most important nodes of the graphts are used, by default 25
        load_from_pickle : str, optional
            path where to load pickle, by default None
        """
        if load_from_pickle:
            # path = os.path.join("..", "..", "data_out", "graph_data", study, "counts_full.pkl")
            AG_dict = pickle.load(open(load_from_pickle, "rb"))
        else:
            AG_dict = load_graphs_postgis(study=study)

        self.nr_nodes = nr_nodes

        self.users = []
        self.nx_graphs = []
        self.adjacency = []
        for user_id, ag in AG_dict.items():
            self.users.append(user_id)
            # TODO: rewrite k importance nodes such that it is filtered by the fraction of occurence, not the abs number
            important_nodes = ag.get_k_importance_nodes(nr_nodes)
            ag_sub = ag.G.subgraph(important_nodes)
            self.nx_graphs.append(ag_sub)
            # get adjacency and reduce to important nodes
            adjacency_ag = ag.adjacency_dict["A"][0]
            adjacency_ag = adjacency_ag.tocsr()[important_nodes, :]
            adjacency_ag = adjacency_ag.tocsr()[:, important_nodes]
            # preprocess adjacency matrix
            adjacency = self.preprocess(adjacency_ag)  # nx.adjacency_matrix(ag_sub)
            self.adjacency.append(adjacency.float())

        # TODO: node features
        self.nr_graphs = len(self.adjacency)

    @staticmethod
    def preprocess(adjacency_matrix):
        return sparse_mx_to_torch_sparse_tensor(preprocess_adj(adjacency_matrix))

    def __len__(self):
        return self.nr_graphs

    def __getitem__(self, idx):
        feats = torch.ones(self.nr_nodes, 1)
        return self.adjacency[idx], feats


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
