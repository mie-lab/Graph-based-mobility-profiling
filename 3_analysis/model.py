import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0.0, act=torch.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inp, adj):
        inp = F.dropout(inp, self.dropout, self.training)
        support = torch.mm(inp, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class GAE(nn.Module):
    def __init__(
        self, input_feat_dim, adj_dim=None, hidden_gc_1=16, hidden_gc_2=32, hidden_dec_1=32, hidden_dec_2=64, dropout=0
    ):
        """ Adjecency dim (= number of nodes) only required for autoencoder"""
        super(GAE, self).__init__()
        self.adj_dim = adj_dim
        self.gc1 = GraphConvolution(input_feat_dim, hidden_gc_1, dropout, act=torch.relu)
        self.gc2 = GraphConvolution(hidden_gc_1, hidden_gc_2, dropout, act=lambda x: x)
        # self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # decoding is simply inner product?!
        self.enc_out_layer = nn.Linear(hidden_gc_2, 1)
        if adj_dim is not None:
            self.dec_hidden_1 = nn.Linear(hidden_gc_2, hidden_dec_1)
            self.dec_hidden_2 = nn.Linear(hidden_dec_1, hidden_dec_2)
            self.dec_out = nn.Linear(hidden_dec_2, adj_dim ** 2)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        z = self.gc2(hidden1, adj)  # , self.gc3(hidden1, adj)
        z_pooled = torch.sum(z, dim=0)
        return z_pooled

    def decode(self, z):
        hidden_dec_1 = torch.relu(self.dec_hidden_1(z))
        hidden_dec_2 = torch.relu(self.dec_hidden_2(hidden_dec_1))
        out_dec = torch.sigmoid(self.dec_out(hidden_dec_2))
        return torch.reshape(out_dec, (self.adj_dim, self.adj_dim))

    def decode_dot_product(self, z):
        z_uns = torch.unsqueeze(z, 1)
        return torch.sigmoid(torch.mm(z_uns, z_uns.t()))

    def forward(self, x, adj):
        # feed forward for classification
        z = self.encode(x, adj)
        out = torch.sigmoid(self.enc_out_layer(z))
        return out
