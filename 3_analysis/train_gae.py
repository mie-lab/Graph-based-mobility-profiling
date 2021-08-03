import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite.json_graph import adjacency
import numpy as np
import pickle
import os
import torch

from model import GAE
from utils import RandomGraphDataset


def train_classify():
    dataset = RandomGraphDataset(adj_norm_factor=0.01)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = GAE(1, 16, 32, 0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(300):
        # iterate over data
        epoch_loss = 0
        for i, (adj, feat, label) in enumerate(dataloader):
            optimizer.zero_grad()
            out = model(feat[0].float(), adj[0].float())
            # print(model.out_layer.weight)
            loss = (out - label) ** 2
            loss.backward()
            # possibly to that only after a few samples (gradient accumulation)
            # print(label, "loss", loss.item())
            # print("grad", model.out_layer.weight.grad)
            optimizer.step()
            abs_loss = torch.abs(out - label)
            epoch_loss += abs_loss.item()

        if epoch % 10 == 0:
            print(epoch_loss / i)


def train_ae():
    # need to pad
    nr_nodes = 10
    optim_every = 10
    dataset = RandomGraphDataset(nr_nodes=10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = GAE(1, adj_dim=nr_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        # iterate over data
        epoch_loss = 0
        for i, (adj, feat, label) in enumerate(dataloader):
            if i % optim_every == 0:
                optimizer.zero_grad()
            z = model.encode(feat[0].float(), adj[0].float())
            out = model.decode(z)
            # print(out)
            # print(adj / adj_norm_factor)
            loss = torch.sum((out - adj[0]) ** 2)
            loss.backward()
            # # possibly to that only after a few samples (gradient accumulation)
            if (i + 1) % optim_every == 0:
                optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            # print(out)
            print(epoch_loss / i)


if __name__ == "__main__":
    train_ae()
