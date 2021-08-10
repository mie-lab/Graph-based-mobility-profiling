import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite.json_graph import adjacency
import numpy as np
import pickle
import os
import torch

from model import GAE
from utils import RandomGraphDataset, MobilityGraphDataset

MODEL_PATH = os.path.join("3_analysis", "graph_ae", "trained_models")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


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


def train_ae(model_name="test", nr_nodes=10):
    # need to pad
    optim_every = 10
    dataset = MobilityGraphDataset("gc2", nr_nodes=nr_nodes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model = GAE(1, adj_dim=nr_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        # iterate over data
        epoch_loss = 0
        for i, (adj, feat) in enumerate(dataloader):
            if i % optim_every == 0:
                optimizer.zero_grad()
            # Todo: batch graphs? encoder and decoder atm just made for single graph
            z = model.encode(feat[0], adj[0])
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

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, model_name))


def test_ae(model_name="test", nr_nodes=50):
    dataset = MobilityGraphDataset("gc2", nr_nodes=10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = GAE(1, adj_dim=nr_nodes)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_name)))

    embedding_dict = {}
    with torch.no_grad():
        for i, (adj, feat) in enumerate(dataloader):
            user = dataset.users[i]
            # pass through model
            z = model.encode(feat[0].float(), adj[0].float())
            out = model.decode(z)
            print("user", user)
            print("loss this", torch.sum((out - adj[0]) ** 2))
            print("avg absolute difference", torch.mean(torch.abs(out - adj[0])))
            # print("loss random:", torch.sum((torch.randn(out.size()) - adj[0]) ** 2))
            # save embedding
            embedding_dict[user] = z.numpy()
            # print(out.numpy())

    # save embeddings to file
    with open(os.path.join(MODEL_PATH, model_name + "_embeddings.pkl"), "wb") as outfile:
        pickle.dump(embedding_dict, outfile)


if __name__ == "__main__":
    test_ae(nr_nodes=10)
    # train_ae()
