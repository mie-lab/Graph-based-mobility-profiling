import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

studies = ["yumuv_graph_rep"]  # , 'gc1']  # , 'geolife',]# 'tist_u1000', 'tist_b100', 'tist_b200', 'tist_u10000']

for study in studies:
    print("start {}".format(study))
    # define output for graphs
    GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)

graph_enumeration_dict = pickle.load(open(GRAPH_OUTPUT + "_daily_graphlets.pkl", "rb"))

to_pandas_dict = {}

for nb_nodes, edges_dict in graph_enumeration_dict.items():
    for nb_edges, graph_enum_dict in edges_dict.items():
        for graph_id, graph_dict in graph_enum_dict.items():
            to_pandas_dict[(nb_nodes, nb_edges, graph_id)] = graph_dict

df = pd.DataFrame(to_pandas_dict).transpose()
df.sort_values("count", inplace=True, ascending=False)

fig, axes = plt.subplots(5, 5, figsize=(20, 20))
total_count = df["count"].sum()
ix = 0
for i, axs in enumerate(axes):
    for j, ax in enumerate(axs):
        G = df["G"].iloc[ix]
        count = df["count"].iloc[ix]

        pos = nx.spring_layout(G)
        nx.draw(G, pos=pos, connectionstyle="arc3, rad = 0.2", ax=ax)
        #
        # self_loops = list(nx.classes.function.selfloop_edges(G))
        # nx.draw_networkx_edges(G, pos, edgelist=self_loops, arrowstyle="<|-", style="dashed",
        #                        connectionstyle="arc3, rad = 0.2")

        ax.set_title(f"{count}, " + "{:.2f})%".format(count / total_count * 100))
        ax.axis("on")
        ix += 1
        if ix == df.shape[0]:
            break
plt.show()
