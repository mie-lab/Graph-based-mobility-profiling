import networkx as nx
import os
import pickle

import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic

studies = [
    "yumuv_graph_rep",
]  # 'gc1']  # , 'geolife',]# 'tist_u1000', 'tist_b100', 'tist_b200', 'tist_u10000']

for study in studies:
    print("start {}".format(study))

    # define output for graphs
    GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)
    AG_dict = pickle.load(open(GRAPH_OUTPUT + "_daily_graphs.pkl", "rb"))

    # all_graphs of 1 user
    graph_enumeration_dict = {}
    graph_enumeration_dict_user = {}
    per_user = False
    omitted_counter = 0
    for user, all_user_graphs in AG_dict.items():
        all_graphs = list(all_user_graphs.values())

        # all_graphs = {nb_nodes: {nb_edges: graph_type}}
        # for all graphs
        for AG in all_graphs:
            match = False
            G = AG.G
            n_nodes = len(G.nodes)
            n_edges = len(G.edges)

            GG = nx.Graph(G)
            if not nx.is_connected(GG):  # or n_edges == 0:
                omitted_counter += 1
                continue
            del GG

            if n_nodes not in graph_enumeration_dict:
                graph_enumeration_dict[n_nodes] = {n_edges: {0: {"G": G, "count": 1}}}
                continue
            elif n_edges not in graph_enumeration_dict[n_nodes]:
                graph_enumeration_dict[n_nodes][n_edges] = {0: {"G": G, "count": 1}}
                continue

            # iterate candidate graphs
            for graph_id, graph_dict in graph_enumeration_dict[n_nodes][n_edges].items():
                G_ref = graph_dict["G"]

                # check for isomorphism
                # if fast_could_be_isomorphic(G_ref, G):
                if is_isomorphic(G_ref, G):
                    graph_dict["count"] = graph_dict["count"] + 1
                    graph_enumeration_dict[n_nodes][n_edges][graph_id] = graph_dict

                    # todo: consider edge weights
                    match = True

                if match:
                    break

            if not match:
                # append new isomorphism
                graph_enumeration_dict[n_nodes][n_edges][graph_id + 1] = {"G": G, "count": 1}

        if per_user:
            graph_enumeration_dict_user[user] = graph_enumeration_dict
            graph_enumeration_dict = {}

    if per_user:
        pickle.dump(graph_enumeration_dict_user, open(GRAPH_OUTPUT + "_daily_graphlets_user.pkl", "wb"))
    else:
        pickle.dump(graph_enumeration_dict, open(GRAPH_OUTPUT + "_daily_graphlets.pkl", "wb"))

print("omitted: ", omitted_counter)
