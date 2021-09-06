import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict

from clustering import ClusterWrapper
from utils import load_question_mapping, load_user_info
from analyze_graph_vs_raw import cluster_characteristics, sort_clusters_into_groups


def plot_longitudinal(before_after_cluster):
    assert "cluster_before" in before_after_cluster.columns and "cluster_after" in before_after_cluster.columns
    cluster_names = np.unique(before_after_cluster["cluster_before"])
    n_clusters = len(cluster_names)

    transition_matrix = np.zeros((n_clusters, n_clusters))
    edges = []
    for c1 in cluster_names:
        users_before = before_after_cluster[before_after_cluster["cluster_before"] == c1]
        for c2 in cluster_names:
            transferred = len(users_before[users_before["cluster_after"] == c2])
            transition_matrix[c1, c2] = transferred
            if transferred > 0:
                edges.append([c1, c2, {"weight": transferred}])
    print("Transitions between clusters:")
    print(transition_matrix)

    # put into graph
    G = nx.DiGraph()
    G.add_edges_from(edges, weight="weight")

    weights = [d[2]["weight"] for d in G.edges(data=True)]

    dist_spring_layout = 10
    norm_width = np.log(weights) * 2

    deg = nx.degree(G)
    # node_sizes = [10 * deg[iata] for iata in G.nodes]

    # draw spring layout
    plt.figure()
    pos = nx.spring_layout(G, k=dist_spring_layout / np.sqrt(len(G)))
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        width=norm_width / 2,
        node_size=500,
        connectionstyle="arc3, rad = 0.2",
    )
    plt.show()


def longitudinal_labels(test_group):
    # ---------- Compare to user interviews ------------
    # get info when someone switched cluster over time
    test_group["switched"] = test_group["cluster_before"] != test_group["cluster_after"]
    # load answers to questions
    user_info = load_user_info("yumuv_graph_rep")
    tg_user_info = user_info[user_info["study_id"] == 22]
    # load questions
    q_tg = load_question_mapping(before_after="after", group="tg")
    for i, row in q_tg.iterrows():
        q = row["question"]
        # only use questions about yumuv
        if "umuv" in q:
            q_id = "q" + i[1:]
            try:
                question_col = tg_user_info[q_id]
            except KeyError:
                print("Question not found in user_info", q_id)
                continue
            question_col = question_col[~pd.isna(question_col)]
            if len(np.unique(question_col.values)) > 4:
                print("Too many unique values", q_id)
                continue
            print("\n", q)
            for value in np.unique(question_col.values):
                print("Reply:", value)
                users_reply = tg_user_info[question_col == value].index
                # filter the ones that did this reply
                filtered_switched = test_group[test_group.index.isin(users_reply)]
                nr_switched = sum(filtered_switched["switched"] == True) / len(filtered_switched)
                print(f"Out of those, {nr_switched} have switched cluster")
            print()


def cross_sectional(data):
    features_cg_bef = data["cg"]["before"].drop(columns=["cluster"])
    labels_cg_bef = data["cg"]["before"]["cluster"]
    # try to characterize clusters
    characteristics = cluster_characteristics(features_cg_bef, labels_cg_bef, printout=False)
    cluster_assigment = sort_clusters_into_groups(characteristics, printout=False)
    groups_cg_bef = [cluster_assigment[cluster] for cluster in labels_cg_bef]
    uni, counts = np.unique(groups_cg_bef, return_counts=True)
    print("CG bef:", {u: round(c / np.sum(counts), 2) for u, c in zip(uni, counts)})
    groups_tg_bef = [cluster_assigment[cluster] for cluster in data["tg"]["before"]["cluster"]]
    uni, counts = np.unique(groups_tg_bef, return_counts=True)
    print("TG aft:", {u: round(c / np.sum(counts), 2) for u, c in zip(uni, counts)})


if __name__ == "__main__":
    path = "out_features/final_1_cleaned"
    node_importance = 0
    n_clusters = 4

    data = defaultdict(dict)
    for group in ["cg", "tg"]:
        for before_after in ["before", "after"]:
            data[group][before_after] = pd.read_csv(
                os.path.join(path, f"yumuv_{before_after}_{group}_graph_features_{node_importance}.csv"),
                index_col="user_id",
            )

    # fit control group before
    clustering = ClusterWrapper()
    data["cg"]["before"]["cluster"] = clustering(data["cg"]["before"], n_clusters=n_clusters)
    # transform control group after
    data["cg"]["after"]["cluster"] = clustering.transform(data["cg"]["after"])
    # merge both (inner join)
    control_group = pd.merge(
        data["cg"]["before"], data["cg"]["after"], on="user_id", how="inner", suffixes=("_before", "_after")
    )
    print(
        "Ratio of CONTROL group that did not switch cluster:",
        np.sum(control_group["cluster_before"] == control_group["cluster_after"]) / len(control_group),
    )
    plot_longitudinal(control_group)

    # Fit test group:
    data["tg"]["before"]["cluster"] = clustering.transform(data["tg"]["before"])
    data["tg"]["after"]["cluster"] = clustering.transform(data["tg"]["after"])
    test_group = pd.merge(
        data["tg"]["before"], data["tg"]["after"], on="user_id", how="inner", suffixes=("_before", "_after")
    )
    print(
        "Ratio of TEST group that did not switch cluster:",
        np.sum(test_group["cluster_before"] == test_group["cluster_after"]) / len(test_group),
    )

    # ---------- Compare to user interviews ------------
    print("\nLONGITUDINAL\n")
    longitudinal_labels(test_group)

    print("\nCROSS SECTIONAL\n")
    cross_sectional(data)
