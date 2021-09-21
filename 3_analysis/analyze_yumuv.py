import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
from collections import defaultdict
from matplotlib.lines import Line2D

from utils import sort_images_by_cluster
from plotting import barplot_clusters, print_chisquare
from clustering import ClusterWrapper
from utils import load_question_mapping, load_user_info
from find_groups import cluster_characteristics, sort_clusters_into_groups


def plot_longitudinal(before_after_cluster, out_path=None):
    assert "cluster_before" in before_after_cluster.columns and "cluster_after" in before_after_cluster.columns
    cluster_names = np.unique(before_after_cluster["cluster_before"])
    n_clusters = len(cluster_names)
    cluster_to_ind = {c: i for i, c in enumerate(cluster_names)}

    transition_matrix = np.zeros((n_clusters, n_clusters))
    edges = []
    for c1 in cluster_names:
        users_before = before_after_cluster[before_after_cluster["cluster_before"] == c1]
        for c2 in cluster_names:
            transferred = len(users_before[users_before["cluster_after"] == c2])
            transition_matrix[cluster_to_ind[c1], cluster_to_ind[c2]] = transferred
            if transferred > 0:
                edges.append([c1, c2, {"weight": transferred}])
    print("Transitions between clusters:")
    print(transition_matrix)

    # plot confusion matrix
    plt.figure(figsize=(15, 10))
    print(np.sum(transition_matrix, axis=1))
    print(transition_matrix / np.sum(transition_matrix, axis=0))
    normalized = np.swapaxes(np.swapaxes(transition_matrix, 1, 0) / np.sum(transition_matrix, axis=1), 1, 0)
    df = pd.DataFrame(normalized, columns=cluster_names, index=cluster_names)
    df_unnormalized = pd.DataFrame(transition_matrix, columns=cluster_names, index=cluster_names)
    sns.heatmap(df, annot=True, cmap="Blues")
    # plt.xticks(rotation=20)
    if out_path:
        plt.savefig(out_path.replace("longitudinal", "longitudinal_matrix"), dpi=600)
    else:
        plt.show

    # put into graph
    G = nx.DiGraph()
    G.add_edges_from(edges, weight="weight")

    weights = [d[2]["weight"] for d in G.edges(data=True)]

    dist_spring_layout = 10
    norm_width = np.log(weights) * 2

    # deg = nx.degree(G)
    # node_sizes = [10 * deg[iata] for iata in G.nodes]

    color_map = ["green", "red", "blue", "purple", "black", "yellow", "orange"]
    # draw spring layout
    plt.figure()
    pos = nx.circular_layout(G)  # , k=dist_spring_layout / np.sqrt(len(G)))
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        width=norm_width / 2,
        node_size=500,
        node_color=color_map[: G.number_of_nodes()],
        connectionstyle="arc3, rad = 0.2",
    )
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=lab, markerfacecolor=col, markersize=10)
        for col, lab in zip(color_map, list(G.nodes))
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=10)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=600)
    else:
        plt.show()
    return df_unnormalized


def questions_longitudinal(test_group, out_path=None):
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
                # print("Question not found in user_info", q_id)
                continue
            question_col = question_col[~pd.isna(question_col)]
            if len(np.unique(question_col.values)) > 4 or len(np.unique(question_col.values)) < 2:
                # print("Too many unique values", q_id)
                continue
            print("\n", q)
            answer_groups, answers = [], []
            for value in np.unique(question_col.values):
                print("Reply:", value)
                users_reply = tg_user_info[question_col == value].index
                # filter the ones that did this reply
                filtered_switched = test_group[test_group.index.isin(users_reply)]
                nr_switched = sum(filtered_switched["switched"] == True) / len(filtered_switched)
                print(f"Out of those, {nr_switched} have switched cluster")
                print()
                # barplot
                answers.append(value)
                answer_groups.append(test_group[test_group.index.isin(users_reply)]["cluster_before"])
            barplot_clusters(
                answer_groups[0],
                answer_groups[1],
                q_id + " " + answers[0],
                q_id + " " + answers[1],
                out_path=out_path,
                title=q,
            )


def print_cross_sectional(groups_cg_bef, groups_tg_bef):
    uni, counts = np.unique(groups_cg_bef, return_counts=True)
    print("CG bef:", {u: round(c / np.sum(counts), 2) for u, c in zip(uni, counts)})
    uni, counts = np.unique(groups_tg_bef, return_counts=True)
    print("TG aft:", {u: round(c / np.sum(counts), 2) for u, c in zip(uni, counts)})

def load_yumuv_features():
    data = defaultdict(dict)
    for group in ["cg", "tg"]:
        for before_after in ["before", "after"]:
            data[group][before_after] = pd.read_csv(
                os.path.join(path, f"yumuv_{before_after}_{group}_graph_features_{node_importance}.csv"),
                index_col="user_id",
            )
    return data

if __name__ == "__main__":
    path = os.path.join("out_features", "final_6_n0_cleaned")
    out_path = "figures/results_yumuv"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    node_importance = 0
    n_clusters = 8

    if not os.path.exists(path):
        raise RuntimeError("First need to download out_features folder")

    data = load_yumuv_features()

    # fit control group before
    clustering = ClusterWrapper()
    cg_both = pd.concat((data["cg"]["before"].reset_index(), data["cg"]["after"].reset_index())).drop("user_id", axis=1)
    # print(cg_both, len(cg_both), len(data["cg"]["before"]))
    int_labels_cg_both = clustering(cg_both, n_clusters=n_clusters)
    # sort clusters into groups
    characteristics = cluster_characteristics(cg_both, int_labels_cg_both, printout=False)
    cluster_assignment = sort_clusters_into_groups(characteristics, add_groups=False, printout=False)
    clustering.cluster_assignment = cluster_assignment
    labels_cg_both = [cluster_assignment[ind] for ind in int_labels_cg_both]

    # # SORT IMAGES INTO FOLDERS
    # print(np.unique(labels_cg_both))
    # cg_both_wo = pd.concat((data["cg"]["before"], data["cg"]["after"]))
    # for type in ["coords", "spring"]:
    #     sort_images_by_cluster(
    #         list(cg_both_wo.index),
    #         labels_cg_both,
    #         name_mapping={lab: lab for lab in np.unique(labels_cg_both)},
    #         in_img_path=os.path.join("graph_images", "yumuv_graph_rep", type),
    #         out_img_path=os.path.join(out_path, f"yumuv_sorted_images_" + type),
    #     )
    # exit()

    # # write terminal output to file:
    f = open(os.path.join(out_path, "terminal.txt"), "w")
    sys.stdout = f

    print("\nLONGITUDINAL\n")

    # transform control group after
    data["cg"]["before"]["cluster"] = clustering.transform(data["cg"]["before"])
    data["cg"]["after"]["cluster"] = clustering.transform(data["cg"]["after"])
    # merge both (inner join)
    control_group = pd.merge(
        data["cg"]["before"], data["cg"]["after"], on="user_id", how="inner", suffixes=("_before", "_after")
    )
    print(
        "Ratio of CONTROL group that switched cluster:",
        np.sum(control_group["cluster_before"] != control_group["cluster_after"]) / len(control_group),
    )
    long_cg = plot_longitudinal(control_group, out_path=os.path.join(out_path, "longitudinal_control_group.png"))

    # Fit test group:
    data["tg"]["before"]["cluster"] = clustering.transform(data["tg"]["before"])
    data["tg"]["after"]["cluster"] = clustering.transform(data["tg"]["after"])
    test_group = pd.merge(
        data["tg"]["before"], data["tg"]["after"], on="user_id", how="inner", suffixes=("_before", "_after")
    )
    print(
        "Ratio of TEST group that switched cluster:",
        np.sum(test_group["cluster_before"] != test_group["cluster_after"]) / len(test_group),
    )
    long_tg = plot_longitudinal(test_group, out_path=os.path.join(out_path, "longitudinal_test_group.png"))

    # chi square analysis long
    print("------------ Chi square test for movement between clustern -----------")
    for col in long_cg.columns:
        if col not in long_tg.columns:
            long_tg[col] = 0
    long_tg = long_tg.reindex(columns=long_cg.columns)
    # print(long_tg)
    for feat, row in long_cg.iterrows():
        # print(feat, row)
        print("--------------- ", feat)
        if feat not in long_tg.index:
            print("feat not there")
            continue
        occ1 = list(row.values)
        occ2 = list(long_tg.loc[feat].values)
        print_chisquare(occ1, occ2)

    # ---------- Compare to user interviews ------------
    questions_longitudinal(test_group, out_path=out_path)

    # ---------- Cross sectional ------------
    print("\nCROSS SECTIONAL\n")
    print_cross_sectional(np.array(data["cg"]["before"]["cluster"]), np.array(data["tg"]["before"]["cluster"]))
    barplot_clusters(
        data["cg"]["before"]["cluster"],
        data["tg"]["before"]["cluster"],
        "Kontrollgruppe (vorher)",
        "Testgruppe (vorher)",
        out_path=out_path,
    )
    barplot_clusters(
        data["tg"]["before"]["cluster"],
        data["tg"]["after"]["cluster"],
        "Test Gruppe vorher",
        "Test Gruppe nachher",
        out_path=out_path,
    )
    f.close()
