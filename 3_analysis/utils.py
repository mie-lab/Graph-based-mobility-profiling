import numpy as np
import pickle
import os
import json
import psycopg2
import functools
import warnings
import pandas as pd
import networkx as nx

import trackintel as ti
from future_trackintel.utils import read_graphs_from_postgresql


def sort_images_by_cluster(
    users, labels, name_mapping={}, in_img_path="graph_images/gc2/spring", out_img_path="sorted_by_cluster"
):
    """
    users: list of strings
    labels: list of same lengths containig assigned cluster for each user
    in_img_path: path to the saved images that we want to sort by cluster
    out_img_path: where to save the sorted images
    """
    import shutil

    # make dictionary
    map_dict = {user_id: cluster for (user_id, cluster) in zip(users, labels)}

    # make out dir and subdirs for each cluster
    if not os.path.exists(out_img_path):
        os.makedirs(out_img_path)
    else:
        warnings.warn("WARNING: out dir already exists")
    for cluster in np.unique(labels):
        # to name the directory, use the name from the mapping dict or by default "cluster_1" etc
        cluster_dir_name = name_mapping.get(cluster, "cluster_" + str(cluster))
        if not os.path.exists(os.path.join(out_img_path, cluster_dir_name)):
            os.makedirs(os.path.join(out_img_path, cluster_dir_name))

    # copy the images
    for user, assigned_cluster in map_dict.items():
        in_path = os.path.join(in_img_path, user + ".png")
        cluster_dir_name = name_mapping.get(assigned_cluster, "cluster_" + str(assigned_cluster))
        out_path = os.path.join(out_img_path, cluster_dir_name, user + ".png")
        print("copying from", in_path, "to", out_path)
        shutil.copy(in_path, out_path)


def normalize_features(feature_matrix):
    """
    Normalize features per column
    feature_matrix: np array of size (number of graphs x number of features)
    """
    std_cols = np.std(feature_matrix, axis=0)
    means_cols = np.mean(feature_matrix, axis=0)
    print(std_cols.shape, means_cols.shape)


def dist_to_stats(node_feats):
    """
    Summarize a distribution of features at vertex level into statistics
    """
    return [
        np.mean(node_feats),
        np.median(node_feats),
        np.std(node_feats),
        np.min(node_feats),
        np.max(node_feats),
        np.quantile(node_feats, 0.25),
        np.quantile(node_feats, 0.75),
    ]


def dist_names(feature):
    """Get features names for the above distribution function"""
    dist_feats = ["mean", "median", "std", "min", "max", "1st_quantile", "3rd_quantile"]
    return [feature + "_" + d for d in dist_feats]


# ----------------- DISTRIBUTIONS TO FIT ----------------------
def func_simple_powerlaw(x, beta):
    return x ** (-(1 + beta))


def func_truncated_powerlaw(x, delta_x, beta, kappa):
    return (x + delta_x) ** (-beta) * np.exp(-delta_x / kappa)


def log_normal(x, mu, sigma):
    factor = 1 / (x * np.sqrt(2 * np.pi) * sigma)
    exponent = (-1) * ((np.log(x) - mu) ** 2 / (2 * sigma ** 2))
    return factor * np.exp(exponent)


# ----------------- DECORATORS----------------------

# Distribution decorator
def get_distribution(func):
    """
    Get distribution features of node level features
    """

    @functools.wraps(func)
    def call_and_dist(self, *args, **kwargs):

        node_level_features = func(self, *args, **kwargs)

        return dist_to_stats(node_level_features)

    return call_and_dist


# Distribution decorator
def get_mean(func):
    """
    Get distribution features of node level features
    """

    @functools.wraps(func)
    def call_and_mean(self, *args, **kwargs):

        node_level_features = func(self, *args, **kwargs)

        return np.mean(node_level_features)

    return call_and_mean


def get_point_dist(p1, p2, crs_is_projected=False):
    if crs_is_projected:
        dist = p1.distance(p2)
    else:
        dist = ti.geogr.point_distances.haversine_dist(p1.x, p1.y, p2.x, p2.y)[0]
    return dist


# ----------------- COUNT CYCLES IN LIST OF LOCATIONS ----------------------


def count_cycles(location_list, cycle_len):
    """Count number of cycles of length cycle_len in a list of locations"""
    cycle_counter = 0
    for i in range(cycle_len, len(location_list)):
        current_loc = location_list[i]
        # cycle found
        if current_loc == location_list[i - cycle_len]:
            # check if the cycles is not smaller
            inbetween = [location_list[j] == current_loc for j in range(i - cycle_len + 1, i - 1)]
            if not any(inbetween):
                cycle_counter += 1
    return cycle_counter


def all_cycle_lengths(nodes_on_rw, resets):
    """
    nodes_on_rw: list of integers, list of nodes encountered on a random walk
    resets: List of positions when the random walk was reset to home

    Returns: cycle_lengths, list of length per encountered cycles
    """
    assert (
        len(resets) == 0 or len(np.unique(np.array(nodes_on_rw)[resets])) == 1
    ), "reset indices must always be a home node"

    last_seen = {}
    cycle_lengths = []
    for pos, node in enumerate(nodes_on_rw):
        # check if the arry was resetted - then no cycle is closed, we need to start from new
        if pos in resets:
            last_seen = {node: pos}  # only the current node is in the last seen tracker
            continue
        # for the current node, check when it was encountered last - closes any loop?
        node_last_seen_pos = last_seen.get(node, -1)
        if node_last_seen_pos >= 0:
            # print(nodes_on_rw[node_last_seen_pos:pos+1], pos - node_last_seen_pos)
            cycle_lengths.append(pos - node_last_seen_pos)

        # update last_seen
        last_seen[node] = pos
    return cycle_lengths


def old_count_cycles(location_list, max_len=5):
    """
    Compute histogram of how often cycles of size x occur
    NOTE: atm the sequence 2 0 2 1 2 would count as 2 cycles of length 2 AND one cycle of length 5
    """
    location_list = np.array(location_list)
    out_counts = np.zeros(max_len)
    for cycle_len in range(1, max_len + 1):
        # shift by x and compare
        rolled = np.roll(location_list, cycle_len)
        compare = rolled[cycle_len:] == location_list[cycle_len:]
        # count how often a cycle of cost x appears
        out_counts[cycle_len - 1] = np.sum(compare.astype(int))
    return out_counts


def clean_equal_cols(feature_df):
    cols_to_be_removed = []
    unique_value = []
    for col in feature_df.columns:
        std = np.std(feature_df[col].values)
        if std == 0:
            cols_to_be_removed.append(col)
            unique_value.append(feature_df[col].values[0])
    feature_df_rem = feature_df.drop(columns=cols_to_be_removed)
    print("Delete features because all values are the same:")
    print(list(zip(cols_to_be_removed, unique_value)))
    return feature_df_rem


def clean_equal_cols_matrix(feature_matrix, feat_names):
    """Delete all features that contain only one value"""
    std = np.std(feature_matrix, axis=0)
    zero_std = std == 0
    print("Delete features because all values are the same:")
    print([(i, j) for i, j in zip(np.array(feat_names)[zero_std], feature_matrix[0, zero_std])])
    return feature_matrix[:, ~zero_std], np.array(feat_names)[~zero_std]


def normalize_features(feature_matrix):
    """
    Normalize features per column
    feature_matrix: np array of size (number of graphs x number of features)
    """
    std_cols = np.std(feature_matrix, axis=0)
    means_cols = np.mean(feature_matrix, axis=0)
    return (feature_matrix - means_cols) / std_cols


def graph_dict_to_list(graph_dict, node_importance=50):
    users = []
    nx_graphs = []
    for user_id, ag in graph_dict.items():
        if node_importance == 0:
            ag_sub = ag.G
        else:
            important_nodes = ag.get_k_importance_nodes(node_importance)
            ag_sub = nx.DiGraph(ag.G.subgraph(important_nodes))

        # delete edges with transition weight 0:
        edges_to_delete = [(a, b) for a, b, attrs in ag_sub.edges(data=True) if attrs["weight"] < 1]
        if len(edges_to_delete) > 0:
            # print("delete edges of user", user_id, "nr edges", len(edges_to_delete))
            ag_sub.remove_edges_from(edges_to_delete)
        if ag_sub.number_of_edges() == 0:
            print("zero edges for user", user_id, " --> skip!")
            continue

        # get only the largest connected component:
        cc = sorted(nx.connected_components(ag_sub.to_undirected()), key=len, reverse=True)
        graph_cleaned = ag_sub.subgraph(cc[0])

        users.append(user_id)
        nx_graphs.append(graph_cleaned)
    return nx_graphs, users


# ----------------- I/O ----------------------


def load_graphs_pkl(path, node_importance=50):
    AG_dict = pickle.load(open(path, "rb"))
    nx_graphs, users = graph_dict_to_list(AG_dict, node_importance=node_importance)
    return nx_graphs, users


def get_con():
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
    return con


def load_graphs_cross_sectional(before_or_after="before", node_importance=0):
    con = get_con()
    graph_table_name = "before_after" if before_or_after != "full" else "full_graph"
    graph_dict = read_graphs_from_postgresql(
        graph_table_name=graph_table_name,
        psycopg_con=con,
        graph_schema_name="yumuv_graph_rep",
        file_name=before_or_after,
        decompress=True,
    )
    nx_graphs, users = graph_dict_to_list(graph_dict, node_importance=node_importance)
    return nx_graphs, users


def load_graphs_postgis(study, node_importance=0, decompress=True):
    # load login data
    con = get_con()
    graph_dict = read_graphs_from_postgresql(
        graph_table_name="full_graph",
        psycopg_con=con,
        graph_schema_name=study,
        file_name="graph_data",
        decompress=decompress,
    )
    nx_graphs, users = graph_dict_to_list(graph_dict, node_importance=node_importance)
    return nx_graphs, users


def load_user_info(study, index_col="user_id"):
    con = get_con()
    user_info = pd.read_sql_query(sql=f"SELECT * FROM {study}.user_info".format(study), con=con, index_col=index_col)
    return user_info


def load_all_questions(path="yumuv_data/yumuv_questions_all.csv"):
    return pd.read_csv(path, index_col="qname")


def load_question_mapping(before_after="before", group="cg"):
    if before_after == "before":
        group = ""
    question_mapping = pd.read_csv(f"yumuv_data/yumuv_{before_after}_{group}.csv", delimiter=";").drop(
        columns="Unnamed: 0"
    )
    # only the qname leads to unique questions
    return question_mapping.set_index("qname")


def split_yumuv_control_group(df):
    """
    Splits dataframe into users which are in treatment group (study_id: 22) and the ones that are in control group
    (study_id: 23)
    """
    user_info = load_user_info("yumuv_graph_rep", index_col="user_id")
    users_tg = user_info[user_info["study_id"] == 22].index
    users_cg = user_info[user_info["study_id"] == 23].index
    print("users in control group:", len(users_cg), "users in treatment group:", len(users_tg))
    df_cg = df[df.index.isin(users_cg)]
    df_tg = df[df.index.isin(users_tg)]
    return df_tg, df_cg
