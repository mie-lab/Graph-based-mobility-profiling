import numpy as np
import pickle
import os
import json
import psycopg2

from future_trackintel.utils import read_graphs_from_postgresql


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


def count_cycles(location_list, max_len=5):
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


def pca(feature_matrix, n_components=2):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(feature_matrix)
    return projected


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
        users.append(user_id)
        # TODO: rewrite k importance nodes such that it is filtered by the fraction of occurence, not the abs number
        important_nodes = ag.get_k_importance_nodes(node_importance)
        ag_sub = ag.G.subgraph(important_nodes)
        nx_graphs.append(ag_sub)
    return nx_graphs, users


def load_graphs_pkl(path, node_importance=50):
    AG_dict = pickle.load(open(path, "rb"))
    nx_graphs, users = graph_dict_to_list(AG_dict, node_importance=node_importance)
    return nx_graphs, users


def load_graphs_postgis(study, node_importance=50):
    # load login data
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
    nx_graphs, users = graph_dict_to_list(graph_dict, node_importance=node_importance)
    return nx_graphs, users