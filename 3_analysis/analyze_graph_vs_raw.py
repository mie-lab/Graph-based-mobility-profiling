from networkx.algorithms.operators.unary import reverse
from numpy.lib.stride_tricks import broadcast_shapes
import sklearn
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import scipy
import sys

from clustering import ClusterWrapper, decision_tree_cluster
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import adjusted_rand_score
from plotting import plot_correlation_matrix, plot_cluster_characteristics
from find_groups import cluster_characteristics


def print_correlated_features(graph_features, raw_features):
    for raw_feat in raw_features.columns:
        # exclude waiting times, not useful
        if "waiting_time" in raw_feat:
            continue
        for graph_feat in graph_features.columns:
            r, p = scipy.stats.pearsonr(raw_features[raw_feat], graph_features[graph_feat])
            if abs(r) > 0.2:
                print("Correlation {:<25} and {:<25}: {:<5}".format(raw_feat, graph_feat, r))


def get_important_features(features, labels, n_important=4, method="forest"):
    assert method in ["tree", "forest"]
    if method == "forest":
        forest = RandomForestClassifier()
        forest.fit(np.array(features), labels)
        feature_importances = forest.feature_importances_
    elif method == "tree":
        feature_importances = decision_tree_cluster(features, labels)

    important_feature_inds = np.flip(np.argsort(feature_importances)[-n_important:])
    return np.array(features.columns)[important_feature_inds], feature_importances[important_feature_inds]


def predict_cluster_RF(graph_features, raw_features):
    assert all(raw_features.index == graph_features.index)

    for n_clusters in range(2, 9):
        # cluster with both separately
        cluster_wrapper = ClusterWrapper()
        labels_graph = cluster_wrapper(graph_features, n_clusters=n_clusters)
        labels_raw = cluster_wrapper(raw_features, n_clusters=n_clusters)

        # predict graph clusters with raw features
        print("------------", n_clusters)
        for feats_in, feat_name in zip([np.array(raw_features), np.array(graph_features)], ["raw", "graph"]):
            for clusters_in, cluster_name in zip([labels_raw, labels_graph], ["raw", "graph"]):
                forest = RandomForestClassifier(oob_score=True)
                forest.fit(feats_in, clusters_in)
                print(
                    f"Ability to predict {cluster_name} clusters with {feat_name} features:",
                    round(forest.oob_score_, 2),
                )


def single_feature_rf(graph_features, raw_features):
    raw_feat_arr = np.array(raw_features)
    for feat in graph_features.columns:
        forest = RandomForestRegressor(oob_score=True)
        forest.fit(raw_feat_arr, graph_features[feat].values)
        print(
            f"Ability to predict feature {feat}",
            round(forest.oob_score_, 2),
        )


def returner_explorers(path_to_returner, graph_features):
    k_returners = pd.read_csv(path_to_returner, index_col="user_id")
    # fill Nans with highest k
    k_returners.loc[k_returners[pd.isna(k_returners["k_returner"])].index, "k_returner"] = 50
    median_k = np.median(k_returners["k_returner"].values)
    k_returners["explorer"] = k_returners["k_returner"].apply(lambda x: x > median_k)
    # align index
    k_returners = k_returners.loc[graph_features.index]
    # set the returner vs explorer feature as index
    print("NOTE: cluster 0 are returners, cluster 1 are explorers")
    graph_features.loc[k_returners.index, "cluster"] = k_returners["explorer"].astype(int)

    print("Features that are significantly different between returners and explorers:")
    cluster_characteristics(graph_features)

    # print("Feature importances to predict returners and explorers according to decision tree:")
    # feature_importances = decision_tree_cluster(
    #     graph_features.drop(columns=["cluster"]), graph_features["cluster"].values
    # )
    # important_feature_inds = np.flip(np.argsort(feature_importances)[-4:])
    # print(np.array(graph_features.columns)[important_feature_inds], feature_importances[important_feature_inds])


def graph_raw_all_datasets(base_path, graph_feat_path, studies_raw=["gc1", "gc2", "geolife"]):
    graph_feats = pd.read_csv(
        os.path.join(graph_feat_path, "all_datasets_clustering.csv"), index_col=["user_id", "study"]
    )

    # collect all raw feats
    raw_feats = []
    for study in studies_raw:
        feat = pd.read_csv(os.path.join(base_path, f"{study}_raw_features_0.csv"))
        feat["user_id"] = feat["user_id"].astype(str)
        feat["study"] = study
        raw_feats.append(feat)
    raw_feats = pd.concat(raw_feats).set_index(["user_id", "study"])

    # filter for common
    together = pd.merge(
        graph_feats,
        raw_feats,
        on=("user_id", "study"),
        how="inner",
        suffixes=("_before", "_after"),
    )
    # # check the number of data
    # print(together.columns)
    # print(len(together), len(graph_feats), len(raw_feats))
    # print(together.reset_index().groupby("study").agg({"study": "count"}))
    # print(graph_feats.reset_index().groupby("study").agg({"study": "count"}))
    # print(raw_feats.reset_index().groupby("study").agg({"study": "count"}))

    graph_feats_filtered = together[graph_feats.columns]
    raw_feats_filtered = together[raw_feats.columns]
    return graph_feats_filtered, raw_feats_filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--study", type=str, required=True, help="study - one of gc1, gc2, geolife or all_datasets"
    )
    parser.add_argument(
        "-i", "--inp_dir", type=str, default=os.path.join("out_features", "test"), help="Output directory"
    )
    parser.add_argument("-n", "--nodes", type=int, default=0, help="number of x important nodes. Set -1 for all nodes")
    parser.add_argument("-o", "--out_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    path = args.inp_dir
    study = args.study
    node_importance = args.nodes
    out_dir = args.out_dir

    n_clusters = 5
    algorithm = "kmeans"

    f = open(os.path.join(out_dir, "graph_vs_raw.txt"), "w")
    sys.stdout = f

    if study == "all_datasets":
        # load features ALL STUDIES
        graph_features, raw_features = graph_raw_all_datasets(args.inp_dir, args.out_dir)
        # graph_features = pd.read_csv("out_features/final_6_n0_cleaned/all_graph_clustering.csv", index_col="user_id")
        labels_graph_orig = graph_features["cluster"]
        graph_features.drop("cluster", axis=1, inplace=True)
        # raw_features = pd.read_csv("out_features/final_6_n0_cleaned/all_raw.csv", index_col="user_id")
    else:
        # load features SINGLE STUDY
        graph_features = pd.read_csv(
            os.path.join(path, f"{study}_graph_features_{node_importance}.csv"), index_col="user_id"
        )
        labels_graph_orig = None
        raw_features = pd.read_csv(
            os.path.join(path, f"{study}_raw_features_{node_importance}.csv"), index_col="user_id"
        )
        raw_features = raw_features.loc[graph_features.index]
    # rename because of clash with raw features
    graph_features.rename(columns={"mean_trip_distance": "mean_trip_dist_graph"}, inplace=True)
    raw_features.drop("mean_waiting_time", axis=1, inplace=True)  # too much noise in waiting time
    assert all(raw_features.index == graph_features.index)
    print("features shape:", graph_features.shape, raw_features.shape)

    # # CORRELATIONS
    # plot correlation matrix of all features to each other
    both = raw_features.join(graph_features)
    plot_correlation_matrix(both, both, fontsize=12, save_path=os.path.join(out_dir, "correlations_w_raw.pdf"))
    # plot_correlation_matrix(graph_features, raw_features)
    # print_correlated_features(graph_features, raw_features)

    # Use random forest RF to predict graph clusters with raw features and the other way round:
    print("Predict CLUSTER with random forest:")
    predict_cluster_RF(graph_features, raw_features)

    print("\n ----------------------------------- \n")
    print("Predict FEATURE with random forest (regressor):")
    single_feature_rf(graph_features, raw_features)

    print("\n ----------------------------------- \n")

    cluster_wrapper = ClusterWrapper()

    # cluster both with their features, compute similarity:
    labels_graph = cluster_wrapper(graph_features, algorithm=algorithm)
    if labels_graph_orig is not None:
        labels_graph = labels_graph_orig
    labels_raw = cluster_wrapper(raw_features, algorithm=algorithm, n_clusters=8)
    print("rand score before", adjusted_rand_score(labels_raw, labels_graph))

    # get best raw features to explain graph features
    selected_features, importances = get_important_features(raw_features, labels_graph)

    raw_filtered = raw_features[selected_features]
    print("Selected raw features to predict graph features:", list(raw_filtered.columns))
    raw_labels_filtered = cluster_wrapper(raw_filtered, algorithm=algorithm)
    print("rand score after filtering", adjusted_rand_score(raw_labels_filtered, labels_graph))

    # get five most important features:
    # important_feature_inds = np.argsort(feature_importances)[-5:]
    # print(np.array(raw_features.columns)[important_feature_inds], feature_importances[important_feature_inds])
    returner_path = os.path.join(path, f"{study}_returner_explorer.csv")
    if os.path.exists(returner_path):
        print("\n ------------------ Returner explorer ----------------")
        returner_explorers(returner_path, graph_features)

    # Plot which raw features are significantly higher or lower than the graph features
    new_feats = raw_features.copy()
    new_feats["study"] = "all_relevant"
    new_feats["cluster"] = labels_graph
    feat_columns = raw_features.columns
    plot_cluster_characteristics(
        new_feats,
        out_path=os.path.join(out_dir, "raw_cluster_characteristics.pdf"),
        feat_columns=feat_columns,
        fontsize_dict={"font.size": 26, "axes.labelsize": 30},
    )

    f.close()
