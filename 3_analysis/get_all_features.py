import argparse
import os
import time
import pandas as pd
import numpy as np

from utils import split_yumuv_control_group
from clustering import ClusterWrapper
from plotting import scatterplot_matrix
from graph_features import GraphFeatures
from raw_features import RawFeatures


def clean_features(path, cutoff=4):
    """Delete users with nans or with outliers (cutoff * std from mean)"""

    out_path = path + "_cleaned"
    if not os.path.exists(out_path):
        print(out_path)
        os.makedirs(out_path)

    for f in os.listdir(path):
        # skip raw features or after features becuase they must be matched
        if f[-3:] != "csv" or "raw" in f or "after" in f:
            continue
        print("---------", f)
        graph_path = os.path.join(path, f)
        raw_path = os.path.join(path, f).replace("graph", "raw")

        graph_features = pd.read_csv(graph_path, index_col="user_id")

        if "yumuv" not in f:
            raw_features = pd.read_csv(raw_path, index_col="user_id")
            feature_df = pd.merge(graph_features, raw_features, on="user_id", how="inner")
        else:
            if "before" in f:
                raw_path = os.path.join(path, f).replace("before", "after")
                after_features = pd.read_csv(raw_path, index_col="user_id")
                feature_df = pd.merge(graph_features, after_features, on="user_id", how="inner")
            else:
                feature_df = graph_features
        print("len prev", len(feature_df))

        feature_df = feature_df.dropna()
        features = np.array(feature_df)
        outlier_arr = []
        for i in range(features.shape[1]):
            col_vals = features[:, i].copy()
            mean, std = (np.mean(col_vals), np.std(col_vals))
            # outliers are above or below cutoff times the std
            outlier_thresh = (mean - cutoff * std, mean + cutoff * std)
            outlier = (col_vals < outlier_thresh[0]) | (col_vals > outlier_thresh[1])
            outlier_arr.append(outlier)

        outlier_arr = np.array(outlier_arr)
        #     print(outlier_arr.shape)
        outlier_arr = np.any(outlier_arr, axis=0)
        print("removed users", list(feature_df[outlier_arr].index))
        feature_df = feature_df[~outlier_arr]
        print(len(feature_df))

        graph_features = graph_features[graph_features.index.isin(feature_df.index)]
        graph_features.to_csv(os.path.join(out_path, f))
        if "yumuv" not in f:
            raw_features = raw_features[raw_features.index.isin(feature_df.index)]
            raw_features.to_csv(os.path.join(out_path, f).replace("graph", "raw"))
        elif "before" in f:
            after_features = after_features[after_features.index.isin(feature_df.index)]
            after_features.to_csv(os.path.join(out_path, f).replace("before", "after"))


def get_graph_and_raw(out_dir, node_importance):

    for study in ["gc1", "gc2", "geolife", "tist_toph100"]:
        for feat_type in ["graph"]:

            print(" -------------- PROCESS", study, feat_type, " ---------------")

            # Generate feature matrix
            tic = time.time()
            if feat_type == "raw":
                trips_available = "tist" not in study  # for tist, the trips are missing
                feat_class = RawFeatures(study, trips_available=trips_available)
                select_features = "all"
            else:
                feat_class = GraphFeatures(study, node_importance=node_importance)
                select_features = "default"

            features = feat_class(features=select_features)
            print(features)
            print("time for feature generation", time.time() - tic)

            out_path = os.path.join(out_dir, f"{study}_{feat_type}_features_{node_importance}")

            features.to_csv(out_path + ".csv")

            # geolife has nan rows, drop them first
            features.dropna(inplace=True)
            cluster_wrapper = ClusterWrapper()
            labels = cluster_wrapper(features, n_clusters=2)
            try:
                scatterplot_matrix(features, features.columns, clustering=labels, save_path=out_path + ".pdf")
            except:
                continue


def get_yumuv(out_dir, node_importance):

    print("Run full yumuv")
    runner_before_feat = GraphFeatures("yumuv_graph_rep", node_importance=node_importance)
    full_features = runner_before_feat(features="default")
    full_features.to_csv(os.path.join(out_dir, f"yumuv_graph_rep_graph_features_{node_importance}.csv"))

    print("Run yumuv before")
    runner_before_feat = GraphFeatures("yumuv_before", node_importance=node_importance)
    before_features = runner_before_feat(features="default")

    print("Run yumuv after")
    runner_after_feat = GraphFeatures("yumuv_after", node_importance=node_importance)
    after_features = runner_after_feat(features="default")

    # split in cg and tg
    for features, name in zip([before_features, after_features], ["before", "after"]):
        tg, cg = split_yumuv_control_group(features)
        cg.to_csv(os.path.join(out_dir, f"yumuv_{name}_cg_graph_features_{node_importance}.csv"))
        tg.to_csv(os.path.join(out_dir, f"yumuv_{name}_tg_graph_features_{node_importance}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, default="out_features/test", help="output directory")
    parser.add_argument("-n", "--nodes", type=int, default=0, help="number of x important nodes. Set 0 for all nodes")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Process graph and raw features for all studies, then add yumuv, then clean
    get_graph_and_raw(args.out_dir, args.nodes)
    get_yumuv(args.out_dir, args.nodes)
    clean_features(args.out_dir)
