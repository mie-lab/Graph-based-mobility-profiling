import argparse
import os
import time
import pandas as pd
import numpy as np

from utils import split_yumuv_control_group
from clustering import normalize_and_cluster
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
        if f[-3:] != "csv" or "raw" in f:
            continue
        print("---------", f)
        graph_path = os.path.join(path, f)
        raw_path = os.path.join(path, f).replace("graph", "raw")

        graph_features = pd.read_csv(graph_path, index_col="user_id")

        if "yumuv" not in f:
            raw_features = pd.read_csv(raw_path, index_col="user_id")
            feature_df = pd.merge(graph_features, raw_features, on="user_id", how="inner")
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
        feature_df = feature_df[~outlier_arr]
        print(len(feature_df))

        graph_features = graph_features[graph_features.index.isin(feature_df.index)]
        graph_features.to_csv(os.path.join(out_path, f))
        if "yumuv" not in f:
            raw_features = raw_features[raw_features.index.isin(feature_df.index)]
            graph_features.to_csv(os.path.join(out_path, f).replace("graph", "raw"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, default="out_features", help="output directory")
    parser.add_argument("-n", "--nodes", type=int, default=0, help="number of x important nodes. Set -1 for all nodes")
    args = parser.parse_args()

    out_dir = args.out_dir
    node_importance = args.nodes

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for study in ["gc1", "gc2", "yumuv_graph_rep", "geolife", "tist_toph100"]:
        for feat_type in ["raw", "graph"]:
            # for yumuv I don't have the raw data
            if study == "yumuv_graph_rep" and feat_type == "raw":
                continue

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

            # for yumuv: split in cg and tg
            if study == "yumuv_graph_rep":
                cg, tg = split_yumuv_control_group(features)
                cg.to_csv(out_path.replace("yumuv_graph_rep", "yumuv_cg") + ".csv")
                tg.to_csv(out_path.replace("yumuv_graph_rep", "yumuv_tg") + ".csv")

            # geolife has nan rows, drop them first
            features.dropna(inplace=True)
            labels = normalize_and_cluster(features, n_clusters=2)
            try:
                scatterplot_matrix(features, features.columns, clustering=labels, save_path=out_path + ".pdf")
            except:
                continue

    clean_features(out_dir)
