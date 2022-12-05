import argparse
import os
import time
import pandas as pd
import numpy as np
import sys

from analysis_utils import split_yumuv_control_group, get_con
from clustering import ClusterWrapper
from plotting import scatterplot_matrix
from graph_features import GraphFeatures
from raw_features import RawFeatures


def remove_outliers(feature_df, cutoff=4):
    print("length before", len(feature_df))
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
    outlier_arr = np.any(outlier_arr, axis=0)
    removed_outliers = list(feature_df[outlier_arr].index)
    print("Removed users", len(removed_outliers), removed_outliers)
    feature_df = feature_df[~outlier_arr]
    print("length after", len(feature_df))
    return feature_df


def clean_features(path, cutoff=4):
    """Delete users with nans or with outliers (cutoff * std from mean)"""

    out_path = path + "_cleaned"
    if not os.path.exists(out_path):
        print(out_path)
        os.makedirs(out_path)

    for f in os.listdir(path):
        # quarters: only remove outliers:
        if "quarter" in f:  # TODO: remove the ones that are not in raw features or not?
            # if f[-3:] == "csv" and "raw" not in f:  #
            #     print()
            #     print(f)
            feature_df = pd.read_csv(os.path.join(path, f), index_col="user_id")
            feature_df = remove_outliers(feature_df)
            feature_df.to_csv(os.path.join(out_path, f))
            # continue

        # skip raw features or after features becuase they must be matched
        if f[-3:] != "csv" or "raw" in f or "after" in f or "quarter" in f:
            continue
        print("---------", f)
        graph_path = os.path.join(path, f)
        raw_path = os.path.join(path, f).replace("graph", "raw")

        graph_features = pd.read_csv(graph_path, index_col="user_id")
        # drop certain columns
        # graph_features.drop(columns=["mean_trip_distance", "quantile9_trip_distance"], inplace=True)

        if "yumuv" not in f:
            raw_features = pd.read_csv(raw_path, index_col="user_id")
            feature_df = pd.merge(graph_features, raw_features, on="user_id", how="inner")
        else:
            if "before" in f:
                raw_path = os.path.join(path, f).replace("before", "after")
                raw_features = pd.read_csv(raw_path, index_col="user_id")
                feature_df = pd.merge(graph_features, raw_features, on="user_id", how="inner")
            else:
                raw_features = graph_features.copy()
                feature_df = graph_features
        print("len prev", len(feature_df))

        u_o1 = [u_id for u_id in graph_features.index if u_id not in feature_df.index]
        print("Users in graph features but not in raw (bzw for yumuv, in before but not in after):", len(u_o1), u_o1)
        u_o2 = [u_id for u_id in raw_features.index if u_id not in feature_df.index]
        print("Users in raw features but not in graph (bzw for yumuv, in after but not in before):", len(u_o2), u_o2)

        feature_df = remove_outliers(feature_df)

        graph_features = graph_features[graph_features.index.isin(feature_df.index)]
        graph_features.to_csv(os.path.join(out_path, f))
        if "yumuv" not in f:
            raw_features = raw_features[raw_features.index.isin(feature_df.index)]
            raw_features.to_csv(os.path.join(out_path, f).replace("graph", "raw"))
        elif "before" in f:
            after_features = raw_features[raw_features.index.isin(feature_df.index)]
            after_features.to_csv(os.path.join(out_path, f).replace("before", "after"))


def get_graph_and_raw(inp_dir, out_dir, node_importance, feat_type="graph"):
    # if no database access, just process the three public datasets
    studies_to_process = ["geolife", "tist_toph100", "tist_random100"]
    if inp_dir == "postgis":
        studies_to_process = studies_to_process + ["gc1", "gc2"]

    for study in studies_to_process:

        print("\n -------------- PROCESS", study, feat_type, " ---------------")

        # Generate feature matrix
        tic = time.time()
        if feat_type == "raw":
            trips_available = "tist" not in study  # for tist, the trips are missing
            feat_class = RawFeatures(inp_dir, study, trips_available=trips_available)
            select_features = "all"
        else:
            feat_class = GraphFeatures(inp_dir, study, node_importance=node_importance)
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

    print("\n----------------- PROCESS YUMUV --------------------")
    runner_all_feat = GraphFeatures("postgis", "yumuv_graph_rep", node_importance=node_importance)
    full_features = runner_all_feat(features="default")
    full_features.to_csv(os.path.join(out_dir, f"yumuv_graph_rep_graph_features_{node_importance}.csv"))

    print("\n----------------- GET YUMUV BEF AND AFT --------------------")

    print("Run yumuv before")
    runner_before_feat = GraphFeatures("postgis", "yumuv_before", node_importance=node_importance)
    before_features = runner_before_feat(features="default")

    print("\nRun yumuv after")
    runner_after_feat = GraphFeatures("postgis", "yumuv_after", node_importance=node_importance)
    after_features = runner_after_feat(features="default")

    yumuv_dir = out_dir + "_long_yumuv"
    os.makedirs(yumuv_dir, exist_ok=True)

    # split in cg and tg
    for features, name in zip([before_features, after_features], ["before", "after"]):
        tg, cg = split_yumuv_control_group(features)
        cg.to_csv(os.path.join(yumuv_dir, f"yumuv_{name}_cg_graph_features_{node_importance}.csv"))
        tg.to_csv(os.path.join(yumuv_dir, f"yumuv_{name}_tg_graph_features_{node_importance}.csv"))

    # save cg for yumuv with whole time period
    # tg, cg_all = split_yumuv_control_group(full_features)
    # cg_all.to_csv(os.path.join(out_dir, f"yumuv_cg_graph_features_{node_importance}.csv"))


def get_gc_quarters(out_dir, node_importance):
    quarter_dir = out_dir + "_quarter"
    os.makedirs(quarter_dir, exist_ok=True)
    for quarter_ind in range(1, 5):
        quarter = "gc1_quarter" + str(quarter_ind)
        runner_all_feat = GraphFeatures("postgis", quarter, node_importance=node_importance)
        full_features = runner_all_feat(features="default")
        full_features.to_csv(os.path.join(quarter_dir, quarter + f"_graph_features_{node_importance}.csv"))


def get_timebins(out_dir, node_importance=0):
    print("----------------- GET TIME BINS FOR GC 1 and 2 --------------------")
    con = get_con()

    for study in ["gc1", "gc2"]:
        # Make new directory for this duration data
        timebin_dir = out_dir + "_long_" + study
        os.makedirs(timebin_dir, exist_ok=True)
        # Run
        for weeks in [4 * (i + 1) for i in range(7)]:
            print("processing weeks:", weeks, "STUDY", study)
            cur = con.cursor()
            # get the timebin names
            cur.execute(f"SELECT name FROM {study}.dur_{weeks}w")
            all_names = [f"dur_{weeks}w_{name[0]}_{study}" for name in cur.fetchall()]
            for name in all_names:
                if os.path.exists(os.path.join(timebin_dir, name + f"_graph_features_{node_importance}.csv")):
                    print("already done", name)
                    continue
                print("processing graphs from", name)
                runner_all_feat = GraphFeatures("postgis", name, node_importance=node_importance)
                full_features = runner_all_feat(features="default")
                full_features.to_csv(os.path.join(timebin_dir, name + f"_graph_features_{node_importance}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, default=os.path.join("data", "graph_data"))
    parser.add_argument(
        "-o", "--out_dir", type=str, default=os.path.join("out_features", "test"), help="output directory"
    )
    parser.add_argument("-n", "--nodes", type=int, default=0, help="number of x important nodes. Set 0 for all nodes")
    parser.add_argument(
        "-f", "--feat_type", type=str, default="graph", help="Compute graph features (graph) or basic features (raw)"
    )
    args = parser.parse_args()

    if not os.path.exists("out_features"):
        os.makedirs("out_features")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # write terminal output to file:
    f = open(os.path.join(args.out_dir, "terminal.txt"), "w")
    sys.stdout = f

    # Process graph and raw features for all studies, then add yumuv, then clean
    get_graph_and_raw(args.in_path, args.out_dir, args.nodes, feat_type=args.feat_type)
    if args.in_path == "postgis":
        get_yumuv(args.out_dir, args.nodes)
        get_timebins(args.out_dir)
    # clean_features(args.out_dir)

    f.close()
