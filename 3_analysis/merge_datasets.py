import os
import numpy as np
import pandas as pd
import scipy
import argparse
import sys

from clustering import ClusterWrapper
from compare_clustering import compute_all_scores
from label_analysis import entropy
from plotting import plot_correlation_matrix


def load_all(path, feature_type="graph", node_importance=0):
    """
    type: one of graph, raw
    """
    all_together = []
    study_labels = []
    for f in os.listdir(path):
        if feature_type not in f or f[-3:] != "csv":
            continue
        study = f.split(f"_{feature_type}_features")[0]
        # print("loading", f, "study:", study)
        graph_features = pd.read_csv(os.path.join(path, f), index_col="user_id")
        all_together.append(graph_features)
        # for study in STUDIES (old version)
        # graph_features = pd.read_csv(
        #     os.path.join(path, f"{study}_graph_features_{node_importance}.csv"), index_col="user_id"
        # )
        # if feature_type == "graph":
        #     all_together.append(graph_features)
        # elif feature_type == "raw":
        #     raw_features = pd.read_csv(
        #         os.path.join(path, f"{study}_raw_features_{node_importance}.csv"), index_col="user_id"
        #     )
        #     raw_features = raw_features[raw_features.index.isin(graph_features.index)]
        #     all_together.append(raw_features)

        study_labels.extend([study for _ in range(len(graph_features))])
    # concatenate
    features_all_datasets = pd.concat(all_together)
    features_all_datasets["study"] = study_labels
    print("Samples per study:", np.unique(study_labels, return_counts=True))
    return features_all_datasets


def mean_features_by_study(features, out_path=None):
    # leave away last column because it's the study label
    agg = {feat: ["mean", "std"] for feat in features.columns[:-1]}
    # group and aggregate
    mean_features = features.groupby("study").agg(agg).round(2)
    if out_path:
        mean_features.to_csv(out_path)
    else:
        print(mean_features)


def remove_outliers(feature_df, outlier_thresh, out_dir, name="", feature_type="graph", node_importance=0):

    # add study to index
    feature_df = feature_df.reset_index().set_index(["user_id", "study"])

    out_below, out_above = outlier_thresh

    filtered_above = feature_df[feature_df < out_above]
    filtered_below = filtered_above[filtered_above > out_below]

    name = "_datasets" if name == "" else name

    # save outliers for reporting in the paper
    nan_rows = filtered_below[filtered_below.isna().any(axis=1)]
    nan_rows.to_csv(os.path.join(out_dir, f"outliers{name}.csv"))

    features_cleaned = filtered_below.dropna().reset_index().set_index("user_id")
    # save in cleaned folder
    features_cleaned.to_csv(os.path.join(out_dir, f"all{name}_{feature_type}_features_{node_importance}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inp_dir", type=str, default=os.path.join("out_features", "test"), help="input directory"
    )
    args = parser.parse_args()

    # parameters
    nodes = 0
    path = args.inp_dir
    feature_type = "graph"
    cutoff = 4

    out_dir = path + "_cleaned"
    os.makedirs(out_dir, exist_ok=True)

    features_main_datasets = load_all(path, feature_type=feature_type, node_importance=nodes)
    print(len(features_main_datasets))
    features_main_datasets.drop("mean_waiting_time", axis=1, inplace=True, errors="ignore")
    features_main_datasets.dropna(inplace=True)
    print("after dropping NaNs:", features_main_datasets.shape)

    # Remove outliers
    main_arr = np.array(features_main_datasets.drop("study", axis=1))
    mean, std = (np.mean(main_arr, axis=0), np.std(main_arr, axis=0))
    print("mean and std:", mean, std)
    # outliers are above or below cutoff times the std
    outlier_thresh = (mean - cutoff * std, mean + cutoff * std)
    # Quantile Methode:
    # quan_75 = np.quantile(main_arr, 0.75, axis=0)
    # quan_25 = np.quantile(main_arr, 0.25, axis=0)
    # box_length = quan_75 - quan_25
    # print(np.around(quan_25, 2))
    # print(np.around(quan_75, 2))
    # print(np.around(quan_25 - 1.5 * box_length, 2))
    # outlier_thresh = (quan_25 - 1.5 * box_length, quan_75 + 1.5 * box_length)
    print("outlier thresh", outlier_thresh)

    # all others use the same outlier threshold!
    for name in ["", "_long_yumuv", "_long_gc1", "_long_gc2"]:
        remove_outliers(features_main_datasets, outlier_thresh, out_dir, name=name, feature_type=feature_type)

    # print mean and std: needs to be adopted to new code structure
    # mean_features_by_study(features_all_datasets, out_path=os.path.join(out_dir, f"dataset_{nodes}.csv"))

    # Plot correlation matrix with subset of studies:
    features_all_datasets = pd.read_csv(
        os.path.join(out_dir, f"all_datasets_{feature_type}_features_{nodes}.csv"), index_col="user_id"
    )
    feats_wostudy = features_all_datasets.drop(columns=["study"])
    plot_correlation_matrix(feats_wostudy, feats_wostudy, save_path=os.path.join(out_dir, f"correlation_{nodes}.pdf"))
