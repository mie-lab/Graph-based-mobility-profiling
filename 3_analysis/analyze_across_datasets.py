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


def load_all(path, type="graph", node_importance=50):
    """
    type: one of graph, raw
    """
    all_together = []
    study_labels = []
    for study in STUDIES:
        graph_features = pd.read_csv(
            os.path.join(path, f"{study}_graph_features_{node_importance}.csv"), index_col="user_id"
        )
        if type == "graph":
            all_together.append(graph_features)
        elif type == "raw":
            raw_features = pd.read_csv(
                os.path.join(path, f"{study}_raw_features_{node_importance}.csv"), index_col="user_id"
            )
            raw_features = raw_features[raw_features.index.isin(graph_features.index)]
            all_together.append(raw_features)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inp_dir", type=str, default=os.path.join("out_features", "test"), help="input directory"
    )
    parser.add_argument("-o", "--out_dir", type=str, default=os.path.join("results"), help="output directory")
    args = parser.parse_args()

    STUDIES = [
        "gc1",
        "gc2",
        "tist_toph100",
        "geolife",
        "yumuv_graph_rep",
        "yumuv_before_cg",
        "yumuv_after_cg",
        "yumuv_before_tg",
        "yumuv_after_tg",
    ]
    # parameters
    nodes = 0
    path = args.inp_dir  # os.path.join("out_features", f"final_{feat_id}_n{nodes}_cleaned")
    feature_type = "graph"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # tist does not have trip data
    if feature_type == "raw" and "tist_toph100" in STUDIES:
        STUDIES.remove("tist_toph100")

    features_all_datasets = load_all(path, type=feature_type, node_importance=nodes)
    features_all_datasets.to_csv(os.path.join(path, f"all_datasets_{feature_type}_features_{nodes}.csv"))

    mean_features_by_study(features_all_datasets, out_path=os.path.join(args.out_dir, f"dataset_{nodes}.csv"))

    # Plot correlation matrix with subset of studies:
    features_all_datasets = features_all_datasets[features_all_datasets["study"].isin(STUDIES)]
    feats_wostudy = features_all_datasets.drop(columns=["study"])
    plot_correlation_matrix(
        feats_wostudy, feats_wostudy, save_path=os.path.join(args.out_dir, f"correlation_{nodes}.pdf")
    )
