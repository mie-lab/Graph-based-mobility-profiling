import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict

from clustering import ClusterWrapper
from find_groups import cluster_characteristics, sort_clusters_into_groups
from plotting import barplot_clusters
from analyze_study import entropy
from utils import load_user_info


def cross_sectional_yumuv(cluster_wrapper, out_path):
    # load and fit all
    yumuv_data = defaultdict(dict)
    for group in ["cg", "tg"]:
        for before_after in ["before", "after"]:
            yumuv_data[group][before_after] = pd.read_csv(
                os.path.join(path, f"yumuv_{before_after}_{group}_graph_features_0.csv"),
                index_col="user_id",
            )
            yumuv_data[group][before_after]["cluster"] = cluster_wrapper.transform(yumuv_data[group][before_after])
    barplot_clusters(
        yumuv_data["cg"]["before"]["cluster"],
        yumuv_data["tg"]["before"]["cluster"],
        "Kontrollgruppe (vorher)",
        "Testgruppe (vorher)",
        save_name="crosssectional_yumuv",
        out_path=out_path,
        rotate=False,
    )
    barplot_clusters(
        yumuv_data["tg"]["before"]["cluster"],
        yumuv_data["tg"]["after"]["cluster"],
        "Test Gruppe vorher",
        "Test Gruppe nachher",
        save_name="longitudinal_yumuv",
        out_path=out_path,
        rotate=False,
    )


def cross_sectional_gc(graph_features, out_path, gc_num):
    study_name = "gc" + str(gc_num)
    feats_gc1 = graph_features[graph_features["study"] == study_name]
    feats_others = graph_features[graph_features["study"] != study_name]
    barplot_clusters(
        feats_gc1["cluster"],
        feats_others["cluster"],
        f"Green Class {gc_num}",
        "Other studies",
        save_name="crosssectional_gc_" + str(gc_num),
        out_path=out_path,
        rotate=False,
    )


def iterate_columns_entropy(joined, user_info):
    for col in user_info.columns:
        if "id" in col:
            continue
        not_nan = pd.isna(joined[col]).sum()
        if not_nan / len(joined) > 0.5:
            # print("Skipping because too many missing values:", col)
            continue

        # entropy is not symmetric, compute both
        entropy_1 = entropy(joined, col, "cluster")

        # if entropy_1 < 0.95:
        print("\n------", col, "------")
        entropy_1 = entropy(joined, col, "cluster", print_parts=True)
        print("\nEntropy:", round(entropy_1, 2), "\n")


if __name__ == "__main__":
    path = os.path.join("out_features", "final_6_n0_cleaned")
    out_path = "figures/results_cross_sectional"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Fit all datasets
    n_clusters = 8
    graph_features = pd.read_csv(os.path.join(path, f"all_datasets_graph_features_0.csv"), index_col="user_id")
    graph_features_numerical = graph_features.drop("study", axis=1)
    cluster_wrapper = ClusterWrapper()
    labels = cluster_wrapper(graph_features_numerical, impute_outliers=False, n_clusters=n_clusters, algorithm="kmeans")
    characteristics = cluster_characteristics(graph_features_numerical, labels, printout=False)
    cluster_assignment = sort_clusters_into_groups(characteristics, add_groups=False, printout=False)
    cluster_wrapper.cluster_assignment = cluster_assignment
    graph_features["cluster"] = [cluster_assignment[lab] for lab in labels]

    # # YUMUV
    # cross_sectional_yumuv(cluster_wrapper, out_path)

    # # GC1:
    # cross_sectional_gc(graph_features, out_path, 1)
    # # GC2:
    # cross_sectional_gc(graph_features, out_path, 2)

    # LABELS GC1
    feats_gc1 = graph_features[graph_features["study"] == "gc1"]
    user_info = load_user_info("gc1", index_col="user_id")
    user_info.index = user_info.index.astype(str)
    print(feats_gc1.columns)
    # merge into one table
    joined = feats_gc1.merge(user_info, how="left", left_on="user_id", right_on="user_id")
    iterate_columns_entropy(joined, user_info)

    # # LABELS YUMUV
    # feats_yumuv = graph_features[graph_features["study"] == "yumuv_graph_rep"]
    # user_info = load_user_info(study, index_col="app_user_id")
    # user_info = user_info.reset_index().rename(columns={"app_user_id": "user_id"})
    # # merge into one table
    # joined = feats_yumuv.merge(user_info, how="left", left_on="user_id", right_on="user_id")
    # iterate_columns_entropy(joined, user_info)
