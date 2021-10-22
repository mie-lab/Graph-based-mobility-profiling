import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from collections import defaultdict

from clustering import ClusterWrapper
from find_groups import cluster_characteristics, sort_clusters_into_groups
from plotting import barplot_clusters


def cross_sectional_yumuv(graph_features, out_path):
    yumuv_cg_before = graph_features[graph_features["study"] == "yumuv_before_cg"]["cluster"]
    yumuv_tg_before = graph_features[graph_features["study"] == "yumuv_before_tg"]["cluster"]
    yumuv_tg_after = graph_features[graph_features["study"] == "yumuv_after_tg"]["cluster"]
    barplot_clusters(
        yumuv_cg_before,
        yumuv_tg_before,
        "Kontrollgruppe (vorher)",
        "Testgruppe (vorher)",
        save_name="crosssectional_yumuv",
        out_path=out_path,
        rotate=False,
    )
    barplot_clusters(
        yumuv_tg_before,
        yumuv_tg_after,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_dir", type=str, default="results", help="input and output directory")
    args = parser.parse_args()

    path = args.inp_dir

    # load data with clusters already assigned
    try:
        graph_features = pd.read_csv(os.path.join(path, "all_datasets_clustering.csv"))
    except FileNotFoundError:
        print("ERROR: all_dataset_clustering.csv file does not exist yet. Run script analyze_study.py first")
        exit()

    # YUMUV
    cross_sectional_yumuv(graph_features, path)
    # NOTE: For yumuv, it might be better to fit the clusters again on just the yumuv data. Then the differences between
    # test and control group might be larger. The code for this can however be found in the script analyze_yumuv.py

    # GC1:
    cross_sectional_gc(graph_features, path, 1)
    # GC2:
    cross_sectional_gc(graph_features, path, 2)
