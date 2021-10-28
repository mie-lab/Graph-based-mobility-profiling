import argparse
import os
import pandas as pd
import numpy as np

from analyze_yumuv import plot_longitudinal


def merge_two(graph_features, before, after):
    return pd.merge(
        graph_features[graph_features["study"] == before],
        graph_features[graph_features["study"] == after],
        on="user_id",
        how="inner",
        suffixes=("_before", "_after"),
    )


def run_longitudinal(graph_features, name, before, after, out_path):
    merged_groups = merge_two(graph_features, before, after)
    print(
        f"Ratio of {name} group that switched cluster:",
        np.sum(merged_groups["cluster_before"] != merged_groups["cluster_after"]) / len(merged_groups),
    )
    long_df = plot_longitudinal(merged_groups, out_path=os.path.join(out_path, f"longitudinal_{name}.png"))
    long_df.to_csv(os.path.join(out_path, f"longitudinal_{name}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_dir", type=str, default="results", help="input and output directory")
    args = parser.parse_args()

    path = args.inp_dir

    # load data with clusters already assigned
    try:
        graph_features = pd.read_csv(os.path.join(path, "all_datasets_clustering.csv"), index_col="user_id")
    except FileNotFoundError:
        print("ERROR: all_dataset_clustering.csv file does not exist yet. Run script analyze_study.py first")
        exit()

    # YUMUV CG
    run_longitudinal(graph_features, "yumuv_control_group", "yumuv_before_cg", "yumuv_after_cg", path)

    # YUMUV TG:
    run_longitudinal(graph_features, "yumuv_test_group", "yumuv_before_tg", "yumuv_after_tg", path)

    # GC1 first and second quarter
    # run_longitudinal(graph_features, "gc1_1st_2nd", "gc1_quarter1", "gc1_quarter2")
    # ... other quarters
