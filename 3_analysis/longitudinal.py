import argparse
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

from analyze_yumuv import plot_longitudinal
from find_groups import cluster_characteristics, sort_clusters_into_groups
from clustering import ClusterWrapper
from plotting import print_chisquare

# --------------------------- Investigations of time consistency for GC ----------------------------------------


def concat_studies(in_path_normal, in_path_dur):
    first_part = pd.read_csv(os.path.join(in_path_normal, f"all_datasets_graph_features_0.csv"), index_col="user_id")
    all_together = []
    study_labels = []
    for f in os.listdir(in_path_dur):
        if not f[-3:] == "csv":
            continue
        study = f.split("graph")[0]
        graph_features = pd.read_csv(os.path.join(in_path_dur, f), index_col="user_id")
        all_together.append(graph_features)
        study_labels.extend([study for _ in range(len(graph_features))])
    # concatenate
    features_for_clustering = pd.concat([first_part, pd.concat(all_together)])
    return features_for_clustering


def fit_on_before():
    STUDY = "gc1"
    dur_graph_path = "out_features/dur_graphs_" + STUDY

    n_clusters = 6
    mean_changed = []
    time_bin_list = [4 * (i + 1) for i in range(7)]
    plt.figure(figsize=(15, 8))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, len(time_bin_list))]
    for ind, k in enumerate(time_bin_list):  # TODO
        print("------------ ", k, "------------------")
        sorted_files = sorted([f for f in os.listdir(dur_graph_path) if f"_{k}w_" in f])
        mean_changed_timebin = []
        x_times = []
        for i in range(len(sorted_files) - 1):
            bef = pd.read_csv(os.path.join(dur_graph_path, sorted_files[i]), index_col="user_id")
            cluster_wrapper = ClusterWrapper(random_state=None)
            labels = cluster_wrapper(bef, impute_outliers=False, n_clusters=n_clusters, algorithm="kmeans")
            bef["cluster"] = labels

            aft = pd.read_csv(os.path.join(dur_graph_path, sorted_files[i + 1]), index_col="user_id")
            print("number in before and after", len(bef), len(aft))
            aft["cluster"] = cluster_wrapper.transform(aft)

            merged_groups = pd.merge(bef, aft, on="user_id", how="inner", suffixes=("_before", "_after"))
            print("number overlapping", len(merged_groups))
            if len(merged_groups) < 5:
                print("skipping not enough equal users")
                continue
            name = f"{k}w_{i}"
            ratio_changed = np.sum(merged_groups["cluster_before"] != merged_groups["cluster_after"]) / len(
                merged_groups
            )
            print(f"Ratio of {name} group that switched cluster:", ratio_changed)
            mean_changed_timebin.append(ratio_changed)
            x_times.append((i + 1) * k)
        x_times.append(x_times[-1] + k)
        mean_changed_timebin.append(mean_changed_timebin[-1])

        plt.plot(x_times, mean_changed_timebin, label=str(k) + " weeks", c=colors[ind])

        mean_changed.append(np.mean(mean_changed_timebin))
    plt.legend(ncol=3)
    plt.savefig(f"results/fit_bef_mean_changed_over_time_{STUDY}.png")
    plt.figure(figsize=(10, 5))
    plt.plot(time_bin_list, mean_changed)
    plt.savefig(f"results/fit_bef_mean_changed_{STUDY}.png")


def fit_all_timebins(path, out_dir):
    from clustering import ClusterWrapper

    # TODO: what to fit on?
    # fit on all datasets
    # graph_features = concat_studies("out_features/final_9_n0_cleaned", "out_features/dur_graphs_gc1")
    graph_features = pd.read_csv(os.path.join(path, f"all_datasets_graph_features_0.csv"), index_col="user_id")

    # STUDIES = ["gc1", "gc2", "tist_toph100", "geolife", "yumuv_graph_rep"]
    # graph_features = graph_features[graph_features["study"].isin(STUDIES)]
    # print("fitting on features of", np.unique(graph_features["study"].values))

    # fit multiple times and save to groups.json file
    cluster_wrapper = ClusterWrapper(random_state=None)
    if "study" in graph_features.columns:
        in_features = graph_features.drop(columns=["study"])
    n_clusters = 7
    labels = cluster_wrapper(in_features, impute_outliers=False, n_clusters=n_clusters, algorithm="kmeans")
    characteristics = cluster_characteristics(in_features, labels, printout=False)
    cluster_assignment = sort_clusters_into_groups(characteristics, add_groups=False, printout=False)
    cluster_wrapper.cluster_assignment = cluster_assignment
    labels_named = [cluster_assignment[label] for label in labels]
    graph_features["cluster"] = labels_named

    consistently_assigned = pd.read_csv(os.path.join(out_dir, "all_datasets_clustering.csv"), index_col="user_id")
    both = pd.merge(
        graph_features, consistently_assigned, on=("user_id", "study"), how="inner", suffixes=("_now", "_consistent")
    )
    print("Same assignment percent:", sum(both["cluster_now"] == both["cluster_consistent"]) / len(both))

    STUDY = "gc1"
    dur_graph_path = "out_features/final_0_n0_long_" + STUDY

    mean_changed = []
    time_bin_list = [4 * (i + 1) for i in range(7)]
    plt.figure(figsize=(15, 8))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, len(time_bin_list))]
    for ind, k in enumerate(time_bin_list):  # TODO
        print("------------ ", k, "------------------")
        sorted_files = sorted([f for f in os.listdir(dur_graph_path) if f"_{k}w_" in f])
        mean_changed_timebin = []
        x_times = []
        for i in range(len(sorted_files) - 1):
            bef = pd.read_csv(os.path.join(dur_graph_path, sorted_files[i]), index_col="user_id")
            aft = pd.read_csv(os.path.join(dur_graph_path, sorted_files[i + 1]), index_col="user_id")
            print("number in before and after", len(bef), len(aft))
            bef["cluster"] = cluster_wrapper.transform(bef)
            aft["cluster"] = cluster_wrapper.transform(aft)

            # TODO: decide whether filtering out other-group or not
            bef = bef[bef["cluster"] != "other"]
            aft = aft[aft["cluster"] != "other"]

            merged_groups = pd.merge(bef, aft, on="user_id", how="inner", suffixes=("_before", "_after"))
            print("number overlapping", len(merged_groups))
            if len(merged_groups) < 5:
                print("skipping not enough equal users")
                continue
            name = f"{k}w_{i}"
            ratio_changed = np.sum(merged_groups["cluster_before"] != merged_groups["cluster_after"]) / len(
                merged_groups
            )
            print(f"Ratio of {name} group that switched cluster:", ratio_changed)
            mean_changed_timebin.append(ratio_changed)
            x_times.append(i * k)
        x_times.append(x_times[-1] + k)
        mean_changed_timebin.append(mean_changed_timebin[-1])

        plt.plot(x_times, mean_changed_timebin, label=str(k) + " weeks", c=colors[ind])

        mean_changed.append(np.mean(mean_changed_timebin))
    plt.legend(ncol=3)
    plt.savefig(f"results/mean_changed_over_time_{STUDY}.png")
    plt.figure(figsize=(15, 8))
    plt.plot(time_bin_list, mean_changed)
    plt.xlabel("Tracking period (weeks)")
    plt.ylabel("Percentage of cluster changes from one bin to the next")
    plt.savefig(f"results/mean_changed_{STUDY}.png")


# if __name__=="__main__":
#     path = "results"
#     in_dir = "out_features/final_9_n0_cleaned"
#     fit_all_timebins(in_dir, path)
#     fit_on_before()
#     exit()

# ---------------------------------- Main code for longitudinal study in final version --------------------------


def merge_two(graph_features, before, after):
    return pd.merge(
        graph_features[graph_features["study"] == before],
        graph_features[graph_features["study"] == after],
        on="user_id",
        how="inner",
        suffixes=("_before", "_after"),
    )


def chi_square_longitudinal(long_tg, long_cg):
    print("------------ Chi square test for movement between clustern -----------")
    for col in long_cg.columns:
        if col not in long_tg.columns:
            long_tg[col] = 0
    long_tg = long_tg.reindex(columns=long_cg.columns)
    # print(long_tg)
    for feat, row in long_cg.iterrows():
        # print(feat, row)
        print("--------------- ", feat)
        if feat not in long_tg.index:
            print("feat not there")
            continue
        occ1 = list(row.values)
        occ2 = list(long_tg.loc[feat].values)
        print_chisquare(occ1, occ2)


def run_longitudinal(graph_features, name, before, after, out_path):
    merged_groups = merge_two(graph_features, before, after)
    print(
        f"Ratio of {name} group that switched cluster:",
        np.sum(merged_groups["cluster_before"] != merged_groups["cluster_after"]) / len(merged_groups),
    )
    long_df = plot_longitudinal(merged_groups, out_path=os.path.join(out_path, f"longitudinal_{name}.pdf"))
    # long_df.to_csv(os.path.join(out_path, f"longitudinal_{name}.csv"))
    return long_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_dir", type=str, default="results", help="input and output directory")
    parser.add_argument(
        "-p",
        "--time_period",
        type=str,
        default="12w",
        help="Time period of GC longitudinal analysis. One of 4w, 8w, 12w, 16w, 20w, 24w, 28w",
    )
    args = parser.parse_args()

    path = args.inp_dir

    # To run the time-bin analysis for GC
    # fit_all_timebins("out_features/final_0_n0_cleaned", path)
    # exit()

    out_path = os.path.join(path, "long")
    os.makedirs(out_path, exist_ok=True)

    f = open(os.path.join(path, "log_longitudinal.txt"), "w")
    sys.stdout = f

    # YUMUV
    # load features with clusters already assigned
    try:
        graph_features_yumuv = pd.read_csv(os.path.join(path, "long_yumuv_clustering.csv"), index_col="user_id")
    except FileNotFoundError:
        print("ERROR: all_long_yumuv_clustering.csv does not exist yet. Run script transform_new_features.py first")
        exit()

    # YUMUV CG
    df_cg = run_longitudinal(graph_features_yumuv, "yumuv_control_group", "yumuv_before_cg", "yumuv_after_cg", out_path)

    # YUMUV TG:
    df_tg = run_longitudinal(graph_features_yumuv, "yumuv_test_group", "yumuv_before_tg", "yumuv_after_tg", out_path)
    # Make chi square test
    chi_square_longitudinal(df_tg, df_cg)

    # GC1 first and second quarter
    graph_features_gc1 = pd.read_csv(os.path.join(path, "long_gc1_clustering.csv"), index_col="user_id")
    dur = args.time_period
    dur_dates = sorted(
        np.unique(graph_features_gc1[graph_features_gc1["study"].str.contains("dur_" + dur)]["study"].values).tolist()
    )
    for i in range(len(dur_dates) - 1):
        print(dur_dates[i], dur_dates[i + 1])
        run_longitudinal(graph_features_gc1, f"gc1_{i+1}_{i+2}", dur_dates[i], dur_dates[i + 1], out_path)
        # prev version with quarters:
        # run_longitudinal(graph_features, f"gc1_{i+1}_{i+2}", f"gc1_quarter{i+1}", f"gc1_quarter{i+2}", out_path)

    f.close()
