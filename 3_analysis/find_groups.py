import os
import numpy as np
import pandas as pd
import scipy
import json
import argparse
import sys
import shutil

from clustering import ClusterWrapper
from utils import sort_images_by_cluster
from plotting import scatterplot_matrix, plot_cluster_characteristics, cluster_by_study


def cluster_characteristics(in_features, cluster_labels=None, printout=True):
    features = in_features.copy()
    if cluster_labels is not None:
        features["cluster"] = cluster_labels
    labels = features["cluster"]
    characteristics = {}
    for cluster in np.unique(labels):
        if printout:
            print(f"------- Cluster {cluster} of {np.sum(labels==cluster)} samples -------------")
        characteristics[cluster] = {}
        for column in features.columns:
            # skip cluster column
            if column == "cluster":
                continue
            this_cluster = features.loc[features["cluster"] == cluster, column]
            other_clusters = features.loc[features["cluster"] != cluster, column]
            #         print(this_cluster)
            #         print(other_clusters)
            # TODO: other test?
            res, p_value = scipy.stats.mannwhitneyu(this_cluster, other_clusters)
            direction = "low" if np.mean(this_cluster) < np.mean(other_clusters) else "high"
            if p_value < 0.05:
                if printout:
                    print(f"{direction} {column} (p-value:{round(p_value, 3)})")
                # print(interpret_dict[column][direction])
                characteristics[cluster][column] = direction
            else:
                # TODO: middle features? compare to each cluster?
                pass
    return characteristics


def sort_clusters_into_groups(characteristics, min_equal=1, add_groups=False, printout=True):
    with open("groups.json", "r") as infile:
        groups = json.load(infile)
    other_groups = [int(k.split("_")[-1]) for k in groups.keys() if "other" in k]
    num_other_groups = max(other_groups) if len(other_groups) > 0 else 0

    # iterate over each cluster
    cluster_assignment = {}
    for cluster, cluster_characteristics in characteristics.items():
        # check in which groups we could put it
        possible_groups = []
        for group_name, group in groups.items():
            is_group = True
            equal_feats = 0
            for key, val in cluster_characteristics.items():
                # if the key is in the group, check whether high/low is same
                if group.get(key, val) != val:
                    is_group = False
                    break
                elif key in group:
                    equal_feats += 1
            if is_group:
                # remember the group and how many equal feats we found
                possible_groups.append((group_name, equal_feats))

        sorted_possible_groups = sorted(possible_groups, key=lambda x: x[1], reverse=True)
        if printout:
            print(f"Cluster {cluster} could be part of", sorted_possible_groups)
        no_tie = len(sorted_possible_groups) < 2 or sorted_possible_groups[0][1] != sorted_possible_groups[1][1]
        # Conditions: at least one group, where at least min_equal elems are the same, and there is no tie
        if len(possible_groups) > 0 and sorted_possible_groups[0][1] >= min_equal:  # and no_tie: # TODO: allow tie?
            most_fitting_group = sorted_possible_groups[0][0]
            if printout:
                print(f"Cluster {cluster} is part of", most_fitting_group)
            cluster_assignment[cluster] = most_fitting_group
        else:
            # make new group
            num_other_groups += 1
            groups["other_" + str(num_other_groups)] = cluster_characteristics

            if printout:
                print("No group possible for cluster", cluster, ", assign to other", num_other_groups)
            cluster_assignment[cluster] = "other"

    # # save updated groups
    if add_groups:
        with open("groups.json", "w") as outfile:
            json.dump(groups, outfile)
    return cluster_assignment


def group_consistency(graph_features, out_path=None, nr_iters=20, n_clusters=5, algorithm="kmeans"):
    res = np.empty((len(graph_features), nr_iters), dtype="<U30")
    for i in range(nr_iters):
        n_clusters = np.random.choice([6, 7, 8])
        cluster_wrapper = ClusterWrapper(random_state=None)
        labels = cluster_wrapper(graph_features, impute_outliers=False, n_clusters=n_clusters, algorithm=algorithm)

        # try to characterize clusters
        characteristics = cluster_characteristics(graph_features, labels, printout=False)
        cluster_assigment = sort_clusters_into_groups(characteristics, printout=False)
        groups = [cluster_assigment[lab] for lab in labels]
        res[:, i] = groups
    df = pd.DataFrame(res, columns=[i for i in range(nr_iters)], index=graph_features.index)

    assigned_most_often = []
    consistency = []
    for row in res:
        uni, counts = np.unique(row, return_counts=True)
        counts_wo_other = counts[uni != "other"]
        uni_wo_other = uni[uni != "other"]
        assigned_most_often.append(uni_wo_other[np.argmax(counts_wo_other)])
        consistency.append(np.max(counts_wo_other) / np.sum(counts_wo_other))
    df["most often"] = assigned_most_often
    df["consistency_" + str(round(np.mean(consistency), 2))] = consistency

    if out_path:
        df.to_csv(out_path)

    print("Average consistency:", np.mean(consistency))
    print(
        "Explanation: This means, the group that is assigned to a user is assigned to this user on avg in x% of the clusterings"
    )
    return assigned_most_often


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--study", type=str, default="all_datasets", help="study - one of gc1, gc2, geolife or all_datasets"
    )
    parser.add_argument(
        "-i", "--inp_dir", type=str, default=os.path.join("out_features", "test"), help="feature inputs"
    )
    parser.add_argument("-o", "--out_dir", type=str, default="results", help="Path where to output all results")
    args = parser.parse_args()

    path = args.inp_dir
    study = args.study
    node_importance = 0
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    algorithm = "kmeans"

    # load features
    graph_features = pd.read_csv(
        os.path.join(path, f"{study}_graph_features_{node_importance}.csv"), index_col="user_id"
    )
    if args.study == "all_datasets":
        STUDIES = ["gc1", "gc2", "tist_toph100", "geolife", "yumuv_graph_rep"]
        print("current studies:", np.unique(graph_features["study"].values))
        print("Warning: To find the groups, we only use the data from the following studies")
        graph_features = graph_features[graph_features["study"].isin(STUDIES)]
        print(np.unique(graph_features["study"].values))

    # delete previous content of groups.json
    with open("groups.json", "w") as outfile:
        json.dump({}, outfile)

    # fit multiple times and save to groups.json file
    cluster_wrapper = ClusterWrapper()
    if "study" in graph_features.columns:
        in_features = graph_features.drop(columns=["study"])
    # Run clustering multiple times, and add the identified groups to the file 3_analysis/groups.json
    for i in range(3):
        for n_clusters in [6, 7, 8]:
            labels = cluster_wrapper(in_features, impute_outliers=False, n_clusters=n_clusters, algorithm=algorithm)
            characteristics = cluster_characteristics(in_features, labels, printout=False)
            cluster_assignment = sort_clusters_into_groups(characteristics, add_groups=True, printout=False)
    # copy the resulting groups to the results folder
    shutil.copy(os.path.join("groups.json"), os.path.join(out_dir, "groups.json"))
