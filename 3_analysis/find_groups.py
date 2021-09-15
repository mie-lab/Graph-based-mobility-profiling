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
from plotting import scatterplot_matrix


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


def group_consistency(graph_features, out_path=None, nr_iters=20, n_clusters=5):
    res = np.empty((len(graph_features), nr_iters), dtype="<U30")
    for i in range(nr_iters):
        cluster_wrapper = ClusterWrapper()
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
    df["consistency"] = np.mean(consistency)

    if out_path:
        df.to_csv(out_path)

    print("Average consistency:", np.mean(consistency))
    print(
        "Explanation: This means, the group that is assigned to a user is assigned to this user on avg in x% of the clusterings"
    )
    return assigned_most_often


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--study", type=str, required=True, help="study - one of gc1, gc2, geolife")
    parser.add_argument("-v", "--version", type=int, default=3, help="feature version")
    parser.add_argument("-n", "--nodes", type=int, default=0, help="number of x important nodes. Set -1 for all nodes")
    parser.add_argument("-o", "--out_dir", type=str, default="results", help="Path where to output all results")
    args = parser.parse_args()

    path = os.path.join("out_features", f"final_{args.version}_n{args.nodes}_cleaned")
    study = args.study
    node_importance = args.nodes

    add_groups = False

    # FOR ACTUALLY FINDING THE GROUPS:
    # add_groups = True
    # cluster_wrapper = ClusterWrapper()
    # labels = cluster_wrapper(graph_features, impute_outliers=False, n_clusters=n_clusters, algorithm=algorithm)

    # AFTER GROUPS ARE FOUND, ANALYZE ASSIGNMENT
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, f"{study}_{args.version}_")

    # write terminal output to file:
    f = open(out_path + "terminal.txt", "w")
    sys.stdout = f

    n_clusters = 8
    algorithm = "kmeans"

    # load features
    graph_features = pd.read_csv(
        os.path.join(path, f"{study}_graph_features_{node_importance}.csv"), index_col="user_id"
    )

    # CLUSTER CONSISTENCY
    labels = group_consistency(graph_features, out_path=out_path + "consistency.csv", n_clusters=n_clusters)

    # copy the groups into the output folder
    shutil.copy("groups.json", out_dir)

    # try to characterize clusters --> Only for terminal output!! already done in consistency function
    print()
    characteristics = cluster_characteristics(graph_features, labels)
    print("\n--------- Sorting cluster into predefined groups ------------")
    cluster_assignment = sort_clusters_into_groups(characteristics, add_groups=add_groups)
    print("\n ----------------------------------- \n")

    # SCATTERPLOT
    scatterplot_matrix(
        graph_features,
        graph_features.columns,
        clustering=labels,
        save_path=os.path.join(out_path + "scatterplot.pdf"),
    )

    # MAKE IMAGES
    for type in ["coords", "spring"]:
        sort_images_by_cluster(
            list(graph_features.index),
            labels,
            name_mapping={lab: lab for lab in np.unique(labels)},
            in_img_path=os.path.join("graph_images", study, type),
            out_img_path=os.path.join(out_dir, f"{study}_{type}_" + algorithm),
        )

    # CHARACTERIZE WITH RAW FEATURES
    if "yumuv" not in study:
        raw_features = pd.read_csv(
            os.path.join(path, f"{study}_raw_features_{node_importance}.csv"), index_col="user_id"
        )
        assert all(raw_features.index == graph_features.index)
        print("features shape:", graph_features.shape, raw_features.shape)
        print()
        print("Cluster characteristics by raw feature")
        _ = cluster_characteristics(raw_features, labels)
    f.close()
