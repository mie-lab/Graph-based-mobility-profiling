import os
import numpy as np
import pandas as pd
import scipy
import json
import argparse

from clustering import ClusterWrapper


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


def sort_clusters_into_groups(characteristics, min_equal=1, printout=True):
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

        # print(f"Cluster {cluster} could be part of", sorted(possible_groups, key=lambda x: x[1], reverse=True))
        if len(possible_groups) > 0:
            most_fitting_group = sorted(possible_groups, key=lambda x: x[1], reverse=True)[0][0]
            if printout:
                print(f"Cluster {cluster} is part of", most_fitting_group)
            cluster_assignment[cluster] = most_fitting_group
        else:
            if printout:
                print("No group possible for cluster", cluster, ", assign to other")
            cluster_assignment[cluster] = "other"
        # if is_group:
        #     print(f"Cluster {cluster} is part of group", group_name)
        #     break
        if len(possible_groups) == 0 and len(cluster_characteristics) > 1:
            num_other_groups += 1
            groups["other_" + str(num_other_groups)] = cluster_characteristics

    # # save updated groups
    # with open("groups.json", "w") as outfile:
    #     json.dump(groups, outfile)
    return cluster_assignment


def group_consistency(graph_features, out_path=None, nr_iters=20):
    res = np.empty((len(graph_features), nr_iters), dtype="<U10")
    for i in range(nr_iters):
        cluster_wrapper = ClusterWrapper()
        labels = cluster_wrapper(graph_features, impute_outliers=False, n_clusters=n_clusters, algorithm=algorithm)

        # try to characterize clusters
        characteristics = cluster_characteristics(graph_features, labels, printout=False)
        cluster_assigment = sort_clusters_into_groups(characteristics, printout=False)
        groups = [cluster_assigment[lab] for lab in labels]
        res[:, i] = groups
    df = pd.DataFrame(res, columns=[i for i in range(nr_iters)], index=graph_features.index)
    if out_path:
        df.to_csv(out_path)

    consistency = []
    for i, row in df.iterrows():
        vals = row.values
        _, counts = np.unique(vals, return_counts=True)
        consistency.append(np.max(counts) / len(vals))

    print("Average consistency:", np.mean(consistency))
    print(
        "Explanation: This means, the group that is assigned to a user is assigned to this user on avg in x% of the clusterings"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--study", type=str, required=True, help="study - one of gc1, gc2, geolife")
    parser.add_argument("-v", "--version", type=int, default=3, help="feature version")
    parser.add_argument("-n", "--nodes", type=int, default=0, help="number of x important nodes. Set -1 for all nodes")
    args = parser.parse_args()

    path = os.path.join("out_features", f"final_{args.version}_n{args.nodes}_cleaned")
    study = args.study
    node_importance = args.nodes

    n_clusters = 5
    algorithm = "kmeans"

    # load features
    graph_features = pd.read_csv(
        os.path.join(path, f"{study}_graph_features_{node_importance}.csv"), index_col="user_id"
    )

    # CLUSTER CONSISTENCY
    group_consistency(graph_features)

    cluster_wrapper = ClusterWrapper()
    labels = cluster_wrapper(graph_features, impute_outliers=False, n_clusters=n_clusters, algorithm=algorithm)

    # try to characterize clusters
    characteristics = cluster_characteristics(graph_features, labels)
    print()
    print("--------- Sorting cluster into predefined groups ------------")
    cluster_assigment = sort_clusters_into_groups(characteristics)
    print("\n ----------------------------------- \n")

    # sort_images_by_cluster(
    #     list(graph_features.index),
    #     labels,
    #     name_mapping=cluster_assigment,
    #     in_img_path="graph_images/gc2/coords",
    #     out_img_path="graph_images/gc2_coords_" + algorithm,
    # )

    # # CHARACTERIZE WITH RAW FEATURES
    # if "yumuv" not in study:
    #     raw_features = pd.read_csv(
    #         os.path.join(path, f"{study}_raw_features_{node_importance}.csv"), index_col="user_id"
    #     )
    #     assert all(raw_features.index == graph_features.index)
    #     print("features shape:", graph_features.shape, raw_features.shape)
    # print()
    # print("Cluster characteristics by raw feature")
    # _ = cluster_characteristics(raw_features, labels)
