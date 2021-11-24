import os
from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
import json
import sys
import shutil
import argparse
from scipy.sparse.construct import rand
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, rand_score

from clustering import ClusterWrapper
from analysis_utils import sort_images_by_cluster
from plotting import scatterplot_matrix, plot_cluster_characteristics, cluster_by_study
from find_groups import cluster_characteristics, sort_clusters_into_groups, group_consistency


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
    # Use only the five studies for identifying the user groups
    STUDIES = ["gc1", "gc2", "tist_toph100", "tist_random100", "geolife", "yumuv_graph_rep"]
    graph_features = graph_features[graph_features["study"].isin(STUDIES)]

    # Save groups file
    shutil.copy("groups.json", "bl_groups.json")

    np.random.seed(20)  # use the seed from analyze study (from our results) here
    # BASELINE labels: current clustering
    labels = group_consistency(
        graph_features.drop("study", axis=1),
        k_choices=[6, 7, 8, 9],
        nr_iters=3,
        out_path=None,
    )
    labels_BL = labels.copy()

    f = open(os.path.join(args.out_dir, "stability_groups.txt"), "w")
    sys.stdout = f

    rand_scores = []
    for iter in range(20):
        print("\nSEED", iter)
        np.random.seed(iter)

        # ------------ GROUP FINDING ----------------
        # plot_features = graph_features.copy()
        # plot_features["cluster"] = labels
        # plot_cluster_characteristics(plot_features)

        # delete previous content of groups.json
        with open("groups.json", "w") as outfile:
            json.dump({}, outfile)

        # fit multiple times and save to groups.json file
        cluster_wrapper = ClusterWrapper()
        if "study" in graph_features.columns:
            in_features = graph_features.drop(columns=["study"])
        else:
            in_features = graph_features.copy()
        # Run clustering multiple times, and add the identified groups to the file 3_analysis/groups.json
        for i in range(3):
            for n_clusters in [6, 7, 8, 9]:
                labels = cluster_wrapper(in_features, impute_outliers=False, n_clusters=n_clusters, algorithm=algorithm)
                characteristics = cluster_characteristics(in_features, labels, printout=False)
                cluster_assignment = sort_clusters_into_groups(
                    characteristics, add_groups=True, printout=False, min_equal=2, allow_tie=True
                )

        labels = group_consistency(
            graph_features.drop("study", axis=1),
            k_choices=[6, 7, 8, 9],
            nr_iters=3,
            out_path=None,
            printout=True,
        )

        score = adjusted_rand_score(labels_BL, labels)
        rand_scores.append(score)
        print("RAND SCORE:", score)
        print("Rand score", rand_score(labels_BL, labels))
        # between 0 and 1, proportional to numbers of pairs ending in same and different clusters
        print("mutual info", adjusted_mutual_info_score(labels_BL, labels))

    print("------- Average rand score:", np.mean(rand_scores), "-----------------")

    # Reset groups file
    shutil.copy("bl_groups.json", "groups.json")
    f.close()
