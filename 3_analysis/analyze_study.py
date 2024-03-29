import os
import numpy as np
import pandas as pd
import argparse
import sys
import shutil
import pickle
from sklearn.metrics import silhouette_score

from analysis_utils import sort_images_by_cluster
from clustering import ClusterWrapper
from find_groups import cluster_characteristics, sort_clusters_into_groups, group_consistency
from plotting import plot_cluster_characteristics, cluster_by_study, scatterplot_matrix
from label_analysis import entropy
from compare_clustering import compute_all_scores


def find_k(features):
    test_k = np.arange(2, 7, 1)
    scores = []
    for n_clusters in test_k:
        cluster_wrapper = ClusterWrapper()
        labels, normed_feature_matrix = cluster_wrapper(features, n_clusters=n_clusters, return_normed=True)
        print()
        print(n_clusters)
        print("Number of samples per cluster", np.unique(labels, return_counts=True))
        score = silhouette_score(normed_feature_matrix, labels)
        scores.append(score)
        print("Silhuette score", score)
    # return k with highest score
    return test_k[np.argmax(scores)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--study", type=str, default="all_datasets", help="study - one of gc1, gc2, geolife")
    parser.add_argument(
        "-i", "--inp_dir", type=str, default=os.path.join("out_features", "test"), help="feature inputs"
    )
    parser.add_argument("-o", "--out_dir", type=str, default="results", help="outputs")
    parser.add_argument("-t", "--feature_type", type=str, default="graph", help="Using graph or raw feature set")
    parser.add_argument("-m", "--min_equal", type=int, default=2, help="Mininimum corresponding features parameter")
    args = parser.parse_args()

    path = args.inp_dir
    study = args.study
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, f"{study}_")

    node_importance = 0
    add_groups = False
    n_clusters = 8
    algorithm = "kmeans"

    # load features
    graph_features = pd.read_csv(
        os.path.join(path, f"{study}_{args.feature_type}_features_{node_importance}.csv"), index_col="user_id"
    )

    # ------------------ ANALYSE MERGED STUDIES WITH GIVEN GROUPS -------------------------------
    if study == "all_datasets":
        # Use only the main studies!! otherwise biased
        STUDIES = ["gc1", "gc2", "tist_toph100", "tist_random100", "geolife", "yumuv_graph_rep"]
        graph_features = graph_features[graph_features["study"].isin(STUDIES)]

        # # Scatterplot with colours showing study
        scatter_features = graph_features.reset_index().set_index(["user_id", "study"])
        scatterplot_matrix(
            scatter_features,
            scatter_features.columns,
            clustering=list(graph_features["study"]),
            save_path=os.path.join(out_dir, "scatterplot_study.pdf"),
        )

        # log to file
        f = open(os.path.join(args.out_dir, "analyze_study.txt"), "w")
        sys.stdout = f

        np.random.seed(20)

        # get most consistent labels
        labels = group_consistency(
            graph_features.reset_index().set_index(["user_id", "study"]),
            k_choices=[6, 7, 8, 9],
            nr_iters=3,
            out_path=os.path.join(out_dir, "consistency.csv"),
            min_equal=args.min_equal,
        )
        graph_features["cluster"] = labels
        graph_features.to_csv(os.path.join(out_path + "clustering.csv"))

        # scatterplot matrix
        scatter_features = graph_features.drop(columns=["cluster"]).reset_index().set_index(["user_id", "study"])
        scatterplot_matrix(
            scatter_features,
            scatter_features.columns,
            clustering=labels,
            save_path=os.path.join(out_dir, "scatterplot.pdf"),
        )

        # # PLOTTING
        plot_cluster_characteristics(
            graph_features.copy(), out_path=os.path.join(out_dir, "cluster_characteristics.pdf")
        )
        cluster_by_study(graph_features.copy(), out_path=os.path.join(out_dir, "dataset_clusters.pdf"))

        # Save cluster wrapper with the best k!
        best_overlap = 0
        for n_clusters_final in [6, 7, 8, 9]:
            cluster_wrapper = ClusterWrapper()
            labels_new = cluster_wrapper(
                scatter_features, impute_outliers=False, n_clusters=n_clusters_final, algorithm="kmeans"
            )
            characteristics = cluster_characteristics(scatter_features, labels_new, printout=False)
            cluster_assignment = sort_clusters_into_groups(characteristics, add_groups=False, printout=False)
            cluster_wrapper.cluster_assignment = cluster_assignment
            group_labels_new = [cluster_assignment[label] for label in labels_new]
            overlap_saved_clustering = sum(graph_features["cluster"].values == np.array(group_labels_new)) / len(
                graph_features
            )
            print("Saved cluster wrapper labels == consistent labels?", overlap_saved_clustering)
            if overlap_saved_clustering > best_overlap:
                best_overlap = overlap_saved_clustering
                print("SAVED WITH k", n_clusters_final)
                with open(os.path.join(out_dir, "clustering.pkl"), "wb") as outfile:
                    pickle.dump(cluster_wrapper, outfile)

        # Entropy calculation:
        features_all_datasets = graph_features.copy()
        features_all_datasets = features_all_datasets[features_all_datasets["study"].isin(STUDIES)]

        print("\n -------  Entropy analysis ------------")
        study_entropy = entropy(features_all_datasets, "study", "cluster", print_parts=True)
        print("ENTROPY studies over groups", study_entropy, "\n")

        study_entropy = entropy(features_all_datasets, "cluster", "study", print_parts=True)
        print("ENTROPY groups over studies", study_entropy, "\n")

        print("\n Entropy without tist...")
        study_entropy = entropy(
            features_all_datasets[~features_all_datasets["study"].str.contains("tist")],
            "cluster",
            "study",
            print_parts=True,
        )
        print("ENTROPY groups over studies without tist", study_entropy, "\n")

        # compare relation between cluster and study labels
        compute_all_scores(features_all_datasets["cluster"].values, features_all_datasets["study"].values)

        print("\nSIGNIFICANT FEATURES (for plot labeling)")
        features_w_labels = pd.read_csv(os.path.join(out_path + "clustering.csv"), index_col=("user_id", "study"))
        cluster_characteristics(features_w_labels, printout=True)

        f.close()

        exit()

    # ------------------ ANALYSE SINGLE STUDY WITH GIVEN GROUPS -------------------------------

    # write terminal output to file:
    f = open(out_path + "terminal.txt", "w")
    sys.stdout = f

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
