import numpy as np
import os
import argparse
import pandas as pd

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from clustering import normalize_and_cluster

"""
Functions to evaluate clustering: 

compare_clusters_... functions for comparing two different clusterings of the same data. 
Implemented metrics according to https://i11www.iti.kit.edu/extra/publications/ww-cco-06.pdf
"""


def compare_clusters_count_pairs(cluster_labels_1, cluster_labels_2):
    """
    Compare two clusterings by computing number of pairs in same cluster and in different cluster etc

    Arguments:
        cluster_labels_1: List or 1d array of length n and type int, containing cluster assignment for each user
        cluster_labels_2: List or 1d array of length n and type int, containing other cluster assignment for each user
    Returns:
        2x2 Matrix indicating number of pairs ending up in same or different cluster according to the 2 clusterings
    """
    nr_users = len(cluster_labels_1)
    # result: #pairs in same cluster in both clusterings, #pairs in different in both,
    # #pairs in same in 1 but different in 2, and #pairs in same in 2 but different in 1
    pairs_same = np.zeros(4)
    # iterate over all pairs of users
    for i in range(nr_users):
        for j in range(i + 1, nr_users):
            same_cluster_1 = cluster_labels_1[i] == cluster_labels_1[j]
            same_cluster_2 = cluster_labels_2[i] == cluster_labels_2[j]
            if same_cluster_1 and same_cluster_2:
                pairs_same[0] += 1
            elif same_cluster_1 and not same_cluster_2:
                pairs_same[1] += 1
            elif same_cluster_2 and not same_cluster_1:
                pairs_same[2] += 1
            elif not same_cluster_2 and not same_cluster_1:
                pairs_same[3] += 1
    return pairs_same.reshape((2, 2))


def compare_clusters_rand_index(cluster_labels_1, cluster_labels_2):
    """Compute Rand Index to compare two clusterings

    Parameters
    ----------
        cluster_labels_1: List or 1d array of length n and type int
            containing cluster assignment for each user
        cluster_labels_2: List or 1d array of length n and type int
            containing other cluster assignment for each user

    Returns
    -------
    Float between 0 and 1
        Index of how much the two clusterings correspond
    """
    pairs_same_matrix = compare_clusters_count_pairs(cluster_labels_1, cluster_labels_2)
    # number of pairs that are in same cluster in both clusterings plus #pairs in different
    # cluster in both clusterings divided by overall number
    n = len(cluster_labels_1)
    return (pairs_same_matrix[0, 0] + pairs_same_matrix[1, 1]) / np.sum(pairs_same_matrix)


def compare_clusters_confusion_matrix(cluster_labels_1, cluster_labels_2):
    """
    Compute a matrix with intersections of elements of cluster i of clustering 1 with cluster j of clustering 2
    Parameters
    ----------
        cluster_labels_1: List or 1d array of length n and type int
            containing cluster assignment for each user
        cluster_labels_2: List or 1d array of length n and type int
            containing other cluster assignment for each user

    Returns
    -------
    2d numpy array
        Intersection of elements in clusters of clustering 1 and clusters of clustering 2
    """
    # use np arrays
    cluster_labels_1 = np.array(cluster_labels_1)
    cluster_labels_2 = np.array(cluster_labels_2)
    # get cluster label names (must be ints!)
    clusters_1 = np.unique(cluster_labels_1)
    clusters_2 = np.unique(cluster_labels_2)
    # output: array of intersections
    intersect_array = np.zeros((len(clusters_1), len(clusters_2)))
    for i, clu_i in enumerate(clusters_1):
        for j, clu_j in enumerate(clusters_2):
            users_in_i = np.where(cluster_labels_1 == clu_i)[0]
            users_in_j = np.where(cluster_labels_2 == clu_j)[0]
            num_intersect = len(np.intersect1d(users_in_i, users_in_j))
            intersect_array[i, j] = num_intersect
    #             print(clu_i,clu_j)
    #             print(users_in_i, users_in_j)
    #             print("intersection", np.intersect1d(users_in_i, users_in_j))
    return intersect_array


def compare_clusters_chi_square(cluster_labels_1, cluster_labels_2):
    """
    Chi Square coefficient for cluster comparison
    Note: assumes independency of clusters, which is usually not given!

    Parameters
    ----------
        cluster_labels_1: List or 1d array of length n and type int
            containing cluster assignment for each user
        cluster_labels_2: List or 1d array of length n and type int
            containing other cluster assignment for each user

    Returns
    -------
    float
        Index of how much the two clusterings correspond
    """
    # use np arrays
    cluster_labels_1 = np.array(cluster_labels_1)
    cluster_labels_2 = np.array(cluster_labels_2)

    # get intersection matrix
    intersections = compare_clusters_confusion_matrix(cluster_labels_1, cluster_labels_2)
    # print(intersections)

    # get cluster label names (must be ints!)
    clusters_1 = np.unique(cluster_labels_1)
    clusters_2 = np.unique(cluster_labels_2)

    # n is number of users
    n = len(cluster_labels_1)

    chi_square = 0
    for i in range(len(clusters_1)):
        for j in range(len(clusters_2)):
            m_ij = intersections[i, j]
            # number of elements in i-th cluster of clustering 1
            num_c1 = np.sum(cluster_labels_1 == clusters_1[i])
            # number of elements in j-th cluster of clustering 2
            num_c2 = np.sum(cluster_labels_2 == clusters_2[j])
            e_ij = num_c1 * num_c2 / n
            # print(i, j, num_c1, num_c2, e_ij, m_ij)

            chi_square += (m_ij - e_ij) ** 2 / e_ij
    return chi_square


def load_two(path, study, node_importance):
    """Load the extracted features from path for study"""
    graph_features = pd.read_csv(
        os.path.join(path, f"{study}_graph_features_{node_importance}.csv"), index_col="user_id"
    )
    raw_features = pd.read_csv(os.path.join(path, f"{study}_raw_features.csv"), index_col="user_id")
    raw_features = raw_features[raw_features.index.isin(graph_features.index)]
    return graph_features, raw_features


def compute_all_scores(clustering_1, clustering_2):
    """Print out the comparison scores of two different clusterings
    clustering_1, clustering 2: lists of same length containing cluster for each datapoint
    """
    print("Adjusted rand score: %0.2f"%adjusted_rand_score(clustering_1, clustering_2))
    print("Adjusted mutual information score: %0.2f"%adjusted_mutual_info_score(clustering_1, clustering_2))
    print("Rand score: %0.2f"%compare_clusters_rand_index(clustering_1, clustering_2))
    print("Chi square score: %0.2f"%compare_clusters_chi_square(clustering_1, clustering_2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--study", type=str, required=True, help="study - one of gc1, gc2, geolife")
    parser.add_argument("-n", "--nodes", type=int, default=-1, help="number of x important nodes. Set -1 for all nodes")
    parser.add_argument("-c", "--clusters", type=int, default=2, help="number of clusters")
    args = parser.parse_args()

    study = args.study
    node_importance = args.nodes
    n_clusters = args.clusters

    # load features and cluster
    graph_features, raw_features = load_two("out_features", study, node_importance)
    graph_labels = normalize_and_cluster(np.array(graph_features), n_clusters=n_clusters)
    raw_labels = normalize_and_cluster(np.array(raw_features), n_clusters=n_clusters)
    compute_all_scores(graph_labels, raw_labels)
