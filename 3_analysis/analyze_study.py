import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from utils import load_user_info
from clustering import normalize_and_cluster, decision_tree_cluster


def find_k(features):
    feature_matrix = np.array(features)

    test_k = np.arange(2, 7, 1)
    scores = []
    for n_clusters in test_k:
        labels, normed_feature_matrix = normalize_and_cluster(feature_matrix, n_clusters=n_clusters, return_normed=True)
        print()
        print(n_clusters)
        print("Number of samples per cluster", np.unique(labels, return_counts=True))
        score = silhouette_score(normed_feature_matrix, labels)
        scores.append(score)
        print("Silhuette score", score)
    # return k with highest score
    return test_k[np.argmax(scores)]


def entropy(df_in, label, cluster):
    """Entropy of labels over the clusters. See Ben-Gal et al paper"""
    # clean the nans:
    df_in = df_in[~pd.isna(df_in[label])]
    df_in = df_in[df_in[label] != "nan"]
    df_in = df_in[~pd.isna(df_in[cluster])]
    df = df_in[df_in[cluster] != "nan"]

    n = len(df)  # total number of points
    uni, counts = np.unique(cluster, return_counts=True)
    entropy = 0
    for c, cluster_df in df.groupby(cluster):
        # number of points in this cluster
        n_k = len(cluster_df)
        uni, counts = np.unique(cluster_df[label].values, return_counts=True)
        # compute entropy of
        inner_entropy = 0
        for (u, c) in zip(uni, counts):
            # print(u, c/n_k)
            inner_entropy += (c / n_k) * np.log2(c / n_k)
        # overall entropy is weighted sum of inner entropy
        norm_factor = np.log2(len(uni)) if len(uni) > 2 else 1
        entropy += inner_entropy * n_k / norm_factor

    return -entropy / n


def get_numeric_columns(df):
    all_numeric = df._get_numeric_data().columns
    # exclude id columns
    return [d for d in all_numeric if not "id" in d]


if __name__ == "__main__":
    study = "gc1"
    name = f"{study}_raw_features.csv"
    features = pd.read_csv(os.path.join("out_features", name), index_col="user_id")
    # find optimal k
    opt_k = find_k(features)
    print("Optimal k", opt_k)

    labels = normalize_and_cluster(np.array(features), n_clusters=opt_k)

    # print decision tree:
    decision_tree_cluster(features, labels)

    # load labels
    user_info = load_user_info(study)
    # merge into one table and add cluster labels
    joined = features.merge(user_info, how="left", left_on="user_id", right_on="user_id")
    joined["cluster"] = labels

    # Decision tree for NUMERIC data - first fill nans
    numeric_columns = get_numeric_columns(user_info)
    tree_input = joined[numeric_columns]
    tree_input = tree_input.fillna(value=tree_input.mean())
    feature_importances = decision_tree_cluster(tree_input, labels)
    # get five most important features:
    important_feature_inds = np.argsort(feature_importances)[-5:]
    print(np.array(numeric_columns)[important_feature_inds], feature_importances[important_feature_inds])

    # Entropy for CATEGORICAL data
    for col in user_info.columns:
        if col in numeric_columns or "id" in col:
            continue
        not_nan = pd.isna(joined[col]).sum()
        if not_nan / len(joined) > 0.5:
            print("Skipping because too many missing values:", col)
            continue

        print("------", col, "------")
        # entropy is not symmetric, compute both
        entropy_1 = entropy(joined, col, "cluster")
        entropy_2 = entropy(joined, "cluster", col)
        print("Entropy:", round(entropy_1, 2), round(entropy_2, 2))