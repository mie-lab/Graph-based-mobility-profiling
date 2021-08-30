import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from utils import load_user_info
from clustering import normalize_and_cluster, decision_tree_cluster


def find_k(features):
    test_k = np.arange(2, 7, 1)
    scores = []
    for n_clusters in test_k:
        labels, normed_feature_matrix = normalize_and_cluster(features, n_clusters=n_clusters, return_normed=True)
        print()
        print(n_clusters)
        print("Number of samples per cluster", np.unique(labels, return_counts=True))
        score = silhouette_score(normed_feature_matrix, labels)
        scores.append(score)
        print("Silhuette score", score)
    # return k with highest score
    return test_k[np.argmax(scores)]


def _rm_nans(df_raw, label, cluster):
    df_in = df_raw.copy()
    df_in = df_in[~pd.isna(df_in[label])]
    df_in = df_in[df_in[label] != "nan"]
    df_in = df_in[~pd.isna(df_in[cluster])]
    return df_in[df_in[cluster] != "nan"]


def _rename_nans(df_raw, label, cluster):
    df_in = df_raw.copy()
    df_in[label] = df_in[label].fillna("nan")
    df_in[cluster] = df_in[cluster].fillna("nan")
    return df_in


def entropy(df_in, label, cluster, treat_nans="remove"):
    """Entropy of labels over the clusters. See Ben-Gal et al paper"""
    # clean the nans:
    if treat_nans == "remove":
        df = _rm_nans(df_in, label, cluster)
    elif treat_nans == "rename":
        df = _rename_nans(df_in, label, cluster)
    else:
        df = df_in.copy()

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
        entropy -= inner_entropy * (n_k / n)

    # factor out the label entropy that is expected:
    uni, counts = np.unique(df[label].values, return_counts=True)
    label_entropy = -1 * sum([(c / n) * np.log2(c / n) for (u, c) in zip(uni, counts)])

    return entropy / label_entropy


def combine_columns(df, list_of_columns, combined_name):
    """Combine several columns into one, by using the value of the first column of list_of_columns that is not nan"""
    combined = []
    for i, row in df.iterrows():
        # take value from first column that is not nan
        found_val = False
        for col in list_of_columns:
            if ~pd.isna(row[col]) and (row[col] is not None) and row[col] != "nan":
                combined.append(row[col])
                found_val = True
                break
        # otherwise fill with nans
        if not found_val:
            combined.append(pd.NA)

    # we must collect the same number of values
    assert len(combined) == len(df)

    df_out = df.drop(columns=list_of_columns)
    df_out[combined_name] = combined

    return df_out


def get_numeric_columns(df):
    all_numeric = df._get_numeric_data().columns
    # exclude id columns
    return [d for d in all_numeric if not "id" in d]


if __name__ == "__main__":
    study = "yumuv_graph_rep"
    feat_type = "graph"
    node_importance = 0
    path = "final_1"

    name = f"{study}_{feat_type}_features_{node_importance}.csv"
    features = pd.read_csv(os.path.join(path, name), index_col="user_id")
    features.dropna(inplace=True)
    # find optimal k
    opt_k = find_k(features)
    print("Optimal k", opt_k)

    labels = normalize_and_cluster(features, n_clusters=opt_k)

    # print decision tree:
    feature_importances = decision_tree_cluster(features, labels)
    # get five most important features:
    important_feature_inds = np.argsort(feature_importances)[-5:]
    print(np.array(features.columns)[important_feature_inds], feature_importances[important_feature_inds])

    # load labels
    # GC1
    user_info = load_user_info(study, index_col="user_id")
    # YUMUV:
    # user_info = load_user_info(study, index_col="app_user_id")
    # user_info = user_info.reset_index().rename(columns={"app_user_id": "user_id"})

    # merge into one table and add cluster labels
    joined = features.merge(user_info, how="left", left_on="user_id", right_on="user_id")
    joined["cluster"] = labels

    # # Decision tree for NUMERIC data - first fill nans
    numeric_columns = get_numeric_columns(user_info)
    # if len(numeric_columns) > 0:
    #     tree_input = joined[numeric_columns]
    #     tree_input = tree_input.fillna(value=tree_input.mean())
    #     feature_importances = decision_tree_cluster(tree_input, labels)
    #     # get five most important features:
    #     important_feature_inds = np.argsort(feature_importances)[-5:]
    #     print(np.array(numeric_columns)[important_feature_inds], feature_importances[important_feature_inds])

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
