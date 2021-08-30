import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

algorithm_dict = {"kmeans": KMeans, "hierarchical": AgglomerativeClustering, "dbscan": DBSCAN}


def normalize_and_cluster(feature_matrix, algorithm="kmeans", n_clusters=2, impute_outliers=False, return_normed=False):
    """
    Normalize feature matrix and cluster it
    algorithm: One of kmeans, hierarchical, dbscan
    """
    if isinstance(feature_matrix, pd.DataFrame):
        feature_matrix = feature_matrix.dropna()
        feature_matrix = np.array(feature_matrix)
    if impute_outliers:
        feature_matrix = outlier_imputation(feature_matrix)
    std_cols = np.std(feature_matrix, axis=0)
    means_cols = np.mean(feature_matrix, axis=0)
    normed_feature_matrix = (feature_matrix - means_cols) / std_cols
    kmeans = algorithm_dict[algorithm](n_clusters=n_clusters).fit(normed_feature_matrix)
    if return_normed:
        return kmeans.labels_, normed_feature_matrix
    return kmeans.labels_


def outlier_imputation(features, cutoff=3):
    """Fill in outliers with the mean value

    Parameters
    ----------
    features : DataFrame
    cutoff : int, optional
        remove values that are above or below the mean+cutoff*std, by default 3
    """
    imputed_features = np.zeros(features.shape)
    for i in range(features.shape[1]):
        col_vals = features[:, i].copy()
        mean, std = (np.mean(col_vals), np.std(col_vals))
        # outliers are above or below cutoff times the std
        outlier_thresh = (mean - cutoff * std, mean + cutoff * std)
        outlier = (col_vals < outlier_thresh[0]) | (col_vals > outlier_thresh[1])
        col_vals[outlier] = mean
        imputed_features[:, i] = col_vals
    return imputed_features


def decision_tree_cluster(features, cluster_labels):
    from sklearn import tree

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(np.array(features), cluster_labels)
    r = tree.export_text(clf, feature_names=list(features.columns))
    print(r)
    return clf.feature_importances_


def pca(feature_matrix, n_components=2):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(feature_matrix)
    return projected
