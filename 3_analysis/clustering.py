import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

algorithm_dict = {"kmeans": KMeans, "hierarchical": AgglomerativeClustering, "dbscan": DBSCAN}

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


class ClusterWrapper:
    def __init__(self, random_state=None):
        # random state for reproducability
        self.random_state = random_state
        self.algorithm_dict = {"kmeans": KMeans, "hierarchical": AgglomerativeClustering, "dbscan": DBSCAN}
        self.cluster_centers = None
        # to sort clusters into groups
        self.cluster_assignment = None

    def __call__(self, feature_matrix, algorithm="kmeans", n_clusters=2, impute_outliers=False, return_normed=False):
        """
        Normalize feature matrix and cluster it
        algorithm: One of kmeans, hierarchical, dbscan
        """
        assert isinstance(feature_matrix, pd.DataFrame)
        prev_len = len(feature_matrix)
        # feature_matrix = feature_matrix.dropna()
        # print("Dropped nans, length now", len(feature_matrix), "vs prev length", prev_len)
        if "study" in feature_matrix.columns:
            feature_matrix = feature_matrix.drop(columns=["study"])
        feature_matrix = np.array(feature_matrix)
        if impute_outliers:
            feature_matrix = outlier_imputation(feature_matrix)
        self.std_cols = np.std(feature_matrix, axis=0)
        self.means_cols = np.mean(feature_matrix, axis=0)
        normed_feature_matrix = self.normalize(feature_matrix)
        if algorithm == "dbscan":
            kmeans = algorithm_dict[algorithm](min_samples=3, eps=1)
        else:
            kmeans = algorithm_dict[algorithm](n_clusters=n_clusters, random_state=self.random_state)  # TODO
        kmeans = kmeans.fit(normed_feature_matrix)
        # save cluster centers
        try:
            self.cluster_centers = kmeans.cluster_centers_
        except AttributeError:
            pass

        if return_normed:
            return kmeans.labels_, normed_feature_matrix
        return kmeans.labels_

    def normalize(self, data):
        return (data - self.means_cols) / self.std_cols

    def transform(self, data):
        if self.cluster_centers is None:
            raise RuntimeError("Must first call the class to create a clustering!")
        normed = self.normalize(np.array(data))
        labels = []
        for feat in normed:
            labels.append(np.argmin([np.linalg.norm(feat - center) for center in self.cluster_centers]))
        if self.cluster_assignment:
            return [self.cluster_assignment[lab] for lab in labels]
        return labels


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
