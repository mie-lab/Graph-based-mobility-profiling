import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

algorithm_dict = {"kmeans": KMeans, "hierarchical": AgglomerativeClustering, "dbscan": DBSCAN}

def normalize_and_cluster(feature_matrix, algorithm="kmeans", n_clusters=2, return_normed=False):
    """
    Normalize feature matrix and cluster it 
    algorithm: One of kmeans, hierarchical, dbscan
    """
    std_cols = np.std(feature_matrix, axis=0)
    means_cols = np.mean(feature_matrix, axis=0)
    normed_feature_matrix = (feature_matrix - means_cols) / std_cols
    kmeans = algorithm_dict[algorithm](n_clusters=n_clusters).fit(normed_feature_matrix)
    if return_normed:
        return kmeans.labels_, normed_feature_matrix
    return kmeans.labels_


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
