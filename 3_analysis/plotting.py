import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 15


def scatterplot_matrix(feature_df, use_features, col_names=None, clustering=None, save_path=None):
    """
    Scatterplot matrix for selected features

    Arguments
        clustering: List of cluster labels for each item
    """
    assert len(use_features) <= 6, "more than 6 features does not make sense in scatterplot matrix"
    # define col names
    if col_names is None:
        col_names = use_features
    # transform to df
    feature_df = feature_df.loc[:, use_features]
    if clustering is not None:
        feature_df["cluster"] = clustering
        col_dict = {cluster: sns.color_palette()[cluster] for cluster in clustering}
        sns.pairplot(feature_df, hue="cluster", palette=col_dict)
    else:
        sns.pairplot(feature_df)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
