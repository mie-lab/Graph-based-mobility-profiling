import warnings
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 15, "axes.labelsize": 15})

from future_trackintel.activity_graph import activity_graph
import os
import pickle


def plot_all_graphs(AG_dict, study):
    """Originally code of example_for_nina file, now function to plot the graphs"""
    output_spring = os.path.join(".", "graph_images", study, "spring")
    if not os.path.exists(output_spring):
        os.makedirs(output_spring)

    output_coords = os.path.join(".", "graph_images", study, "coords")
    if not os.path.exists(output_coords):
        os.makedirs(output_coords)

    for user_id_this, AG in AG_dict.items():

        AG.plot(
            filename=os.path.join(output_spring, str(user_id_this)),
            filter_node_importance=25,
            draw_edge_label=False,
        )
        AG.plot(
            filename=os.path.join(output_coords, str(user_id_this)),
            filter_node_importance=25,
            draw_edge_label=False,
            layout="coordinate",
        )


def scatterplot_matrix(feature_df, use_features, col_names=None, clustering=None, save_path=None):
    """
    Scatterplot matrix for selected features

    Arguments
        clustering: List of cluster labels for each item
    """
    map_cluster_to_col = {cluster: i for i, cluster in enumerate(np.unique(clustering))}
    if len(use_features) > 6:
        warnings.warn("More than 6 features does not make sense in scatterplot matrix, only using first 6")
        use_features = use_features[:6]
    # define col names
    if col_names is None:
        col_names = use_features
    # transform to df
    feature_df = feature_df.loc[:, use_features]
    if clustering is not None:
        feature_df["cluster"] = clustering
        col_dict = {cluster: sns.color_palette()[map_cluster_to_col[cluster]] for cluster in clustering}
        print(col_dict)
        sns.pairplot(feature_df, hue="cluster", palette=col_dict)
    else:
        sns.pairplot(feature_df)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_correlation_matrix(feat1, feat2, save_path=None):
    correlations = np.zeros((len(feat1.columns), len(feat1.columns)))
    for i, raw_feat in enumerate(feat1.columns):
        for j, graph_feat in enumerate(feat2.columns):
            r, p = scipy.stats.pearsonr(feat1[raw_feat], feat2[graph_feat])
            correlations[i, j] = r
    plt.figure(figsize=(20, 10))
    for i in range(len(correlations)):
        correlations[i, i] = 0
    df = pd.DataFrame(correlations, columns=feat1.columns, index=feat2.columns)
    sns.heatmap(df, annot=True, cmap="PiYG")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_extremes_features(
    feat_path=os.path.join("out_features", "final_5_n0_cleaned", "gc1_graph_features_0.csv"), nr_plot=1
):
    feats = pd.read_csv(feat_path, index_col="user_id")
    for feature in feats.columns:
        vals = feats[feature].values
        smallest = feats[feats[feature] < np.quantile(vals, 0.05)].index
        highest = feats[feats[feature] > np.quantile(vals, 0.95)].index
        print("FEATURE", feature)
        for i in range(nr_plot):
            img_small = plt.imread(os.path.join("graph_images", "gc1", "coords", str(smallest[i]) + ".png"))
            img_high = plt.imread(os.path.join("graph_images", "gc1", "coords", str(highest[i]) + ".png"))
            img_small_spring = plt.imread(os.path.join("graph_images", "gc1", "spring", str(smallest[i]) + ".png"))
            img_high_spring = plt.imread(os.path.join("graph_images", "gc1", "spring", str(highest[i]) + ".png"))
            plt.figure(figsize=(20, 8))
            plt.subplot(1, 4, 1)
            plt.imshow(img_small)
            plt.title("small " + feature)
            plt.axis("off")
            plt.subplot(1, 4, 2)
            plt.imshow(img_small_spring)
            plt.title("small " + feature)
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(img_high)
            plt.title("high " + feature)
            plt.axis("off")
            plt.subplot(1, 4, 4)
            plt.imshow(img_high_spring)
            plt.title("high " + feature)
            plt.axis("off")
            plt.tight_layout()
            plt.show()


def plot_powerlaw_rank_fit(g):
    from scipy.optimize import curve_fit

    def func_simple_powerlaw(x, beta):
        return x ** (-beta)

    vals = np.array(list(dict(g.out_degree()).values()))
    # degrees: np.array(list(dict(graph.out_degree()).values()))
    # transitions: np.array([edge[2]["weight"] for edge in g.edges(data=True)])
    sorted_vals = (sorted(vals)[::-1])[:20]
    normed_vals = sorted_vals / np.sum(sorted_vals)
    normed_vals = normed_vals / normed_vals[0]
    params = curve_fit(func_simple_powerlaw, np.arange(len(normed_vals)) + 1, normed_vals, maxfev=3000, bounds=(0, 5))
    beta = params[0]
    x = np.arange(1, 20, 0.1)
    y = func_simple_powerlaw(x, beta)
    plt.plot(np.arange(len(normed_vals)) + 1, normed_vals)
    plt.plot(x, y)
    plt.title("Beta is " + str(beta))
    plt.show()


if __name__ == "__main__":
    study = "yumuv_graph_rep"
    from utils import get_con
    from future_trackintel.utils import read_graphs_from_postgresql

    con = get_con()
    graph_dict = read_graphs_from_postgresql(
        graph_table_name="full_graph", psycopg_con=con, graph_schema_name=study, file_name="graph_data"
    )
    # path_to_pickle = os.path.join(".", "data_out", "graph_data", study, "counts_full.pkl")
    plot_all_graphs(graph_dict, study)
