import warnings
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from future_trackintel.activity_graph import activity_graph
import os
import pickle

plt.rcParams["axes.labelsize"] = 15


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
        col_dict = {cluster: sns.color_palette()[cluster] for cluster in clustering}
        sns.pairplot(feature_df, hue="cluster", palette=col_dict)
    else:
        sns.pairplot(feature_df)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    study = "gc1"
    from utils import get_con
    from future_trackintel.utils import read_graphs_from_postgresql

    con = get_con()
    graph_dict = read_graphs_from_postgresql(
        graph_table_name="full_graph", psycopg_con=con, graph_schema_name=study, file_name="graph_data"
    )
    # path_to_pickle = os.path.join(".", "data_out", "graph_data", study, "counts_full.pkl")
    plot_all_graphs(graph_dict, study)
