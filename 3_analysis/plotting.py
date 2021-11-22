import warnings
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import chi2_contingency, contingency

matplotlib.rcParams.update({"font.size": 15, "axes.labelsize": 15})

from future_trackintel.activity_graph import activity_graph
import os
import pickle


def plot_all_graphs(AG_dict, study, filter_node=100):
    """Originally code of example_for_nina file, now function to plot the graphs"""
    output_spring = os.path.join(".", "graph_images", study, f"spring_{filter_node}")
    if not os.path.exists(output_spring):
        os.makedirs(output_spring)

    output_coords = os.path.join(".", "graph_images", study, f"coords_{filter_node}")
    if not os.path.exists(output_coords):
        os.makedirs(output_coords)

    # Use the following in activtitiy_graph_utils in order to plot only on switzerland
    # lon_min, lon_max, lat_min, lat_max = (6.218109005202716, 8.968002536801063, 45.87257606616743, 47.03243181641454)

    for user_id_this, AG in AG_dict.items():

        AG.plot(
            filename=os.path.join(output_spring, str(user_id_this)),
            filter_node_importance=filter_node,
            filter_extent=False,
            draw_edge_label=False,
        )
        AG.plot(
            filename=os.path.join(output_coords, str(user_id_this)),
            filter_node_importance=filter_node,
            filter_dist=400,
            draw_edge_label=False,
            layout="coordinate",
        )


def get_percentage(df, var1, var2):
    return (
        df.groupby([var1])[var2]
        .value_counts(normalize=True)
        .rename("Percentage")
        .mul(100)
        .reset_index()
        .sort_values(var2)
    )


def cluster_by_study(feats, out_path=None, fontsize_dict={"font.size": 28, "axes.labelsize": 30}):
    """
    Feats requires column study, column clustering,
    """
    matplotlib.rcParams.update(fontsize_dict)
    study_mapping = {
        "gc1": "Green Class 1",
        "gc2": "Green Class 2",
        "yumuv_graph_rep": "YUMUV",
        "geolife": "Geolife",
        "tist_toph100": "Foursquare",
        "tist_random100": "Foursquare",
    }
    filtered = feats[feats["study"].isin(study_mapping.keys())]
    filtered["study"] = filtered["study"].apply(lambda x: study_mapping[x])
    df_perc = get_percentage(filtered, "study", "cluster")
    plt.figure(figsize=(20, 10))
    p = sns.barplot(x="study", y="Percentage", hue="cluster", data=df_perc)
    plt.xlabel("")
    plt.legend(ncol=3, framealpha=0.8, loc="upper center")
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()


def plot_cluster_characteristics(
    feats,
    out_path=None,
    feat_columns=[
        "degree_beta",
        "journey_length",
        "hub_size",
        "highest_decile_distance",
        "median_trip_distance",
        "transition_beta",
    ],
    fontsize_dict={"font.size": 28, "axes.labelsize": 30},
    plot_mode="english",
):
    matplotlib.rcParams.update(fontsize_dict)
    feat_rn_dict = {group: group.replace(" ", "\n") for group in np.unique(feats["cluster"])}
    feats["cluster"] = feats["cluster"].apply(lambda x: feat_rn_dict[x])
    if plot_mode == "german":
        rn_dict = {"cluster": "User group", "value": "Standardabweichungen vom Durchschnitt", "variable": "Feature"}
    else:
        rn_dict = {"cluster": "User group", "value": "Standard deviations from mean", "variable": "Feature"}

    # filter out the ones that are double
    if "study" in feats.columns:
        feats_by_cluster = feats[
            ~feats["study"].isin(["yumuv_after_cg", "yumuv_after_tg", "yumuv_before_cg", "yumuv_before_tg"])
        ]
    else:
        feats_by_cluster = feats.copy()
    # NORMALIZE
    for col in feat_columns:
        feats_by_cluster[col] = (feats_by_cluster[col] - np.mean(feats_by_cluster[col].values)) / np.std(
            feats_by_cluster[col].values
        )
    # MELT AND PLOT
    feats_by_cluster = pd.melt(feats_by_cluster, id_vars=["cluster"], value_vars=feat_columns)
    feats_by_cluster.rename(columns=rn_dict, inplace=True)
    plt.figure(figsize=(20, 10))
    p = sns.barplot(x=rn_dict["cluster"], y=rn_dict["value"], hue=rn_dict["variable"], data=feats_by_cluster)
    plt.xlabel("")
    if plot_mode == "german":
        plt.ylim(-1.5, 3)
        plt.legend(ncol=2, framealpha=1)
    else:
        plt.ylim(-1, 3)
        plt.legend(ncol=3, framealpha=1)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()


def scatterplot_matrix(feature_df_in, use_features, clustering=None, save_path=None):
    """
    Scatterplot matrix for selected features

    Arguments
        clustering: List of cluster labels for each item
    """
    map_cluster_to_col = {cluster: i for i, cluster in enumerate(np.unique(clustering))}
    if len(use_features) > 6:
        warnings.warn("More than 6 features does not make sense in scatterplot matrix, only using first 6")
        use_features = use_features[:6]

    feature_df = feature_df_in.rename(columns=column_mapping)
    use_features = [column_mapping[feat] for feat in use_features]

    # transform to df
    feature_df = feature_df.loc[:, use_features]
    if clustering is not None:
        feature_df["Group"] = clustering
        col_dict = {cluster: sns.color_palette()[map_cluster_to_col[cluster]] for cluster in clustering}
        sns.pairplot(feature_df, hue="Group", palette=col_dict)
    else:
        sns.pairplot(feature_df)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


column_mapping = {
    "degree_beta": "degree\nbeta",
    "transition_beta": "transition\nbeta",
    "journey_length": "journey\nlength",
    "mean_trip_distance": "mean trip\ndistance",
    "median_trip_distance": "median trip\ndistance",
    "mean_clustering_coeff": "mean\nclustering\ncoefficient",
    "distance_ht_index": "ht index",
    "highest_decile_distance": "highest decile\ndistance",
    "hub_size": "hub\nsize",
}


def plot_correlation_matrix(feat1, feat2, save_path=None, fontsize=25):
    correlations = np.zeros((len(feat1.columns), len(feat1.columns)))
    for i, raw_feat in enumerate(feat1.columns):
        for j, graph_feat in enumerate(feat2.columns):
            r, p = scipy.stats.pearsonr(feat1[raw_feat], feat2[graph_feat])
            correlations[i, j] = r
    plt.figure(figsize=(20, 10))
    for i in range(len(correlations)):
        correlations[i, i] = 0
    # if the feature is in the mapping dictionary, map it to the new name
    col_labs = [column_mapping.get(col, col) for col in feat1.columns]
    ind_labs = [column_mapping.get(col, col) for col in feat2.columns]
    df = pd.DataFrame(correlations, columns=col_labs, index=ind_labs)
    sns.heatmap(df, annot=True, cmap="PiYG", annot_kws={"size": fontsize})
    plt.xticks(np.arange(len(col_labs)) + 0.5, col_labs, fontsize=fontsize)
    plt.yticks(np.arange(len(col_labs)) + 0.5, ind_labs, fontsize=fontsize, rotation=0)
    plt.tight_layout()
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


def print_chisquare(occ1, occ2):
    # occ1 = np.array(occ1)
    # occ2 = np.array(occ2)
    # print("chisquare input:", occ1 / np.sum(occ1) * 100 + 1, occ2 / np.sum(occ2) * 100 + 1)
    # print("CHISQUARE", chisquare(occ1 / np.sum(occ1) * 100 + 1, occ2 / np.sum(occ2) * 100 + 1))
    contingency_table = np.array([occ1, occ2])
    print("Contingency table")
    print("Kontrollgruppe:", occ1)
    print("Testgruppe:", occ2)
    # print(contingency_table)
    contingency_table = contingency_table[:, np.any(contingency_table, axis=0)]
    stat, p, dof, expected = chi2_contingency(contingency_table)
    print(f"CHI SQUARE TEST: With p={round(p, 2)}, the groups are significantly different")
    return p


def barplot_clusters(
    labels1,
    labels2,
    name1="Group 1",
    name2="Group 2",
    save_name="test",
    title="",
    out_path=None,
    rotate=True,
    yesno=False,
):
    occuring_labels = np.unique(list(labels1) + list(labels2))
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)

    if yesno:
        occ1 = [sum(labels1 == lab) for lab in occuring_labels]
        occ2 = [sum(labels2 == lab) for lab in occuring_labels]
        chisquare_text = ""
        ylabel = "Anzahl an Nutzern mit Antwort Ja / Nein"
    else:
        occ1 = [sum(labels1 == lab) / len(labels1) for lab in occuring_labels]
        occ2 = [sum(labels2 == lab) / len(labels2) for lab in occuring_labels]
        occ1_unnormalized = [sum(labels1 == lab) for lab in occuring_labels]
        occ2_unnormalized = [sum(labels2 == lab) for lab in occuring_labels]

        print("Do chi square test for ", name1, name2)
        p_val = print_chisquare(occ1_unnormalized, occ2_unnormalized)
        chisquare_text = "_chisquare" + str(round(p_val, 2))
        ylabel = "Ratio of users"

    x = np.arange(len(occuring_labels))
    plt.figure(figsize=(10, 8))
    plt.bar(x - 0.2, occ1, 0.4, label=name1)
    plt.bar(x + 0.2, occ2, 0.4, label=name2)
    rot = 90 if rotate else 0

    labs_with_absatz = [lab.replace(" ", "\n") for lab in occuring_labels]
    plt.xticks(x, labs_with_absatz, rotation=rot)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title, fontsize=15)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(os.path.join(out_path, save_name + chisquare_text + ".png"), dpi=600)
    else:
        plt.show()


if __name__ == "__main__":
    study = "gc2"
    from analysis_utils import get_con
    from utils import read_graphs_from_postgresql

    con = get_con()
    graph_dict = read_graphs_from_postgresql(
        graph_table_name="full_graph", psycopg_con=con, graph_schema_name=study, file_name="graph_data"
    )
    # path_to_pickle = os.path.join(".", "data_out", "graph_data", study, "counts_full.pkl")
    plot_all_graphs(graph_dict, study)
