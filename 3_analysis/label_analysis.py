from genericpath import exists
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

fontsize_dict = {"font.size": 15, "axes.labelsize": 15}
matplotlib.rcParams.update(fontsize_dict)
from utils import load_user_info, load_all_questions


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


def entropy(df_in, label, cluster, treat_nans="remove", print_parts=False, nr_bins=4):
    """Entropy of labels over the clusters. See Ben-Gal et al paper"""
    # only need these two columns
    df_in = df_in[[cluster, label]]
    # clean the nans:
    if treat_nans == "remove":
        df = _rm_nans(df_in, label, cluster)
    elif treat_nans == "rename":
        df = _rename_nans(df_in, label, cluster)
    else:
        df = df_in.copy()

    # bin labels for numeric columns
    col_vals = df[label]
    mapping = {}
    if len(np.unique(col_vals)) > 4 and not isinstance(col_vals.values[0], str):
        lower_cutoff = 0
        for bin in range(nr_bins):
            upper_cutoff = np.quantile(df[label], (bin + 1) / nr_bins)
            cond = (col_vals <= upper_cutoff) & (col_vals >= lower_cutoff)
            df.loc[cond, label] = bin
            mapping[bin] = f"{lower_cutoff}-{upper_cutoff}"
            lower_cutoff = upper_cutoff

    if len(mapping) > 0 and print_parts:
        print(mapping)
    n = len(df)  # total number of points
    uni, counts = np.unique(cluster, return_counts=True)
    entropy = 0
    for c, cluster_df in df.groupby(cluster):
        # number of points in this cluster
        n_k = len(cluster_df)
        uni, counts = np.unique(cluster_df[label].values, return_counts=True)
        if print_parts:
            print(c, np.around(counts / n_k, 2), uni)
        # compute entropy of
        inner_entropy = 0
        for (u, c) in zip(uni, counts):
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


def get_q_for_col(col, questions):
    if col[0] == "q":
        col_to_qname = "Q" + col.split("_")[0][1:]
    else:
        col_to_qname = col
    try:
        corresponding_q = questions.loc[col_to_qname]["question"]
    except KeyError:
        corresponding_q = col
    if not isinstance(corresponding_q, str):
        corresponding_q = corresponding_q.iloc[0]
    return corresponding_q


def plot_and_entropy(joined, user_info, out_figures, questions=None):
    os.makedirs(out_figures, exist_ok=True)
    # make labels readable
    joined["cluster_to_plot"] = joined["cluster"].apply(lambda x: "\n".join(x.split(" ")))
    for col in user_info.columns:
        if "id" in col:
            continue
        not_nan = pd.isna(joined[col]).sum()
        if not_nan / len(joined) > 0.5:
            # print("Skipping because too many missing values:", col)
            continue

        corresponding_q = col
        if questions is not None:
            corresponding_q = get_q_for_col(col, questions)

        # if entropy_1 < 0.95:
        print("\n------", corresponding_q, "------")
        entropy1 = entropy(joined, col, "cluster", print_parts=True)
        rounded_entropy = round(entropy1, 2)
        print("\nEntropy:", round(entropy1, 2), "\n")

        # PLOTTING
        col_vals = joined[col]
        col_vals = col_vals[~pd.isna(col_vals)]
        # skip the ones where plotting does not make sense
        if (
            len(np.unique(col_vals)) < 2
            or pd.isna(entropy1)
            or entropy1 > 0.99
            or "click" in col
            or "page_submit" in col
        ):
            continue
        if len(np.unique(col_vals)) < 4:
            plt.figure(figsize=(10, 5))
            sns.countplot(x="cluster_to_plot", hue=col, data=joined)
            plt.title(corresponding_q, fontsize=12)
            plt.savefig(os.path.join(out_figures, f"{col}_{rounded_entropy}.png"))
        elif not isinstance(col_vals.values[0], str):
            plt.figure(figsize=(10, 5))
            sns.boxplot(x="cluster_to_plot", y=col, data=joined)
            plt.title(corresponding_q, fontsize=12)
            plt.savefig(os.path.join(out_figures, col + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_dir", type=str, default="results", help="input and output directory")
    parser.add_argument("-s", "--study", type=str, default="yumuv", help="which study to analyze")
    args = parser.parse_args()

    path = args.inp_dir
    study = args.study

    # plot_for_sure = ["age", "hhsize", "hhinc"]

    # load data with clusters already assigned
    try:
        graph_features = pd.read_csv(os.path.join(path, "all_datasets_clustering.csv"), index_col="user_id")
    except FileNotFoundError:
        print("ERROR: all_dataset_clustering.csv file does not exist yet. Run script analyze_study.py first")
        exit()

    # Load the question mapping
    if "yumuv" in study:
        questions = load_all_questions()
    else:
        questions = pd.DataFrame()

    feats_study = graph_features[graph_features["study"] == study]
    user_info = load_user_info(study, index_col="user_id")
    user_info = user_info[~pd.isna(user_info.index)]
    user_info.index = user_info.index.astype(int).astype(str)
    # merge into one table
    joined = feats_study.merge(user_info, how="left", left_index=True, right_index=True)
    # print(joined)
    plot_and_entropy(joined, user_info, os.path.join(path, study + "_label_analysis"), questions=questions)
