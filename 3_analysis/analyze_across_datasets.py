import os
import numpy as np
import pandas as pd

from clustering import normalize_and_cluster
from compare_clustering import compute_all_scores


def load_all(type="graph", node_importance=50):
    """
    type: one of graph, raw
    """
    all_together = []
    study_labels = []
    for study in STUDIES:  # , "yumuv_graph_rep"]: # _{node_importance}
        graph_features = pd.read_csv(
            os.path.join("out_features", f"{study}_graph_features_{node_importance}.csv"), index_col="user_id"
        )
        # TODO: need to filter raw features?
        raw_features = pd.read_csv(os.path.join("out_features", f"{study}_raw_features.csv"), index_col="user_id")
        raw_features = raw_features[raw_features.index.isin(graph_features.index)]

        if type == "graph":
            all_together.append(graph_features)
        elif type == "raw":
            all_together.append(raw_features)

        study_labels.extend([study for _ in range(len(graph_features))])
    # concatenate
    features_all_datasets = pd.concat(all_together)
    features_all_datasets["study"] = study_labels
    print("Samples per study:", np.unique(study_labels, return_counts=True))
    return features_all_datasets


def mean_features_by_study(features, out_path=None):
    # leave away last column because it's the study label
    agg = {feat: "mean" for feat in features.columns[:-1]}
    # group and aggregate
    mean_features = features.groupby("study").agg(agg)
    if out_path:
        mean_features.to_csv(out_path)
    else:
        print(mean_features)


if __name__ == "__main__":
    STUDIES = ["gc1", "gc2", "geolife"]

    n_clusters = 3
    features_all_datasets = load_all()
    cluster_labels = normalize_and_cluster(
        np.array(features_all_datasets.drop(columns=["study"])), n_clusters=n_clusters
    )
    # compare relation between cluster and study labels
    compute_all_scores(cluster_labels, np.array(features_all_datasets["study"]))
    mean_features_by_study(features_all_datasets)
