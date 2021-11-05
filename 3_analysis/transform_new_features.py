import pickle
import os
import argparse
import pandas as pd

from clustering import ClusterWrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inp_dir", type=str, default=os.path.join("out_features", "test"), help="feature inputs"
    )
    parser.add_argument("-o", "--out_dir", type=str, default="results", help="Path where to output all results")
    args = parser.parse_args()

    inp_dir = args.inp_dir
    results_dir = args.out_dir

    with open(os.path.join(results_dir, "clustering.pkl"), "rb") as infile:
        cluster_wrapper = pickle.load(infile)

    for study in ["yumuv", "gc1", "gc2"]:
        in_features = pd.read_csv(
            os.path.join(inp_dir, f"all_long_{study}_graph_features_0.csv"), index_col=("user_id", "study")
        )

        labels = cluster_wrapper.transform(in_features)
        in_features["cluster"] = labels

        in_features.to_csv(os.path.join(results_dir, f"long_{study}_clustering.csv"))
