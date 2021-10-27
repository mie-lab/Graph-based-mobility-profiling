import networkx as nx
import numpy as np
import os
import time
from numpy.core.fromnumeric import sort
import pandas as pd
import argparse

from scipy.optimize import curve_fit

from joblib import Parallel, delayed


from utils import *


class GraphFeatures:
    def __init__(self, study, node_importance=50, random_walk_iters=500):
        """
        study: str, study name
        node_importance: int, only keep x most important nodes for each graph
        """
        # Load from pickle
        # self._graphs, self._users = load_graphs_pkl(
        #     os.path.join(".", "data_out", "graph_data", "gc2", "counts_full.pkl"), node_importance=50
        # )
        self._debug = False
        if "yumuv" in study and study != "yumuv_graph_rep":
            # for yumuv: get before or after
            assert study[:6] == "yumuv_", "must be named yumuv_before, yumuv_after or yumuv_full"
            before_or_after = study.split("_")[1]
            self._graphs, self._users = load_graphs_cross_sectional(before_or_after, node_importance)
        else:
            self._graphs, self._users = self._load_graphs(study, node_importance)
        print("Loaded data", len(self._graphs))

        # specify necessary parameters for the feature extraction
        self._random_walk_iters = random_walk_iters

        self.all_features = [f for f in dir(self) if not f.startswith("_")]

    def _load_graphs(self, study, node_importance):
        graphs, users = load_graphs_postgis(study, node_importance=node_importance, decompress=True)
        print("loaded graphs", len(graphs))
        return graphs, users

    def _check_implemented(self, features):
        # check if all required features are implemented
        for feat in features:
            if not hasattr(self, feat):
                raise NotImplementedError(f"Feature {feat} ist not implemented!")

    def __call__(self, features="default", parallelize=False, **kwargs):
        """Compute desired features for all graphs and possibly parallelize over graphs"""
        # if features == "default":
        #     features = self.default_features
        # if features == "all":
        #     features = self.all_features
        # if features == "random_walk":
        #     features = self.random_walk_features
        features = self.all_features
        # Check that the features are computable
        self._check_implemented(features)

        # get names of output features from the dictionary - by default its simply the name itself
        # feat_names = [e for feat in features for e in self.feature_dict[feat].get("feature_names", [feat])]
        print("Computing the following features")
        print(features)

        # helper function: compute all features of one graph
        def compute_feat(graph, features):
            feature_output = []
            for feat in features:
                feature_function = getattr(self, feat)
                this_feat_out = feature_function(graph)
                feature_output.append(this_feat_out)
            return np.array(feature_output)

        if parallelize:
            feature_matrix = Parallel(n_jobs=4, prefer="threads")(
                delayed(compute_feat)(graph, features) for graph in self._graphs
            )
        else:
            feature_matrix = []
            for user, graph in zip(self._users, self._graphs):
                feature_matrix.append(compute_feat(graph, features))
        feature_matrix = np.array(feature_matrix)
        print("feature matrix shape", feature_matrix.shape)
        # convert to dataframe
        feature_df = pd.DataFrame(feature_matrix, index=self._users, columns=features)
        feature_df.index.set_names("user_id", inplace=True)

        return feature_df

    def _test_feature(self, feature, nr_do="all"):
        for feature in self.all_features:
            # print(feature)
            tic = time.time()
            # self._debug = True
            if nr_do == "all":
                nr_do = len(self._users)
            feature_avg = []
            for user, graph in zip(self._users[:nr_do], self._graphs[:nr_do]):
                feature_function = getattr(self, feature)
                this_feat_out = feature_function(graph)
                feature_avg.append(this_feat_out)
                # print("Feature output:", this_feat_out)
            # print(time.time() - tic)
            print("AVG", feature, np.mean(feature_avg), np.std(feature_avg))
        exit()

    def _random_walk(self, graph, return_resets=False):
        # start at node with highest degree
        all_degrees = np.array(graph.out_degree())
        start_node = all_degrees[np.argmax(all_degrees[:, 1]), 0]
        current_node = start_node

        # check if we can walk somewhere at all
        if np.max(all_degrees[:, 1]) == 0:
            return 0  # TODO

        encountered_locations = [current_node]
        number_of_walks = 0
        # keep track of when we reset the position to home --> necessary for cycle count
        reset_to_home = []
        for step in range(self._random_walk_iters):
            # get out neighbors with corresponding transition number
            neighbor_edges = graph.out_edges(current_node, data=True)
            # check if we are at a dead end OR if we get stuck at one node and only make cycles of len 1 there
            at_dead_end = len(neighbor_edges) == 0
            at_inf_loop = len(neighbor_edges) == 1 and [n[1] for n in neighbor_edges][0] == current_node
            # or we have a transition weight of 0
            at_zero_transition = (
                len(neighbor_edges) > 0 and np.sum(np.array([n[2]["weight"] for n in neighbor_edges])) == 0
            )
            if at_dead_end or at_inf_loop or at_zero_transition:
                # increase number of walks counter
                number_of_walks += 1
                # reset current node
                current_node = start_node
                neighbor_edges = graph.out_edges(current_node, data=True)
                # we are again at the start node
                encountered_locations.append(start_node)
                # reset location is step + 2 because in encountered_locations
                prev_added = len(reset_to_home)
                reset_to_home.append(step + 1 + prev_added)

            out_weights = np.array([n[2]["weight"] for n in neighbor_edges])
            out_weights = out_weights[~np.isnan(out_weights)]
            out_probs = out_weights / np.sum(out_weights)
            next_node = [n[1] for n in neighbor_edges]
            if np.any(np.isnan(out_probs)):
                print(out_probs, out_weights, at_zero_transition)
            # draw one neightbor randomly, weighted by transition count
            current_node = np.random.choice(next_node, p=out_probs)
            # print("used node with weight", out_weights[next_node.index(current_node)])
            # collect node (features)
            encountered_locations.append(current_node)

        if return_resets:
            return encountered_locations, reset_to_home
        # simply save the encountered nodes here
        return encountered_locations
        # extract features from random walk
        # 1) distribution of cycles
        # cycles_on_walk = count_cycles(encountered_locations, max_len=self.max_cycle_len)
        # 2) number of encountered nodes
        # self.nr_encountered_nodes = len(np.unique(encountered_locations))
        # 3) TODO: distribution of means of tranport or other node/edge features?
        # return list(cycles_on_walk)

    def _home_cycle_lengths(self, graph):
        """Get cycle lengths of journeys (starting and ending at home"""
        nodes_on_rw, resets = self._random_walk(graph, return_resets=True)
        assert (
            len(resets) == 0 or len(np.unique(np.array(nodes_on_rw)[resets])) == 1
        ), "reset indices must always be a home node"
        cycle_lengths = []
        home_node = nodes_on_rw[0]
        at_home = np.where(np.array(nodes_on_rw) == home_node)[0]
        for i in range(len(at_home) - 1):
            if at_home[i + 1] not in resets:
                cycle_lengths.append(at_home[i + 1] - at_home[i])
        return cycle_lengths

    def journey_length(self, graph):
        cycle_lengths = self._home_cycle_lengths(graph)
        if self._debug:
            print(cycle_lengths)
        if len(cycle_lengths) == 0:
            return 0
        return np.mean(cycle_lengths)

    def _transitions(self, graph):
        """Get all edge weights"""
        transition_counts = [edge[2]["weight"] for edge in graph.edges(data=True)]
        return transition_counts

    def _weighted_dists(self, graph):
        dist_list = []
        for (u, v, data) in graph.edges(data=True):
            loc_u = graph.nodes[u]["center"]
            loc_v = graph.nodes[v]["center"]
            weight = data["weight"]
            dist = get_point_dist(loc_u, loc_v, crs_is_projected=False)
            dist_list.extend([dist for _ in range(int(weight))])
        return dist_list

    def median_trip_distance(self, graph):
        dist_list = self._weighted_dists(graph)
        return np.median(dist_list)

    def highest_decile_distance(self, graph):
        dist_list = self._weighted_dists(graph)
        return np.quantile(dist_list, 0.9)

    def _degree(self, graph, mode="out"):
        """
        Degree distribution of graph
        """
        # one function for in, out and all degrees
        use_function = {"all": graph.degree(), "out": graph.out_degree(), "in": graph.in_degree()}
        degrees = list(dict(use_function[mode]).values())
        return degrees

    def _fit_powerlaw(self, item_list):
        if len(item_list) == 0 or np.sum(item_list) == 0:
            return 0

        sorted_vals = sorted(item_list)[::-1]
        # get relative probability
        normed_vals = sorted_vals / np.sum(sorted_vals)
        # Normalize by first value! Because: power function 1/x^beta always passes through (1,1) - we want to fit this
        normed_vals = normed_vals / normed_vals[0]
        params, _ = curve_fit(
            func_simple_powerlaw, np.arange(len(normed_vals)) + 1, normed_vals, maxfev=3000, bounds=(0, 5)
        )

        # Prev version: with cutoff and no normalization
        # sorted_vals = (sorted(item_list)[::-1])[:cutoff]
        # normed_degrees = sorted_vals / np.sum(sorted_vals)
        return params[0]

    def degree_beta(self, graph):
        degrees = np.array(list(dict(graph.out_degree()).values()))
        return self._fit_powerlaw(degrees)

    def transition_beta(self, graph):
        transitions = np.array([edge[2]["weight"] for edge in graph.edges(data=True)])
        return self._fit_powerlaw(transitions)

    def mean_clustering_coeff(self, graph):
        clusterings = nx.algorithms.cluster.clustering(nx.DiGraph(graph), weight="weight")
        return np.mean(list(dict(clusterings).values()))


if __name__ == "__main__":
    """Test on example data"""
    # todo: move to separate process script?
    from plotting import scatterplot_matrix
    from utils import normalize_features, clean_equal_cols, load_graphs_pkl

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--study", type=str, required=True, help="study - one of gc1, gc2, geolife")
    parser.add_argument("-n", "--nodes", type=int, default=0, help="number of x important nodes. Set -1 for all nodes")
    args = parser.parse_args()

    study = args.study
    node_importance = args.nodes
    out_dir = os.path.join("out_features", "test")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Generate feature matrix
    tic = time.time()
    graph_feat = GraphFeatures(study, node_importance=node_importance)
    # TESTING:
    graph_feat._test_feature("unique_journeys")

    feat_matrix = graph_feat(features="default", parallelize=False)
    print(feat_matrix)
    print("time for feature generation", time.time() - tic)

    out_path = os.path.join(out_dir, f"{study}_graph_features_{node_importance}")

    feat_matrix.to_csv(out_path + ".csv")
    if study == "yumuv_graph_rep":
        cg, tg = split_yumuv_control_group(feat_matrix)
        cg.to_csv(out_path.replace("yumuv_graph_rep", "yumuv_cg") + ".csv")
        tg.to_csv(out_path.replace("yumuv_graph_rep", "yumuv_tg") + ".csv")

    # Clean and normalize
    cleaned_feat_df = clean_equal_cols(feat_matrix)
    print(np.array(cleaned_feat_df).shape)
    normed_feat_matrix = normalize_features(np.array(cleaned_feat_df))

    # KMeans
    from sklearn.cluster import KMeans

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normed_feat_matrix)
    labels = kmeans.labels_
    print("Kmeans labels", labels)

    # Plot scatterplot matrix

    use_features = graph_feat.default_features
    scatterplot_matrix(cleaned_feat_df, use_features, clustering=kmeans.labels_, save_path=out_path + ".pdf")
