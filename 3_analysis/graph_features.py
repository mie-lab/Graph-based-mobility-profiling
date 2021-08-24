import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
import numpy as np
import os
import time
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
        # self.graphs, self.users = load_graphs_pkl(
        #     os.path.join(".", "data_out", "graph_data", "gc2", "counts_full.pkl"), node_importance=50
        # )
        self.graphs, self.users = self.load_graphs(study, node_importance)

        # specify necessary parameters for the feature extraction
        self.random_walk_iters = random_walk_iters

        self.prev_features = [
            "nr_edges",
            "cycles_2_random_walk",
            "cycles_3_random_walk",
            "simple_powerlaw_transitions",
            "mean_distance_random_walk",
            "mean_sp_length",
            "mean_node_degree",
            "mean_betweenness_centrality",
        ]
        self.random_walk_features = [
            "mean_distance_random_walk",
            "cycle_length_mu",
            "cycle_length_sigma",
            "ratio_nodes_random_walk",
            "core_periphery_random_walk",
        ]
        # default: Use random walk and power law features
        self.default_features = self.random_walk_features + ["simple_powerlaw_transitions"]
        self.all_features = [f for f in dir(self) if not f.startswith("_")]

    def load_graphs(self, study, node_importance):        
        graphs, users = load_graphs_postgis(study, node_importance=node_importance)
        print("loaded graphs", len(graphs))
        return graphs, users
        
    def _check_implemented(self, features):
        # check if all required features are implemented
        for feat in features:
            if not hasattr(self, feat):
                raise NotImplementedError(f"Feature {feat} ist not implemented!")

    def __call__(self, features="default", parallelize=False, **kwargs):
        """Compute desired features for all graphs and possibly parallelize over graphs"""
        if features == "default":
            features = self.default_features
        if features == "all":
            features = self.all_features
        if features == "random_walk":
            features = self.random_walk_features
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
                delayed(compute_feat)(graph, features) for graph in self.graphs
            )
        else:
            feature_matrix = [compute_feat(graph, features) for graph in self.graphs]
        feature_matrix = np.array(feature_matrix)
        print("feature matrix shape", feature_matrix.shape)
        # convert to dataframe
        feature_df = pd.DataFrame(feature_matrix, index=self.users, columns=features)
        feature_df.index.set_names("user_id", inplace=True)

        return feature_df

    # --------------------- GRAPH LEVEL FEATURES --------------------
    def nr_nodes(self, graph):
        return graph.number_of_nodes()

    def nr_edges(self, graph):
        return graph.number_of_edges()

    def components(self, graph):
        """
        Count number of connected components
        """
        comp = nx.connected_components(graph)
        # I thought about measuring the graph diameter, but this is already part of the sp emb
        # max([nx.algorithms.distance_measures.diameter(test_graph.subgraph(c).copy()) for c in comp])
        return sum(1 for c in comp)  # count elements in generator

    def _graphlets(self, graph):
        # TODO
        # https://github.com/KirillShmilovich/graphlets
        pass

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
        for step in range(self.random_walk_iters):
            # get out neighbors with corresponding transition number
            neighbor_edges = graph.out_edges(current_node, data=True)
            # check if we are at a dead end OR if we get stuck at one node and only make cycles of len 1 there
            at_dead_end = len(neighbor_edges) == 0
            at_inf_loop = len(neighbor_edges) == 1 and [n[1] for n in neighbor_edges][0] == current_node
            if at_dead_end or at_inf_loop:
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
            out_probs = out_weights / np.sum(out_weights)
            next_node = [n[1] for n in neighbor_edges]
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

    def cycles_2_random_walk(self, graph):
        random_walk_sequence = self._random_walk(graph)
        return count_cycles(random_walk_sequence, cycle_len=2)

    def cycles_3_random_walk(self, graph):
        random_walk_sequence = self._random_walk(graph)
        return count_cycles(random_walk_sequence, cycle_len=3)

    def mean_cycle_len_random_walk(self, graph):
        nodes_on_rw, resets = self._random_walk(graph, return_resets=True)
        cycle_lengths = all_cycle_lengths(nodes_on_rw, resets)
        return np.mean(cycle_lengths)

    def _lognormal_cycle_len_random_walk(self, graph):
        nodes_on_rw, resets = self._random_walk(graph, return_resets=True)
        cycle_lengths = all_cycle_lengths(nodes_on_rw, resets)
        # print(cycle_lengths)
        # get distribution
        uni, counts = np.unique(cycle_lengths, return_counts=True)
        # fix: if we only have cycles of length x, then we need to more zero-datapoints
        if len(uni) == 0:
            uni = np.arange(10) +1
            counts = np.array([1] + [0 for _ in range(9)])
        elif len(uni) ==1:
            uni = np.array(list(uni) + [uni[0] + i for i in range(1, 11)])
            counts = np.array(list(counts) + [0 for _ in range(10)])
        # normalize counts
        normed_counts = counts / np.sum(counts)
        # print(uni, normed_counts)
        # fit log normal
        params, _ = curve_fit(log_normal, uni, normed_counts, maxfev=2000, bounds=([-5, 0], [5, 5]))
        # return mu and sigma
        return params

    def cycle_length_mu(self, graph):
        return self._lognormal_cycle_len_random_walk(graph)[0]

    def cycle_length_sigma(self, graph):
        return self._lognormal_cycle_len_random_walk(graph)[1]

    def _distances_random_walk(self, graph, crs_is_projected=False):
        random_walk_sequence, resets = self._random_walk(graph, return_resets=True)
        # get all shapely Point centers on the random walk
        locs_on_rw = [graph.nodes[node_ind]["center"] for node_ind in random_walk_sequence]
        # get all distances on the random walk
        distances = [
            get_point_dist(locs_on_rw[i], locs_on_rw[i + 1], crs_is_projected=crs_is_projected)
            for i in range(len(locs_on_rw) - 1) if i+1 not in resets
        ]
        return distances

    def mean_distance_random_walk(self, graph, cutoff=300000):
        distances = self._distances_random_walk(graph)
        # filter out 0 distances and far trips
        distances = [d for d in distances if d > 0 and d < cutoff]
        if len(distances)==0:
            distances = [0]
        # return median distance (in m)
        return np.median(distances)

    def ratio_nodes_random_walk(self, graph):
        """Ratio of the number of nodes that are encountered on a random walk"""
        total_nodes = graph.number_of_nodes()
        nodes_on_rw = self._random_walk(graph)
        uni = np.unique(nodes_on_rw)
        return len(uni) / total_nodes

    def core_periphery_random_walk(self, graph, thresh=0.85):
        nodes_on_rw = self._random_walk(graph)
        _, counts = np.unique(nodes_on_rw, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        cumulative_counts = np.cumsum(sorted_counts)
        # number of nodes needed to cover thresh times the traffic
        nodes_in_core = np.where(cumulative_counts > thresh * np.sum(counts))[0][0] + 1
        return nodes_in_core

    # ---------- EDGE FEATURES ---------------------------

    def _transitions(self, graph):
        """Get all edge weights"""
        transition_counts = [edge[2]["weight"] for edge in graph.edges(data=True)]
        return transition_counts

    @get_distribution
    def dist_transitions(self, graph):
        """Compute distribution of edge weights"""
        return self._transitions(graph)

    @get_mean
    def mean_transitions(self, graph):
        return self._transitions(graph)

    def simple_powerlaw_transitions(self, graph):
        """Fit simple power law curve and return beta"""
        transition_counts = self._transitions(graph)
        uni, counts = np.unique(transition_counts, return_counts=True)
        # normalize counts
        counts = counts / np.sum(counts)
        params, _ = curve_fit(func_simple_powerlaw, uni, counts, maxfev=3000, bounds=(0, 4))
        # returns only beta
        return params[0]

    def truncated_powerlaw_transitions(self, graph, return_param="beta"):
        """Fit simple power law curve and return beta"""
        transition_counts = self._transitions(graph)
        uni, counts = np.unique(transition_counts, return_counts=True)
        # normalize counts
        counts = counts / np.sum(counts)
        params, _ = curve_fit(
            func_truncated_powerlaw, uni, counts, maxfev=3000, bounds=([-np.inf, 0, 0], [np.inf, 4, np.inf])
        )
        # returns list of 3 parameters
        return params

    # ---------- NODE FEATURES ---------------------------

    def _sp_length(self, graph):
        """
        Returns discrete histogram of path length occurences
        """
        all_sp = nx.floyd_warshall(graph)
        all_sp_lens = [v for sp_dict in all_sp.values() for v in list(sp_dict.values())]
        return all_sp_lens

    @get_distribution
    def dist_sp_length(self, graph):
        all_sp_lens = self._sp_length(graph)
        sp_lengths = [v for v in all_sp_lens if v < np.inf]
        return sp_lengths

    def not_inf_sp_length(self, graph):
        all_sp_lens = self._sp_length
        all_not_inf = [v for v in all_sp_lens if v < np.inf]
        percent_not_inf = len(all_not_inf) / len(all_sp_lens)
        return percent_not_inf

    def bins_sp_length(self, graph, nr_bins=4):
        all_sp = nx.floyd_warshall(graph)
        # only for bins: get distribution of paths (only possible if no edge weights)
        path_len_counts = np.zeros(nr_bins)
        for node1 in sorted(all_sp.keys()):
            for node2 in range(node1 + 1, max(all_sp.keys())):
                sp_len = all_sp[node1][node2]
                if sp_len <= nr_bins:
                    path_len_counts[int(sp_len) - 1] += 1
        return path_len_counts

    @get_mean
    def mean_sp_length(self, graph):
        all_sp_lens = self._sp_length(graph)
        sp_lengths = [v for v in all_sp_lens if v < np.inf]
        return sp_lengths

    def _degree(self, graph, mode="out"):
        """
        Degree distribution of graph
        """
        # one function for in, out and all degrees
        use_function = {"all": graph.degree(), "out": graph.out_degree(), "in": graph.in_degree()}
        degrees = list(dict(use_function[mode]).values())
        return degrees

    def bins_degree(self, graph, nr_bins=None, mode="out"):
        degrees = self._degree(graph, mode=mode)
        # return degree in bins
        if nr_bins is None:
            nr_bins = max(degrees) + 1
        degree_count_bins = np.zeros(nr_bins)
        uni, degree_counts = np.unique(degrees, return_counts=True)
        degree_count_bins[uni] = degree_counts
        return degree_count_bins

    @get_distribution
    def dist_node_degree(self, graph, mode="out"):
        # return statistics of degree distribution
        return self._degree(graph, mode=mode)

    @get_mean
    def mean_node_degree(self, graph, mode="out"):
        return self._degree(graph, mode=mode)

    def _eigenvector_centrality(self, graph):
        """
        Compute centrality of each node and return distribution statistics
        TODO: could use decorator
        """
        if isinstance(graph, nx.classes.multidigraph.MultiDiGraph):
            graph = nx.DiGraph(graph)
        centrality = nx.eigenvector_centrality_numpy(graph)
        return list(centrality.values())

    @get_distribution
    def dist_eigenvector_centrality(self, graph):
        return self._eigenvector_centrality(graph)

    def _betweenness_centrality(self, graph):
        if isinstance(graph, nx.classes.multidigraph.MultiDiGraph):
            graph = nx.DiGraph(graph)
        centrality = nx.algorithms.centrality.betweenness_centrality(graph)
        return list(centrality.values())

    @get_mean
    def mean_betweenness_centrality(self, graph):
        return self._betweenness_centrality(graph)


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
    out_dir = "test_get_all"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Generate feature matrix
    tic = time.time()
    graph_feat = GraphFeatures(study, node_importance=node_importance)
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

    use_features = graph_feat.random_walk_features
    scatterplot_matrix(cleaned_feat_df, use_features, clustering=kmeans.labels_, save_path=out_path + ".pdf")
