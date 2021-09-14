import networkx as nx
import numpy as np
from numpy.core.fromnumeric import sort
import argparse

from scipy.optimize import curve_fit

from joblib import Parallel, delayed


from utils import *
from graph_features import GraphFeatures


class OldGraphFeatures(GraphFeatures):
    def __init__(self):
        super().__init__(self)

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
        self.prev_rw_features = [
            "mean_distance_random_walk",
            "cycle_length_mu",
            "cycle_length_sigma",
            "ratio_nodes_random_walk",
            "hub_size",
        ]
        self.default_features = [
            "unique_journeys",
            "journey_length",
            "hub_size",
            "transition_hhi",
            "trip_distance",
            "degree_hhi",
            "log_hub_size",
            "degree_powerlaw",
            "transition_powerlaw",
            "power_betweenness_centrality",
        ]

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

    def ratio_short_journeys(self, graph, cutoff=3):
        """Compute ratio of cycles of length smaller cutoff
        By default, compute ratio of cycles of length 1 or 2 (simple detour visits) to longer cycles
        """
        cycle_lengths = self._home_cycle_lengths(graph)
        cycle_lengths = np.array(cycle_lengths)
        if len(cycle_lengths) == 0:
            return 1
        return sum(cycle_lengths < cutoff) / len(cycle_lengths)

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
            uni = np.arange(10) + 1
            counts = np.array([1] + [0 for _ in range(9)])
        elif len(uni) == 1:
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
            for i in range(len(locs_on_rw) - 1)
            if i + 1 not in resets
        ]
        return distances

    def mean_distance_random_walk(self, graph, cutoff=300000):
        distances = self._distances_random_walk(graph)
        # filter out 0 distances and far trips
        distances = [d for d in distances if d > 0 and d < cutoff]
        if len(distances) == 0:
            distances = [0]
        # return median distance (in m)
        return np.median(distances)

    def median_distance_journeys(self, graph):
        nodes_on_rw, resets = self._random_walk(graph, return_resets=True)

        home_node = nodes_on_rw[0]
        at_home = np.where(np.array(nodes_on_rw) == home_node)[0]

        journey_distances = []
        # iterate over journeys
        for i in range(len(at_home) - 1):
            # found journey
            if at_home[i + 1] not in resets:
                start_ind = at_home[i]
                end_ind = at_home[i + 1]
                # skip journeys that are only loops at one location
                if end_ind == start_ind + 1:
                    continue
                # get sequence of locations on journey
                journey_sequence = nodes_on_rw[start_ind : end_ind + 1]
                locs_on_journey = [graph.nodes[node_ind]["center"] for node_ind in journey_sequence]
                # get all distances on the random walk
                distances = [
                    get_point_dist(locs_on_journey[i], locs_on_journey[i + 1], crs_is_projected=False)
                    for i in range(len(locs_on_journey) - 1)
                ]
                if self._debug:
                    print(journey_sequence)
                    print(distances)
                journey_distances.append(np.sum(distances))
        if len(journey_distances) == 0:
            return 0
        return np.median(journey_distances)

    def _hhi_old_version(self, item_list, N=20):
        """Compute HHI on the N most often occuring items"""
        uni, counts = np.unique(item_list, return_counts=True)
        if self._debug:
            print("HHI:", uni, counts)
        sorted_counts = np.sort(counts)[-N:]
        sum_first_N = np.sum(sorted_counts)

        hhi_out = 0
        for c in sorted_counts:
            hhi_out += (c / sum_first_N) ** 2
        return hhi_out

    def ratio_nodes_random_walk(self, graph):
        """Ratio of the number of nodes that are encountered on a random walk"""
        total_nodes = graph.number_of_nodes()
        nodes_on_rw = self._random_walk(graph)
        uni = np.unique(nodes_on_rw)
        return len(uni) / total_nodes

    def random_walk_hhi(self, graph):
        nodes_on_rw = self._random_walk(graph)
        return self._hhi(nodes_on_rw)

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

    def bins_degree(self, graph, nr_bins=None, mode="out"):
        degrees = self._degree(graph, mode=mode)
        # return degree in bins
        if nr_bins is None:
            nr_bins = max(degrees) + 1
        degree_count_bins = np.zeros(nr_bins)
        uni, degree_counts = np.unique(degrees, return_counts=True)
        degree_count_bins[uni] = degree_counts
        return degree_count_bins

    def degree_hhi(self, graph, mode="out"):
        degrees = self._degree(graph, mode=mode)
        return self._hhi(degrees)

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

    def hub_size(self, graph, thresh=0.8):
        nodes_on_rw = self._random_walk(graph)
        _, counts = np.unique(nodes_on_rw, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        cumulative_counts = np.cumsum(sorted_counts)
        # number of nodes needed to cover thresh times the traffic
        nodes_in_core = np.where(cumulative_counts > thresh * np.sum(counts))[0][0] + 1
        return nodes_in_core / np.sqrt(graph.number_of_nodes())

    def log_hub_size(self, graph, thresh=0.8):
        nodes_on_rw = self._random_walk(graph)
        _, counts = np.unique(nodes_on_rw, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        sorted_counts = np.log(sorted_counts)
        cumulative_counts = np.cumsum(sorted_counts)
        # number of nodes needed to cover thresh times the traffic
        nodes_in_core = np.where(cumulative_counts > thresh * np.sum(sorted_counts))[0][0] + 1
        return nodes_in_core / graph.number_of_nodes()

    def _hhi(self, item_list):
        """Compute HHI on the N most often occuring items"""
        item_list = np.array(item_list)
        shares = item_list / np.sum(item_list) * 100
        if self._debug:
            print("HHI:", shares)
        return np.sum(shares ** 2)

    def transition_hhi(self, graph):
        transitions = self._transitions(graph)
        if self._debug:
            print(np.array(transitions).astype(int))
        return self._hhi(transitions)

    def degree_hhi(self, graph, mode="in"):
        degrees = self._degree(graph, mode=mode)
        return self._hhi(degrees)

    def _betweenness_centrality(self, graph):
        if isinstance(graph, nx.classes.multidigraph.MultiDiGraph):
            graph = nx.DiGraph(graph)
        centrality = nx.algorithms.centrality.betweenness_centrality(graph)
        return list(centrality.values())

    def betweenness_beta(self, graph):
        centrality_vals = self._betweenness_centrality(graph)
        return self._fit_powerlaw(np.array(centrality_vals))

    def _sp_length(self, graph):
        """
        Returns discrete histogram of path length occurences
        """
        all_sp = nx.floyd_warshall(graph)
        all_sp_lens = [v for sp_dict in all_sp.values() for v in list(sp_dict.values())]
        return all_sp_lens

    def mean_sp_length(self, graph):
        all_sp_lens = self._sp_length(graph)
        sp_lengths = [v for v in all_sp_lens if v < np.inf]
        return np.mean(sp_lengths)

    def mean_trip_distance(self, graph):
        sum_of_weights = 0
        weighted_distance = 0
        for (u, v, data) in graph.edges(data=True):
            loc_u = graph.nodes[u]["center"]
            loc_v = graph.nodes[v]["center"]
            weight = data["weight"]
            sum_of_weights += weight
            dist = get_point_dist(loc_u, loc_v, crs_is_projected=False)
            weighted_distance += dist * weight
        return weighted_distance / sum_of_weights
