import networkx as nx
import numpy as np
import os
import time
import pandas as pd
from joblib import Parallel, delayed

from utils import dist_names, dist_to_stats, count_cycles, load_graphs_postgis, load_graphs_pkl


class GraphFeatures:
    def __init__(self, graphs, users, random_walk_iters=100, max_cycle_len=5):
        """
        graphs: List of nx graph objects
        users: List of same length as graphs, containing the corresponding user ids
        """
        self.users = users
        self.graphs = graphs

        # specify necessary parameters for the feature extraction
        self.random_walk_iters = random_walk_iters
        self.max_cycle_len = max_cycle_len

        random_walk_feats = ["nr_encountered_locs"] + ["nr_cycles_" + str(i + 1) for i in range(max_cycle_len)]
        sp_feats = ["percent_not_inf"] + dist_names("sp_length")
        # All available graph features
        self.feature_dict = {
            "size": {"function": self.size_emb, "feature_names": ["nr_nodes", "nr_edges"]},
            "connected": {"function": self.components_emb, "feature_names": ["nr_connected_components"]},
            "random_walk": {"function": self.random_walk_emb, "feature_names": random_walk_feats},
            "transitions": {"function": self.edge_weight_emb, "feature_names": dist_names("transitions")},
            "shortest_paths": {"function": self.shortest_path_emb, "feature_names": sp_feats},
            "degrees": {"function": self.degree_emb, "feature_names": dist_names("degree")},
            "centrality": {"function": self.centrality_emb, "feature_names": dist_names("centrality")},
        }
        # print("GraphFeature object initialized, available features:", self.feature_dict)

    def __call__(self, features="all", parallelize=False):
        if features == "all":
            features = list(self.feature_dict.keys())
        # Check that the features are computable
        assert all(
            [feat in self.feature_dict.keys() for feat in features]
        ), "Only features in feature_dict are allowed!"

        # helper function: compute all features of one graph
        def compute_feat(graph, features, check_len):
            feature_output = []
            for feat in features:
                feature_function = self.feature_dict[feat]["function"]
                this_feat_out = feature_function(graph)
                feature_output.extend(this_feat_out)
                # print(len(this_feat_out))
            # print(len(feature_output), feature_output, check_len)
            assert len(feature_output) == check_len
            return np.array(feature_output)

        feat_names = [e for feat in features for e in self.feature_dict[feat]["feature_names"]]
        print("Computing the following features")
        print(feat_names)

        check_len = len(feat_names)  # check later that len of names is same as len of features

        if parallelize:
            feature_matrix = Parallel(n_jobs=4, prefer="threads")(
                delayed(compute_feat)(graph, features, check_len) for graph in self.graphs
            )
        else:
            feature_matrix = [compute_feat(graph, features, check_len) for graph in self.graphs]
        feature_matrix = np.array(feature_matrix)
        print("feature matrix shape", feature_matrix.shape)
        # convert to dataframe
        feature_df = pd.DataFrame(feature_matrix, index=self.users, columns=feat_names)

        return feature_df

    # --------------------- GRAPH LEVEL FEATURES --------------------
    def size_emb(self, graph):
        """Very general features of graph size"""
        feats = [graph.number_of_nodes(), graph.number_of_edges()]
        return feats

    def components_emb(self, graph):
        """
        Count number of connected components
        """
        test_graph = nx.Graph(graph)
        comp = nx.connected_components(test_graph)
        # I thought about measuring the graph diameter, but this is already part of the sp emb
        # max([nx.algorithms.distance_measures.diameter(test_graph.subgraph(c).copy()) for c in comp])
        return [sum(1 for c in comp)]  # count elements in generator

    def graphlets_emb(self, graph):
        # TODO
        # https://github.com/KirillShmilovich/graphlets
        pass

    def random_walk_emb(self, graph, steps=100, max_cycle_len=5):
        # start at node with highest degree
        all_degrees = np.array(graph.out_degree())
        start_node = all_degrees[np.argmax(all_degrees[:, 1]), 0]
        current_node = start_node

        # check if we can walk somewhere at all
        if np.max(all_degrees[:, 1]) == 0:
            return 0  # TODO

        encountered_locations = [current_node]
        number_of_walks = 0
        for step in range(steps):
            # get out neighbors with corresponding transition number
            neighbor_edges = graph.out_edges(current_node, data=True)
            # check if we are at a dead end
            if len(neighbor_edges) == 0:
                # increase number of walks counter
                number_of_walks += 1
                # reset current node
                current_node = start_node
                neighbor_edges = graph.out_edges(current_node, data=True)
            out_weights = np.array([n[2]["weight"] for n in neighbor_edges])
            out_probs = out_weights / np.sum(out_weights)
            next_node = [n[1] for n in neighbor_edges]
            # draw one neightbor randomly, weighted by transition count
            current_node = np.random.choice(next_node, p=out_probs)
            # print("used node with weight", out_weights[next_node.index(current_node)])
            # collect node (features)
            encountered_locations.append(current_node)

        # extract features from random walk
        # 1) distribution of cycles
        cycles_on_walk = count_cycles(encountered_locations, max_len=self.max_cycle_len)
        # 2) number of encountered nodes
        nr_encountered_nodes = len(np.unique(encountered_locations))
        # 3) TODO: distribution of means of tranport or other node/edge features?

        return [nr_encountered_nodes] + list(cycles_on_walk)

    def edge_weight_emb(self, graph):
        """Compute distribution of edge weights"""
        transition_counts = [edge[2]["weight"] for edge in graph.edges(data=True)]
        return dist_to_stats(transition_counts)

    # ---------- NODE FEATURES ---------------------------

    def shortest_path_emb(self, graph, bins=False, nr_bins=4):
        """
        Returns discrete histogram of path length occurences
        """
        # TODO: check what floyd warshall does with a weighted graph! --> bzw how do we bin it then?
        all_sp = nx.floyd_warshall(graph)
        # only for bins: get distribution of paths (only possible if no edge weights)
        if bins:
            path_len_counts = np.zeros(nr_bins)
            for node1 in sorted(all_sp.keys()):
                for node2 in range(node1 + 1, max(all_sp.keys())):
                    sp_len = all_sp[node1][node2]
                    if bins and sp_len <= nr_bins:
                        path_len_counts[int(sp_len) - 1] += 1
            return path_len_counts
        else:
            all_sp_lens = [v for sp_dict in all_sp.values() for v in list(sp_dict.values())]
            # get non_infs:
            all_not_inf = [v for v in all_sp_lens if v < np.inf]
            percent_not_inf = len(all_not_inf) / len(all_sp_lens)
            return [percent_not_inf] + dist_to_stats(all_not_inf)

    def degree_emb(self, graph, bins=False, nr_bins=None, mode="out"):
        """
        Degree distribution of graph
        """
        # one function for in, out and all degrees
        use_function = {"all": graph.degree(), "out": graph.out_degree(), "in": graph.in_degree()}
        degrees = list(dict(use_function[mode]).values())
        # return degree in bins
        if bins:
            if nr_bins is None:
                nr_bins = max(degrees) + 1
            degree_count_bins = np.zeros(nr_bins)
            uni, degree_counts = np.unique(degrees, return_counts=True)
            degree_count_bins[uni] = degree_counts
            return degree_count_bins
        # return statistics of bin distribution
        else:
            return dist_to_stats(degrees)

    def centrality_emb(self, graph):
        """
        Compute centrality of each node and return distribution statistics
        TODO: could use decorator
        """
        if isinstance(graph, nx.classes.multidigraph.MultiDiGraph):
            graph = nx.DiGraph(graph)
        centrality = nx.eigenvector_centrality_numpy(graph)
        return dist_to_stats(list(centrality.values()))


if __name__ == "__main__":
    """Test on example data"""
    import pickle
    from plotting import scatterplot_matrix
    from utils import normalize_features, clean_equal_cols, load_graphs_pkl

    # TODO: node features
    # Load graphs as nx graphs into list
    graphs, users = load_graphs_pkl(
        os.path.join(".", "data_out", "graph_data", "gc2", "counts_full.pkl"), node_importance=50
    )
    # graphs, users = load_graphs_postgis("gc2", node_importance=50)

    print("loaded graphs", len(graphs))

    # Generate feature matrix
    tic = time.time()
    graph_feat = GraphFeatures(graphs, users)
    feat_matrix = graph_feat(parallelize=False)
    print(feat_matrix)
    print("time for feature generation", time.time() - tic)

    # Save feature matrix to pickle
    # with open("features_test.pkl", "wb") as outfile:
    #     pickle.dump((feat_matrix, feat_names, users), outfile)

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

    use_features = ["transitions_mean", "transitions_std", "sp_length_mean", "degree_mean", "nr_cycles_3"]
    col_names = [
        "Avg no.\n of transitions",
        "Std no. of \n transitions",
        "Avg. shortest \n path length",
        "Avg. degree",
        "No. of cycles \n of length 3",
    ]
    scatterplot_matrix(cleaned_feat_df, use_features, clustering=kmeans.labels_, col_names=col_names)
