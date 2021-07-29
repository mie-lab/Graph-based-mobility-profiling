import ntpath

import pandas as pd
from future_trackintel import tigraphs
import numpy as np
import os
import networkx as nx
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import functools
from future_trackintel.activity_graphs_utils import draw_smopy_basemap, nx_coordinate_layout_smopy
from future_trackintel.activity_graphs_utils import haversine_dist_of_shapely_objs as h_dist
from pathlib import Path


class activity_graph:
    def __init__(self, staypoints, locations, node_feature_names=[], trips=None):
        self.validate_user(staypoints, locations)
        self.node_feature_names = node_feature_names
        self.user_id = staypoints["user_id"].iloc[0]
        self.init_activity_dict()
        if trips is not None:
            self.validate_user(trips, locations)
            self.weights_transition_count_trips(trips=trips, staypoints=staypoints)
        else:
            self.weights_transition_count(staypoints)
        self.G = self.generate_activity_graphs(locations)

    def init_activity_dict(self):
        self.adjacency_dict = {}
        self.adjacency_dict["A"] = []
        self.adjacency_dict["location_id_order"] = []
        self.adjacency_dict["edge_name"] = []

    def validate_user(self, staypoints, locations):
        """Test if only a single user id was given"""
        assert len(staypoints["user_id"].unique()) == 1, (
            "An activity graph has to be user specific but your "
            "staypoints have"
            f" these users: {staypoints['user_id'].unique()}"
        )
        assert len(staypoints["user_id"].unique()) == 1, (
            "An activity graph has to be user specific but your "
            "locations have"
            f" these users: {locations['user_id'].unique()}"
        )
        user_staypoints = staypoints["user_id"].unique()
        user_locations = locations["user_id"].unique()

        assert (user_staypoints == user_locations).all(), (
            f"staypoints and locations need the same user_id but your "
            f"data have staypoints: {user_locations} and locations: {user_locations}"
        )

    def weights_transition_count_trips(self, trips, staypoints, adjacency_dict=None):
        """
        # copy of weights_transition_count
        Calculate the number of transition between locations as graph weights.
        Graphs based on the activity locations (trackintel locations) can have several
        types of weighted edges. This function calculates the edge weight based
        on the number of transitions of an individual user between locations.
        The function requires the staypoints to have a cluster id field (e.g.
        staypoints.as_staypoints.extract_locations() was already used.

        Parameters
        ----------
        staypoints : GeoDataFrame

        Returns
        -------
        adjacency_dict : dictionary
                A dictionary of adjacency matrices of type scipy.sparse.coo_matrix
        """
        trips_a = trips.copy()
        staypoints_a = staypoints.copy()
        # join location_id to trips:

        origin_not_na = ~trips_a["origin_staypoint_id"].isna()
        dest_not_na = ~trips_a["destination_staypoint_id"].isna()

        trips_a.loc[origin_not_na, "location_id"] = staypoints_a.loc[
            trips_a.loc[origin_not_na, "origin_staypoint_id"], "location_id"
        ].values
        trips_a.loc[dest_not_na, "location_id_end"] = staypoints_a.loc[
            trips_a.loc[dest_not_na, "destination_staypoint_id"], "location_id"
        ].values

        trips_a = trips_a.sort_values(["started_at"])

        # delete trips with unknown start/end location
        trips_a.dropna(subset=["location_id", "location_id_end"], inplace=True)

        try:
            counts = trips_a.groupby(by=["user_id", "location_id", "location_id_end"]).size().reset_index(name="counts")
        except ValueError:
            # If there are only rows with nans, groupby throws an error but should
            # return an empty dataframe
            counts = pd.DataFrame(columns=["user_id", "location_id", "location_id_end", "counts"])

        # create Adjacency matrix
        A, location_id_order, name = _create_adjacency_matrix_from_transition_counts(counts)

        self.adjacency_dict["A"].append(A)
        self.adjacency_dict["location_id_order"].append(location_id_order)
        self.adjacency_dict["edge_name"].append("transition_counts")

        return adjacency_dict

    def weights_transition_count(self, staypoints, adjacency_dict=None):
        """
        Calculate the number of transition between locations as graph weights.
        Graphs based on the activity locations (trackintel locations) can have several
        types of weighted edges. This function calculates the edge weight based
        on the number of transitions of an individual user between locations.
        The function requires the staypoints to have a cluster id field (e.g.
        staypoints.as_staypoints.extract_locations() was already used.

        Parameters
        ----------
        staypoints : GeoDataFrame

        Returns
        -------
        adjacency_dict : dictionary
                A dictionary of adjacency matrices of type scipy.sparse.coo_matrix
        """

        staypoints_a = staypoints.sort_values(["user_id", "started_at"])
        # Deleting staypoints without cluster means that we count non-direct
        # transitions between two clusters e.g., 1 -> -1 -> 2 as direct transitions
        # between two clusters!
        # E.g., 1 -> 2

        staypoints_a.dropna(subset=["location_id"], inplace=True)
        staypoints_a = staypoints_a.loc[staypoints_a["location_id"] != -1]

        # count transitions between cluster
        staypoints_a["location_id_end"] = staypoints_a.groupby("user_id")["location_id"].shift(-1)

        # drop transitions without locations.
        # this means we only count locations between two valid locations
        staypoints_a.dropna(subset=["location_id", "location_id_end"], inplace=True)

        try:
            counts = (
                staypoints_a.groupby(by=["user_id", "location_id", "location_id_end"]).size().reset_index(name="counts")
            )
        except ValueError:
            # If there are only rows with nans, groupby throws an error but should
            # return an empty dataframe
            counts = pd.DataFrame(columns=["user_id", "location_id", "location_id_end", "counts"])

        # create Adjacency matrix
        A, location_id_order, name = _create_adjacency_matrix_from_transition_counts(counts)

        self.adjacency_dict["A"].append(A)
        self.adjacency_dict["location_id_order"].append(location_id_order)
        self.adjacency_dict["edge_name"].append("transition_counts")

        return adjacency_dict

    @property
    @functools.lru_cache()
    def edge_types(self):
        edge_type_list = []

        for n, nbrsdict in self.G.adjacency():  # iter all nodes
            for nbr, keydict in nbrsdict.items():  # iter all neighbors
                for edge_type_name, _ in keydict.items():  # iter all edges
                    if edge_type_name not in edge_type_list:
                        # append edge attribute to list
                        edge_type_list.append(edge_type_name)

        return edge_type_list

    def to_file(self, path):
        pass

    def get_k_importance_nodes(self, k):
        node_in_degree = np.asarray([(n, self.G.in_degree(n)) for n in self.G.nodes])
        best_ixs = np.argsort(node_in_degree[:, 1],)[
            ::-1
        ][:k]
        # we readdress the first column of node_in_degree with best_ixs in case that node degree are not
        # a serial starting from 0
        return node_in_degree[:, 0][best_ixs]

    def generate_activity_graphs(self, locations):
        """
        Generate user specific graphs based on activity locations (trackintel locations).
        This function creates a networkx graph per user based on the locations of
        the user as nodes and a set of (weighted) edges defined in adjacency dict.
        Parameters
        ----------
        locations : GeoDataFrame
            Trackintel dataframe of type locations
        adjacency_dict : dictionary or list of dictionaries
             A dictionary with adjacendy matrices of type: {user_id:
             scipy.sparse.coo_matrix}.
        edgenames : List
            List of names (stings) given to edges in a multigraph
        Returns
        -------
        G_dict : dictionary
            A dictionary of type: {user_id: networkx graph}.
        """
        # Todo: Enable multigraph input. E.g. adjacency_dict[user_id] = [edges1,
        #  edges2]
        # Todo: Should we do a check if locations is really a dataframe of trackintel
        #  type?

        locations = locations.copy()
        G_dict = {}
        # we want the location id
        locations.index.name = "location_id"
        locations.reset_index(inplace=True)
        locations = locations.set_index("user_id", drop=False)
        locations.index.name = "user_id_ix"
        locations.sort_index(inplace=True)

        if "extent" not in locations.columns:
            locations["extent"] = pd.NA

        G = tigraphs.initialize_multigraph(self.user_id, locations, self.node_feature_names)
        G.graph["edge_keys"] = []

        # todo: put edge creation in extra function
        A_list = self.adjacency_dict["A"]
        location_id_order_list = self.adjacency_dict["location_id_order"]
        edge_name_list = self.adjacency_dict["edge_name"]

        for ix in range(len(A_list)):
            A = A_list[ix]
            location_id_order = location_id_order_list[ix]
            edge_name = edge_name_list[ix]
            # todo: assert location_id_order
            G_temp = nx.from_scipy_sparse_matrix(A, create_using=nx.MultiDiGraph())
            edge_list = nx.to_edgelist(G_temp)

            # target structure for edge list:
            # [(0, 0, 'transition_counts', {'weight': 1.0, 'edge_name': 'transition_counts'}),
            # (0, 1, 'transition_counts', {'weight': 7.0, 'edge_name': 'transition_counts'}
            edge_list = [(x[0], x[1], edge_name, {**x[2], **{"edge_name": edge_name}}) for x in edge_list]

            G.add_edges_from(edge_list, weight="weight")
            G.graph["edge_keys"].append(edge_name)

        return G

    def plot(
        self,
        filename,
        layout="spring",
        edge_attributes=None,
        filter_node_importance=None,
        filter_extent=True,
        filter_dist=100,
        dist_spring_layout=10,
        draw_edge_label=False,
        draw_edge_label_type="transition_counts",
    ):
        """

        Parameters
        ----------
        image_folder
        layout [spring, coordinate]
        edge_attributes
        filter_node_importance
        filter_extent
        filter_dist
        dist_spring_layout

        Returns
        -------

        """
        folder_name = ntpath.dirname(filename)
        Path(folder_name).mkdir(parents=True, exist_ok=True)

        if filter_node_importance is not None:
            important_nodes = self.get_k_importance_nodes(filter_node_importance)
        else:
            important_nodes = self.G.nodes()
        # filter graph extent:
        if filter_extent:
            center_node_id = int(self.get_k_importance_nodes(1))
            c_geom = self.G.nodes[center_node_id]["center"]
            filtered_nodes = [n for n in self.G.nodes if h_dist(self.G.nodes[n]["center"], c_geom) < filter_dist * 1000]
            important_nodes = np.intersect1d(filtered_nodes, important_nodes)

        G = self.G.subgraph(important_nodes)

        # edge color management
        if edge_attributes is not None:
            for edge_attribute in edge_attributes:
                weights = [G[u][v][edge_attribute]["weight"] + 1 for u, v in G.edges()]
        else:
            # list(self.G[u][v])[0] is the edge_attribute key (e.g., 'transition_counts' of the edge
            weights = [G[u][v][list(G[u][v])[0]]["weight"] + 1 for u, v in G.edges()]

        norm_width = np.log(weights) * 2

        deg = nx.degree(G)
        node_sizes = [10 * deg[iata] for iata in G.nodes]

        if layout == "coordinate":
            # draw geographic representation
            ax, smap = draw_smopy_basemap(G)
            nx.draw_networkx(
                G,
                ax=ax,
                font_size=20,
                width=1,
                linewidths=norm_width,
                with_labels=False,
                node_size=node_sizes,
                pos=nx_coordinate_layout_smopy(G, smap),
                connectionstyle="arc3, rad = 0.1",
            )

        elif layout == "spring":

            # draw spring layout
            plt.figure()
            pos = nx.spring_layout(G, k=dist_spring_layout / np.sqrt(len(G)))
            nx.draw(
                G,
                pos=pos,
                width=norm_width / 2,
                node_size=node_sizes,
                connectionstyle="arc3, rad = 0.2",
            )

        if draw_edge_label:
            edges_new = {}
            edges = nx.get_edge_attributes(G, "weight")
            # edges have to be recoded for drawing. Multigraph edges have the format: (n1, n2, edge_type): weight but
            # the drawing function only accepts (n1 n2): weight as input

            for (u, v, enum), weight in edges.items():
                if enum == draw_edge_label_type:
                    edges_new[(u, v)] = str(int(weight))
            GG = nx.Graph()
            GG.add_edges_from(edges_new)
            nx.draw_networkx_edge_labels(GG, pos, edge_labels=edges_new, label_pos=0.2)

        plt.savefig(filename)
        plt.close()

    def get_adjecency_matrix_by_type(self, edge_type):
        assert edge_type in self.adjacency_dict["edge_name"], (
            f"Only {self.adjacency_dict['edge_name']} are available " f"but you provided {edge_type}"
        )
        edge_type_ix = self.adjacency_dict["edge_name"].index(edge_type)
        return self.adjacency_dict["A"][edge_type_ix]

    def get_adjecency_matrix(self):
        return nx.linalg.graphmatrix.adjacency_matrix(self.G).tocoo()


def _create_adjacency_matrix_from_transition_counts(counts):
    """
    Transform transition counts into a adjacency matrix per user.
    The input provides transition counts between locations of a user. These
    counts are transformed into a weighted adjacency matrix.
    Parameters
    ----------
    counts : DataFrame
        pandas DataFrame that has at least the columns ['user_id',
        'location_id', 'location_id_end', 'counts']. Counts represents the
        number of transitions between two locations.
    Returns
    -------
    adjacency_dict : dictionary
            A dictionary of adjacency matrices of type scipy.sparse.coo_matrix
    """

    row_ix = counts["location_id"].values.astype("int")
    col_ix = counts["location_id_end"].values.astype("int")
    values = counts["counts"].values.astype("float")

    if len(values) == 0:
        A = coo_matrix((0, 0))
        location_id_order = np.asarray([])

    else:
        # ix transformation to go from 0 to n
        org_ix = np.unique(np.concatenate((row_ix, col_ix)))
        new_ix = np.arange(0, len(org_ix))
        ix_tranformation = dict(zip(org_ix, new_ix))
        ix_backtranformation = dict(zip(new_ix, org_ix))

        row_ix = [ix_tranformation[row_ix_this] for row_ix_this in row_ix]
        row_ix = np.asarray(row_ix)
        col_ix = [ix_tranformation[col_ix_this] for col_ix_this in col_ix]
        col_ix = np.asarray(col_ix)

        # set shape and create sparse matrix
        max_ix = np.max([np.max(row_ix), np.max(col_ix)]) + 1
        shape = (max_ix, max_ix)

        A = coo_matrix((values, (row_ix, col_ix)), shape=shape)
        location_id_order = org_ix

    return A, location_id_order, "transition_counts"
