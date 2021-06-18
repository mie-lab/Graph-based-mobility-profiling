import pandas as pd
import geopandas as gpd
import pytest
import trackintel as ti
from sqlalchemy import create_engine
from future_trackintel import tigraphs
from future_trackintel.activity_graph import activity_graph
import numpy as np
import os
import pickle
import ntpath
import json
import datetime
from shapely.geometry import Point

CRS_WGS84 = 'epsg:4326'
#
studies = ['geolife']  # ,'gc1', 'geolife',]# 'tist_u1000', 'tist_b100', 'tist_b200', 'tist_u10000']
n = 'fconn'  # number of neighbors for neighbor weights

@pytest.fixture
def example_staypoints():
    """Staypoints to load into the database."""
    l1 = Point(8.5067847, 47.4)
    l2 = Point(8.5067847, 47.6)
    l4 = Point(8.5067847, 47.8)
    l6 = Point(8.5067847, 47.0)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 01:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-01 02:00:00", tz="utc")
    t4 = pd.Timestamp("1971-01-01 03:00:00", tz="utc")
    t5 = pd.Timestamp("1971-01-01 04:00:00", tz="utc")
    t6 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t7 = pd.Timestamp("1971-01-01 06:00:00", tz="utc")
    t8 = pd.Timestamp("1971-01-01 07:00:00", tz="utc")

    s1 = "Home"
    s2 = "work"
    s4 = "sport"
    s5 = "park"
    s6 = "friend"

    c1 = "a"
    c2 = "b"
    c3 = "c"
    c4 = "d"
    c5 = "e"
    c6 = "f"
    c7 = "g"


    one_hour = datetime.timedelta(hours=1)

    list_dict = [
        {"user_id": 0, "started_at": t1, "finished_at": t2, "geometry": l1, "label": s1, "context": c1,
         "location_id": 1},
        {"user_id": 0, "started_at": t2, "finished_at": t3, "geometry": l2, "label": s2, "context": c2,
         "location_id": 2},
        {"user_id": 0, "started_at": t3, "finished_at": t4, "geometry": l2, "label": s2, "context": c3,
         "location_id": 2},
        {"user_id": 0, "started_at": t4, "finished_at": t5, "geometry": l4, "label": s4, "context": c4,
         "location_id": 4},
        {"user_id": 0, "started_at": t5, "finished_at": t6, "geometry": l2, "label": s5, "context": c5,
         "location_id": 2},
        {"user_id": 0, "started_at": t6, "finished_at": t7, "geometry": l6, "label": s6, "context": c6,
         "location_id": 6},
        {"user_id": 0, "started_at": t7, "finished_at": t2, "geometry": l1, "label": s1, "context": c7,
         "location_id": 1},
    ]
    stps = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    stps.index.name = "id"
    assert stps.as_staypoints
    return stps

@pytest.fixture
def example_locations():
    """Locations to load into the database."""
    l1 = Point(8.5067847, 47.4)
    l2 = Point(8.5067847, 47.6)
    l4 = Point(8.5067847, 47.8)
    l6 = Point(8.5067847, 47.0)

    list_dict = [
        {"id": 1, "user_id": 0, "center": l1},
        {"id": 2, "user_id": 0, "center": l2},
        {"id": 4, "user_id": 0, "center": l4},
        {"id": 6, "user_id": 0, "center": l6}
    ]
    locs = gpd.GeoDataFrame(data=list_dict, geometry="center", crs="EPSG:4326")
    locs.set_index("id", inplace=True)
    assert locs.as_locations
    return locs


@pytest.fixture
def db_engine():
    # build database login string from file
    DBLOGIN_FILE = os.path.join("dblogin.json")
    with open(DBLOGIN_FILE) as json_file:
        LOGIN_DATA = json.load(json_file)

    conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}" \
        .format(**LOGIN_DATA)

    return create_engine(conn_string)

@pytest.fixture
def geolife_user_1(db_engine):
    engine = db_engine

    sp = gpd.GeoDataFrame.from_postgis("SELECT * FROM geolife.staypoints where user_id = 1", engine,
                                       geom_col="geom", index_col="id")
    for col in ['started_at', 'finished_at']:
        sp[col] = pd.to_datetime(sp[col], utc=True)
    locs = gpd.GeoDataFrame.from_postgis("SELECT * FROM geolife.locations where user_id = 1", engine,
                                       geom_col="center", index_col="id")
    return sp, locs

@pytest.fixture
def gc1_user(db_engine):
    engine = db_engine

    sp = gpd.GeoDataFrame.from_postgis("SELECT * FROM gc1.staypoints where user_id = 1595", engine,
                                       geom_col="geom", index_col="id")
    for col in ['started_at', 'finished_at']:
        sp[col] = pd.to_datetime(sp[col], utc=True)
    locs = gpd.GeoDataFrame.from_postgis("SELECT * FROM gc1.locations where user_id = 1595", engine,
                                       geom_col="center", index_col="id")
    return sp, locs

class TestValidate_user:
    def test1(self):
        pass


# activity graph
class TestActivtyGraph:
    def test_create_activty_graph_example(self, example_staypoints, example_locations):
        """Test if adjecency matrix gets corretly reproduced"""
        A_true = np.asarray([[0, 1, 0, 0],
                              [0, 1, 1, 1],
                              [0, 1, 0, 0],
                              [1, 0, 0, 0]])

        spts = example_staypoints
        locs = example_locations
        AG = activity_graph(spts, locs)
        # AG.plot(os.path.join(".", "tests"))
        A = np.asarray(AG.get_adjecency_matrix().todense())
        assert np.allclose(A, A_true)

    def test_create_activity_graph_1_geolife_user(self, geolife_user_1):
        """test if activity graph runs with geolife data"""
        sp, locs = geolife_user_1
        AG = activity_graph(sp, locs)

# plot
class TestPlot:
    def test_plot_graph_1_geolife_user(self, geolife_user_1):
        """create a plot of a geolife user"""
        sp, locs = geolife_user_1
        AG = activity_graph(sp, locs)
        AG.plot(filename=os.path.join(".", "tests", "geolife_spring"), layout="spring")
        AG.plot(filename=os.path.join(".", "tests", "geolife_coordinate"), layout="coordinate")


    def test_plot_graph_1_gc1_user(self, gc1_user):
        """create a plot of a sbb gc user"""
        sp, locs = gc1_user
        AG = activity_graph(sp, locs)
        AG.plot(filename=os.path.join(".", "tests", "gc1_spring"), layout="spring")
        AG.plot(filename=os.path.join(".", "tests", "gc1_coordinate"), layout="coordinate")

    def test_plot_example(self, example_staypoints, example_locations):
        """create a plot of a sbb gc user"""
        sp = example_staypoints
        locs = example_locations
        AG = activity_graph(sp, locs)
        AG.plot(filename=os.path.join(".", "tests", "example_spring"), layout="spring")
        AG.plot(filename=os.path.join(".", "tests", "example_coordinate"), layout="coordinate")

    def test_plot_filter(self, gc1_user):
        """test calling the filter_node_importance argument"""
        sp, locs = gc1_user
        AG = activity_graph(sp, locs)
        AG.plot(filename=os.path.join(".", "tests", "gc1_spring25"), layout="spring", filter_node_importance=25)
        AG.plot(filename=os.path.join(".", "tests", "gc1_coordinate25"), layout="coordinate",
                filter_node_importance=25)


def test_show_A_graph_1_gc1_user(gc1_user):
    sp, locs = gc1_user
    AG = activity_graph(sp, locs)
    edge_type = AG.edge_types[0]
    A = AG.get_adjecency_matrix_by_type(edge_type)
    A2 = AG.get_adjecency_matrix()
    pass

def test_show_A_graph_1_gc1_user2(gc1_user):
    sp, locs = gc1_user
    AG = activity_graph(sp, locs)
    A = AG.get_adjecency_matrix()
    pass
# edge_types

def test_edge_attr_graph_1_geolife_user(geolife_user_1):
    sp, locs = geolife_user_1
    AG = activity_graph(sp, locs)
    print(AG.edge_types)

# get_adjecency_matrix_by_type

def test_get_adjacency_matrix_wrong_edge_name(gc1_user):
    sp, locs = gc1_user
    AG = activity_graph(sp, locs)
    edge_type = AG.edge_types[0]
    with pytest.raises(AssertionError):
        A = AG.get_adjecency_matrix_by_type("test")

# get_k_importance_nodes

def test_get_k_importance_nodes_1(gc1_user):
    sp, locs = gc1_user
    AG = activity_graph(sp, locs)
    assert len(AG.get_k_importance_nodes(1)) == 1
