from future_trackintel.tigraphs import _create_adjacency_matrix_from_counts
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import pytest
import datetime
from shapely.geometry import Point
import geopandas as gpd
from future_trackintel.activity_graph import activity_graph

@pytest.fixture
def example_staypoints():
    """Staypoints to load into the database."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")
    one_hour = datetime.timedelta(hours=1)

    list_dict = [
        {"user_id": 0, "started_at": t1, "finished_at": t2, "geometry": p1, "location_id": 0},
        {"user_id": 0, "started_at": t2, "finished_at": t3, "geometry": p2, "location_id": 0},
        {"user_id": 0, "started_at": t3, "finished_at": t3 + one_hour, "geometry": p3, "location_id": 1},
    ]
    stps = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    stps.index.name = "id"
    assert stps.as_staypoints
    return stps

@pytest.fixture
def example_locations():
    """Locations to load into the database."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    list_dict = [
        {"user_id": 0, "center": p1},
        {"user_id": 0, "center": p2},
        {"user_id": 0, "center": p3},
    ]
    locs = gpd.GeoDataFrame(data=list_dict, geometry="center", crs="EPSG:4326")
    locs.index.name = "id"
    assert locs.as_locations
    return locs

def test_val_activity_graph_stp(example_staypoints, example_locations):
    spts = example_staypoints
    locs = example_locations

    ag = activity_graph(spts, locs)

    spts['user_id'].iloc[0] = 1
    with pytest.raises(AssertionError):
        ag = activity_graph(spts, locs)


def test_val_activity_graph_locs(example_staypoints, example_locations):
    spts = example_staypoints
    locs = example_locations

    ag = activity_graph(spts, locs)

    locs['user_id'].iloc[0] = 1
    with pytest.raises(AssertionError):
        ag = activity_graph(spts, locs)

def test_val_activity_graph_different_users(example_staypoints, example_locations):
    spts = example_staypoints
    locs = example_locations

    ag = activity_graph(spts, locs)

    locs['user_id'] = 1
    with pytest.raises(AssertionError):
        ag = activity_graph(spts, locs)

def test_adjencency_matrix():
    "check if correct adjacency matrix is created from counts input"
    counts = pd.DataFrame(np.array([['a', 1, 2, 5],
                                 ['a', 1, 7, 2],
                                 ['a', 2, 1, 4],
                                 ['a', 2, 3, 1],
                                 ['a', 3, 1, 1],
                                 ['a', 7, 1, 2]]),
    columns=['user_id', 'location_id', 'location_id_end', 'counts'], )
    counts.loc[:, ['location_id', 'location_id_end', 'counts']] = counts.loc[:, ['location_id',
                                                                                           'location_id_end',
                                                                                 'counts']].astype(int)

    user_list = ['a']
    adjacency_dict = _create_adjacency_matrix_from_counts(counts, user_list)

    A = np.asarray(adjacency_dict['a']['A'][0].todense())
    df = pd.DataFrame(data=A, index=adjacency_dict['a']['location_id_order'][0], columns=adjacency_dict['a'][
        'location_id_order'][0])


    A_manuel = np.asarray([[0.00000, 5.00000, 0.00000, 2.00000],
    [4.00000, 0.00000, 1.00000, 0.00000],
    [1.00000, 0.00000, 0.00000, 0.00000],
    [2.00000, 0.00000, 0.00000, 0.00000]])
    df_manuel = pd.DataFrame(data=A_manuel, index=[1,2,3,7], columns=[1,2,3,7])

    assert_frame_equal(df, df_manuel)

