import geopandas as gpd
import pandas as pd
import trackintel as ti
from sqlalchemy import create_engine
import numpy as np
import os
import pickle
import ntpath
import json
import datetime
import numpy as np
import sys
from future_trackintel.activity_graph import activity_graph
from future_trackintel.utils import write_graphs_to_postgresql, read_graphs_from_postgresql
import copy
from trackintel.analysis.tracking_quality import _split_overlaps
from collections import defaultdict
import pytz
import psycopg2
from tqdm import tqdm
import sparse
import networkx as nx

CRS_WGS84 = "epsg:4326"


def get_engine(study, return_con=False):

    if study == "yumuv_graph_rep":
        sys.path.append(r"C:\Users\e527371\OneDrive\Programming\yumuv")
        from db_login import DSN  # database login information

        engine = create_engine("postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}".format(**DSN))
        if return_con:
            con = psycopg2.connect(
                dbname=DSN["db_database"],
                user=DSN["db_user"],
                password=DSN["db_password"],
                host=DSN["db_host"],
                port=DSN["db_port"],
            )
    else:

        # build database login string from file
        DBLOGIN_FILE = os.path.join("./dblogin.json")
        with open(DBLOGIN_FILE) as json_file:
            LOGIN_DATA = json.load(json_file)

        conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA)
        engine = create_engine(conn_string)
        if return_con:
            con = psycopg2.connect(
                dbname=LOGIN_DATA["database"],
                user=LOGIN_DATA["user"],
                password=LOGIN_DATA["password"],
                host=LOGIN_DATA["host"],
                port=LOGIN_DATA["port"],
            )

    if return_con:
        return con
    else:
        return engine



def get_trips(study, engine, limit=""):
    trips = pd.read_sql(sql="select * from {}.trips {}".format(study, limit), con=engine, index_col="id")
    trips["started_at"] = pd.to_datetime(trips["started_at"], utc=True)
    trips["finished_at"] = pd.to_datetime(trips["finished_at"], utc=True)

    return trips


def get_locations(study, engine, limit=""):
    locs = ti.io.read_locations_postgis(
        sql="select * from {}.locations {}".format(study, limit),
        con=engine,
        center="center",
        index_col="id"
    )
    return locs

def get_staypoints(study, engine, limit=""):
    sp = gpd.read_postgis(
        sql="select * from {}.staypoints {}".format(study, limit),
        con=engine,
        geom_col="geom",
        index_col="id",
    )
    sp["started_at"] = pd.to_datetime(sp["started_at"], utc=True)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"], utc=True)

    return sp

# globals
# study name is used as schema name in database
# studies = ["gc2", "gc1", "geolife"]
# studies = ['geolife']
#studies = ["yumuv_graph_rep"]
study = 'gc1'
limit = ""
single_user = False


def calc_entropy(Px):
    return - np.sum(Px * np.log2(Px))


def get_k_sequence_distribution(data, k_max, start_node=None):
    k_list = []
    for k in range(k_max+1):
        if k == 0:
            data['k0'] = data['location_id']
        else:
            data['k{}'.format(str(k))] = data['k{}'.format(str(k-1))].shift(-1)
        k_list.append('k{}'.format(str(k)))

    if start_node is not None:
        start_node_ix = data['location_id'] == start_node
        data = data[start_node_ix]

    return data.groupby(k_list).size()


if __name__ == "__main__":


    engine = get_engine(study)
    con = get_engine(study, return_con=True)
    trips =get_trips(study, engine=engine)
    # locs = get_locations(study, engine)
    sp = get_staypoints(study, engine)
    trips = trips.join(sp['location_id'], on="destination_staypoint_id")

    entropy_dict = {}
    for user in trips.user_id.unique():
        trips_user = trips[trips['user_id'] == user].copy()

        start_node = trips_user.groupby('location_id').size().sort_values(ascending=False)
        start_node = start_node.index[0]
        # start_node = None

        # trips['k0'] = trips['location_id']
        # trips['k1'] = trips['k0'].shift(-1)
        # trips['k2'] = trips['k1'].shift(-1)
        #
        # b = trips.groupby(['k0', 'k1', 'k2']).size()
        k0 = get_k_sequence_distribution(trips_user, 0, start_node=start_node)
        k1 = get_k_sequence_distribution(trips_user, 1, start_node=start_node)
        k2 = get_k_sequence_distribution(trips_user, 2, start_node=start_node)
        k3 = get_k_sequence_distribution(trips_user, 3, start_node=start_node)
        k4 = get_k_sequence_distribution(trips_user, 4, start_node=start_node)
        k5 = get_k_sequence_distribution(trips_user, 5, start_node=start_node)

        e0 = calc_entropy(k0/np.sum(k0))
        e1 = calc_entropy(k1/np.sum(k1))
        e2 = calc_entropy(k2/np.sum(k2))
        e3 = calc_entropy(k3/np.sum(k3))
        e4 = calc_entropy(k4 / np.sum(k4))
        e5 = calc_entropy(k5 / np.sum(k5))

        entropy_dict[user] = [e0, e1, e2, e3, e4, e5]

    df = pd.DataFrame(entropy_dict).transpose()

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.displot(data=df, kind="kde")
    plt.xlim(0, 18)
    plt.show()

        # print(calc_entropy(b/np.sum(b)))




