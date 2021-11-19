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

def sample_tist_users(nb_users, engine):
    """
    Sample nb_users from tist.
    Where statement:
    homecount: 75 percentile
    totalcount: 25 percentile
    nb_locs: 25 percentile

    returns list with user_ids
    """
    query = """select user_id from tist.user_data where
                homecount > 24 and totalcount > 81 and nb_locs > 40 
                order by random() limit {}""".format(nb_users)

    return list(pd.read_sql(query, con=engine))


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


def get_locations(study, engine, limit=""):
    locs = ti.io.read_locations_postgis(
        sql="select * from {}.locations {}".format(study, limit),
        con=engine,
        center="center",
        index_col="id"
    )
    return locs


def get_triplegs(study, engine, limit=""):
    tpls = pd.read_sql(
        sql="select id, user_id, started_at, finished_at from {}.triplegs {}".format(study, limit),
        con=engine,
        index_col="id",
    )
    tpls["started_at"] = pd.to_datetime(tpls["started_at"], utc=True)
    tpls["finished_at"] = pd.to_datetime(tpls["finished_at"], utc=True)
    return tpls


def get_trips(study, engine, limit=""):
    trips = pd.read_sql(sql="select * from {}.trips {}".format(study, limit), con=engine, index_col="id")
    trips["started_at"] = pd.to_datetime(trips["started_at"], utc=True)
    trips["finished_at"] = pd.to_datetime(trips["finished_at"], utc=True)

    return trips




# globals
# study name is used as schema name in database
# studies = ["gc2", "gc1", "geolife"]
# studies = ['geolife']
#studies = ["yumuv_graph_rep"]
study = 'gc1'
limit = ""
single_user = False

if __name__ == "__main__":


    engine = get_engine(study)
    con = get_engine(study, return_con=True)


    AG_dict = read_graphs_from_postgresql(
        graph_table_name="full_graph", psycopg_con=con, graph_schema_name=study, file_name="graph_data", decompress=True)


    def calc_entropy(Px):
        return - np.sum(Px * np.log2(Px))

    entropy_dict = {}
    for user, AG in AG_dict.items():
        print(user)
        filter_node_importance=50
        important_nodes = AG.get_k_importance_nodes(filter_node_importance)
        G = AG.G
        # G = G.subgraph(important_nodes)
        # G.plot(r"C:\Users\henry\OneDrive\Programming\21_mobility-graph-representation\graph_images\gc1\spring" + "\\{"
        #                                                                                                          "}".format(str(user)),
        #     # filter_node_importance=50
        #        )
        A = nx.linalg.graphmatrix.adjacency_matrix(G)
        # A = G.get_adjecency_matrix()
        col_sum = A.sum(axis=1)
        home_node_ix = np.argmax(col_sum)

        from sklearn.preprocessing import normalize

        A = normalize(A, norm='l1', axis=1)
        A_ = sparse.COO(A)

        # p0 for home start
        p0 = np.zeros((A.shape[1]))
        p0[home_node_ix] = 1

        #p0 for all starts
        # p0 = np.asarray(col_sum/np.sum(col_sum))

        # A1_ = A.multiply(p0[..., None])
        A1 = A_ * p0[..., None]
        A2 = A_ * (A1[..., None])
        A3 = A_ * (A2[..., None])
        A4 = A_ * (A3[..., None])
        A5 = A_ * (A4[..., None])
        #
        # x = A4.coords
        # y = A4.data

        #
        entropy_dict[user] = list(map(calc_entropy, [A1.data,
                                                     A2.data,
                                                     A3.data,
                                                A4.data
                                                     ,A5.data
                                                     ]))


    df = pd.DataFrame(entropy_dict).transpose()
    df.columns = ['1', '2', '3', '4', '5']
    df['0'] = 0
    df = df[['0', '1', '2', '3', '4', '5']]

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.displot(data=df, kind="kde")
    plt.xlim(0, 18)
    plt.show()


    #
    # # test
    # b = np.asarray([1, 0, 0])
    # B = np.asarray([[0, 0.66, 0.33], [0, 0, 1], [1, 0, 0]])
    #
    # B1 = B * b[..., None]
    # B1_ = B1[..., None]
    # B2 = B * B1_
    #
    # b1 = np.matmul(b[..., None].transpose(), B).transpose()
    # b2 = np.matmul(b1.transpose(), B).transpose()
    # import matplotlib.pyplot as plt
    #
    #
    #
    #
    #




