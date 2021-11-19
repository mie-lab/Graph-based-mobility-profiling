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
from utils import write_graphs_to_postgresql, read_graphs_from_postgresql, get_engine, \
    get_staypoints, get_locations, get_triplegs, get_trips
import copy
from trackintel.analysis.tracking_quality import _split_overlaps
from collections import defaultdict
import pytz
import psycopg2
from tqdm import tqdm
import matplotlib.pyplot as plt

CRS_WGS84 = "epsg:4326"

# globals
# study name is used as schema name in database
# studies = ["gc2", "gc1", "geolife"]
# studies = ['geolife']
#studies = ["yumuv_graph_rep"]
studies = ['tist_top10'] #, 'tist_toph10', 'tist_top100', 'tist_toph100', 'tist_top500', 'tist_toph500',
#                      'tist_top1000', 'tist_toph1000']
# limit = "where user_id > 1670"
limit = ""
single_user = False

# nb_users =

if __name__ == "__main__":

    for study in studies:
        print("start {}".format(study))

        # define output for graphs
        GRAPH_OUTPUT = os.path.join("../2_preprocessing", "data_out", "graph_data", study)
        # GRAPH_FOLDER, _ = ntpath.split(GRAPH_OUTPUT)
        if not os.path.exists(GRAPH_OUTPUT):
            os.mkdir(GRAPH_OUTPUT)

        engine = get_engine(study)

        # download data
        print("\t download staypoints")
        sp = get_staypoints(study=study, engine=engine)

        print("\t download locations")
        locs = get_locations(study=study, engine=engine)

        sp["finished_at"] = sp.groupby("user_id")["started_at"].shift(-1)
        sp['duration'] = sp['finished_at'] - sp['started_at']

        user_data = pd.read_sql("select * from tist.user_data", con=engine)

    print(sp['purpose'].unique())
    exclude_purpose = ['Light Rail', 'Subway', 'Platform', 'Trail', 'Road']
    a = sp.groupby('purpose').size().sort_values()


    a = user_data[['totalcount', 'homecount', 'duration', 'nb_locs']].copy()
    a = a[a['totalcount'] > 1]
    a.loc[a.index, 'duration'] = a['duration'].dt.total_seconds()
    a.describe()
    pd.plotting.scatter_matrix(a)

    a = np.log10(a+1)
    pd.plotting.scatter_matrix(a)
    plt.show()
    a.describe()
