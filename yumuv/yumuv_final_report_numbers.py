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

CRS_WGS84 = "epsg:4326"


def get_engine(study, return_con=False):
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



study = 'yumuv_graph_rep'
engine = get_engine(study)
con = get_engine(study, return_con=True)

AG_dict = read_graphs_from_postgresql(graph_table_name="before_after", psycopg_con=con, graph_schema_name=study,
                                      file_name="before", decompress=True)

user_info = pd.read_sql("select * from yumuv_graph_rep.user_info", con=engine)
user_info = user_info.loc[:, ['user_id', 'study_id']]
user_info_tg = user_info[user_info['study_id'] == 22]
users_tg = user_info_tg['user_id'].to_list()

user_info_cg = user_info[user_info['study_id'] == 23]
users_cg = user_info_cg['user_id'].to_list()


users_in_graph = list(AG_dict.keys())

users_tg_in_graph = list(set(users_tg) & set(users_in_graph))
users_cg_in_graph = list(set(users_cg) & set(users_in_graph))
