"""Script to upload the processed yumuv data to the database. Script was necessary because raw yumuv data
 had to be processed on the hardware of the data owner. """

import json
import os
import pickle

import psycopg2
from sqlalchemy import create_engine

from config import config

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


study = "yumuv_graph_rep"
pkl_name = open(config["yumuv_user_info"], "rb")
user_info = pickle.load(pkl_name)

engine = get_engine(study, return_con=False)

user_info.to_sql(con=engine, schema="yumuv_graph_rep", name="user_info", index=False, if_exists="replace")
