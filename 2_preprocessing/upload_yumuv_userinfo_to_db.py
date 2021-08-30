from sqlalchemy import create_engine
import os
import pickle
import json
from future_trackintel.utils import write_graphs_to_postgresql
import psycopg2

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

pkl_name = open(os.path.join("D:/", "temp", "yumuv_userinfo.pkl"), "rb")
user_info = pickle.load(pkl_name)

engine = get_engine(study, return_con=False)

user_info.to_sql(con=engine, schema="yumuv_graph_rep", name="user_info", index=False, if_exists="replace")
con = engine.connect()
con.execute(
    """GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA yumuv_graph_rep
TO wnina;"""
)
