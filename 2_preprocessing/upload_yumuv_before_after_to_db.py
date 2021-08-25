from sqlalchemy import create_engine
import os
import pickle
import json
from future_trackintel.utils import write_graphs_to_postgresql, read_graphs_from_postgresql
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

GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)

pkl_name_before = open(os.path.join(GRAPH_OUTPUT, "counts_full_before.pkl"), "rb")
pkl_name_after = open(os.path.join(GRAPH_OUTPUT, "counts_full_after.pkl"), "rb")

AG_dict_before = pickle.load(pkl_name_before)
AG_dict_after = pickle.load(pkl_name_after)

con = get_engine(study, return_con=True)

write_graphs_to_postgresql(
    graph_data=AG_dict_before,
    graph_table_name="before_after",
    graph_schema_name=study,
    psycopg_con=con,
    file_name="before",
    drop_and_create=True)

write_graphs_to_postgresql(
    graph_data=AG_dict_after,
    graph_table_name="before_after",
    graph_schema_name=study,
    psycopg_con=con,
    file_name="after",
    drop_and_create=False
)