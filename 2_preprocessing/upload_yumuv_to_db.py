"""Script to upload the processed yumuv data to the database. Script was necessary because raw yumuv data
 had to be processed on the hardware of the data owner. """

from sqlalchemy import create_engine
import os
import pickle
import json
from general_utils import write_graphs_to_postgresql, read_graphs_from_postgresql
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
# GRAPH_FOLDER, _ = ntpath.split(GRAPH_OUTPUT)
if not os.path.exists(GRAPH_OUTPUT):
    os.mkdir(GRAPH_OUTPUT)

pkl_name = open(os.path.join(GRAPH_OUTPUT, "counts_full.pkl"), "rb")
AG_dict = pickle.load(pkl_name)

con = get_engine(study, return_con=True)

write_graphs_to_postgresql(
    graph_data=AG_dict,
    graph_table_name="full_graph",
    graph_schema_name=study,
    psycopg_con=con,
    file_name="graph_data",
)

AG_dict2 = read_graphs_from_postgresql(graph_table_name="full_graph",
    graph_schema_name=study,
    psycopg_con=con,
    file_name="graph_data")