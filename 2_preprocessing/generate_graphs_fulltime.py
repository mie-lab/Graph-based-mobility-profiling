import copy
import json
import ntpath
import os
import pickle

import trackintel as ti
from sqlalchemy import create_engine

from future_trackintel.activity_graph import activity_graph

CRS_WGS84 = "epsg:4326"
studies = ["gc2", "gc1"]  # , 'geolife',]# 'tist_u1000', 'tist_b100', 'tist_b200', 'tist_u10000']

for study in studies:
    print("start {}".format(study))
    # define output for graphs
    GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)
    GRAPH_FOLDER, _ = ntpath.split(GRAPH_OUTPUT)
    if not os.path.exists(GRAPH_FOLDER):
        os.mkdir(GRAPH_FOLDER)

    # build database login string from file
    DBLOGIN_FILE = os.path.join("./dblogin.json")
    with open(DBLOGIN_FILE) as json_file:
        LOGIN_DATA = json.load(json_file)

    conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA)

    engine = create_engine(conn_string)
    conn = engine.connect()

    print("\t download staypoints")
    sp = ti.io.read_staypoints_postgis(conn_string, table_name="{}.staypoints".format(study), geom_col="geom")

    print("\t download locations")
    locs = ti.io.read_locations_postgis(conn_string, table_name="{}.locations".format(study), geom_col="center")

    AG_dict = {}

    for user_id_this in locs["user_id"].unique():
        sp_user = sp[sp["user_id"] == user_id_this]
        locs_user = locs[locs["user_id"] == user_id_this]
        AG = activity_graph(sp_user, locs_user)
        AG.plot(
            os.path.join(".", "graph_images", "new", study, "spring", str(user_id_this)),
            filter_node_importance=25,
            draw_edge_label=False,
        )
        AG.plot(
            os.path.join(".", "graph_images", "new", study, "coords", str(user_id_this)),
            filter_node_importance=25,
            draw_edge_label=False,
            layout="coordinate",
        )
        AG_dict[user_id_this] = copy.deepcopy(AG)

    # create graphs of the full period
    print("\t create full graph with counts")
    pickle.dump(AG_dict, open(GRAPH_OUTPUT + "_counts_full.pkl", "wb"))
