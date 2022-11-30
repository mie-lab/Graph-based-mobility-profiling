"""Script to upload the processed yumuv data to the database. Script was necessary because raw yumuv data
 had to be processed on the hardware of the data owner. """

from sqlalchemy import create_engine
import os
import json
import psycopg2
import config

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

data_folder = config['yumuv_import_data_folder']

con = get_engine(study, return_con=False)

import geopandas as gpd
import pandas as pd

sp = gpd.read_file(os.path.join(data_folder, "staypoint.GeoJSON"))
tl = gpd.read_file(os.path.join(data_folder, "tripleg.GeoJSON"))

sp["started_at"] = pd.to_datetime(sp["started_at"], utc=True)
sp["finished_at"] = pd.to_datetime(sp["finished_at"], utc=True)
sp.to_postgis(name="staypoints", con=con, schema=study, if_exists="replace")


tl["started_at"] = pd.to_datetime(tl["started_at"], utc=True)
tl["finished_at"] = pd.to_datetime(tl["finished_at"], utc=True)
tl.to_postgis(name="triplegs", con=con, schema=study, if_exists="replace")

locs = gpd.read_file(os.path.join(data_folder, "locations.GeoJSON"))
trips = pd.read_json(os.path.join(data_folder, "trips.json"))

trips["started_at"] = pd.to_datetime(trips["started_at"], utc=True)
trips["finished_at"] = pd.to_datetime(trips["finished_at"], utc=True)
trips.to_sql(name="trips", con=con, schema=study, if_exists="replace")

locs.to_postgis(name="locations", con=con, schema=study, if_exists="replace")
