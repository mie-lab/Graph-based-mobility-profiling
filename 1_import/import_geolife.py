# -*- coding: utf-8 -*-
"""
This script reads in the geolife data (as it can be downloaded from
https://www.microsoft.com/en-us/download/details.aspx?id=52367) and loads it
in a postgis database
"""

import os
import time
import json
import ntpath
import glob
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
import trackintel as ti

# connect to postgis database
config_file = os.path.join("..", "dblogin.json")
with open(config_file) as json_file:
    login_data = json.load(json_file)

conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**login_data)
engine = create_engine(conn_string)
conn = engine.connect()

geolife_path = r"E:\Geolife Trajectories 1.3\Data"
# geolife_path = r"D:\temp\geolife65"
# geolife_path = r"D:\temp\geolife_13_users"
# geolife_path = r"C:\Users\henry\OneDrive\Programming\21_mobility-graph-representation\data_in\data_geolife"
schema_name = "geolife"

pfs, mode_labels = ti.io.dataset_reader.read_geolife(geolife_path, print_progress=True)

print("extract staypoints")
pfs, spts = pfs.as_positionfixes.generate_staypoints(gap_threshold=15, include_last=True, print_progress=True)

print("extract triplegs")
pfs, tpls = pfs.as_positionfixes.generate_triplegs(spts, method="between_staypoints", gap_threshold=15)

print("attach labels to triplegs")
tpls = ti.io.dataset_reader.geolife_add_modes_to_triplegs(tpls, mode_labels)

print("extract locations")
spts, locs = spts.as_staypoints.generate_locations(
    method="dbscan", epsilon=50, num_samples=1, distance_metric="haversine", agg_level="user"
)

# spts = ti.analysis.location_identifier(spts, method="FREQ", pre_filter=True)

print("add activity flag to staypoints")
spts = spts.as_staypoints.create_activity_flag(method="time_threshold", time_threshold=15)

print("extract trips")
spts, triplegs, trips = ti.preprocessing.triplegs.generate_trips(spts, tpls, gap_threshold=15, print_progress=True)

print("write to database")

print("write positionfixes to database")
pfs.as_positionfixes.to_postgis(conn_string, "positionfixes", schema=schema_name, if_exists="replace")

print("write staypoints to database")
ti.io.write_staypoints_postgis(spts, conn_string, "staypoints", schema=schema_name, if_exists="replace")

print("write triplegs to database")
ti.io.write_triplegs_postgis(tpls, conn_string, "triplegs", schema=schema_name, if_exists="replace")

print("write trips to database")
trips.as_trips.to_postgis(conn_string, "trips", schema=schema_name, if_exists="replace")

print("write locations to database")
locs = locs.drop("extent", axis=1)
locs.as_locations.to_postgis(conn_string, "locations", schema=schema_name, if_exists="replace")