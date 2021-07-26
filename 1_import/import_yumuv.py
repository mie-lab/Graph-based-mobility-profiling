import csv
import datetime
import logging
import os
import sys
import geopandas as gpd
import pandas as pd
import pytz
from shapely.geometry import Point
from sqlalchemy import create_engine
import trackintel as ti

sys.path.append(r"C:\Users\e527371\OneDrive\Programming\yumuv")
from db_login import DSN  # database login information
import numpy as np

engine = create_engine(
    "postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}".format(
        **DSN
    )
)
#
data_folder = os.path.join("C:/", "yumuv", "data")  # todo move to config file
cache_folder = os.path.join(data_folder, "cache")  # todo move to config file
max_date = datetime.datetime(year=2021, month=3, day=1, tzinfo=pytz.utc)

print("Download staypoints")
sp = gpd.read_postgis("select * from yumuv.staypoint", engine, geom_col="geometry")
sp = ti.io.read_staypoints_gpd(sp, user_id="user_fk", geom_col="geometry", tz="UTC")

sp["elevation"] = np.nan
# Add activity: Everything longer than 25 minutes or meaningful purpose
sp = sp.as_staypoints.create_activity_flag(
    time_threshold=25, activity_column_name="activity"
)
meaningful_purpose = ~sp["stay_purpose"].isin(["wait", "unknown"])
sp["activity"] = sp["activity"] | meaningful_purpose
sp = sp.rename(columns={"geometry": "geom", "stay_purpose": "purpose"})


print("Download triplegs")
tpls = gpd.read_postgis("select * from yumuv.tripleg", engine, geom_col="geometry")
tpls = tpls.drop("geometry_raw", axis=1)
tpls = tpls.rename(columns={"geometry": "geom"})

geom_not_valid = ~tpls.geometry.is_valid
print("invalid triplegs", sum(geom_not_valid))
tpls = tpls[tpls.geometry.is_valid]
tpls = ti.io.read_triplegs_gpd(
    tpls, user_id="story_line_uuid", geom_col="geometry", tz="UTC"
)

sp, locs = sp.as_staypoints.generate_locations(
    method="dbscan",
    epsilon=30,
    num_samples=2,
    distance_metric="haversine",
    agg_level="user",
)


print("write staypoints to database")
ti.io.write_staypoints_postgis(
    staypoints=sp,
    con=engine,
    name="staypoints",
    schema="yumuv_graph_rep",
    if_exists="replace",
)

print("write triplegs")

ti.io.write_triplegs_postgis(
    triplegs=tpls,
    con=engine,
    name="triplegs",
    schema="yumuv_graph_rep",
    if_exists="replace",
)

loc_extent = gpd.GeoDataFrame(locs["extent"], geometry="extent")
loc_extent.to_file("test_locs3.shp")

print("write locations to database")
ti.io.write_locations_postgis(
    locs, name="locations", con=engine, schema="yumuv_graph_rep", if_exists="replace"
)
