# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:31:37 2019

@author: martinhe
"""

import geopandas as gpd
import pandas as pd
import trackintel as ti
from sqlalchemy import create_engine
import numpy as np
import os
import json
from future_trackintel.utils import horizontal_merge_staypoints

CRS_WGS84 = "epsg:4326"
#
studies = ["gc1"]

DBLOGIN_FILE = os.path.join(".", "dblogin.json")
DBLOGIN_FILE_SOURCE = os.path.join(".", "dblogin_source.json")

with open(DBLOGIN_FILE) as json_file:
    LOGIN_DATA = json.load(json_file)

with open(DBLOGIN_FILE_SOURCE) as json_file_source:
    LOGIN_DATA_SOURCE = json.load(json_file_source)

for study in studies:
    # build database login string from file
    print("start study {}".format(study))

    conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA)
    conn_string_source = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA_SOURCE)

    engine = create_engine(conn_string)
    conn = engine.connect()

    engine_source = create_engine(conn_string_source)
    conn_source = engine_source.connect()

    print("download staypoints")
    sp = gpd.GeoDataFrame.from_postgis(
        sql="SELECT * FROM {}.staypoints where user_id <= 1600".format(study),
        con=conn_source,
        crs=CRS_WGS84,
        geom_col="geometry_raw",
        index_col="id",
    )
    print("download triplegs")
    tpls = gpd.GeoDataFrame.from_postgis(
        sql="SELECT * FROM {}.triplegs where ST_isValid(geometry) and user_id <= 1600 limit 1000".format(study),
        con=conn_source,
        crs=CRS_WGS84,
        geom_col="geometry",
        index_col="id",
    )

    conn_source.close()
    sp = sp.drop("geometry", axis=1)
    tpls = tpls.drop("geometry_raw", axis=1)

    sp = sp.rename(columns={"geometry_raw": "geom", "purpose_validated": "purpose"})
    tpls = tpls.rename(columns={"geometry": "geom"})

    sp = sp.set_geometry("geom")
    tpls = tpls.set_geometry("geom")

    sp["elevation"] = np.nan
    sp["started_at"] = sp["started_at"].dt.tz_localize("UTC")
    sp["finished_at"] = sp["finished_at"].dt.tz_localize("UTC")

    tpls["started_at"] = tpls["started_at"].dt.tz_localize("UTC")
    tpls["finished_at"] = tpls["finished_at"].dt.tz_localize("UTC")

    print("create places")
    sp, locs = sp.as_staypoints.generate_locations(
        method="dbscan", epsilon=30, num_samples=1, distance_metric="haversine", agg_level="user"
    )
    # merge horizontal staypoints
    sp = horizontal_merge_staypoints(sp)
    sp = ti.io.read_staypoints_gpd(sp)

    sp, tpls, trips = ti.preprocessing.generate_trips(sp, tpls)
    tpls.index.name = "id"

    print("write staypoints to database")
    ti.io.write_staypoints_postgis(
        staypoints=sp,
        con=conn_string,
        name="staypoints",
        schema=study,
        if_exists="replace",
        index_label=locs.index.name,
    )

    print("write triplegs")
    ti.io.write_triplegs_postgis(
        triplegs=tpls, con=conn_string, name="triplegs", schema=study, if_exists="replace", index_label=locs.index.name
    )

    print("write trips")
    ti.io.write_trips_postgis(
        trips=trips, con=engine, name="trips", schema=study, if_exists="replace", index_label=trips.index.name
    )

    print("write locations to database")
    locs = locs.drop("extent", axis=1)
    ti.io.write_locations_postgis(
        locs, con=conn_string, schema=study, name="locations", if_exists="replace", index_label=locs.index.name
    )
