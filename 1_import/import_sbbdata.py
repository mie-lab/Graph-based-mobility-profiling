"""
Script to import sbb green class data into a postgis database. Also applies trackintel
data model
"""

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import trackintel as ti
from sqlalchemy import create_engine

from general_utils import horizontal_merge_staypoints

CRS_WGS84 = "epsg:4326"
#
studies = ["gc2", "gc1"]

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

    print("\tdownload staypoints")
    sp = gpd.GeoDataFrame.from_postgis(
        sql="SELECT * FROM {}.staypoints".format(study),
        con=conn_source,
        crs=CRS_WGS84,
        geom_col="geometry_raw",
        index_col="id",
    )

    print("\tdownload triplegs")
    tpls = gpd.GeoDataFrame.from_postgis(
        sql="SELECT * FROM {}.triplegs where ST_isValid(geometry)".format(study),
        con=conn_source,
        crs=CRS_WGS84,
        geom_col="geometry",
        index_col="id",
    )

    # drop entries with invalid timestamps
    valid_tstamp_flag_sp = sp.started_at <= sp.finished_at
    valid_tstamp_flag_tpls = tpls.started_at <= tpls.finished_at
    print(
        "\t\ttimestamps of {} sp and {} tpls corrupted. corrupted ts are dropped".format(
            sp.shape[0] - sum(valid_tstamp_flag_sp), tpls.shape[0] - sum(valid_tstamp_flag_tpls)
        )
    )

    sp = sp[valid_tstamp_flag_sp]
    tpls = tpls[valid_tstamp_flag_tpls]

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

    print("\tcreate locations")
    sp, locs = sp.as_staypoints.generate_locations(
        method="dbscan", epsilon=30, num_samples=1, distance_metric="haversine", agg_level="user"
    )
    # merge horizontal staypoints
    sp = horizontal_merge_staypoints(sp, custom_add_dict={"purpose_detected": list})
    sp = ti.io.read_staypoints_gpd(sp, geom_col="geom")

    sp, tpls, trips = ti.preprocessing.generate_trips(sp, tpls, gap_threshold=25)
    tpls.index.name = "id"

    # time threshld for activities
    meaningful_purpose = ~sp["purpose"].isin(["wait", "unknown"])
    meaningful_duration = sp["finished_at"] - sp["started_at"] >= pd.Timedelta("25min")
    sp["activity"] = sp["activity"] | meaningful_purpose | meaningful_duration

    print("\twrite staypoints to database")
    ti.io.write_staypoints_postgis(
        staypoints=sp,
        con=conn_string,
        name="staypoints",
        schema=study,
        if_exists="replace",
        index_label=locs.index.name,
    )

    print("\twrite triplegs")
    ti.io.write_triplegs_postgis(
        triplegs=tpls, con=conn_string, name="triplegs", schema=study, if_exists="replace", index_label=locs.index.name
    )

    print("\twrite trips")
    ti.io.write_trips_postgis(
        trips=trips, con=engine, name="trips", schema=study, if_exists="replace", index_label=trips.index.name
    )

    print("\twrite locations to database")
    locs = locs.drop("extent", axis=1)
    ti.io.write_locations_postgis(
        locs, con=conn_string, schema=study, name="locations", if_exists="replace", index_label=locs.index.name
    )
