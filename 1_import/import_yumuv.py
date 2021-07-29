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
from trackintel.preprocessing.triplegs import generate_trips

sys.path.append(r"C:\Users\e527371\OneDrive\Programming\yumuv")
from db_login import DSN  # database login information
import numpy as np


def horizontal_merge_staypoints(sp, gap_threshold=20):
    """merge staypoints that are consecutive at the same place"""
    # merge consecutive staypoints

    sp_merge = sp.copy()
    assert sp_merge.index.name == "id", "expected index name to be 'id'"

    sp_merge = sp_merge.reset_index()
    sp_merge.sort_values(inplace=True, by=["user_id", "started_at"])
    sp_merge[["next_started_at", "next_location_id"]] = sp_merge[["started_at", "location_id"]].shift(-1)
    cond = pd.Series(data=False, index=sp_merge.index)
    cond_old = pd.Series(data=True, index=sp_merge.index)
    cond_diff = cond != cond_old

    while np.sum(cond_diff) >= 1:
        # .values is important otherwise the "=" would imply a join via the new index
        sp_merge["next_id"] = sp_merge["id"].shift(-1).values

        # identify rows to merge
        cond1 = sp_merge["next_started_at"] - sp_merge["finished_at"] < datetime.timedelta(minutes=gap_threshold)
        cond2 = sp_merge["location_id"] == sp_merge["next_location_id"]
        cond = cond1 & cond2

        # assign index to next row
        sp_merge.loc[cond, "id"] = sp_merge.loc[cond, "next_id"]
        cond_diff = cond != cond_old
        cond_old = cond.copy()

        print("\t", np.sum(cond_diff))

    # aggregate values
    sp_merged = sp_merge.groupby(by="id").agg(
        {
            "user_id": "first",
            "trip_id": list,
            "prev_trip_id": list,
            "next_trip_id": list,
            "started_at": "first",
            "finished_at": "last",
            "geom": "first",
            "elevation": "first",
            "location_id": "first",
            "activity": "first",
            "purpose": list
            # "purpose_detected": list,
            # "purpose_validated": list,
            # "validated": "first",
            # "validated_at": "first",
        }
    )

    return sp_merged


engine = create_engine("postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}".format(**DSN))
#
data_folder = os.path.join("C:/", "yumuv", "data")  # todo move to config file
cache_folder = os.path.join(data_folder, "cache")  # todo move to config file
max_date = datetime.datetime(year=2021, month=3, day=1, tzinfo=pytz.utc)
# limit = "where user_fk < 4980"
limit = ""

print("Download staypoints")
sp = gpd.read_postgis(
    """select staypoint.*, study_code.study_id from
                        yumuv.staypoint left join raw_myway.study_code
                         on app_user_id = user_fk {}""".format(
        limit
    ),
    engine,
    geom_col="geometry",
    index_col="id",
)

sp = ti.io.read_staypoints_gpd(sp, user_id="user_fk", geom_col="geometry", tz="UTC")

sp["elevation"] = np.nan
# Add activity: Everything longer than 25 minutes or meaningful purpose
sp = sp.as_staypoints.create_activity_flag(time_threshold=25, activity_column_name="activity")
meaningful_purpose = ~sp["stay_purpose"].isin(["wait", "unknown"])
sp["activity"] = sp["activity"] | meaningful_purpose
sp = sp.rename(columns={"geometry": "geom", "stay_purpose": "purpose"})
sp = sp.set_geometry("geom")

sp, locs = sp.as_staypoints.generate_locations(
    method="dbscan", epsilon=30, num_samples=1, distance_metric="haversine", agg_level="user",
)
sp = horizontal_merge_staypoints(sp)
sp = ti.io.read_staypoints_gpd(sp)


print("Download triplegs")
tpls = gpd.read_postgis(
    """select tripleg.*, study_code.study_id
 FROM yumuv.tripleg left join raw_myway.study_code ON
  app_user_id = user_fk {}""".format(
        limit
    ),
    engine,
    geom_col="geometry",
    index_col="id",
)

tpls = tpls.drop("geometry_raw", axis=1)
tpls = tpls.rename(columns={"geometry": "geom"})
tpls = tpls.set_geometry("geom")

geom_not_valid = ~tpls.geometry.is_valid
print("invalid triplegs", sum(geom_not_valid))
tpls = tpls[tpls.geometry.is_valid]
tpls = ti.io.read_triplegs_gpd(tpls, user_id="user_fk", geom_col="geom", tz="UTC")


print("generate trips")
sp, tpls, trips = generate_trips(sp, tpls)
tpls.index.name = "id"
# sp.iloc[0:5000, :].to_file("sp_debug_30_1_yumuv.gpk", driver='GPKG')

print("write staypoints to database")
ti.io.write_staypoints_postgis(
    staypoints=sp,
    con=engine,
    name="staypoints",
    schema="yumuv_graph_rep",
    if_exists="replace",
    index_label=sp.index.name,
)

print("write triplegs")
ti.io.write_triplegs_postgis(
    triplegs=tpls,
    con=engine,
    name="triplegs",
    schema="yumuv_graph_rep",
    if_exists="replace",
    index_label=tpls.index.name,
)

print("write trips")
ti.io.write_trips_postgis(
    trips=trips, con=engine, name="trips", schema="yumuv_graph_rep", if_exists="replace", index_label=trips.index.name
)

print("write locations to database")
ti.io.write_locations_postgis(
    locs, name="locations", con=engine, schema="yumuv_graph_rep", if_exists="replace", index_label=locs.index.name
)
