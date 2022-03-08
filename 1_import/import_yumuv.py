import datetime
import os
import sys
import geopandas as gpd
import pytz
from sqlalchemy import create_engine
import trackintel as ti
from trackintel.preprocessing.triplegs import generate_trips

sys.path.append(r"C:\Users\e527371\OneDrive\Programming\yumuv")
from db_login import DSN  # database login information
import numpy as np
from future_trackintel.utils import horizontal_merge_staypoints


min_date = datetime.datetime(year=2020, month=7, day=13, tzinfo=pytz.utc)
max_date = datetime.datetime(year=2020, month=11, day=15, tzinfo=pytz.utc)

engine = create_engine("postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}".format(**DSN))
#
data_folder = os.path.join("C:/", "yumuv", "data")
cache_folder = os.path.join(data_folder, "cache")
limit = ""  # optional: can be used to download subset

sp_sql = """SELECT staypoint.*, study_code_sorted.study_id FROM
                        yumuv.staypoint LEFT JOIN 
						(select distinct ON(app_user_id) app_user_id, study_code.study_id
						 FROM raw_myway.study_code ORDER BY app_user_id, study_code.study_id)
						AS study_code_sorted
                         ON study_code_sorted.app_user_id = user_fk {}""".format(
    limit
)

print("Download staypoints")
sp = gpd.read_postgis(sp_sql, engine, geom_col="geometry", index_col="id")

sp = ti.io.read_staypoints_gpd(sp, user_id="user_fk", geom_col="geometry", tz="UTC")

sp["elevation"] = np.nan

# Add activity: Everything longer than 25 minutes or meaningful purpose
sp = sp.as_staypoints.create_activity_flag(time_threshold=25, activity_column_name="activity")
meaningful_purpose = ~sp["stay_purpose"].isin(["wait", "unknown"])
sp["activity"] = sp["activity"] | meaningful_purpose

sp = sp.rename(columns={"geometry": "geom", "stay_purpose": "purpose"})
sp = sp.set_geometry("geom")

# filter study duration
sp_date_flag = (sp["started_at"] >= min_date) & (sp["finished_at"] <= max_date)
sp = sp[sp_date_flag]

sp, locs = sp.as_staypoints.generate_locations(
    method="dbscan",
    epsilon=30,
    num_samples=1,
    distance_metric="haversine",
    agg_level="user",
)
sp = horizontal_merge_staypoints(sp)
sp = ti.io.read_staypoints_gpd(sp, geom_col="geom")


print("Download triplegs")
tpls = gpd.read_postgis(
    """select *  FROM yumuv.tripleg {}""".format(limit),
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


tpls_date_flag = (tpls["started_at"] >= min_date) & (tpls["finished_at"] <= max_date)
tpls = tpls[tpls_date_flag]

print("generate trips")
assert sp.index.is_unique
assert tpls.index.is_unique
sp, tpls, trips = generate_trips(sp, tpls, gap_threshold=25)
tpls.index.name = "id"


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
