"""
Script to import tist data into a postgis database. Also applies trackintel
data model
"""

import datetime
import json
import os

import geopandas as gpd
import pandas as pd
import trackintel as ti
from dateutil import tz
from sqlalchemy import create_engine

from config import config

schema_name = "tist"

dblogin_file = os.path.join("dblogin.json")
with open(dblogin_file) as json_file:
    login_data = json.load(json_file)

conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**login_data)

engine = create_engine(conn_string)
conn = engine.connect()


path_checkins = config["path_checkins"]
path_pois = config["path_pois"]


# https://sites.google.com/site/yangdingqi/home/foursquare-dataset
# Dingqi Yang, Daqing Zhang, Bingqing Qu. Participatory Cultural Mapping Based on
# Collective Behavior Data in Location Based Social Networks. ACM Trans. on Intelligent
# Systems and Technology (TIST), 2015. [PDF]

# load raw data
print("Reading raw data")

checkins = pd.read_csv(path_checkins, sep="\t", header=None, names=["user_id", "venue_id", "started_at", "timezone"])
venues = pd.read_csv(path_pois, sep="\t", header=None, names=["venue_id", "lat", "lon", "purpose", "country_code"])
print("Prepare time stamps")
checkins["started_at"] = pd.to_datetime(
    checkins["started_at"], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce", utc=True
)
checkins["started_at_local"] = checkins.apply(
    lambda x: x["started_at"].tz_convert(tz=tz.tzoffset(None, datetime.timedelta(minutes=x["timezone"]))), axis=1
)
checkins["finished_at"] = pd.Series([pd.NaT], dtype=pd.api.types.DatetimeTZDtype(tz="utc"))

print("Prepare venues")
# repair unicode error
venues.loc[venues["purpose"] == "Caf", "purpose"] = "Café"

# prepare venues
venues = gpd.GeoDataFrame(venues, geometry=gpd.points_from_xy(venues.lon, venues.lat))
venues.drop(["lat", "lon"], inplace=True, axis=1)
venues = venues.set_index("venue_id")

# prepare checkins as staypoins
sp = checkins.join(venues, on="venue_id", how="left")
sp = gpd.GeoDataFrame(sp, geometry="geometry")
del venues, checkins
print("Create locations")

# create locations
# 10e-6 is a very small search radius to ensure that every venue gets detected as a location (if it has a unique
# location)
sp, locations = sp.as_staypoints.generate_locations(epsilon=10e-6, num_samples=1, distance_metric="euclidean")
locations.drop("extent", axis=1, inplace=True)


# tist is now in trackintel format.
print("Write back locations ")
ti.io.write_locations_postgis(
    locations, con=conn_string, schema=schema_name, name="locations", if_exists="replace", chunksize=100000
)
print("Write back staypoints")
ti.io.write_staypoints_postgis(
    sp, con=conn_string, schema=schema_name, name="staypoints", if_exists="replace", chunksize=100000
)
