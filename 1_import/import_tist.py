"""
Script to import tist data into a postgis database. Also applies trackintel
data model
"""

import os
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import geopandas as gpd
import json
import trackintel as ti
import datetime as dt
from shapely.geometry import Point

schema_name = 'tist2'

dblogin_file = os.path.join("..", "dblogin.json")
with open(dblogin_file) as json_file:
    login_data = json.load(json_file)

conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**login_data)

engine = create_engine(conn_string)
conn = engine.connect()


path_checkins = os.path.join(r'E:\data_tist\dataset_TIST2015_Checkins.txt')
path_pois = os.path.join(r'E:\data_tist\dataset_TIST2015_POIs.txt')



# https://sites.google.com/site/yangdingqi/home/foursquare-dataset
# Dingqi Yang, Daqing Zhang, Bingqing Qu. Participatory Cultural Mapping Based on Collective Behavior Data in Location Based Social Networks. ACM Trans. on Intelligent Systems and Technology (TIST), 2015. [PDF]

# load raw data
print('Reading raw data')

checkins = pd.read_csv(path_checkins, sep='\t', header=None,
                      names=['user_id', 'venue_id', 'started_at', 'timezone'])
venues = pd.read_csv(path_pois, sep='\t', header=None,
                    names=['venue_id', 'lat', 'lon', 'category', 'country_code'])

checkins['started_at'] = pd.to_datetime(checkins['started_at'], format='%a %b %d %H:%M:%S +0000 %Y',
		errors='coerce')

# repair unicode error
venues.loc[venues["category"] == 'Caf', "category"] = 'Café'

venues = venues.set_index('venue_id')
spts = checkins.join(venues, on="venue_id", how='left')

spts['started_at'] = spts['started_at'].astype(pd.api.types.DatetimeTZDtype(tz='utc'))
spts['finished_at'] = pd.Series([pd.NaT], dtype=pd.api.types.DatetimeTZDtype(tz='utc'))


# creating a geometry column
geometry = [Point(xy) for xy in zip(spts['lon'], spts['lat'])]
spts = gpd.GeoDataFrame(spts, crs='epsg:4326', geometry=geometry)
spts.drop(['lat', 'lon'], inplace=True, axis=1)

# tist is now in trackintel format.


print('generate locations')
locations, spts = spts.as_staypoints.generate_locations(epsilon=50, num_samples=4, distance_metric='haversine')
print('Write back staypoints')
ti.io.write_staypoints_postgis(spts, conn_string, schema=schema_name, table_name="staypoints", if_exists='replace')
print('Write back locations ')
ti.io.write_locations_postgis(locations, conn_string, schema=schema_name, table_name="locations", if_exists='replace')





