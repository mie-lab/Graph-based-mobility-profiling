"""
Script to import tist data into a postgis database. Also applies trackintel
data model
"""

import os
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import json


dblogin_file = os.path.join("..", "dblogin.json")
with open(dblogin_file) as json_file:
    login_data = json.load(json_file)

conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**login_data)

engine = create_engine(conn_string)
conn = engine.connect()

path_checkins = os.path.join('data_tist', 'dataset_TIST2015_Checkins.txt')
path_pois = os.path.join('data_tist', 'dataset_TIST2015_POIs.txt')

# create schema
with psycopg2.connect(conn_string) as conn2:
    cur = conn2.cursor()

    query = """CREATE SCHEMA if not exists tist;"""
    cur.execute(query)
    conn2.commit()

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

print('Writing to database')
checkins.to_sql('checkins', engine, schema="tist", if_exists='append', index=False, chunksize=50000)
venues.to_sql('venues', engine, schema="tist", if_exists='append', index=False, chunksize=50000)

del checkins, venues


with psycopg2.connect(conn_string) as conn2:
    cur = conn2.cursor()

    # create staypoints table
    print('Create staypoint table')
    query = """CREATE TABLE tist.staypoints as
                    SELECT checkins.user_id, checkins.started_at,
                            venues.lat, venues.lon, venues.category,
                            checkins.timezone, venues.country_code
                    FROM tist.checkins
                    INNER JOIN tist.venues on checkins.venue_id = venues.venue_id;"""
    cur.execute(query)
    conn2.commit()

    # create and fill geometry
    print('Create staypoint geometries')
    query = """select AddGeometryColumn(
                'tist', 'staypoints', 'geom', 4326, 'Point', 2);"""
    cur.execute(query)

    query = """update tist.staypoints SET
                geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);"""
    cur.execute(query)
    conn2.commit()

print('Done')
