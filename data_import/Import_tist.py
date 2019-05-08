import numpy as np
import glob
import pandas as pd
import geopandas as gpd
import sys
from sqlalchemy import create_engine
import psycopg2
import time

crs_wgs = {'init' :'epsg:4326'}
with open('login.json') as json_file:  
    login_data = json.load(json_file)
    
conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**login_data)

engine = create_engine(conn_string)
conn = engine.connect()

data_folder = ".\data_tist\*"
user_folder = glob.glob(data_folder)


with psycopg2.connect(conn_string) as conn2:
    cur = conn2.cursor()    
    
    query = """CREATE SCHEMA if not exists tist;"""    
    cur.execute(query)
    conn2.commit()

checkins = pd.read_csv('dataset_TIST2015_Checkins.txt', sep='\t', header=None, names=['user_id,', 'venue_id', 'started_at'])
venues = pd.read_csv('dataset_TIST2015_POIs.txt', sep='\t', header=None, names=['venue_id', 'lat', 'lon', 'category', 'country_code'])

checkins.to_sql('checkins', engine, schema="tist", if_exists='append', index=False, chunksize=50000)
venues.to_sql('venues', engine, schema="tist", if_exists='append', index=False, chunksize=50000)
