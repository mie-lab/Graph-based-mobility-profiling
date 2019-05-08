# -*- coding: utf-8 -*-
"""
This script reads in the geolife data (as it can be downloaded from
 https://www.microsoft.com/en-us/download/details.aspx?id=52367) and loads it in a postgis database
"""

import time
import json
import numpy as np
import glob
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
import psycopg2



crs_wgs = {'init' :'epsg:4326'}
feet2meter = 0.3048

# connect to postgis database
with open('login.json') as json_file:  
    login_data = json.load(json_file)
    
conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**login_data)
engine = create_engine(conn_string)
conn = engine.connect()

data_folder = ".\data_geolife\*"
user_folder = glob.glob(data_folder)

# create schema for the data
with psycopg2.connect(conn_string) as conn2:
    cur = conn2.cursor()    
    
    query = """CREATE SCHEMA if not exists geolife;"""    
    cur.execute(query)
    conn2.commit()

# In the geolife data, every user has a folder with a file with tracking data for every day. We iterate every folder
# concatenate all files of 1 user into a single pandas dataframe and send it to the postgres database.
    
for user_folder_this in user_folder:
    t_start = time.time()
    user_id = int(user_folder_this.split("\\")[-1])
    print("start user_id: ", user_id)
    
    input_files = glob.glob(user_folder_this+"\Trajectory\*.plt")
    df_list = []
    
    # read every day of every user
    for input_file_this in input_files:
        # todo: Rename columns in correct trackintel way
        data_this = pd.read_csv(input_file_this, skiprows=6, header=None,
                                names=['lat','lon','zeros','altitude meter','date days','date','time'])
        
        data_this['timestamp'] = pd.to_datetime(data_this['date'] + ' ' + data_this['time'])
        
        data_this.drop(['zeros','date days','date','time'], axis=1, inplace=True)
        data_this['user_id'] = user_id
        data_this['altitude meter'] = data_this['altitude meter'] * feet2meter
    
        df_list.append(data_this)
        
    data = pd.concat(df_list, axis=0, ignore_index=True)
    del df_list
    data.to_sql('positionfixes', engine, schema="geolife", if_exists='append', index=False, chunksize=50000)
    del data, data_this
    
    t_end = time.time()
    print("finished user_id: ", user_id, "Duration: ","{:.0f}".format(t_end-t_start))
    

print("finished all users, start creating geometries")

# add geometry to table
with psycopg2.connect(conn_string) as conn2:
    cur = conn2.cursor()    
    
    query = """select AddGeometryColumn('geolife', 'positionfixes', 'geom', 4326, 'Point', 2);"""
    cur.execute(query)
    conn2.commit()
    
    query = """update geolife.positionfixes SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);"""
    cur.execute(query)
    conn2.commit()
    




        

