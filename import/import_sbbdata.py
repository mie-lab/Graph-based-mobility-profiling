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

CRS_WGS84 = 'epsg:4326'
#
studies = ['gc2', 'gc1']

DBLOGIN_FILE = os.path.join('..','dblogin.json')
DBLOGIN_FILE_SOURCE = os.path.join('..','dblogin_source.json')

with open(DBLOGIN_FILE) as json_file:  
    LOGIN_DATA = json.load(json_file)

with open(DBLOGIN_FILE_SOURCE) as json_file_source:
    LOGIN_DATA_SOURCE = json.load(json_file_source)
        
for study in studies:
    # build database login string from file
    print('start study {}'.format(study))
        
    conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}"\
                    .format(**LOGIN_DATA)
    conn_string_source = "postgresql://{user}:{password}@{host}:{port}/{database}"\
                    .format(**LOGIN_DATA_SOURCE)
    
    engine = create_engine(conn_string)
    conn = engine.connect()

    engine_source = create_engine(conn_string_source)
    conn_source = engine_source.connect()

    print('download staypoints')    
    sp = gpd.GeoDataFrame.from_postgis(sql="SELECT * FROM {}.staypoints".format(study),
                                           con=conn_source,
                                           crs=CRS_WGS84, 
                                           geom_col='geometry_raw')
    conn_source.close()
    sp = sp.drop("geometry", axis=1)
    # create important places 
    sp["elevation"] = np.nan
    sp['started_at'] = sp['started_at'].dt.tz_localize('UTC')
    sp['finished_at'] = sp['finished_at'].dt.tz_localize('UTC')
    print('create places')
    sp, locs = sp.as_staypoints.generate_locations(method='dbscan', epsilon=50, num_samples=1,
                                                       distance_metric='haversine', agg_level='user')

    print('write staypoints to database')

    ti.io.write_staypoints_postgis(staypoints=sp, conn_string=conn_string, table_name="staypoints",
                                   schema=study, if_exists="replace")
    # conn = engine.connect()
    # sp.to_postgis(name="staypoints", con=conn, schema=study, if_exists='replace', index=False, index_label=None)


    print('write locations to database')
    locs = locs.drop('extent', axis=1)
    ti.io.write_locations_postgis(locs, conn_string, schema=study, table_name="locations", if_exists='replace')