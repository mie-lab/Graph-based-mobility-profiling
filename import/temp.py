# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:11:30 2019

@author: martinhe
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:31:37 2019

@author: martinhe
"""

import geopandas as gpd
import trackintel as ti
from sqlalchemy import create_engine
import numpy as np
import os
import json

CRS_WGS84 = {'init' :'epsg:4326'}
#
studies = ['gc1',] # 'gc2', 
study = 'gc1'

DBLOGIN_FILE = os.path.join('..','dblogin.json')
with open(DBLOGIN_FILE) as json_file:  
    LOGIN_DATA = json.load(json_file)
        
conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA)
    
engine = create_engine(conn_string)
conn = engine.connect()

print('download staypoints')    
sp = gpd.GeoDataFrame.from_postgis("""SELECT *, geometry_raw as geom
                                       FROM {}.staypoints limit 1000""".format(study),
                                       conn,
                                       crs=CRS_WGS84, 
                                       geom_col='geom')
conn.close()

# create important places 
sp["elevation"] = np.nan

sp = sp.set_geometry("geom")
    
#    print('create places')
#    places = sp.as_staypoints.extract_places(epsilon=50, num_samples=4,
#                                             distance_matrix_metric='haversine')
#
#    print('write staypoints to database')
#    ti.io.write_staypoints_postgis(sp, conn_string, schema=study, table_name="staypoints")
#    print('write places to database')
#    ti.io.write_places_postgis(places, conn_string, schema=study, table_name="places")
    
