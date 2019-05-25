# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:31:37 2019

@author: martinhe
"""

import geopandas as gpd
import trackintel as ti
from sqlalchemy import create_engine
from trackintel.preprocessing import activity_graphs as tigraphs
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
import ntpath
import json
import datetime
from sklearn.preprocessing import MinMaxScaler

from activity_graphs_utils import draw_smopy_basemap, nx_coordinate_layout_smopy

CRS_WGS84 = {'init' :'epsg:4326'}
#
studies = ['gc1','gc2']

for study in studies:
    # build database login string from file
    DBLOGIN_FILE = os.path.join("dblogin.json")
    with open(DBLOGIN_FILE) as json_file:  
        LOGIN_DATA = json.load(json_file)
        
    conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}"\
                    .format(**LOGIN_DATA)
    
    engine = create_engine(conn_string)
    conn = engine.connect()
    
    sp_org = gpd.GeoDataFrame.from_postgis("""SELECT *, geometry_raw as geometry_new
                                           FROM {}.staypoints""".format(study),
                                           conn,
                                           crs=CRS_WGS84, 
                                           geom_col='geometry_new',
                                           index_col='id')
    conn.close()
    
    # create important places 
    sp = sp_org.copy()
    sp["geom"] = sp["geometry_new"]
    sp["elevation"] = np.nan
    
    sp = sp.set_geometry("geom")
    
    print('create places')
    places = sp.as_staypoints.extract_places(epsilon=50, num_samples=4,
                                             distance_matrix_metric='haversine')

    print('write staypoints to database')
    ti.io.write_staypoints_postgis(sp, conn_string, schema=study, table_name="staypoints")
    print('write places to database')
    ti.io.write_places_postgis(places, conn_string, schema=study, table_name="places")