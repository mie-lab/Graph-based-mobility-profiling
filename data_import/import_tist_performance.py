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
import numpy as np

CRS_WGS = {'init' :'epsg:4326'}
schema_name = 'tist'

dblogin_file = os.path.join("..", "dblogin.json")
with open(dblogin_file) as json_file:
    login_data = json.load(json_file)

conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**login_data)

engine = create_engine(conn_string)
conn = engine.connect()

# create places and bring into trackintel format



import time
t_start_total = time.time()
schema_name = 'tist'
  
    
user_ids = pd.read_sql("select distinct user_id from {}.staypoints order by user_id".format(schema_name),engine).values.ravel()
users_per_iteration = 100
nb_users = len(user_ids)
nb_of_splits = nb_users // users_per_iteration
if nb_of_splits == 0:
    nb_of_splits = 1


for iter_ix, user_ids_this in enumerate(np.array_split(user_ids, nb_of_splits)):
    t_start_iter = time.time()
    
    min_id_this = user_ids_this[0]
    max_id_this = user_ids_this[-1]
    
    user_ids_this = tuple(user_ids_this)
        
    t1 = time.time()
    sp = gpd.GeoDataFrame.from_postgis(
                "select * from {}.staypoints where user_id in {}".format(schema_name,
                               user_ids_this), conn, crs=CRS_WGS, geom_col='geom')
    t1 = time.time()-t1
#    t2 = time.time()
#    sp = gpd.GeoDataFrame.from_postgis(
#            "select * from {}.staypoints where user_id >= {} and user_id <= {} order by user_id".format(schema_name,
#                           min_id_this,max_id_this), conn, crs=CRS_WGS, geom_col='geom')
#    t2 = time.time() - t2
#    
    print('\t \t reading: {:.0f}'.format(t1))
    
    # bring to trackintel format
    # add columns
    
    REQUIRED_COLUMNS = ['user_id', 'started_at', 'finished_at', 'elevation', 'geom']
    columns = list(set(list(sp.columns) + REQUIRED_COLUMNS))
    sp = sp.reindex(columns=columns)
    
    
    # extract places
    
    t2 = time.time()
    places = sp.as_staypoints.extract_places(epsilon=50, num_samples=4, distance_matrix_metric='haversine')
    t2 = time.time() - t2
    print('\t \t extracting: {:.0f}'.format(t2))
    t3 = time.time()
    
    
    # write staypoints and places to database
    ti.io.write_staypoints_postgis(sp, conn_string, schema=schema_name,
                                   table_name="staypoints_temp", if_exists='append')
    ti.io.write_places_postgis(places, conn_string, schema=schema_name,
                               table_name="places_temp", if_exists='append')
    t3 = time.time() - t3
    print('\t \t writing: {:.0f}'.format(t3))
    # print report
    t_iter = time.time()-t_start_iter
    print('\t {}/{} users. Time for this iteration: {:.0f} seconds'.format((iter_ix+1)*users_per_iteration,nb_users,t_iter))
    
    break

print('Done')
t_total = time.time() - t_start_total
print('Total time for {} users: {:.0f} seconds'.format(nb_users,t_total))




