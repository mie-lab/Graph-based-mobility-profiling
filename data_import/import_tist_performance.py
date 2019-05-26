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
import time

CRS_WGS = {'init' :'epsg:4326'}

dblogin_file = os.path.join("..", "dblogin.json")
with open(dblogin_file) as json_file:
    login_data = json.load(json_file)

conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**login_data)
engine = create_engine(conn_string)
conn = engine.connect()


t_start_total = time.time()
schema_name = 'tist_e5'
  
# download user_ids   
user_ids = pd.read_sql("select distinct user_id from {}.staypoints \
                       order by user_id".format(schema_name),
                       engine).values.ravel()
users_per_iteration = 1000
nb_users = len(user_ids)
nb_of_splits = nb_users // users_per_iteration
if nb_of_splits == 0:
    nb_of_splits = 1


for iter_ix, user_ids_this in enumerate(np.array_split(user_ids, nb_of_splits)):
    t_start_iter = time.time()
    
    # download staypoints
    user_ids_this = tuple(user_ids_this)
    sp = gpd.GeoDataFrame.from_postgis(
                "select * from {}.staypoints where user_id in {}".format(schema_name,
                               user_ids_this), conn, crs=CRS_WGS, geom_col='geom')

    # extract places
    places = sp.as_staypoints.extract_places(epsilon=50, num_samples=4, distance_matrix_metric='haversine')
    
    # create and write staypoints_id - place_id mapping to database
    mapping_df = sp.loc[sp['place_id']>-1,['place_id','id']]
    mapping_df.to_sql('mapping_temp', conn_string, schema=schema_name,
                      if_exists='append', index=False)
    
    # write places to database
    ti.io.write_places_postgis(places, conn_string, schema=schema_name,
                               table_name="places", if_exists='append')

    # print iteration report 
    t_iter = time.time()-t_start_iter
    print('\t {}/{} users. Time for this iteration: {:.0f} seconds'.format(
            (iter_ix+1)*users_per_iteration, nb_users, t_iter))

print('Update staypoint ids')
with psycopg2.connect(conn_string) as conn2:
    cur = conn2.cursor()

    QUERY = """UPDATE {}.staypoints as sp
                SET place_id = mapping.place_id
                FROM {}.mapping_temp as mapping
                where mapping.id = sp.id
                """.format(schema_name, schema_name)
    cur.execute(QUERY)
    conn2.commit()
    
print("done")

t_total = time.time() - t_start_total
print('Total time for {} users: {:.0f} seconds'.format(nb_users,t_total))




