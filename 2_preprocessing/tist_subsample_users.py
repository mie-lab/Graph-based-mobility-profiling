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
from dateutil import tz
import datetime
import psycopg2

schema_name = "tist"

dblogin_file = os.path.join("dblogin.json")
with open(dblogin_file) as json_file:
    LOGIN_DATA = json.load(json_file)

conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA)

engine = create_engine(conn_string)
conn = engine.connect()

con = psycopg2.connect(
    dbname=LOGIN_DATA["database"],
    user=LOGIN_DATA["user"],
    password=LOGIN_DATA["password"],
    host=LOGIN_DATA["host"],
    port=LOGIN_DATA["port"],
)

# https://sites.google.com/site/yangdingqi/home/foursquare-dataset
# Dingqi Yang, Daqing Zhang, Bingqing Qu. Participatory Cultural Mapping Based on Collective Behavior Data in Location Based Social Networks. ACM Trans. on Intelligent Systems and Technology (TIST), 2015. [PDF]

# load raw data
print("Reading raw data")
cur = con.cursor()

sql_index = """CREATE INDEX IF NOT EXISTS user_id_ix on tist.staypoints using btree (user_id);
                CREATE INDEX IF NOT EXISTS purpose_ix on tist.staypoints using btree (purpose);"""
cur.execute(sql_index)

con.commit()

k_list = [10, 100, 500, 1000]
for k in k_list:

    print(k)
    sql_drop = """drop table if exists tist_top{}.staypoints;
                  drop table if exists tist_top{}.locations;
                  drop table if exists tist_toph{}.staypoints;
                  drop table if exists tist_toph{}.locations;""".format(k, k, k, k)
    cur.execute(sql_drop)
    sql_schema = """CREATE SCHEMA IF NOT EXISTS tist_top{};
    CREATE SCHEMA IF NOT EXISTS tist_toph{};""".format(k, k)
    sql_sp_topk = """create table tist_top{}.staypoints as SELECT * FROM tist.staypoints WHERE staypoints.user_id in (
                                                select ordering.user_id from (
                                                                        select user_id, count(*) from tist.staypoints group by user_id
                                                                        order by count desc limit {}) as
                                                                        ordering)""".format(
        k, k
    )
    sql_loc_topk = """create table tist_top{}.locations as SELECT * FROM tist.locations WHERE
                                                locations.user_id in (
                                                select ordering.user_id from (
                                                                        select user_id, count(*) from tist.staypoints group by user_id
                                                                        order by count desc limit {}) as
                                                                        ordering)""".format(
        k, k
    )

    sql_sp_topkh = """create table tist_toph{}.staypoints as SELECT * FROM tist.staypoints WHERE staypoints.user_id in (
                                                select ordering.user_id from (
                                                                        select user_id, purpose, count(*) from tist.staypoints where purpose = 'Home (private)'
                                                    group by (user_id, purpose) order by count desc limit {}) as 
                                                                        ordering)""".format(
        k, k
    )
    sql_loc_topkh = """create table tist_toph{}.locations as SELECT * FROM tist.locations WHERE locations.user_id in (
                                                select ordering.user_id from (
                                                                        select user_id, purpose, count(*) from tist.staypoints where purpose = 'Home (private)'
                                                    group by (user_id, purpose) order by count desc limit {}) as 
                                                                        ordering)""".format(
        k, k
    )

    sql_rename = """ALTER TABLE tist_top{}.staypoints RENAME COLUMN geometry TO geom;
                    ALTER TABLE tist_top{}.staypoints RENAME COLUMN index TO id;
                    ALTER TABLE tist_toph{}.staypoints RENAME COLUMN geometry TO geom;
                    ALTER TABLE tist_toph{}.staypoints RENAME COLUMN index TO id;
            """



    cur.execute(sql_schema)
    con.commit()
    cur.execute(sql_sp_topk)
    cur.execute(sql_loc_topk)
    cur.execute(sql_sp_topkh)
    cur.execute(sql_loc_topkh)
    # cur.execute(sql_rename)
    con.commit()

con.close()