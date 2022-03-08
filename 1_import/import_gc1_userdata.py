"""Download user data from main database (commitdb) to the project database (graph_rep)"""

import pandas as pd
import os
import json

CRS_WGS84 = "epsg:4326"
#
DBLOGIN_FILE = os.path.join(".", "dblogin.json")
DBLOGIN_FILE_SOURCE = os.path.join(".", "dblogin_source.json")


with open(DBLOGIN_FILE) as json_file:
    LOGIN_DATA = json.load(json_file)

with open(DBLOGIN_FILE_SOURCE) as json_file_source:
    LOGIN_DATA_SOURCE = json.load(json_file_source)

# build database login string from file
conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA)
conn_string_source = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA_SOURCE)

userdata = pd.read_sql("select * from gc1.users", conn_string_source)
columns_to_drop = [
    "workaddress",
    "homeaddress",
    "work_lon",
    "work_lat",
    "home_lon",
    "home_lat",
    "homebfs",
    "workbfs",
    "work_pc1",
    "work_pc2",
    "work_pc3",
    "work_pc4",
    "pc1",
    "pc2",
    "pc3",
    "pc4",
    "geom_home",
    "geom_work",
]

userdata.drop(columns_to_drop, axis=1, inplace=True)

userdata.to_sql(name="user_info", con=conn_string, schema="gc1", index=False, if_exists="replace")
