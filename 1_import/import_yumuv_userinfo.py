import csv
import datetime
import logging
import os
import sys
import geopandas as gpd
import pandas as pd
import pytz
from shapely.geometry import Point
from sqlalchemy import create_engine
import trackintel as ti
from trackintel.preprocessing.triplegs import generate_trips
import pickle

sys.path.append(r"C:\Users\e527371\OneDrive\Programming\yumuv")
from db_login import DSN  # database login information
import numpy as np
from future_trackintel.utils import horizontal_merge_staypoints

engine = create_engine("postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}".format(**DSN))
#
data_folder = os.path.join("C:/", "yumuv", "data")  # todo move to config file
cache_folder = os.path.join(data_folder, "cache")  # todo move to config file

user_info_sql = """SELECT distinct on (app_user_id) app_user_id,
code, study_id, p_ptmobtool_2 AS ga1, p_ptmobtool_3 AS ga2, p_ptmobtool_4 AS ht
FROM raw_myway.study_code
left join henry_dev.eth_survey on p_id = code"""

user_info = pd.read_sql(user_info_sql, con=engine)
out_name = open(os.path.join(data_folder, "yumuv_userinfo.pkl"), "wb")
pickle.dump(user_info, out_name)
