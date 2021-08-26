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

# user_info_sql = """SELECT distinct on (app_user_id) app_user_id,
# code, study_id, p_ptmobtool_2 AS ga1, p_ptmobtool_3 AS ga2, p_ptmobtool_4 AS ht
# FROM raw_myway.study_code
# left join henry_dev.eth_survey on p_id = code
# left join henry_dev.eth_final_survey_fk on externalreference = code
# left join henry_dev.eth_final_survey_kg on externalreference = code"""
#
# user_info = pd.read_sql(user_info_sql, con=engine)



df_initial_survey = pd.read_sql("""select distinct on(externalreference) * from 
henry_dev.eth_survey order by externalreference, progress desc""", con=engine, index_col="externalreference")

df_final_survey_kg = pd.read_sql("""select distinct on(externalreference) * from 
henry_dev.eth_final_survey_kg order by externalreference, progress desc""", con=engine, index_col="externalreference")

df_final_survey_fk = pd.read_sql("""select distinct on(externalreference) * from 
henry_dev.eth_final_survey_fk order by externalreference, progress desc""", con=engine, index_col="externalreference")


initial_drop = ['index', 'unnamed 0', 'startdate', 'enddate', 'status', 'ipaddress',
       'progress', 'duration in seconds', 'finished', 'recordeddate',
       'responseid', 'recipientlastname', 'recipientfirstname',
       'recipientemail', 'locationlatitude', 'locationlongitude',
       'distributionchannel', 'userlanguage', 'q_recaptchascore',
       'metadata_browser', 'metadata_version', 'metadata_operating system',
       'metadata_resolution', 'participation_timing_first click',
       'participation_timing_last click', 'participation_timing_page submit',
       'participation_timing_click count', 'q221', 'q222', 'q223_first click',
       'q223_last click', 'q223_page submit', 'q223_click count',
       'participation', 'q224_first click', 'q224_last click',
       'q224_page submit', 'q224_click count', 'q225_1', 'q225_2', 'q226',
       'q227', 'q743_first click', 'q743_last click', 'q743_page submit',
       'q743_click count', 'q744_1', 'q744_2', 'q745', 'q746',
       'hh_timing_p1_first click', 'hh_timing_p1_last click',
       'hh_timing_p1_page submit', 'hh_timing_p1_click count', 'q228_1',
       'q228_2', 'q228_3', 'q228_4','q295_1', 'q295_2', 'q295_5', 'q295_3', 'q295_4',
                'vorname', 'nachname', 'primaryemail', 'emailvalidation']

kg_drop = ['index', 'unnamed 0', 'startdate', 'enddate', 'status', 'ipaddress',
       'progress', 'duration in seconds', 'finished', 'recordeddate',
       'responseid', 'recipientlastname', 'recipientfirstname',
       'recipientemail', 'locationlatitude', 'locationlongitude',
       'distributionchannel', 'userlanguage', 'q_recaptchascore', 'q5_browser',
       'q5_version', 'q5_operating system', 'q5_resolution', 'q38_1', 'q38_5', 'q38_6', 'q38_7', 'q38_8', 'q38_2',
       'q38_3', 'q38_4', 'q32']

fk_drop = ['index', 'unnamed 0', 'startdate', 'enddate', 'status', 'ipaddress',
       'progress', 'duration in seconds', 'finished', 'recordeddate',
       'responseid', 'recipientlastname', 'recipientfirstname',
       'recipientemail', 'locationlatitude', 'locationlongitude',
       'distributionchannel', 'userlanguage', 'q_recaptchascore', 'q5_browser',
       'q5_version', 'q5_operating system', 'q5_resolution', 'q11', 'q31_1',
       'q31_2', 'q31_3', 'q31_4', 'q31_4_text', 'q38',
       'q58_1', 'q58_7', 'q58_2', 'q58_3', 'q58_4', 'q58_5', 'q32']


df_initial_survey.drop(initial_drop, inplace=True, axis=1)
df_final_survey_kg.drop(kg_drop, inplace=True, axis=1)
df_final_survey_fk.drop(fk_drop, inplace=True, axis=1)

df_merge_fk = df_initial_survey.merge(df_final_survey_fk, how='right', right_index=True, left_index=True)
df_merge_kg = df_initial_survey.merge(df_final_survey_kg, how='right', right_index=True, left_index=True)

df_all = df_merge_fk.append(df_merge_kg)

# add user id from user_info table
df_user_id = pd.read_sql("""select user_id, study_id, survey_code as externalreference from henry_dev.user_dates""",
                         con=engine, index_col="externalreference")

df_all = df_all.join(df_user_id, how='left')
df_all.reset_index().set_index('user_id')

out_name = open(os.path.join(data_folder, "yumuv_userinfo.pkl"), "wb")
pickle.dump(df_all, out_name)