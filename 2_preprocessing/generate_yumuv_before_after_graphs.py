import csv
import datetime
import logging
import os

import geopandas as gpd
import pandas as pd
import pytz
from shapely.geometry import Point
from sqlalchemy import create_engine
import trackintel as ti
from db_login import DSN  # database login information
from generate_graphs import get_staypoints, get_triplegs, get_trips, get_locations, generate_graphs
from collections import defaultdict
import pickle
engine = create_engine(
    "postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}".format(
        **DSN
    )
)

study = "yumuv_graph_rep"
sp_ = get_staypoints(study=study, engine=engine)
tpls_ = get_triplegs(study=study, engine=engine)
trips_ = get_trips(study=study, engine=engine)
locs = get_locations(study=study, engine=engine)

user_dates_kg = pd.read_sql("select * from henry_dev.user_dates where abo_start is not null and user_id = 5040", con=engine, index_col="user_id")

sp = sp_.reset_index().set_index(['user_id', 'started_at'], drop=False).sort_index()
tpls = tpls_.reset_index().set_index(['user_id', 'started_at'], drop=False).sort_index()
trips = trips_.reset_index().set_index(['user_id', 'started_at'], drop=False).sort_index()

# dfbefore_dict = defaultdict(dict)
# dfafter_dict = defaultdict(dict)
sp_before_list = []
tpls_before_list = []
trips_before_list = []
sp_after_list = []
tpls_after_list = []
trips_after_list = []

users_without_data = []


# filter the data of every control group user by its individual start and end
for user_id_iter, row in user_dates_kg.iterrows():

    study_start = row['study_start']
    abo_start = row['abo_start']
    abo_end = row['abo_end']

    try:
        sp_before_list.append(sp.loc[(user_id_iter, slice(study_start, abo_start)), :])
        tpls_before_list.append(tpls.loc[(user_id_iter, slice(study_start, abo_start)), :])
        trips_before_list.append(trips.loc[(user_id_iter, slice(study_start, abo_start)), :])

        sp_after_list.append(sp.loc[(user_id_iter, slice(abo_start, abo_end)), :])
        tpls_after_list.append(tpls.loc[(user_id_iter, slice(abo_start, abo_end)), :])
        trips_after_list.append(trips.loc[(user_id_iter, slice(abo_start, abo_end)), :])

    except KeyError:
        users_without_data.append(user_id_iter)


sp_before_tg = pd.concat(sp_before_list, axis=0)
tpls_before_tg = pd.concat(tpls_before_list, axis=0)
trips_before_tg = pd.concat(trips_before_list, axis=0)

sp_after_tg = pd.concat(sp_after_list, axis=0)
tpls_after_tg = pd.concat(tpls_after_list, axis=0)
trips_after_tg = pd.concat(trips_after_list, axis=0)

# control group (cg)
# these are filtered by the averages of the treatment group
study_start_mean = user_dates_kg['study_start'].mean()
abo_start_mean = user_dates_kg['abo_start'].mean()
abo_end_mean = user_dates_kg['abo_end'].mean()

# filter to have only cg users
idx = pd.IndexSlice

sp_before_cg = sp.loc[idx[~sp.index.get_level_values(0).isin(user_dates_kg.index), slice(study_start_mean, abo_start_mean)], :]
tpls_before_cg = tpls.loc[idx[~tpls.index.get_level_values(0).isin(user_dates_kg.index), slice(study_start_mean, abo_start_mean)], :]
trips_before_cg = trips.loc[idx[~trips.index.get_level_values(0).isin(user_dates_kg.index), slice(study_start_mean, abo_start_mean)], :]

sp_after_cg = sp.loc[idx[~sp.index.get_level_values(0).isin(user_dates_kg.index), slice(abo_start_mean, abo_end_mean)], :]
tpls_after_cg = tpls.loc[idx[~tpls.index.get_level_values(0).isin(user_dates_kg.index), slice(abo_start_mean, abo_end_mean)], :]
trips_after_cg = trips.loc[idx[~trips.index.get_level_values(0).isin(user_dates_kg.index), slice(abo_start_mean, abo_end_mean)], :]

# merge cg + treatment group
sp_before = sp_before_tg.append(sp_before_cg, ignore_index=True).set_index('id')
tpls_before = tpls_before_tg.append(tpls_before_cg, ignore_index=True).set_index('id')
trips_before = trips_before_tg.append(trips_before_cg, ignore_index=True).set_index('id')

sp_after = sp_after_tg.append(sp_after_cg, ignore_index=True).set_index('id')
tpls_after = tpls_after_tg.append(tpls_after_cg, ignore_index=True).set_index('id')
trips_after = trips_after_tg.append(trips_after_cg, ignore_index=True).set_index('id')

locs_before = locs.loc[sp_before['location_id'].unique()]
locs_after = locs.loc[sp_after['location_id'].unique()]

# postprocessing of gaps
# because we filtered it can now be that a the origin of a trip is no longer in the set of staypoints. This would
# cause an error in the graph generation and these ids have to be set to nan
origin_missing = ~ trips_before['origin_staypoint_id'].isin(sp_before.index)
destination_missing = ~trips_before['destination_staypoint_id'].isin(sp_before.index)
trips_before.loc[origin_missing, 'origin_staypoint_id'] = pd.NA
trips_before.loc[destination_missing, 'destination_staypoint_id'] = pd.NA

origin_missing = ~trips_after['origin_staypoint_id'].isin(sp_after.index)
destination_missing = ~trips_after['destination_staypoint_id'].isin(sp_after.index)
trips_after.loc[origin_missing, 'origin_staypoint_id'] = pd.NA
trips_after.loc[destination_missing, 'destination_staypoint_id'] = pd.NA

# generate graphs
AG_dict_before = generate_graphs(locs=locs_before, sp=sp_before, study=study, trips=trips_before, plotting=True)
AG_dict_after = generate_graphs(locs=locs_after, sp=sp_after, study=study, trips=trips_after, plotting=True)

GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)
out_name_before = open(os.path.join(GRAPH_OUTPUT, "counts_full_before.pkl"), "wb")
out_name_after = open(os.path.join(GRAPH_OUTPUT, "counts_full_after.pkl"), "wb")
pickle.dump(AG_dict_before, out_name_before)
pickle.dump(AG_dict_after, out_name_after)