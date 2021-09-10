import csv
import datetime
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz
from shapely.geometry import Point
from sqlalchemy import create_engine
import trackintel as ti
from db_login import DSN  # database login information
from generate_graphs import get_staypoints, get_triplegs, get_trips, get_locations, generate_graphs, filter_user_by_number_of_days
from collections import defaultdict
import pickle
import numpy as np

engine = create_engine(
    "postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}".format(
        **DSN
    )
)
file_prefix = "210910"
study = "yumuv_graph_rep"
limit = "" #"where user_id in (5652, 5609, 4979, 5008, 6007)"
sp_ = get_staypoints(study=study, engine=engine, limit=limit)
tpls_ = get_triplegs(study=study, engine=engine, limit=limit)
trips_ = get_trips(study=study, engine=engine, limit=limit)
locs_ = get_locations(study=study, engine=engine, limit=limit)


user_dates_kg = pd.read_sql("select * from henry_dev.user_dates where abo_start is not null", con=engine, index_col="user_id")

sp = sp_.reset_index().set_index(['user_id', 'started_at'], drop=False).sort_index()
tpls = tpls_.reset_index().set_index(['user_id', 'started_at'], drop=False).sort_index()
trips = trips_.reset_index().set_index(['user_id', 'started_at'], drop=False).sort_index()

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

locs_before = locs_.loc[sp_before['location_id'].unique()]
locs_after = locs_.loc[sp_after['location_id'].unique()]

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

# tracking quality
print("\t\t drop users with bad coverage")
sp_before, _ = filter_user_by_number_of_days(sp=sp_before, tpls=tpls_before, coverage=0.7, min_nb_good_days=14)
sp_after, _ = filter_user_by_number_of_days(sp=sp_after, tpls=trips_after, coverage=0.7, min_nb_good_days=14)


# only take intersection of users (users that are available in the before and after set)
valid_users = np.intersect1d(sp_before.user_id.unique(), sp_after.user_id.unique(), assume_unique=True)

sp_before = sp_before[sp_before['user_id'].isin(valid_users)]
trips_before = trips_before[trips_before['user_id'].isin(valid_users)]
locs_before = locs_before[locs_before['user_id'].isin(valid_users)]

sp_after = sp_after[sp_after['user_id'].isin(valid_users)]
trips_after = trips_after[trips_after['user_id'].isin(valid_users)]
locs_after = locs_after[locs_after['user_id'].isin(valid_users)]




# generate graphs
AG_dict_before = generate_graphs(locs=locs_before, sp=sp_before, study=study, trips=trips_before, plotting=False)
AG_dict_after = generate_graphs(locs=locs_after, sp=sp_after, study=study, trips=trips_after, plotting=False)

# generate output
GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)

out_name_before = open(os.path.join(GRAPH_OUTPUT, "counts_full_before_{}.pkl".format(file_prefix)), "wb")
out_name_after = open(os.path.join(GRAPH_OUTPUT, "counts_full_after_{}.pkl".format(file_prefix)), "wb")
pickle.dump(AG_dict_before, out_name_before)
pickle.dump(AG_dict_after, out_name_after)
out_name_after.close()
out_name_before.close()

# test if reading works
pkl_name_before = open(os.path.join(GRAPH_OUTPUT, "counts_full_before_{}.pkl".format(file_prefix)), "rb")
pkl_name_after = open(os.path.join(GRAPH_OUTPUT, "counts_full_after_{}.pkl".format(file_prefix)), "rb")
AG_dict_before2 = pickle.load(pkl_name_before)
AG_dict_after2 = pickle.load(pkl_name_after)

nb_users_tg = np.sum(user_dates_kg.index.isin(AG_dict_after.keys()))
nb_users_cg = len(AG_dict_after.keys()) - nb_users_tg
print("\t nb of users treatment group: ", nb_users_tg, "nb of users control group: ", nb_users_cg)
print("\t nb of users treatment group before filtering: ", len(user_dates_kg))