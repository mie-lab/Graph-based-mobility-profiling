import geopandas as gpd
import pandas as pd
import trackintel as ti
from sqlalchemy import create_engine
import numpy as np
import os
import pickle
import ntpath
import json
import datetime
import numpy as np
import sys
from future_trackintel.activity_graph import activity_graph
from utils import get_engine, get_staypoints, get_locations,\
    get_triplegs, get_trips, filter_user_by_number_of_days, filter_days_with_bad_tracking_coverage

import copy
from trackintel.analysis.tracking_quality import _split_overlaps
from collections import defaultdict
import pytz
import psycopg2
from tqdm import tqdm

CRS_WGS84 = "epsg:4326"

def generate_graphs(locs, sp, study, trips=None, plotting=False):
    AG_dict = {}

    for user_id_this in tqdm(locs["user_id"].unique()):
        sp_user = sp[sp["user_id"] == user_id_this]
        if sp_user.empty:
            continue
        locs_user = locs[locs["user_id"] == user_id_this]

        if trips is not None:
            trips_user = trips[trips["user_id"] == user_id_this]
            if trips_user.empty:
                continue
            AG = activity_graph(sp_user, locs_user, trips=trips_user)
        else:
            AG = activity_graph(sp_user, locs_user)

        if study == "geolife":
            AG.add_node_features_from_staypoints(sp, agg_dict={"started_at": list, "finished_at": list})
        else:
            AG.add_node_features_from_staypoints(
                sp, agg_dict={"started_at": list, "finished_at": list, "purpose": list}
            )

        if plotting:
            AG.plot(
                os.path.join(".", "graph_images", "new", study, "spring", str(user_id_this)),
                filter_node_importance=25,
                draw_edge_label=False,
            )
            # AG.plot(
            #     os.path.join(".", "graph_images", "new", study, "coords", str(user_id_this)),
            #     filter_node_importance=25,
            #     draw_edge_label=False,
            #     layout="coordinate",
            # )
        AG_dict[user_id_this] = copy.deepcopy(AG)

    return AG_dict

studies = ["gc2"]
# studies = ['geolife']
# studies = studies + ['tist_top10', 'tist_toph10', 'tist_top100', 'tist_toph100', 'tist_top500', 'tist_toph500',
#                      'tist_top1000', 'tist_toph1000']
# limit = "where user_id < 1600"
limit = ""
single_user = False
quarters = ['quarter1', 'quarter2', 'quarter3', 'quarter4']

if __name__ == "__main__":

    for study in studies:
        print("start {}".format(study))
        # define output for graphs
        GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)
        # GRAPH_FOLDER, _ = ntpath.split(GRAPH_OUTPUT)
        if not os.path.exists(GRAPH_OUTPUT):
            os.mkdir(GRAPH_OUTPUT)

        engine = get_engine(study)

        min_date = get_staypoints(study=study, engine=engine, limit="order by started_at limit 1"
                                  ).iloc[0].loc['started_at']
        max_date = get_staypoints(study=study, engine=engine, limit="order by started_at desc limit 1")['started_at'].iloc[0]
        duration_study = max_date - min_date

        window_granularity = 28
        max_window_length = int((duration_study.days/2) // window_granularity)

        for window_length in range(max_window_length):
            print("window length: ", window_length)
            duration_this = pd.Timedelta("{} d".format(window_granularity)) * (window_length + 1)


            nb_intervals = duration_study // duration_this

            for i in range(nb_intervals):
                continue

                limit = "where started_at >= '{:%Y-%m-%d}' and started_at < '{:%Y-%m-%d}'".format(
                    min_date + i * duration_this, min_date + (i + 1) * duration_this)

                if min_date + (i + 1) * duration_this > max_date:
                    print("max date violation")
                    continue

                # download data
                print("\t download locations")
                locs = get_locations(study=study, engine=engine)
                print("\t download staypoints")
                sp = get_staypoints(study=study, engine=engine, limit=limit)
                print("\t download triplegs")
                tpls = get_triplegs(study=study, engine=engine, limit=limit)
                print("\t download trips")
                trips = get_trips(study=study, engine=engine, limit=limit)

                # postprocessing of gaps
                # because we filtered it can now be that a the origin of a trip is no longer in the set of staypoints. This would
                # cause an error in the graph generation and these ids have to be set to nan
                origin_missing = ~ trips['origin_staypoint_id'].isin(sp.index)
                destination_missing = ~trips['destination_staypoint_id'].isin(sp.index)
                trips.loc[origin_missing, 'origin_staypoint_id'] = pd.NA
                trips.loc[destination_missing, 'destination_staypoint_id'] = pd.NA

                print("\t filter by tracking coverage")
                # todo: make sure that the set of users is the same in all graphs
                sp, user_id_ix = filter_user_by_number_of_days(sp=sp, tpls=tpls, coverage=0.7, min_nb_good_days=int(
                    duration_this.days*0.33))
                print("\t\t drop users with bad coverage")
                tpls = tpls[tpls.user_id.isin(user_id_ix)]
                trips = trips[trips.user_id.isin(user_id_ix)]
                locs = locs[locs.user_id.isin(user_id_ix)]

                print("\tgenerate full graphs (transition counts)")
                AG_dict = generate_graphs(locs=locs, sp=sp, study=study, trips=trips, plotting=False)

                # store graphs in DB
                from sqlalchemy import types

                con = get_engine(study, return_con=True)

                # out_name = open(os.path.join(GRAPH_OUTPUT, "counts_full_" + quarter + ".pkl"), "wb")
                # pickle.dump(AG_dict, out_name)
                # out_name.close()

                print("\t write graph to db")
                # if i == 0:
                #     write_graphs_to_postgresql(
                #         graph_data=AG_dict,
                #         graph_table_name="dur_" + str(duration_this.days // 7) + "w",
                #         graph_schema_name=study,
                #         psycopg_con=con,
                #         file_name="{:%Y-%m-%d}".format(min_date + i * duration_this),
                #         drop_and_create=True,
                #     )
                # else:
                #     write_graphs_to_postgresql(
                #         graph_data=AG_dict,
                #         graph_table_name="dur_" + str(duration_this.days // 7) + "w",
                #         graph_schema_name=study,
                #         psycopg_con=con,
                #         file_name="{:%Y-%m-%d}".format(min_date + i * duration_this),
                #         drop_and_create=False,
                #     )

                # print("\t test reading from db")
                # AG_dict2 = read_graphs_from_postgresql(
                #     graph_table_name="quarters", psycopg_con=con, graph_schema_name=study, file_name=quarter, decompress=True
                # )
