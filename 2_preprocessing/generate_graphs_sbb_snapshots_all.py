"""Create individual mobility graphs based on temporal subsets of a tracking study"""
import os

import pandas as pd

from general_utils import (
    get_engine,
    get_staypoints,
    get_locations,
    get_triplegs,
    get_trips,
    filter_user_by_number_of_days,
    generate_graphs,
)
from graph_trackintel.io import read_graphs_from_postgresql, write_graphs_to_postgresql

CRS_WGS84 = "epsg:4326"

studies = ["gc1", "gc2"]
limit = ""

if __name__ == "__main__":

    for study in studies:

        print("start {}".format(study))

        # define output for graphs
        GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)
        if not os.path.exists(GRAPH_OUTPUT):
            os.mkdir(GRAPH_OUTPUT)

        engine = get_engine(study)

        min_date = (
            get_staypoints(study=study, engine=engine, limit="order by started_at limit 1").iloc[0].loc["started_at"]
        )
        max_date = get_staypoints(study=study, engine=engine, limit="order by started_at desc limit 1")[
            "started_at"
        ].iloc[0]
        duration_study = max_date - min_date

        # the window_granularity variable determines the step-size for the different subsets. E.g, a
        # window_granularity of 28 days = 4 weeks means that the subsets will be 4, 8, 12, 16, ... weeks long
        # a windows_granularity of 14 days = 2 weeks would mean that the subsets would be 2, 4, 6, 8, ... weeks long
        window_granularity = 28
        max_window_length = int((duration_study.days / 2) // window_granularity)

        for window_length in range(max_window_length):
            print("window length: ", window_length)
            duration_this = pd.Timedelta("{} d".format(window_granularity)) * (window_length + 1)

            nb_intervals = duration_study // duration_this

            for i in range(nb_intervals):

                limit = "where started_at >= '{:%Y-%m-%d}' and started_at < '{:%Y-%m-%d}'".format(
                    min_date + i * duration_this, min_date + (i + 1) * duration_this
                )

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
                # because we filtered the data it can now be that the origin of a trip is no longer in the set
                # of staypoints. This would cause an error in the graph generation and these ids have to be set to nan
                origin_missing = ~trips["origin_staypoint_id"].isin(sp.index)
                destination_missing = ~trips["destination_staypoint_id"].isin(sp.index)
                trips.loc[origin_missing, "origin_staypoint_id"] = pd.NA
                trips.loc[destination_missing, "destination_staypoint_id"] = pd.NA

                print("\t filter by tracking coverage")
                # note: The set of users in all graphs is not necessarily the same
                sp, user_id_ix = filter_user_by_number_of_days(
                    sp=sp, tpls=tpls, coverage=0.7, min_nb_good_days=int(duration_this.days * 0.33)
                )
                print("\t\t drop users with bad coverage")
                tpls = tpls[tpls.user_id.isin(user_id_ix)]
                trips = trips[trips.user_id.isin(user_id_ix)]
                locs = locs[locs.user_id.isin(user_id_ix)]

                print("\tgenerate full graphs (transition counts)")
                AG_dict = generate_graphs(
                    locs=locs, sp=sp, study=study, trips=trips, plot_spring=False, plot_coords=False
                )

                # store graphs in DB
                con = get_engine(study, return_con=True)

                print("\t write graph to db")
                if i == 0:
                    write_graphs_to_postgresql(
                        graph_data=AG_dict,
                        graph_table_name="dur_" + str(duration_this.days // 7) + "w",
                        graph_schema_name=study,
                        psycopg_con=con,
                        file_name="{:%Y-%m-%d}".format(min_date + i * duration_this),
                        drop_and_create=True,
                    )
                else:
                    write_graphs_to_postgresql(
                        graph_data=AG_dict,
                        graph_table_name="dur_" + str(duration_this.days // 7) + "w",
                        graph_schema_name=study,
                        psycopg_con=con,
                        file_name="{:%Y-%m-%d}".format(min_date + i * duration_this),
                        drop_and_create=False,
                    )

                print("\t test reading from db")
                AG_dict2 = read_graphs_from_postgresql(
                    graph_table_name="quarters",
                    psycopg_con=con,
                    graph_schema_name=study,
                    file_name=quarter,
                    decompress=True,
                )
