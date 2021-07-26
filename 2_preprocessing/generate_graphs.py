import geopandas as gpd
import pandas as pd
import trackintel as ti
from sqlalchemy import create_engine
from future_trackintel import tigraphs
import numpy as np
import os
import pickle
import ntpath
import json
import datetime
import numpy as np
import sys
from future_trackintel.activity_graph import activity_graph
import copy
from trackintel.analysis.tracking_quality import _split_overlaps
from collections import defaultdict
import pytz

CRS_WGS84 = "epsg:4326"


def get_engine(study):

    if study == "yumuv_graph_rep":
        sys.path.append(r"C:\Users\e527371\OneDrive\Programming\yumuv")
        from db_login import DSN  # database login information

        engine = create_engine("postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}".format(**DSN))

    else:

        # build database login string from file
        DBLOGIN_FILE = os.path.join("./dblogin.json")
        with open(DBLOGIN_FILE) as json_file:
            LOGIN_DATA = json.load(json_file)

        conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA)
        engine = create_engine(conn_string)

    return engine


def get_staypoints(study, engine):
    sp = gpd.read_postgis(
        sql="select * from {}.staypoints {}".format(study, limit), con=engine, geom_col="geom", index_col="id",
    )
    sp["started_at"] = pd.to_datetime(sp["started_at"], utc=True)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"], utc=True)

    return sp


def get_triplegs(study, engine):
    tpls = pd.read_sql(
        sql="select id, user_id, started_at, finished_at from {}.triplegs {}".format(study, limit),
        con=engine,
        index_col="id",
    )
    tpls["started_at"] = pd.to_datetime(tpls["started_at"], utc=True)
    tpls["finished_at"] = pd.to_datetime(tpls["finished_at"], utc=True)
    return tpls


def horizontal_merge_staypoints(sp):
    """merge staypoints that are consecutive at the same place"""
    # merge consecutive staypoints

    sp_merge = sp.copy()
    assert sp_merge.index.name == "id", "expected index name to be 'id'"

    sp_merge = sp_merge.reset_index()
    sp_merge.sort_values(inplace=True, by=["user_id", "started_at"])
    sp_merge[["next_started_at", "next_location_id"]] = sp_merge[["started_at", "location_id",]].shift(-1)
    cond = pd.Series(data=False, index=sp_merge.index)
    cond_old = pd.Series(data=True, index=sp_merge.index)
    cond_diff = cond != cond_old

    while np.sum(cond_diff) >= 1:
        # .values is important otherwise the "=" would imply a join via the new index
        sp_merge["next_id"] = sp_merge["id"].shift(-1).values

        # identify rows to merge
        cond1 = sp_merge["next_started_at"] - sp_merge["finished_at"] < datetime.timedelta(minutes=10)
        cond2 = sp_merge["location_id"] == sp_merge["next_location_id"]
        cond3 = sp_merge["location_id"] == sp_merge["next_location_id"]
        cond = cond1 & cond2 & cond3

        # assign index to next row
        sp_merge.loc[cond, "id"] = sp_merge.loc[cond, "next_id"]
        cond_diff = cond != cond_old
        cond_old = cond.copy()

        print(np.sum(cond_diff))

    # aggregate values
    sp_merged = sp_merge.groupby(by="id").agg(
        {
            "id": "first",
            "index": "first",
            "user_id": "first",
            "trip_id": list,
            "prev_trip_id": list,
            "next_trip_id": list,
            "started_at": "first",
            "finished_at": "last",
            "geom": "first",
            "elevation": "first",
            "location_id": "first",
            "activity": "first",
            "purpose": list
            # "purpose_detected": list,
            # "purpose_validated": list,
            # "validated": "first",
            # "validated_at": "first",
        }
    )

    return sp_merged


def filter_by_tracking_coverage(sp, tpls, coverage=0.99):

    # could be replaced by https://github.com/mie-lab/trackintel/issues/258 once implemented

    # filter by tracking quality
    sp = _split_overlaps(sp, granularity="day")
    sp_tpls = sp.append(tpls)
    sp_tpls = _split_overlaps(sp_tpls.reset_index(), granularity="day")
    # get the tracked day relative to the first day
    sp_tpls["duration"] = sp_tpls["finished_at"] - sp_tpls["started_at"]
    sp_tpls.set_index("started_at", inplace=True)
    sp_tpls.index.name = "started_at_day"

    # calculate daily tracking quality
    sp_tpls_grouper = sp_tpls.groupby(["user_id", pd.Grouper(freq="D")])
    tracking_quality = sp_tpls_grouper["duration"].sum() / datetime.timedelta(days=1)

    # np.sum(tracking_quality < 0.99)/tracking_quality.size
    # delete days with low tracking quality
    sp["started_at_day"] = pd.to_datetime(sp["started_at"].dt.date, utc=True)
    sp = sp.set_index(["user_id", "started_at_day"], drop=False)
    to_del_ix = tracking_quality[tracking_quality < coverage].index

    sp.drop(sp.index.intersection(to_del_ix), axis=0, inplace=True)
    sp.set_index("id", drop=True, inplace=True)
    return sp


def generate_graphs(locs, sp, out_name, plotting=False):
    AG_dict = {}

    for user_id_this in locs["user_id"].unique():
        sp_user = sp[sp["user_id"] == user_id_this]
        if sp_user.empty:
            continue
        locs_user = locs[locs["user_id"] == user_id_this]
        AG = activity_graph(sp_user, locs_user)
        if plotting:
            AG.plot(
                os.path.join(".", "graph_images", "new", study, "spring", str(user_id_this)),
                filter_node_importance=25,
                draw_edge_label=False,
            )
            AG.plot(
                os.path.join(".", "graph_images", "new", study, "coords", str(user_id_this)),
                filter_node_importance=25,
                draw_edge_label=False,
                layout="coordinate",
            )
        AG_dict[user_id_this] = copy.deepcopy(AG)

    # create graphs of the full period
    print("\t create full graph with counts")
    pickle.dump(AG_dict, out_name)


def generate_graphs_daily(locs, sp, out_name, plotting=False):
    # create daily graphs
    print("create index")
    sp.set_index("started_at", drop=False, inplace=True)
    sp.index.name = "started_at_ix"
    sp.sort_index(inplace=True)

    AG_dict = defaultdict(dict)

    sp_grouper = sp.groupby(["user_id", pd.Grouper(freq="D")])
    i = 0

    for (user, day), sp_group in sp_grouper:

        # relevant location ids:
        relevant_locs = sp_group["location_id"].unique()
        locs_this = locs[locs.index.isin(relevant_locs)]

        if locs_this.size == 0 or sp_group.empty:
            continue

        AG = activity_graph(sp_group, locs_this)
        if plotting:
            AG.plot(
                os.path.join(".", "graph_images", "new", "daily", user, "spring", day.strftime("%Y-%m-%d"),),
                draw_edge_label=True,
            )
            AG.plot(
                os.path.join(".", "graph_images", "new", "daily", user, "coords", day.strftime("%Y-%m-%d"),),
                layout="coordinate",
            )

        AG_dict[user][day] = copy.deepcopy(AG)
        i = i + 1
        if i % 500 == 0:
            print(user, day)

    pickle.dump(AG_dict, out_name)


# globals
# study name is used as schema name in database
studies = ["yumuv_graph_rep"]  # , 'gc1']  # , 'geolife',]# 'tist_u1000', 'tist_b100', 'tist_b200', 'tist_u10000']
limit = ""
single_user = False

if __name__ == "__main__":

    for study in studies:
        print("start {}".format(study))

        # define output for graphs
        GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)
        GRAPH_FOLDER, _ = ntpath.split(GRAPH_OUTPUT)
        if not os.path.exists(GRAPH_FOLDER):
            os.mkdir(GRAPH_FOLDER)

        engine = get_engine(study)

        # download data
        print("\t download staypoints")
        sp = get_staypoints(study=study, engine=engine)
        print("\t download triplegs")
        tpls = get_triplegs(study=study, engine=engine)
        print("\t download locations")
        locs = ti.io.read_locations_postgis(
            sql="select * from {}.locations".format(study), con=engine, geom_col="center",
        )

        if single_user:
            sp = sp[sp["user_id"] == "0d45bbc4-6c61-44a0-a7aa-d69311e3db40"]

        sp_merged = horizontal_merge_staypoints(sp)
        sp = filter_by_tracking_coverage(sp=sp_merged, tpls=tpls, coverage=0.99)

        print("generate full graphs (transition counts)")
        generate_graphs(locs=locs, sp=sp, out_name=open(GRAPH_OUTPUT + "_counts_full.pkl", "wb"))
        print("generate daily graphs (transition counts)")
        generate_graphs_daily(locs=locs, sp=sp, out_name=open(GRAPH_OUTPUT + "_daily_graphs.pkl", "wb"))
