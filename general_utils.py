import geopandas as gpd
import pandas as pd
import trackintel as ti
from sqlalchemy import create_engine
import os
import json
import sys
import psycopg2
import pickle
import zlib
from psycopg2 import sql
from trackintel.analysis.tracking_quality import _split_overlaps
from tqdm import tqdm
from graph_trackintel.activity_graph import ActivityGraph
import copy


def filter_user_by_number_of_days(sp, tpls, coverage=0.7, min_nb_good_days=28, filter_sp=True):
    """

    Parameters
    ----------
    sp
    tpls
    coverage
    min_nb_good_days
    filter_sp

    Returns
    -------

    """
    # could be replaced by https://github.com/mie-lab/trackintel/issues/258 once implemented
    nb_users = len(sp.user_id.unique())

    sp_tpls = sp.append(tpls).sort_values(["user_id", "started_at"])

    coverage_df = ti.analysis.tracking_quality.temporal_tracking_quality(sp_tpls, granularity="day", max_iter=1000)

    good_days_count = coverage_df[coverage_df["quality"] >= coverage].groupby(by="user_id")["quality"].count()
    good_users = good_days_count[good_days_count >= min_nb_good_days].index
    if filter_sp:
        sp = sp[sp.user_id.isin(good_users)]
        print("\t\t nb users now: ", len(sp.user_id.unique()), "before: ", nb_users)
    return sp, good_users


def filter_days_with_bad_tracking_coverage(sp, tpls, coverage=0.99):
    """

    Parameters
    ----------
    sp
    tpls
    coverage

    Returns
    -------

    """
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

    # delete days with low tracking quality
    sp["started_at_day"] = pd.to_datetime(sp["started_at"].dt.date, utc=True)
    sp = sp.set_index(["user_id", "started_at_day"], drop=False)
    to_del_ix = tracking_quality[tracking_quality < coverage].index
    nb_sp_old = sp.shape[0]
    sp.drop(sp.index.intersection(to_del_ix), axis=0, inplace=True)
    sp.set_index("id", drop=True, inplace=True)
    print("\t nb dropped: ", nb_sp_old - sp.shape[0], "nb kept: ", sp.shape[0])
    return sp


def get_engine(study, return_con=False):
    """Crete a engine object for database connection

    study: Used to specify the database for the connection. "yumuv_graph_rep" directs to sbb internal database
    return_con: Boolean
        if True, a psycopg connection object is returned
    """
    if study == "yumuv_graph_rep":
        sys.path.append(r"C:\Users\e527371\OneDrive\Programming\yumuv")
        from db_login import DSN  # database login information

        engine = create_engine("postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}".format(**DSN))
        if return_con:
            con = psycopg2.connect(
                dbname=DSN["db_database"],
                user=DSN["db_user"],
                password=DSN["db_password"],
                host=DSN["db_host"],
                port=DSN["db_port"],
            )
    else:
        # build database login string from file
        DBLOGIN_FILE = os.path.join("./dblogin.json")
        with open(DBLOGIN_FILE) as json_file:
            LOGIN_DATA = json.load(json_file)

        conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**LOGIN_DATA)
        engine = create_engine(conn_string)
        if return_con:
            con = psycopg2.connect(
                dbname=LOGIN_DATA["database"],
                user=LOGIN_DATA["user"],
                password=LOGIN_DATA["password"],
                host=LOGIN_DATA["host"],
                port=LOGIN_DATA["port"],
            )

    if return_con:
        return con
    else:
        return engine


def sample_tist_users(nb_users, engine):
    """
    Sample nb_users from tist.
    Where statement:
    homecount: 75 percentile
    totalcount: 25 percentile
    nb_locs: 25 percentile

    returns list with user_ids
    """
    query = """select user_id from tist.user_data where
                homecount > 24 and totalcount > 81 and nb_locs > 40 
                order by random() limit {}""".format(
        nb_users
    )

    return list(pd.read_sql(query, con=engine))


def get_staypoints(study, engine, limit=""):
    """
    Download staypoints and transform to trackintel format
    """
    sp = gpd.read_postgis(
        sql="select * from {}.staypoints {}".format(study, limit),
        con=engine,
        geom_col="geom",
        index_col="id",
    )
    sp["started_at"] = pd.to_datetime(sp["started_at"], utc=True)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"], utc=True)

    return sp


def get_locations(study, engine, limit=""):
    """
    Download locations and transform to trackintel format
    """
    locs = ti.io.read_locations_postgis(
        sql="select * from {}.locations {}".format(study, limit), con=engine, center="center", index_col="id"
    )
    return locs


def get_triplegs(study, engine, limit=""):
    """
    Download triplegs and transform to trackintel format
    """
    tpls = pd.read_sql(
        sql="select id, user_id, started_at, finished_at from {}.triplegs {}".format(study, limit),
        con=engine,
        index_col="id",
    )
    tpls["started_at"] = pd.to_datetime(tpls["started_at"], utc=True)
    tpls["finished_at"] = pd.to_datetime(tpls["finished_at"], utc=True)

    return tpls


def get_trips(study, engine, limit=""):
    """
    Download trips and transform to trackintel format
    """
    trips = pd.read_sql(sql="select * from {}.trips {}".format(study, limit), con=engine, index_col="id")
    trips["started_at"] = pd.to_datetime(trips["started_at"], utc=True)
    trips["finished_at"] = pd.to_datetime(trips["finished_at"], utc=True)

    return trips


def generate_graphs(
    locs,
    sp,
    study,
    trips=None,
    gap_threshold=None,
    plot_spring=True,
    plot_coords=True,
    output_dir=os.path.join(".", "graph_images", "new"),
):
    """
    Wrapper function around graph-trackintel.ActivityGraph to create person specific graphs.

    Implements a per-user iteration, quality checks and adds dataset specific features to the activity graph.

    Parameters
    ----------
    locs: trackintel locations
    sp: trackintel staypoints
    study: str
        name of study
    trips: trackintel trips
        optional input but if provided, activity graphs are created based on trips
    gap_threshold: float
        Maximum time in hours between the start of two staypoints so that they are still considered consecutive.
        Only relevant when trips are not provided
    plot_spring: boolean
        If true a visualization using spring layout will be stored in output_dir
    plot_coords: boolean
        If true a visualization using coordinate layout will be stored in output_dir

    Returns
    -------

    """
    AG_dict = {}

    # loop by user
    for user_id_this in tqdm(locs["user_id"].unique()):
        sp_user = sp[sp["user_id"] == user_id_this]
        if sp_user.empty:
            continue
        locs_user = locs[locs["user_id"] == user_id_this]

        # if trips are provided they are used to create the activity graph
        if trips is not None:
            trips_user = trips[trips["user_id"] == user_id_this]
            if trips_user.empty:
                continue
            AG = ActivityGraph(sp_user, locs_user, trips=trips_user, gap_threshold=gap_threshold)
        else:
            AG = ActivityGraph(sp_user, locs_user, gap_threshold=gap_threshold)

        if study == "geolife":
            AG.add_node_features_from_staypoints(sp, agg_dict={"started_at": list, "finished_at": list})
        else:
            AG.add_node_features_from_staypoints(
                sp, agg_dict={"started_at": list, "finished_at": list, "purpose": list}
            )

        if plot_spring:
            AG.plot(
                os.path.join(output_dir, study, "spring", str(user_id_this)),
                filter_node_importance=25,
                draw_edge_label=False,
            )

        if plot_coords:
            AG.plot(
                os.path.join(output_dir, study, "coords", str(user_id_this)),
                filter_node_importance=25,
                draw_edge_label=False,
                layout="coordinate",
            )

        AG_dict[user_id_this] = copy.deepcopy(AG)

    return AG_dict
