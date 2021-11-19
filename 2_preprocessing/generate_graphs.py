import pandas as pd
import trackintel as ti
import os
import pickle
import datetime
from future_trackintel.activity_graph import activity_graph
import copy
from trackintel.analysis.tracking_quality import _split_overlaps
from utils import get_engine, get_staypoints,get_locations, get_triplegs, \
                    get_trips, write_graphs_to_postgresql, read_graphs_from_postgresql
from tqdm import tqdm

CRS_WGS84 = "epsg:4326"

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


def generate_graphs(locs, sp, study, trips=None, plotting=False, gap_threshold=None):
    """

    Parameters
    ----------
    locs
    sp
    study
    trips
    plotting
    gap_threshold

    Returns
    -------

    """
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
            AG = activity_graph(sp_user, locs_user, trips=trips_user, gap_threshold=gap_threshold)
        else:
            AG = activity_graph(sp_user, locs_user,  gap_threshold=gap_threshold)

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


# def generate_graphs_daily(locs, sp, out_name, study, trips=None, plotting=False):
#     # create daily graphs
#     print("create index")
#     sp.set_index("started_at", drop=False, inplace=True)
#     sp.index.name = "started_at_ix"
#     sp.sort_index(inplace=True)
#
#     AG_dict = defaultdict(dict)
#
#     sp_grouper = sp.groupby(["user_id", pd.Grouper(freq="D")])
#     i = 0
#
#     for (user, day), sp_group in sp_grouper:
#
#         # relevant location ids:
#         relevant_locs = sp_group["location_id"].unique()
#         locs_this = locs[locs.index.isin(relevant_locs)]
#
#         if locs_this.size == 0 or sp_group.empty:
#             continue
#
#         AG = activity_graph(sp_group, locs_this)
#         if study == "geolife":
#             AG.add_node_features_from_staypoints(sp_group, agg_dict={"started_at": list, "finished_at": list})
#         else:
#             AG.add_node_features_from_staypoints(
#                 sp_group, agg_dict={"started_at": list, "finished_at": list, "purpose": list}
#             )
#
#         if plotting:
#             AG.plot(
#                 os.path.join(
#                     ".",
#                     "graph_images",
#                     "new",
#                     "daily",
#                     user,
#                     "spring",
#                     day.strftime("%Y-%m-%d"),
#                 ),
#                 draw_edge_label=True,
#             )
#             AG.plot(
#                 os.path.join(".", "graph_images", "new", "daily", user, "coords", day.strftime("%Y-%m-%d"),),
#                 layout="coordinate",
#             )
#
#         AG_dict[user][day] = copy.deepcopy(AG)
#         i = i + 1
#         if i % 500 == 0:
#             print(user, day)
#
#     pickle.dump(AG_dict, out_name)


# globals
# study name is used as schema name in database
# studies = ["gc2", "gc1", "geolife"]
# studies = ['geolife']
#studies = ["yumuv_graph_rep"]
studies = ['tist_toph10'] #['tist_toph100', 'tist_random100'] #, 'tist_toph10', 'tist_top100', 'tist_toph100', 'tist_top500',
# 'tist_toph500',
#                      'tist_top1000', 'tist_toph1000']
# limit = "where user_id > 1670"
limit = ""
single_user = False

if __name__ == "__main__":

    for study in studies:
        print("start {}".format(study))

        # define output for graphs
        GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)
        # GRAPH_FOLDER, _ = ntpath.split(GRAPH_OUTPUT)
        if not os.path.exists(GRAPH_OUTPUT):
            os.mkdir(GRAPH_OUTPUT)

        engine = get_engine(study)

        # download data
        print("\t download staypoints")
        sp = get_staypoints(study=study, engine=engine)

        print("\t download locations")
        locs = get_locations(study=study, engine=engine)

        if "tist" not in study:

            print("\t download triplegs")
            tpls = get_triplegs(study=study, engine=engine)

            print("\t download trips")
            trips = get_trips(study=study, engine=engine)

            print("\t filter by tracking coverage")
            if study == "geolife":
                sp, user_id_ix = filter_user_by_number_of_days(sp=sp, tpls=tpls, coverage=0.7, min_nb_good_days=14)
            else:
                sp, user_id_ix = filter_user_by_number_of_days(sp=sp, tpls=tpls, coverage=0.7, min_nb_good_days=14)
            print("\t\t drop users with bad coverage")
            tpls = tpls[tpls.user_id.isin(user_id_ix)]
            trips = trips[trips.user_id.isin(user_id_ix)]
            locs = locs[locs.user_id.isin(user_id_ix)]

            print("\tgenerate full graphs (transition counts)")
            AG_dict = generate_graphs(locs=locs, sp=sp, study=study, trips=trips, plotting=True)

        else:
            # exclude_purpose = ['Light Rail', 'Subway', 'Platform', 'Trail', 'Road', 'Train', 'Bus Line']
            # sp = sp[~sp['purpose'].isin(exclude_purpose)]
            # a = pd.DataFrame(sp.groupby('purpose').size().sort_values())
            print("\tgenerate full graphs (transition counts)\n")
            AG_dict = generate_graphs(locs=locs, sp=sp, study=study, plotting=True, gap_threshold=12)

        con = get_engine(study, return_con=True)

        out_name = open(os.path.join(GRAPH_OUTPUT, "counts_full.pkl"), "wb")
        pickle.dump(AG_dict, out_name)
        out_name.close()

        print("\t write graph to db")
        if study == "yumuv_graph_rep":
            pass
        else:
            write_graphs_to_postgresql(
                graph_data=AG_dict,
                graph_table_name="full_graph",
                graph_schema_name=study,
                psycopg_con=con,
                file_name="graph_data",
                drop_and_create=True,
            )

            print("\t test reading from db")
            AG_dict2 = read_graphs_from_postgresql(
                graph_table_name="full_graph", psycopg_con=con, graph_schema_name=study, file_name="graph_data", decompress=True
            )
