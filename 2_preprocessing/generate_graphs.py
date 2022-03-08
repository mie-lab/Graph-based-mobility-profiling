import os
import pickle
from graph_trackintel.activity_graph import ActivityGraph
import copy
from general_utils import (
    get_engine,
    get_staypoints,
    get_locations,
    get_triplegs,
    get_trips,
    write_graphs_to_postgresql,
    read_graphs_from_postgresql,
    filter_user_by_number_of_days,
)
from tqdm import tqdm

CRS_WGS84 = "epsg:4326"


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


# globals
# study name is used as schema name in database
# studies = ["gc2", "gc1", "geolife", "yumuv_graph_rep"]
studies = ["tist_toph10"]  # ['tist_toph100', 'tist_random100']
# #, 'tist_toph10', 'tist_top100', 'tist_toph100', 'tist_top500', 'tist_toph500', 'tist_top1000', 'tist_toph1000']

limit = ""
single_user = False

if __name__ == "__main__":

    for study in studies:
        print("start {}".format(study))

        # define output for graphs
        GRAPH_OUTPUT = os.path.join(".", "data_out", "graph_data", study)
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
            AG_dict = generate_graphs(locs=locs, sp=sp, study=study, trips=trips, plot_spring=True, plot_coords=False)

        else:
            print("\tgenerate full graphs (transition counts)\n")
            AG_dict = generate_graphs(
                locs=locs, sp=sp, study=study, plot_spring=True, plot_coords=False, gap_threshold=12
            )

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
                graph_table_name="full_graph",
                psycopg_con=con,
                graph_schema_name=study,
                file_name="graph_data",
                decompress=True,
            )
