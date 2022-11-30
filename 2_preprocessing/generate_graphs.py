import os
import pickle
from general_utils import (
    get_engine,
    get_staypoints,
    get_locations,
    get_triplegs,
    get_trips,
    write_graphs_to_postgresql,
    read_graphs_from_postgresql,
    filter_user_by_number_of_days,
    generate_graphs,
)

CRS_WGS84 = "epsg:4326"

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
