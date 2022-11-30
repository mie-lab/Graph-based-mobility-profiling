import os
import argparse
import pickle
import trackintel as ti

from general_utils import generate_graphs, filter_user_by_number_of_days

CRS_WGS84 = "epsg:4326"

# public studies
studies = ["tist_toph100", "tist_top100", "geolife"]

limit = ""
single_user = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, default=os.path.join("data", "raw"))
    parser.add_argument("-o", "--out_path", type=str, default=os.path.join("data", "graph_data"))
    args = parser.parse_args()

    for study in studies:
        data_path = os.path.join(data_path, study)
        print("start {}".format(study))

        # define output for graphs
        GRAPH_OUTPUT = os.path.join(args.out_path, study)
        if not os.path.exists(GRAPH_OUTPUT):
            os.mkdir(GRAPH_OUTPUT)

        # download data
        print("\t download staypoints")
        sp = ti.io.read_staypoints_csv(os.path.join(data_path, "staypoins.csv"))

        print("\t download locations")
        sp = ti.io.read_locations_csv(os.path.join(data_path, "locations.csv"))

        if "tist" not in study:

            print("\t download triplegs")
            tpls = ti.io.read_triplegs_csv(os.path.join(data_path, "triplegs.csv"))

            print("\t download trips")
            trips = ti.io.read_trips_csv(os.path.join(data_path, "trips.csv"))

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

        out_name = open(os.path.join(GRAPH_OUTPUT, "counts_full.pkl"), "wb")
        pickle.dump(AG_dict, out_name)
        out_name.close()
