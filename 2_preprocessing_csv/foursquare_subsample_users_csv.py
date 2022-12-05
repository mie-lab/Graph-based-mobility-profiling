"""
Script to apply different subsample methods to the foursquare dataset.
"""

import os

import pandas as pd
import geopandas as gpd
import argparse
import trackintel as ti
from subsampled_users import valid_user_ids


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, default=os.path.join("data", "raw"))
    args = parser.parse_args()

    schema_name = "foursquare"
    data_path = args.data_path

    print("\t read staypoints")
    sp = ti.io.read_staypoints_csv(os.path.join(data_path, schema_name, "staypoints.csv"), geom_col="geom",
                                   index_col="id", columns={"geometry": "geom"})

    print("\t read locations")
    locs = ti.io.read_locations_csv(os.path.join(data_path, schema_name, "locations.csv"), index_col="id")

    for study in ["fs_toph100", "fs_top100"]:
        print("\t", study)
        valid_user_ids_study = valid_user_ids[study]
        valid_user_ids_study = list(map(int, valid_user_ids_study))

        sp2 = sp[sp["user_id"].isin(valid_user_ids_study)]
        locs2 = locs[locs["user_id"].isin(valid_user_ids_study)]

        out_path = os.path.join(data_path, schema_name, study)
        os.makedirs(out_path, exist_ok=True)
        print("\t \t write")
        ti.io.write_staypoints_csv(sp2, os.path.join(out_path, "staypoints.csv"))
        ti.io.write_locations_csv(locs2, os.path.join(out_path, "locations.csv"))
    print("done")