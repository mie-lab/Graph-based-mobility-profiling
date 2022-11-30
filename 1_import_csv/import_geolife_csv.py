# -*- coding: utf-8 -*-
"""
This script reads in the geolife data (as it can be downloaded from
https://www.microsoft.com/en-us/download/details.aspx?id=52367) and loads it
in a postgis database
"""

import os
import pandas as pd
import trackintel as ti
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_path", type=str, required=True)
parser.add_argument("-o", "--out_path", type=str, default=os.path.join("data", "raw", "geolife"))
args = parser.parse_args()

out_path = args.out_path
os.makedirs(out_path, exist_ok=True)

geolife_path = args.data_path

pfs, mode_labels = ti.io.dataset_reader.read_geolife(geolife_path, print_progress=True)

print("extract staypoints")
pfs, spts = pfs.as_positionfixes.generate_staypoints(
    gap_threshold=24 * 60, include_last=True, print_progress=True, dist_threshold=200, time_threshold=30, n_jobs=4
)

print("extract triplegs")
pfs, tpls = pfs.as_positionfixes.generate_triplegs(spts, method="between_staypoints", gap_threshold=25)

print("attach labels to triplegs")
tpls = ti.io.dataset_reader.geolife_add_modes_to_triplegs(tpls, mode_labels)

# drop entries with invalid timestamps
valid_tstamp_flag_sp = spts.started_at <= spts.finished_at
valid_tstamp_flag_tpls = tpls.started_at <= tpls.finished_at
print(
    "\t\ttimestamps of {} sp and {} tpls corrupted. corrupted ts are dropped".format(
        spts.shape[0] - sum(valid_tstamp_flag_sp), tpls.shape[0] - sum(valid_tstamp_flag_tpls)
    )
)

sp = spts[valid_tstamp_flag_sp]
tpls = tpls[valid_tstamp_flag_tpls]
# filter staypoints that are too long
duration_too_long = (sp["finished_at"] - sp["started_at"]) > pd.Timedelta("20h")
sp = sp[(sp["finished_at"] - sp["started_at"]) < pd.Timedelta("20")]


print("extract locations")
spts, locs = spts.as_staypoints.generate_locations(
    method="dbscan", epsilon=30, num_samples=1, distance_metric="haversine", agg_level="user"
)

spts = ti.analysis.location_identifier(spts, method="FREQ", pre_filter=True)

print("add activity flag to staypoints")
spts = spts.as_staypoints.create_activity_flag(method="time_threshold", time_threshold=25)

print("extract trips")
spts, triplegs, trips = ti.preprocessing.triplegs.generate_trips(spts, tpls, gap_threshold=25)

print("write to database")

print("write positionfixes to database")
ti.io.write_positionfixes_csv(pfs, os.path.join(out_path, "positionfixes"))

print("write staypoints to database")
ti.io.write_staypoints_csv(spts, os.path.join(out_path, "staypoints"))

print("write triplegs to database")
ti.io.write_triplegs_csv(tpls, os.path.join(out_path, "triplegs"))

print("write trips to database")
trips.as_trips.to_csv(os.path.join(out_path, "trips"))

print("write locations to database")
locs = locs.drop("extent", axis=1)
locs.as_locations.to_csv(os.path.join(out_path, "locations"))
