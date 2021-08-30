from math import dist
import numpy as np
import os
import time
import json
import pandas as pd
import psycopg2
import geopandas as gpd
import skmob
from functools import reduce
import trackintel as ti
import argparse

from utils import dist_to_stats, dist_names, get_point_dist
from clustering import normalize_and_cluster
from skmob.measures.individual import *
from plotting import scatterplot_matrix


class RawFeatures:
    def __init__(self, study, trips_available=True):
        self._trips_available = trips_available
        print("Loading data...")
        self._load_data(study)
        self._tdf = self._to_skmob(self._sp, self._locations)

        # the features we aim to use
        self._default_features = [
            "number_locations",
            "real_entropy",
            "mean_trip_distance",
            "mean_trip_duration",
            "radius_of_gyration",
        ]
        self._all_features = [f for f in dir(self) if not f.startswith("_")]
        print("Available features", self._all_features)

    @staticmethod
    def _to_skmob(stps, locations):
        sp_with_locs = stps.join(locations, how="left", on="location_id", rsuffix="r")
        assert all(sp_with_locs["user_id"] == sp_with_locs["user_idr"])

        # use longitude and latiude of location, not of staypoints (e.g. for getting k most frequent locations)
        sp_with_locs["longitude"] = sp_with_locs["center"].apply(lambda x: x.x)
        sp_with_locs["latitude"] = sp_with_locs["center"].apply(lambda x: x.y)
        # use started_at as the datetime - TODO!
        sp_with_locs["datetime"] = pd.to_datetime(sp_with_locs["started_at"], utc=True)
        sp_with_locs.reset_index(inplace=True)

        # Make skmob dataframe
        tdf = skmob.TrajDataFrame(
            sp_with_locs,
            latitude="latitude",
            longitude="longitude",
            datetime="datetime",
            user_id="user_id",
            crs={"init": stps.crs},
        )
        return tdf

    def _get_con(self):
        DBLOGIN_FILE = os.path.join("./dblogin.json")
        with open(DBLOGIN_FILE) as json_file:
            LOGIN_DATA = json.load(json_file)

        con = psycopg2.connect(
            dbname=LOGIN_DATA["database"],
            user=LOGIN_DATA["user"],
            password=LOGIN_DATA["password"],
            host=LOGIN_DATA["host"],
            port=LOGIN_DATA["port"],
        )
        return con

    def _load_data(self, study):
        CRS_WGS84 = "epsg:4326"
        con = self._get_con()

        # get staypoints
        self._sp = gpd.GeoDataFrame.from_postgis(
            sql="SELECT * FROM {}.staypoints".format(study),
            con=con,
            crs=CRS_WGS84,
            geom_col="geom",
            index_col="id",
        )
        # get locs
        self._locations = gpd.GeoDataFrame.from_postgis(
            sql="SELECT * FROM {}.locations".format(study),
            con=con,
            crs=CRS_WGS84,
            geom_col="center",
            index_col="id",
        )
        # get trips
        if self._trips_available:
            self._trips = gpd.GeoDataFrame.from_postgis(
                sql="SELECT * FROM {}.trips".format(study),
                con=con,
                crs=CRS_WGS84,
                geom_col="geom",  # center for locations
                index_col="id",
            )

    # ----------- STPS based features -----------------

    def random_entropy(self):
        return random_entropy(self._tdf, show_progress=False)

    def real_entropy(self):
        return real_entropy(self._tdf, show_progress=False)

    def uncorrelated_entropy(self):
        return uncorrelated_entropy(self._tdf, show_progress=False)

    def max_distance_from_home(self):
        return max_distance_from_home(self._tdf, show_progress=False)

    def number_locations(self):
        num_locs = self._locations.groupby("user_id").agg({"center": "count"})
        return num_locs.reset_index().rename(columns={"user_id": "uid", "center": "number_locations"})

    def waiting_time_distribution(self):
        times = waiting_times(self._tdf)
        waiting_time_dist = times["waiting_times"].apply(np.mean)  # dist_to_stats)
        col_names = ["mean_waiting_time"]  # dist_names("waiting_time")
        time_df = pd.DataFrame(waiting_time_dist.tolist(), index=times.index, columns=col_names)
        time_df["uid"] = times["uid"]
        return time_df

    def radius_of_gyration(self):
        return radius_of_gyration(self._tdf, show_progress=False)

    # ----------- Trip based features -----------------

    def mean_trip_distance(self, is_projected=False):

        trips_copy = self._trips.copy()
        # is_projected = ti.geogr.distances.check_gdf_crs(trips_copy)
        trips_copy["trip_distance"] = trips_copy["geom"].apply(lambda x: get_point_dist(x[0], x[1], is_projected))
        grouped = trips_copy.groupby("user_id").agg({"trip_distance": "mean"})
        return grouped.reset_index().rename(columns={"user_id": "uid", "trip_distance": "mean_trip_distance"})

    def _trip_duration(self):
        # compute time duration
        self._trips["time_passed"] = (self._trips.finished_at - self._trips.started_at).astype("timedelta64[m]")
        # get user ids
        grouped_trips = self._trips.groupby("user_id")
        # get list of times for each user
        time_passed_list = grouped_trips.agg({"time_passed": list})
        return time_passed_list

    def mean_trip_duration(self):
        time_passed_list = self._trip_duration()
        # uid_column = [uid for (uid, _) in grouped_trips]
        uid_column = list(time_passed_list.index)
        # get stats for list for each user
        # time_passed_dist = time_passed_list["time_passed"].apply(dist_to_stats)
        # col_names = dist_names("trip_time")
        time_passed_dist = time_passed_list["time_passed"].apply(np.mean)
        col_names = ["mean_trip_duration"]
        dist_df = pd.DataFrame(time_passed_dist.tolist(), index=time_passed_list.index, columns=col_names)
        dist_df["uid"] = uid_column
        return dist_df

    def _check_implemented(self, features):
        # check if all required features are implemented
        for feat in features:
            if not hasattr(self, feat):
                raise NotImplementedError(f"Feature {feat} ist not implemented!")

    def __call__(self, features="default", **kwargs):
        """Collect all desired features"""
        if features == "default":
            features = self._default_features
        elif features == "all":
            features = self._all_features
        # if trips are not available, exclude those features
        if not self._trips_available:
            features = [f for f in features if "trip" not in f]
        self._check_implemented(features)
        print("The following features will be computed:", features)

        collect_features = []
        for feat in features:
            print("----- ", feat, "----------")
            # call corresponding method
            feat_df = getattr(self, feat)()
            print(feat_df.columns)

            collect_features.append(feat_df)

        df_all_features = reduce(lambda left, right: pd.merge(left, right, on=["uid"], how="outer"), collect_features)
        # clean
        df_all_features.rename(columns={"uid": "user_id"}, inplace=True)
        df_all_features.set_index("user_id", inplace=True)
        return df_all_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--study", type=str, required=True, help="study - one of gc1, gc2, geolife")
    args = parser.parse_args()

    study = args.study
    out_dir = "test"

    trips_available = "tist" not in study  # for tist, the trips are missing

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, f"{study}_raw_features")

    raw_feat = RawFeatures(study, trips_available=trips_available)
    raw_feature_df = raw_feat(features="all")
    raw_feature_df.to_csv(out_path + ".csv")
    print(raw_feature_df.head(10))
    print(raw_feature_df.shape)

    raw_feature_df.dropna(inplace=True)
    labels = normalize_and_cluster(raw_feature_df, n_clusters=2)
    scatterplot_matrix(raw_feature_df, raw_feature_df.columns, clustering=labels, save_path=out_path + ".pdf")
