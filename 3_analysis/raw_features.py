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

from utils import dist_to_stats, dist_names
from skmob.measures.individual import *


class RawFeatures:
    def __init__(self, study):
        print("Loading data...")
        self.load_data(study)
        self.tdf = self.to_skmob(self.stps, self.locations)

        self.feature_dict = {
            "entropy_real": self.real_entropy,
            "entropy_random": self.random_entropy,
            "trips_duration": self.trip_len_time,
            "waiting_times": self.waiting_time_distribution,
            "entropy_uncorrelated": self.uncorrelated_entropy,
            "number_locations": self.number_locations,
            "dist_from_home": self.max_distance_from_home,
        }

    @staticmethod
    def to_skmob(stps, locations):
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

    def get_con(self):
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

    def load_data(self, study):
        CRS_WGS84 = "epsg:4326"
        con = self.get_con()

        # get staypoints
        self.stps = gpd.GeoDataFrame.from_postgis(
            sql="SELECT * FROM {}.staypoints".format(study),
            con=con,
            crs=CRS_WGS84,
            geom_col="geom",
            index_col="id",
        )
        # get locs
        self.locations = gpd.GeoDataFrame.from_postgis(
            sql="SELECT * FROM {}.locations".format(study),
            con=con,
            crs=CRS_WGS84,
            geom_col="center",
            index_col="id",
        )
        # get trips
        self.trips = pd.read_sql_query(sql="SELECT * FROM {}.trips".format(study), con=con, index_col="id")

    # ----------- STPS based features -----------------

    def random_entropy(self):
        return random_entropy(self.tdf)

    def real_entropy(self):
        return real_entropy(self.tdf)

    def uncorrelated_entropy(self):
        return uncorrelated_entropy(self.tdf)

    def max_distance_from_home(self):
        return uncorrelated_entropy(self.tdf)

    def number_locations(self):
        num_locs = self.locations.groupby("user_id").agg({"center": "count"})
        return num_locs.reset_index().rename(columns={"user_id": "uid"})

    def waiting_time_distribution(self):
        times = waiting_times(self.tdf)
        waiting_time_dist = times["waiting_times"].apply(dist_to_stats)
        col_names = dist_names("waiting_time")
        time_df = pd.DataFrame(waiting_time_dist.tolist(), index=times.index, columns=col_names)
        time_df["uid"] = times["uid"]
        return time_df

    # # commented out because we have it in the call method directly
    # def k_radius_of_gyration(self, k_most_frequent=[5, 10, 20]):
    #     df_for_each_k = []
    #     for k in k_most_frequent:
    #         krg_df = k_radius_of_gyration(self.tdf, k)
    #         df_for_each_k.append(krg_df)

    #     df_k = reduce(lambda left, right: pd.merge(left, right, on=["uid"], how="outer"), df_for_each_k)
    #     return df_k

    # ----------- Trip based features -----------------

    def max_trip_distance(self):
        # TODO, use trips table
        pass

    def trip_len_time(self):
        # compute time duration
        self.trips["time_passed"] = (self.trips.finished_at - self.trips.started_at).astype("timedelta64[m]")
        # get user ids
        grouped_trips = self.trips.groupby("user_id")
        uid_column = [uid for (uid, _) in grouped_trips]
        # get list of times for each user
        time_passed_list = grouped_trips.agg({"time_passed": list})
        # get stats for list for each user
        time_passed_dist = time_passed_list["time_passed"].apply(dist_to_stats)
        col_names = dist_names("trip_time")
        dist_df = pd.DataFrame(time_passed_dist.tolist(), index=time_passed_list.index, columns=col_names)
        dist_df["uid"] = uid_column
        return dist_df

    def __call__(self, features="all"):
        """Collect all desired features"""
        if features == "all":
            features = list(self.feature_dict.keys()) + ["5_gyration", "10_gyration", "50_gyration"]
        assert all([f in self.feature_dict for f in features if "gyration" not in f])

        collect_features = []
        for feat in features:
            print("----- ", feat, "----------")
            # for radius of gyration, get k
            if "gyration" in feat:
                k = int(feat.split("_")[0])
                feat_df = k_radius_of_gyration(self.tdf, k)
            else:
                feat_df = self.feature_dict[feat]()
            print(feat_df.columns)

            collect_features.append(feat_df)

        df_all_features = reduce(lambda left, right: pd.merge(left, right, on=["uid"], how="outer"), collect_features)
        return df_all_features


if __name__ == "__main__":
    raw_feat = RawFeatures("gc2")
    out = raw_feat(features="all")
    print(out.head(10))
    print(out.shape)
