import pandas as pd
import numpy as np
import datetime
import psycopg2
import pickle
import zlib


def horizontal_merge_staypoints(sp, gap_threshold=20, custom_add_dict={}):
    """merge staypoints that are consecutive at the same place"""
    # merge consecutive staypoints

    sp_merge = sp.copy()
    assert sp_merge.index.name == "id", "expected index name to be 'id'"

    sp_merge = sp_merge.reset_index()
    sp_merge.sort_values(inplace=True, by=["user_id", "started_at"])
    sp_merge[["next_started_at", "next_location_id"]] = sp_merge[["started_at", "location_id"]].shift(-1)
    cond = pd.Series(data=False, index=sp_merge.index)
    cond_old = pd.Series(data=True, index=sp_merge.index)
    cond_diff = cond != cond_old

    while np.sum(cond_diff) >= 1:
        # .values is important otherwise the "=" would imply a join via the new index
        sp_merge["next_id"] = sp_merge["id"].shift(-1).values

        # identify rows to merge
        cond1 = sp_merge["next_started_at"] - sp_merge["finished_at"] < datetime.timedelta(minutes=gap_threshold)
        cond2 = sp_merge["location_id"] == sp_merge["next_location_id"]
        cond = cond1 & cond2

        # assign index to next row
        sp_merge.loc[cond, "id"] = sp_merge.loc[cond, "next_id"]
        cond_diff = cond != cond_old
        cond_old = cond.copy()

        print("\t", np.sum(cond_diff))

    # aggregate values

    agg_dict = {
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
        "purpose": list,
    }

    agg_dict.update(custom_add_dict)

    sp_merged = sp_merge.groupby(by="id").agg(agg_dict)

    return sp_merged


from psycopg2 import sql



