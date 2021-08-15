import pandas as pd
import numpy as np
import datetime
import psycopg2
import pickle


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
        "purpose": list
    }

    agg_dict.update(custom_add_dict)

    sp_merged = sp_merge.groupby(by="id").agg(agg_dict)

    return sp_merged


from psycopg2 import sql


def write_graphs_to_postgresql(
    graph_data, graph_table_name, psycopg_con, graph_schema_name="public", file_name="graph_data", drop_and_create=True
):

    pickle_string = pickle.dumps(graph_data)

    cur = psycopg_con.cursor()
    if drop_and_create:
        cur.execute(
            sql.SQL("drop table if exists {}.{}").format(
                sql.Identifier(graph_schema_name), sql.Identifier(graph_table_name)
            )
        )
        cur.execute(
            sql.SQL("create table {}.{} (name text, data bytea)").format(
                sql.Identifier(graph_schema_name), sql.Identifier(graph_table_name)
            )
        )
        psycopg_con.commit()

    cur.execute(
        sql.SQL("insert into {}.{} values (%s, %s)").format(
            sql.Identifier(graph_schema_name), sql.Identifier(graph_table_name)
        ),
        [file_name, pickle_string],
    )
    psycopg_con.commit()
    cur.close()


def read_graphs_from_postgresql(graph_table_name, psycopg_con, graph_schema_name="public", file_name="graph_data"):
    # retrieve string
    cur = psycopg_con.cursor()
    cur.execute(
        sql.SQL("select data from {}.{} where name = %s").format(
            sql.Identifier(graph_schema_name), sql.Identifier(graph_table_name)
        ),
        (file_name,),
    )
    pickle_string2 = cur.fetchall()[0][0].tobytes()

    cur.close()
    AG_dict2 = pickle.loads(pickle_string2)

    return AG_dict2
