import geopandas as gpd
import trackintel as ti
from sqlalchemy import create_engine
from future_trackintel import tigraphs
import numpy as np
import os
import pickle
import ntpath
import json
import datetime
from future_trackintel.activity_graph import activity_graph
import copy

CRS_WGS84 = 'epsg:4326'
#
studies = ['gc2','gc1' ]#, 'geolife',]# 'tist_u1000', 'tist_b100', 'tist_b200', 'tist_u10000']
n = 'fconn' # number of neighbors for neighbor weights

for study in studies:
    print("start {}".format(study))
    # define output for graphs
    GRAPH_OUTPUT = os.path.join(".", 'data_out', "graph_data", study)
    GRAPH_FOLDER, _ = ntpath.split(GRAPH_OUTPUT)
    if not os.path.exists(GRAPH_FOLDER):
        os.mkdir(GRAPH_FOLDER)
        
    # build database login string from file
    DBLOGIN_FILE = os.path.join("./dblogin.json")
    with open(DBLOGIN_FILE) as json_file:  
        LOGIN_DATA = json.load(json_file)
        
    conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}"\
                    .format(**LOGIN_DATA)
    
    engine = create_engine(conn_string)
    conn = engine.connect()
    
    print('\t download staypoints')
    sp = ti.io.read_staypoints_postgis(conn_string, table_name='{}.staypoints'.format(study), geom_col='geom')

    print('\t download places')
    locs = ti.io.read_locations_postgis(conn_string, table_name='{}.locations'.format(study),
                                          geom_col='center')

    AG_list = []
    for user_id_this in locs["user_id"].unique():
        sp_user = sp[sp['user_id'] == user_id_this]
        locs_user = locs[locs['user_id'] == user_id_this]
        AG = activity_graph(sp_user, locs_user)
        AG.plot(os.path.join(".", "graph_images", "new"),  filter_node_importance=25)
        AG_list.append(copy.deepcopy(AG))

    # create graphs of the full period
    print('\t create full graph with counts')
    pickle.dump(AG_list, open( GRAPH_OUTPUT + "_counts_full.pkl", "wb" ))
    
    # print('\t create full graph with neighbors')
    # A_dict = tigraphs.weights_n_neighbors(locations, n)
    # G_list = list(tigraphs.generate_activity_graphs(locations, A_dict).items())
    # pickle.dump( G_list, open( GRAPH_OUTPUT + "_{}dist_full.pkl".format(n), "wb" ) )
    #
    # print('\t create full graph with delaunay tesselation')
    # A_dict = tigraphs.weights_delaunay(locations, to_crs={'init': 'epsg:3857'})
    # G_list = list(tigraphs.generate_activity_graphs(locations, A_dict).items())
    # pickle.dump( G_list, open( GRAPH_OUTPUT + "_delaunay_full.pkl", "wb" ) )

#
#
#
#     print('filter users without places')
#     user_ids_places = locations['user_id'].unique()
#     sp = sp[sp['user_id'].isin(user_ids_places)]
#
#     print('create index')
#     sp.set_index('started_at', drop=False, inplace=True)
#     sp.index.name = 'started_at_ix'
#     sp.sort_index(inplace=True)
#     # create graphs with temporal window
#     start_date = min(sp['started_at'])
#     end_date = max(sp['started_at'])
#
#     delta_date = end_date - start_date
#
#     date_step = 1
#     date_list = [start_date + datetime.timedelta(days=int(x))
#         for x in np.arange(date_step, delta_date.days, date_step)]
#
#     A_dict_counts = {}
#     G_list_counts = []
#
#     A_dict_ndist = {}
#     G_list_ndist = []
#
#     A_dict_delaunay = {}
#     G_list_delaunay = []
#
#     start_date_this = start_date
#
#     print('\t create time-window graphs')
#     for ix, end_date_this in enumerate(date_list):
#         sp_this = sp[(sp.index >= start_date_this) & (sp.index < end_date_this)]
#         # sp_this = sp[(sp['started_at'] > start_date_this) & (sp['finished_at'] < end_date_this)]
#
#         places_this = locations[locations['place_id'].isin(sp_this['place_id'])]
#
#         A_dict_counts = (tigraphs.weights_transition_count(sp_this))
#         A_dict_ndist = tigraphs.weights_n_neighbors(places_this, n)
#         A_dict_delaunay = tigraphs.weights_delaunay(places_this, to_crs={'init': 'epsg:3857'})
#
#
#         G_list_counts = G_list_counts + list(tigraphs.generate_activity_graphs(places_this, A_dict_counts).items())
#         G_list_ndist = G_list_ndist + list(tigraphs.generate_activity_graphs(places_this, A_dict_ndist).items())
#         G_list_delaunay = G_list_delaunay + list(tigraphs.generate_activity_graphs(places_this, A_dict_delaunay).items())
#
#         start_date_this = end_date_this
#
#         if ix%5 == 0:
#             print('\t \t {}/{} finished'.format(ix,len(date_list)))
#
#
#     print('\t writing time-window graphs...')
#     pickle.dump( G_list_counts, open( GRAPH_OUTPUT + "_counts_{}days.pkl".format(date_step) , "wb" ) )
#     pickle.dump( G_list_ndist, open( GRAPH_OUTPUT + "_{}dist_{}days.pkl".format(n,date_step) , "wb" ) )
#     pickle.dump( G_list_delaunay, open( GRAPH_OUTPUT + "_delaunay_{}days.pkl".format(date_step) , "wb" ) )
#
#     print('finished {}'.format(study))
#
# print('all done')
#
