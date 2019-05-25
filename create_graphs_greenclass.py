import geopandas as gpd
import trackintel as ti
from sqlalchemy import create_engine
from trackintel.preprocessing import activity_graphs as tigraphs
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
import ntpath
import json
import datetime
from sklearn.preprocessing import MinMaxScaler

from activity_graphs_utils import draw_smopy_basemap, nx_coordinate_layout_smopy

CRS_WGS84 = {'init' :'epsg:4326'}
#
studies = ['gc1', 'gc2']
n = 20 # number of neighbors for neighbor weights

for study in studies:

    # define output for images
    IMAGE_OUTPUT = os.path.join(".", "graph_images", study)
    if not os.path.exists(IMAGE_OUTPUT):
        os.mkdir(IMAGE_OUTPUT)
        
    # define output for graphs
    GRAPH_OUTPUT = os.path.join(".", "graph_data", study +".graphs")
    GRAPH_FOLDER, _= ntpath.split(GRAPH_OUTPUT)
    if not os.path.exists(GRAPH_FOLDER):
        os.mkdir(GRAPH_FOLDER)
        
    # build database login string from file
    DBLOGIN_FILE = os.path.join("dblogin.json")
    with open(DBLOGIN_FILE) as json_file:  
        LOGIN_DATA = json.load(json_file)
        
    conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}"\
                    .format(**LOGIN_DATA)
    
    engine = create_engine(conn_string)
    conn = engine.connect()
    
    
    places = ti.io.read_places_postgis(conn_string, table_name='{}.places'.format(study),
                              geom_col='center')
    
    
    #
    # create graphs of the full period
    print('create full graph with counts')
    A_dict = (tigraphs.weights_transition_count(sp))
    G_list = list(tigraphs.generate_activity_graphs(places, A_dict).items())
    pickle.dump( G_list, open( GRAPH_OUTPUT + "_counts_full.pkl", "wb" ) )
    
    print('create full graph with neighbors')
    A_dict = tigraphs.weights_n_neighbors(places, n)
    G_list = list(tigraphs.generate_activity_graphs(places, A_dict).items())
    pickle.dump( G_list, open( GRAPH_OUTPUT + "_{}dist_full.pkl".format(n), "wb" ) )
    
    print('create full graph with delaunay tesselation')
    A_dict = tigraphs.weights_delaunay(places,  to_crs={'init': 'epsg:3857'})
    G_list = list(tigraphs.generate_activity_graphs(places, A_dict).items())
    pickle.dump( G_list, open( GRAPH_OUTPUT + "_delaunay_full.pkl", "wb" ) )
    
    
    # create graphs with temporal window
    start_date = min(sp['started_at'])
    end_date = max(sp['finished_at'])
    
    delta_date = end_date - start_date
    
    date_step = 10
    date_list = [start_date + datetime.timedelta(days=int(x)) 
        for x in np.arange(date_step, delta_date.days, date_step)]
    
    A_dict_counts = {}
    G_list_counts = []
    
    A_dict_ndist = {}
    G_list_ndist = []
    
    A_dict_delaunay = {}
    G_list_delaunay = []
    
    start_date_this = start_date
    
    print('create time-window graphs')
    for ix,end_date_this in enumerate(date_list):
        sp_this = sp[(sp['started_at'] > start_date_this) & (sp['finished_at'] < end_date_this)]
        places_this = places[places['place_id'].isin(sp_this['place_id'])]
        
        A_dict_counts = (tigraphs.weights_transition_count(sp_this))
        A_dict_ndist = tigraphs.weights_n_neighbors(places_this, n)
        A_dict_delaunay = tigraphs.weights_delaunay(places_this, to_crs={'init': 'epsg:3857'})
        
            
        G_list_counts = G_list_counts + list(tigraphs.generate_activity_graphs(places_this, A_dict_counts).items())
        G_list_ndist = G_list_ndist + list(tigraphs.generate_activity_graphs(places_this, A_dict_ndist).items())
        G_list_delaunay = G_list_delaunay + list(tigraphs.generate_activity_graphs(places_this, A_dict_delaunay).items())
    
        start_date_this = end_date_this
        
        if ix%10 == 0:
            print('\t {}/{} finished'.format(ix,len(date_list)))
    
    print('writing time-window graphs...')
    pickle.dump( G_list_counts, open( GRAPH_OUTPUT + "_counts_{}days.pkl".format(date_step) , "wb" ) )
    pickle.dump( G_list_ndist, open( GRAPH_OUTPUT + "_{}dist_{}days.pkl".format(n,date_step) , "wb" ) )
    pickle.dump( G_list_delaunay, open( GRAPH_OUTPUT + "_delaunay_{}days.pkl".format(date_step) , "wb" ) )
    
    print('done')


