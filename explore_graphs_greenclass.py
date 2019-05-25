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
study = "gc1"

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
A_dict = tigraphs.weights_n_neighbors(places, 5)
G_list = list(tigraphs.generate_activity_graphs(places, A_dict).items())
pickle.dump( G_list, open( GRAPH_OUTPUT + "_5dist_full.pkl", "wb" ) )

print('create full graph with delaunay tesselation')
A_dict = tigraphs.weights_delaunay(places,  to_crs={'init': 'epsg:3857'})
G_list = list(tigraphs.generate_activity_graphs(places, A_dict).items())
pickle.dump( G_list, open( GRAPH_OUTPUT + "_counts_full.pkl", "wb" ) )


# create graphs with temporal window
start_date = min(sp['started_at'])
end_date = max(sp['finished_at'])

delta_date = end_date - start_date

date_step = 10
date_list = [start_date + datetime.timedelta(days=int(x)) 
    for x in np.arange(date_step, delta_date.days, date_step)]

A_dict_counts = {}
G_list_counts = []

A_dict_5dist = {}
G_list_5dist = []

A_dict_delaunay = {}
G_list_delaunay = []

start_date_this = start_date

print('create time-window graphs')
for ix,end_date_this in enumerate(date_list):
    sp_this = sp[(sp['started_at'] > start_date_this) & (sp['finished_at'] < end_date_this)]
    places_this = places[places['place_id'].isin(sp_this['place_id'])]
    
    A_dict_counts = (tigraphs.weights_transition_count(sp_this))
    A_dict_5dist = tigraphs.weights_n_neighbors(places_this, 4)
    A_dict_delaunay = tigraphs.weights_delaunay(places_this, to_crs={'init': 'epsg:3857'})
    
        
    G_list_counts = G_list_counts + list(tigraphs.generate_activity_graphs(places_this, A_dict_counts).items())
    G_list_5dist = G_list_5dist + list(tigraphs.generate_activity_graphs(places_this, A_dict_5dist).items())
    G_list_delaunay = G_list_delaunay + list(tigraphs.generate_activity_graphs(places_this, A_dict_delaunay).items())

    start_date_this = end_date_this
    
    if ix%10 == 0:
        print('\t {}/{} finished'.format(ix,len(date_list)))

print('writing time-window graphs...')
pickle.dump( G_list_counts, open( GRAPH_OUTPUT + "_counts_{}days.pkl".format(date_step) , "wb" ) )
pickle.dump( G_list_5dist, open( GRAPH_OUTPUT + "_5dist_{}days.pkl".format(date_step) , "wb" ) )
pickle.dump( G_list_5dist, open( GRAPH_OUTPUT + "_delaunay_{}days.pkl".format(date_step) , "wb" ) )

print('done')








## code for plotting
#
##
##
#def get_color_hash(key_list):
#    cmap = plt.get_cmap('Set1')
#    colors = cmap(np.linspace(0, 1, len(key_list)))
##    colors = np.linspace(0, 1, len(key_list))
#     
#    return dict(zip(key_list, colors))
#
#
#
#
## save graphs to file
#
#
#for ix, id_G_tuple in enumerate(G_list):
#    user_id, G = id_G_tuple
#    edge_color_dict = get_color_hash(G.graph['edge_keys']+[''])
#    if len(G.nodes) == 0:
#        continue
#    
#    print(user_id)
#    # edge color management
#    weights = [1/d for u,v,d in G.edges(data='weight')]
#    edge_colors = np.asarray([edge_color_dict[edge_key]
#                            for u,v, edge_key in G.edges(keys=True)])
#    mm = MinMaxScaler((2,10))
#    norm_weights = np.log(weights)
#    norm_weights = mm.fit_transform(norm_weights.reshape((-1,1)))
#
#    deg = nx.degree(G)
#    node_sizes = [5 * deg[iata] for iata in G.nodes]
#    
#    # draw geographic representation
#    ax, smap = draw_smopy_basemap(G)
#    nx.draw_networkx(G, ax=ax,
#                 edge_color=edge_colors,
#                 font_size=20,
#                 width=norm_weights.ravel(),
#                 with_labels=False,
#                 node_size=node_sizes,
#                 pos=nx_coordinate_layout_smopy(G,smap))
#    
#    


#    plt.close()
##    
##
##    filename = IMAGE_OUTPUT + "\\" + str(user_id) + "_" + str(ix) + "_coordinate_layout" + ".png"
##    plt.savefig(filename)
##    plt.close()
##    
#    # draw spring layout 
#    plt.figure()
#    pos = nx.spring_layout(G)
#    nx.draw(G, pos=pos, width=norm_width/2, node_size=node_sizes)
#    filename = IMAGE_OUTPUT + "\\" + str(user_id) + "_" + str(ix) + "_spring_layout" + ".png"
#    plt.savefig(filename)
#    plt.close()
##    
