"""
A script that explores how to create grpahs based on the tist movement data
"""

import geopandas as gpd
import trackintel as ti
from sqlalchemy import create_engine
from trackintel.preprocessing import activity_graphs as tigraphs
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import pickle
import ntpath

from activity_graphs_utils import draw_smopy_basemap, nx_coordinate_layout_smopy


CRS_WGS = {'init' :'epsg:4326'}

# define output for images
IMAGE_OUTPUT = os.path.join(".", "graph_images", "tist")
if not os.path.exists(IMAGE_OUTPUT):
    os.mkdir(IMAGE_OUTPUT)
    
# define output for graphs
GRAPH_OUTPUT = os.path.join(".", "graph_data", "tist.graphs.pkl")
GRAPH_FOLDER, _= ntpath.split(GRAPH_OUTPUT)
if not os.path.exists(GRAPH_FOLDER):
    os.mkdir(GRAPH_FOLDER)

# build database login string from file
DBLOGIN_FILE = os.path.join("dblogin.json")
with open(DBLOGIN_FILE) as json_file:  
    LOGIN_DATA = json.load(json_file)
    
conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}" \
    .format(**LOGIN_DATA)

engine = create_engine(conn_string)
conn = engine.connect()

sp_org = gpd.GeoDataFrame.from_postgis("""SELECT * from tist.staypoints order by user_id limit 100""",
                                       conn, crs=CRS_WGS, geom_col='geom')
# bring to trackintel format
sp = sp_org.copy()
# add columns
REQUIRED_COLUMNS = ['user_id', 'started_at', 'finished_at', 'elevation', 'geom']
columns = list(set(list(sp.columns) + REQUIRED_COLUMNS))
sp = sp.reindex(columns=columns)
#sp[sp["user_id"]==1].as_staypoints.plot()

places = sp.as_staypoints.extract_places(epsilon=50, num_samples=4, distance_matrix_metric='haversine')
#places[places["user_id"]==1].as_places.plot(plot_osm=True)

A_dict = tigraphs.weights_transition_count(sp)
#
G_dict = tigraphs.generate_activity_graphs(places, A_dict)

# save graphs to file
pickle.dump( G_dict, open( GRAPH_OUTPUT, "wb" ) )


#
## save figures
#
#for user_id, G in G_dict.items():
#    print(user_id)
#    # edge color management
#    weights = [G[u][v]['weight']+1 for u,v in G.edges()]
#    norm_width = np.log(weights)*2
#
#    deg = nx.degree(G)
#    node_sizes = [5 * deg[iata] for iata in G.nodes]
#    
#    # draw geographic representation
#    ax, smap = draw_smopy_basemap(G)
#    nx.draw_networkx(G, ax=ax,
#                 font_size=20,
#                 width=1,
#                 edge_width=norm_width,
#                 with_labels=False,
#                 node_size=node_sizes,
#                 pos=nx_coordinate_layout_smopy(G,smap))
#    
#
#    filename = os.path.join(IMAGE_OUTPUT, str(user_id) + "_coordinate_layout" + ".png")
#    plt.savefig(filename)
#    plt.close()
#    
#    # draw spring layout 
#    plt.figure()
#    pos = nx.spring_layout(G)
#    nx.draw(G, pos=pos, width=norm_width/2, node_size=node_sizes)
#    filename = os.path.join(IMAGE_OUTPUT, str(user_id) + "_spring_layout" + ".png")
#    plt.savefig(filename)
#    plt.close()
#    
    




