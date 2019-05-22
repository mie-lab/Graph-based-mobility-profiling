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

from activity_graphs_utils import draw_smopy_basemap, nx_coordinate_layout_smopy

CRS_WGS84 = {'init' :'epsg:4326'}
#
study = "gc1"

# define output for images
IMAGE_OUTPUT = os.path.join(".", "graph_images", study)
if not os.path.exists(IMAGE_OUTPUT):
    os.mkdir(IMAGE_OUTPUT)
    
# define output for graphs
GRAPH_OUTPUT = os.path.join(".", "graph_data", study +".graphs.pkl")
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

sp_org = gpd.GeoDataFrame.from_postgis("""SELECT *, geometry_raw as geometry_new
                                       FROM gc1.staypoints""",
                                       conn, crs=CRS_WGS84, geom_col='geometry_new', index_col='id')
# create important places 
sp = sp_org.copy()
sp["geom"] = sp["geometry_new"]
sp["elevation"] = np.nan

sp = sp.loc[:,["user_id", "started_at", "finished_at", "elevation", "geom"]]
sp = sp.set_geometry("geom")



#sp = sp.to_crs({'init': 'epsg:2056'})
#sp = ti.trackintel.read_staypoints_postgis(conn_string=conn_string, geom_col="geometry_raw", table_name="gc2.staypoints")
places = sp.as_staypoints.extract_places(epsilon=50, num_samples=4,
                                         distance_matrix_metric='haversine')


start_date = min(sp['started_at']) + datetime.timedelta(days=100)
end_date = max(sp['finished_at'])

delta_date = end_date - start_date

date_step = 10
date_list = [start_date + datetime.timedelta(days=int(x)) for x in np.arange(date_step, delta_date.days, date_step)]

A_dict = {}
G_list = []
start_date_this = start_date

# todo deal with empty graphs/dicts
# todo deal with missing user_ids in A_dict

for end_date_this in date_list:
    sp_this = sp[(sp['started_at'] > start_date_this) & (sp['finished_at'] < end_date_this)]
    places_this = places[places['place_id'].isin(sp_this['place_id'])]
    
    A_dict = (tigraphs.weights_transition_count(sp_this))
    
    G_list = G_list + list(tigraphs.generate_activity_graphs(places_this, A_dict).items())
    
    start_date_this = end_date_this

# save graphs to file
pickle.dump( G_list, open( GRAPH_OUTPUT, "wb" ) )


#for ix, id_G_tuple in enumerate(G_list):
#    user_id, G = id_G_tuple
#    if len(G.nodes) == 0:
#        continue
#    
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
#    filename = IMAGE_OUTPUT + "\\" + str(user_id) + "_" + str(ix) + "_coordinate_layout" + ".png"
#    plt.savefig(filename)
#    plt.close()
#    
#    # draw spring layout 
#    plt.figure()
#    pos = nx.spring_layout(G)
#    nx.draw(G, pos=pos, width=norm_width/2, node_size=node_sizes)
#    filename = IMAGE_OUTPUT + "\\" + str(user_id) + "_" + str(ix) + "_spring_layout" + ".png"
#    plt.savefig(filename)
#    plt.close()
#    
