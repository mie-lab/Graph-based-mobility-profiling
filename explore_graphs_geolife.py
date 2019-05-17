import pandas as pd
import geopandas as gpd
import trackintel as ti
from sqlalchemy import create_engine
from trackintel.preprocessing import activity_graphs as tigraphs
import numpy as np
import networkx as nxa
import matplotlib.pyplot as plt
import pickle
import os
import json
import ntpath
from shapely.geometry import Point

from activity_graphs_utils import draw_smopy_basemap, nx_coordinate_layout_smopy

CRS_WGS84 = {'init' :'epsg:4326'}



# define output for images
IMAGE_OUTPUT = os.path.join(".", "graph_images", "geolife")
if not os.path.exists(IMAGE_OUTPUT):
    os.mkdir(IMAGE_OUTPUT)

# define output for graphs
GRAPH_OUTPUT = os.path.join(".", "graph_data", "geolife.graphs.pkl")
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

posfix = pd.read_sql("""select * from geolife.positionfixes where user_id <= 0""", engine)

posfix = gpd.GeoDataFrame(posfix,
                          geometry=[Point(xy) for xy in 
                                    zip(posfix.lon, posfix.lat)], 
                          crs=CRS_WGS84)

posfix['accuracy'] = np.nan
posfix['geom'] = posfix['geometry']
posfix = posfix.set_geometry("geom")


sp = posfix.as_positionfixes.extract_staypoints()

#sp = sp.to_crs({'init': 'epsg:2056'})
#sp = ti.trackintel.read_staypoints_postgis(conn_string=conn_string, geom_col="geometry_raw", table_name="gc2.staypoints")
# todo: Create reliable staypoints and write them to database!

places = sp.as_staypoints.extract_places(epsilon=0.0001, num_samples=3)
places.as_places.plot()

A_dict = tigraphs.weights_transition_count(sp)
G_dict = tigraphs.generate_activity_graphs(places, A_dict)


# save graphs to file
pickle.dump( G_dict, open( GRAPH_OUTPUT, "wb" ) )


for user_id, G in G_dict.items():
    print(user_id)
    # edge color management
    weights = [G[u][v]['weight']+1 for u,v in G.edges()]
    norm_width = np.log(weights)*2

    deg = nx.degree(G)
    node_sizes = [5 * deg[iata] for iata in G.nodes]
    
    # draw geographic representation
    ax, smap = draw_smopy_basemap(G)
    nx.draw_networkx(G, ax=ax,
                 font_size=20,
                 width=1,
                 edge_width=norm_width,
                 with_labels=False,
                 node_size=node_sizes,
                 pos=nx_coordinate_layout_smopy(G,smap))
    

    filename = IMAGE_OUTPUT + "\\" + str(user_id) + "_coordinate_layout" + ".png"
    plt.savefig(filename)
    plt.close()
    
    # draw spring layout 
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, width=norm_width/2, node_size=node_sizes)
    filename = IMAGE_OUTPUT + "\\" + str(user_id) + "_spring_layout" + ".png"
    plt.savefig(filename)
    plt.close()
    


