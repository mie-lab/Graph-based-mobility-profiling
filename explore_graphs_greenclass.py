import geopandas as gpd
import trackintel as ti
from sqlalchemy import create_engine
from trackintel.preprocessing import activity_graphs as tigraphs
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import smopy

from activity_graphs_utils import draw_smopy_basemap, nx_coordinate_layout_smopy

crs = {'init' :'epsg:4326'}

with open('login.json') as json_file:  
    login_data = json.load(json_file)
    
conn_string = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**login_data)

image_output = r".\graph_images\gc2"

engine = create_engine(conn_string)
conn = engine.connect()

sp_org = gpd.GeoDataFrame.from_postgis("""SELECT *, geometry_raw as geometry_new
                                       FROM gc1.staypoints""",
                                       conn, crs=crs, geom_col='geometry_new', index_col='id')
# create important places 
sp = sp_org.copy()
sp["geom"] = sp["geometry_new"]
sp["elevation"] = np.nan

sp = sp.loc[:,["user_id", "started_at", "finished_at", "elevation", "geom"]]
sp = sp.set_geometry("geom")


#sp = sp.to_crs({'init': 'epsg:2056'})
#sp = ti.trackintel.read_staypoints_postgis(conn_string=conn_string, geom_col="geometry_raw", table_name="gc2.staypoints")
    
places = sp.as_staypoints.extract_places(epsilon=0.0001, num_samples=3)

A_dict = tigraphs.weights_transition_count(sp)

G_dict = tigraphs.generate_activity_graphs(places, A_dict)



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
    

    filename = image_output + "\\" + str(user_id) + "_coordinate_layout" + ".png"
    plt.savefig(filename)
    plt.close()
    
    # draw spring layout 
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, width=norm_width/2, node_size=node_sizes)
    filename = image_output + "\\" + str(user_id) + "_spring_layout" + ".png"
    plt.savefig(filename)
    plt.close()
    


