# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:33:39 2019

@author: martinhe
"""

import smopy
import networkx as nx
from trackintel.preprocessing import activity_graphs as tigraphs

def nx_coordinate_layout_smopy(G, smap):
    """"transforms WGS84 coordinates to pixel coordinates of a smopy map"""
    node_center = nx.get_node_attributes(G, 'center')
    pos = {key: (smap.to_pixels(geometry.y,geometry.x)) for key,geometry in node_center.items()}

    return pos


def draw_smopy_basemap(G, figsize=(8, 6), zoom=10):
    
    pos_wgs = tigraphs.nx_coordinate_layout(G) 
    lon = list_values = [ coords[0] for coords in pos_wgs.values() ]
    lat = list_values = [ coords[1] for coords in pos_wgs.values() ]
    
    lon_min = min(lon)
    lon_max = max(lon)
    lat_min = min(lat)
    lat_max = max(lat)

    smap = smopy.Map(lat_min, lon_min, lat_max, lon_max, tileserver="http://tile.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png", z=zoom)
    #map = smopy.Map((min_y, max_y-min_y, min_x, max_x-min_x), z=5)
    ax = smap.show_mpl(figsize=figsize)
    
    return ax, smap