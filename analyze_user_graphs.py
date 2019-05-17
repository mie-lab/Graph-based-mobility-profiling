# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:18:10 2019

@author: martinhe
"""

import geopandas as gpd
import trackintel as ti
from sqlalchemy import create_engine
from trackintel.preprocessing import activity_graphs as tigraphs
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import json
import os
import pickle
import ntpath
import glob


plt.rcParams["figure.figsize"] = (16,9)
fontSizeAll = 18
font = {     'size'   : fontSizeAll}
matplotlib.rc('font', **font)

GRAPH_DATA_FOLDER = os.path.join(".", "graph_data","*")

GRAPH_FILES = glob.glob(GRAPH_DATA_FOLDER)

graph_data_dict = {}
study_name_list = []
# load all graph data
for graph_file in GRAPH_FILES:
    _, filename = ntpath.split(graph_file)
    study_name = filename.split(".")[0]
    study_name_list.append(study_name)
    graph_data_dict[study_name] = pickle.load( open( graph_file, "rb" ) )


# iterate all graph data sets,
# todo: define G_dict in loop
    


result = {}
result['av_degree'] = {}
result['av_clustering'] = {}
result['av_shortest_path'] = {}
result['av_diameter'] = {}
result['av_shortest_path'] = {}

for study_name in study_name_list:
    
    G_dict = graph_data_dict[study_name]
    
    av_degree_list = []
    av_clustering_list = []
    av_shortest_path_list = []
    av_diameter_list = []
    av_shortest_path_list = []

    for user_id, G in G_dict.items():
        # average degree
        av_degree_list.append(np.mean(list(dict(G.degree).values())))
        
        # average clustering
        av_clustering_list.append(nx.average_clustering(G, weight='weight'))
        
        # Diameter
        av_diameter_list.append(nx.diameter(G))
        
        # average shortest path length
        av_shortest_path_list.append(nx.average_shortest_path_length(G, weight='weight'))
    
    #save to dict for each study
    
    result['av_degree'].update( {study_name: av_degree_list} )
    result['av_clustering'].update( {study_name: av_clustering_list} )
    result['av_shortest_path'].update( {study_name: av_shortest_path_list} )
    result['av_diameter'].update( {study_name: av_diameter_list} )
    


for parameter, results_by_study_dict in result.items():   
    fig, ax = plt.subplots(2,2) 
    fig.suptitle(parameter)
    for i, study_name in enumerate(study_name_list):
        ax_this = ax[i//2][i%2]
        ax_this.hist(results_by_study_dict[study_name], 20)
        ax_this.set_title(study_name)
        
    

