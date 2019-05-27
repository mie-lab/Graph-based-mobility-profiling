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
import re
import itertools


expr1 = ['.*']
weights = ['fconndist_','counts_', 'delaunay_']
subset = ['full', '7days']

expressions = list(itertools.product(expr1,weights,subset))
expressions = [''.join(t) for t in expressions]



plt.rcParams["figure.figsize"] = (16,9)
fontSizeAll = 18
font = {     'size'   : fontSizeAll}
matplotlib.rc('font', **font)

GRAPH_DATA_FOLDER = os.path.join(".", "graph_data","*")
GRAPH_IMAGES_FOLDER = os.path.join(".", "graph_images",'hists','')
GRAPH_FILES = glob.glob(GRAPH_DATA_FOLDER)

graph_data_dict = {}
study_name_list_all = []
# load all graph data
for graph_file in GRAPH_FILES:
    _, filename = ntpath.split(graph_file)
    study_name = filename.split(".")[0]
    study_name_list_all.append(study_name)
    graph_data_dict[study_name] = pickle.load( open( graph_file, "rb" ) )


# iterate all graph data sets,
# todo: define G_dict in loop
    
for expr in expressions:
    print(expr)
    regex = re.compile(expr)
    
    study_name_list = list(filter(regex.search, study_name_list_all))
    
    result = {}
    result['av_degree'] = {}
    result['av_clustering'] = {}
    result['av_shortest_path'] = {}
    result['av_diameter'] = {}
    result['av_shortest_path'] = {}
    result['weights'] = {}
    result['nb_nodes'] = {}
    
    for study_name in study_name_list:
        
        G_list = graph_data_dict[study_name]
        
        av_degree_list = []
        av_clustering_list = []
        av_shortest_path_list = []
        av_diameter_list = []
        av_shortest_path_list = []
        weights_list = []
        nb_nodes_list = []
    
        for user_id, G1 in G_list:
            G = nx.Graph(G1)
            # skip empty graphs
            if len(G.nodes) < 2:
                continue
            
            # average degree
            av_degree_list.append(np.mean(list(dict(G.degree).values())))
            
            # average clustering
            av_clustering_list.append(nx.average_clustering(G, weight='weight'))
             
            try:
                # Diameter
                av_diameter_list.append(nx.diameter(G))
            
                # average shortest path length
                av_shortest_path_list.append(nx.average_shortest_path_length(G, weight='weight'))
            except nx.NetworkXError:
                print('unconnected')
                
            # all weights
            weights_list = weights_list + [w for u,v,w in G.edges(data='weight')]
            
            # nb of nodes
            nb_nodes_list.append(len(G.nodes))
        
        #save to dict for each study
        
        result['av_degree'].update( {study_name: av_degree_list} )
        result['av_clustering'].update( {study_name: av_clustering_list} )
        result['av_shortest_path'].update( {study_name: av_shortest_path_list} )
        result['av_diameter'].update( {study_name: av_diameter_list} )
        result['weights'].update({study_name: weights_list})
        result['nb_nodes'].update({study_name: nb_nodes_list})
        
    
    # plot
        
    # plot parameter
    plt_params = {}
    
    
    for parameter, results_by_study_dict in result.items():   
        fig, ax = plt.subplots(2,2) 
        fig.suptitle(parameter)
        for i, study_name in enumerate(study_name_list):
            ax_this = ax[i//2][i%2]
            data = np.asarray(results_by_study_dict[study_name])
    #        data = np.log10(data+0.01)
            ax_this.hist(data, 50, log=False)
            ax_this.set_title(study_name)
        
        plt.savefig(GRAPH_IMAGES_FOLDER+parameter+'_'+regex.pattern[2:]+'.png')
        plt.close()
        
    # same plot with log axis
    for parameter, results_by_study_dict in result.items():   
        fig, ax = plt.subplots(2,2) 
        fig.suptitle(parameter)
        for i, study_name in enumerate(study_name_list):
            ax_this = ax[i//2][i%2]
            data = np.asarray(results_by_study_dict[study_name])
    #        data = np.log10(data+0.01)
            ax_this.hist(data, 50, log=True)
            ax_this.set_title(study_name)
        
        plt.savefig(GRAPH_IMAGES_FOLDER+parameter+'_log_'+regex.pattern[2:]+'.png')
        plt.close()
    

