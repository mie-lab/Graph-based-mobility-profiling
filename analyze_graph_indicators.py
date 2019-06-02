 """Create histograms with different graph indicators"""

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
import sys

#set overall plot variables
plt.rcParams["figure.dpi"] = 400
plt.rcParams["figure.figsize"] = (16,9)

fontSizeAll = 18
font = {     'size'   : fontSizeAll}
matplotlib.rc('font', **font)


# paths
GRAPH_input = os.path.join(".", 'graph_data','processed','processed_graphs.pkl')
GRAPH_IMAGES_FOLDER = os.path.join(".", "graph_images",'hists','')

# load preprocessed graphs
result =  pickle.load( open( GRAPH_input, "rb" ) )

# study name describes how the graph was created. It is the filename of the 
# file that was created during graph_generation and contains all necessary 
# information: dataset + weights + temporal extent. 
# e.g. 'tist_u1000_counts_7days' = data from tist_u1000, with transition count
# as weights and binned into 7 day bins.
study_name_list_all = list(result['nb_nodes'].keys())
study_name_list_all.sort()

# We want to plot data from all four datasets datasets that were processed in 
# the same way (e.g. the same weights and temporal binning). 
# Therefore we use a regular expression to filter the study names. 

# To plot all possible combinations we now create a list of regular expressions
# that we iterate later.
data = ['.*']
weights = ['fconndist_','counts_',  'delaunay_']
temporal_extent = ['full',  '7days']

expressions = list(itertools.product(data,weights,temporal_extent))
expressions = [''.join(t) for t in expressions]

# There are several different versions of the foursquare (tist) dataset resulting
# from different sampling strategies. We only want to plot 1 version of the 
# foursquare (tist) dataset and therefore we filter all other study_names
ignore_studies = [
 'tist_u10000_counts_7days',
 'tist_u10000_counts_full',
 'tist_u10000_delaunay_7days',
 'tist_u10000_delaunay_full',
 'tist_u10000_fconndist_7days',
 'tist_u10000_fconndist_full',
 'tist_u1000_counts_7days',
 'tist_u1000_counts_full',
 'tist_u1000_delaunay_7days',
 'tist_u1000_delaunay_full',
 'tist_u1000_fconndist_7days',
 'tist_u1000_fconndist_full'
 ]


# ignore studies if we want to compare different tist sample strategies:
#ignore_studies = [
# 'gc2_counts_7days',
# 'gc2_counts_full',
# 'gc2_delaunay_7days',
# 'gc2_delaunay_full',
# 'gc2_fconndist_7days',
# 'gc2_fconndist_full',
# 'geolife_counts_7days',
# 'geolife_counts_full',
# 'geolife_delaunay_7days',
# 'geolife_delaunay_full',
# 'geolife_fconndist_7days',
# 'geolife_fconndist_full']

# iterate all combinations of regular expressions created above. This
# corresponds to the different graph creation methods
for expr in expressions:
     print(expr)
     regex = re.compile(expr)
     
     study_name_list = list(filter(regex.search, study_name_list_all))
     for ignore_item in ignore_studies:
         try:
            study_name_list.remove(ignore_item)
         except ValueError:
            pass
     print("\t", study_name_list)
 
      
     # plot non-log histograms
     for parameter, results_by_study_dict in result.items():   
         if '7days' in expr:
             fig, ax = plt.subplots(2,2, sharex='all', sharey='all')
         else: 
             fig, ax = plt.subplots(2,2, sharex='all')
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
         if '7days' in expr:
             fig, ax = plt.subplots(2,2, sharex='all', sharey='all')
         else: 
             fig, ax = plt.subplots(2,2, sharex='all')
             
         fig.suptitle(parameter+'_log')
         for i, study_name in enumerate(study_name_list):
             ax_this = ax[i//2][i%2]
             data = np.asarray(results_by_study_dict[study_name])
             data = np.log10(data+0.01)
             ax_this.hist(data, 50, log=True)
             ax_this.set_title(study_name)
        
         plt.savefig(GRAPH_IMAGES_FOLDER+parameter+'_'+regex.pattern[2:]+'_log.png')
         plt.close()