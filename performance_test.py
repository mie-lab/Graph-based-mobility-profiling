# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:23:51 2019

@author: martinhe
"""

import pandas as pd
from functools import partial
import timeit
import numpy as np
import matplotlib.pyplot as plt

import sklearn
import trackintel as ti
from trackintel.geogr.distances import calculate_distance_matrix
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import pyproj
import matplotlib

       

coord_in=pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by GPS units and Google Earth
coord_out=pyproj.Proj("+init=EPSG:27700") # UK Ordnance Survey, 1936 datum
plt.rcParams["figure.figsize"] = (16,9)
fontSizeAll = 18
font = {     'size'   : fontSizeAll}
matplotlib.rc('font', **font)

# https://codereview.stackexchange.com/questions/165245/plot-timings-for-a-range-of-inputs
def input_data(n):
    return pd.DataFrame(data=np.random.normal(size=(int(n),2)),
                        columns=['lat','long'])

def input_data_proj(n):
    
    data = pyproj.transform(coord_in, 
                            coord_out, 
                            np.random.normal(size=(int(n),1)),
                            np.random.normal(size=(int(n),1))) 
    data = np.concatenate(data, axis=1) 
    
    return pd.DataFrame(data=data,
                        columns=['lat','long'])
    
def plot_times(functions, inputs, repeats=3, n_tests=1, file_name=""):
    timings = get_timings(functions, inputs, repeats=3, n_tests=1)
    results = aggregate_results(timings)
    fig, ax = plot_results(results)

    return fig, ax, results

def get_timings(functions, inputs, repeats, n_tests):
    for func in functions:
        
        data = (timeit.Timer(partial(func, i)).repeat(repeat=repeats, number=n_tests) for i in inputs)
        
        result = pd.DataFrame(index = inputs, columns = range(repeats), 
            data=data)
        yield func, result
        
def aggregate_results(timings):
    empty_multiindex = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names=['func', 'result'])
    aggregated_results = pd.DataFrame(columns=empty_multiindex)

    for func, timing in timings:
        for measurement in timing:
            aggregated_results[func.__name__, measurement] = timing[measurement]
        aggregated_results[func.__name__, 'avg'] = timing.mean(axis=1)
        aggregated_results[func.__name__, 'yerr'] = timing.std(axis=1)

    return aggregated_results

def plot_results(results):
    fig, ax = plt.subplots()
    x = results.index
    for func in results.columns.levels[0]:
        y = results[func, 'avg']
        yerr = results[func, 'yerr']        
        ax.errorbar(x, y, yerr=yerr, fmt='-o', label=func, linewidth=3, 
                    elinewidth=1)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Input')
    ax.set_ylabel('Time [s]')
    ax.legend()    
    return fig, ax


# --------------------------------


#
#n = 100
#
#z = calculate_distance_matrix(data)
#
#cdist(xy, xy, metric=haversine_dist_cdist)




def distance_matrix_trackintel_haversine(n):
    calculate_distance_matrix(points=input_data(n))

def calculate_distance_matrix_scikit_haversine(n):
    calculate_distance_matrix(points=input_data(n), dist_metric='test_haversine')

def calculate_distance_matrix_scipy_euclidean(n):
    calculate_distance_matrix(points=input_data_proj(n), dist_metric='euclidean', n_jobs=None)
    
def calculate_distance_matrix_scipy_euclidean_multicore(n):
    calculate_distance_matrix(points=input_data_proj(n), dist_metric='euclidean', n_jobs=-1) 

def calculate_distance_matrix_scikit_euclidean(n):
    calculate_distance_matrix(points=input_data_proj(n), dist_metric='l2')

def calculate_distance_matrix_scikit_euclidean_multicore(n):
    calculate_distance_matrix(points=input_data_proj(n), dist_metric='l2', n_jobs=-1) 
    
functions = (distance_matrix_trackintel_haversine,
             calculate_distance_matrix_scikit_haversine,
             calculate_distance_matrix_scipy_euclidean,
#             calculate_distance_matrix_scipy_euclidean_multicore,
#             calculate_distance_matrix_scikit_euclidean,
#             calculate_distance_matrix_scikit_euclidean_multicore
)
fig, ax = plot_times(functions, inputs=np.logspace(1,3,30), repeats=1, n_tests=1, file_name="")

plt.savefig("dist_matrix_performance.png")

