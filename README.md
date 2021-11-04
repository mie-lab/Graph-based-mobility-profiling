# Mobility Profiling Based on Graph Representations

## Preprocessing

### current data format 

dictionary with user id as keys and _activity graph objects_ as values e.g.,

```python
{1597: <future_trackintel.activity_graph.activity_graph at 0x158eee23520>,
1598: <future_trackintel.activity_graph.activity_graph at 0x158eee23fa0>,
1599: <future_trackintel.activity_graph.activity_graph at 0x158f0883c70>}
```

The activity graph class is defined in `future_trackintel.activity_graph`

#### attributes 
- G: the nx graph model
    - node ids start with 0
    - nodes have a location id, geometry and an optional extent geometry
    - It is planned that arbitrary features of staypoints, locations and context will be
    assigned to nodes
    - edges are directed 
    - can be adressed by the 3-tuple (node 1, node 2, edge type) e.g., 
      `G.edges[(0,2, 'transition_counts')]`. _transition_counts_ is currently the only edge type that is used at the moment.


### Analysis

Given the mobility graphs, our analysis is grouped into scripts in the folder `3_analysis`. The workflow is the following:

**1) Extract features**

First, all graph (and raw) mobility features are extracted from the graphs. Run
```
python 3_analysis/get_all_features.py --out_dir='out_features/final_1_n0'
```
The features are dumped in csv Files into a folder called `final_1_n0`, and then cleaned (outlier removal) and saved to a folder `final_1_n0_cleaned`.

Note: The studies can also be processed individually, for this use `python 3_analysis/graph_features.py -s study` and `python 3_analysis/raw_features.py -s study`.

**2) Merge datasets, compute correlation matrix and feature characteristics:**

Adjust the parameters that are hard-coded in the beginning of the file (input directory, studies to merge, etc), and run 
```
python 3_analysis/merge_datasets.py --inp_dir='out_features/final_1_n0_cleaned' --out_dir='results'
```
This will save a csv file with all graph features combined for all datasets (saved to the same folder as the input graph feature csvs), secondly a csv file with the averages per feature, and third a plot for the correlation matrix.
The latter two will be saved to the output directory called `results`.

**3) Identify user groups**

The user groups are identified by clustering multiple times with k for K-Means clustering. 
```
python 3_analysis/find_groups.py -i out_features/final_1_n0_cleaned -o results
```
The resulting user groups are saved in the file `3_analysis/groups.json` and copied to `results/groups.json` to keep everything together in the results folder.

NOTE: At this point, the groups are only named other_1, other_2 etc. They need to be renamed in the file [3_analysis/groups.json](3_analysis/groups.json) for further processing.

**4) Analyse the identified user groups wrt the features**

Run
```
python 3_analysis/analyze_study.py -i out_features/final_1_n0_cleaned -o results -s all_datasets
```
This will run the clustering multiple times again with the identified user groups, and compute the consistency. The user group appearing most often for each user will be saved in the output file `results/all_datasets_clustering.csv`.

Note: It is also possible to analyse a single study with the user groups. To do this, specify for example `-s gc1` in the command above.


**5) Cross sectional study with GC and YUMUV**

For the cross secional study, we use the assigned groups from above (`results/all_datasets_clustering.csv`). In this script we simply compare the assigned groups between control group and test group. Run
```
python 3_analysis/cross_sectional.py -i results
```

**6) Longitudinal study with GC1 and YUMUV**

Run the following to save all longitudinal plots into the results folder:
```
python 3_analysis/longitudinal.py -i results
```

**7) Label analysis**

For GC and YUMUV, the results of a user survey are also available, with questions about demographics and mobility behavior. We compare the replies of each user group vs the other user groups and save the results in a csv file (and plot significant ones). This is done by running
```
python 3_analysis/label_analysis.py -i results -s yumuv
```
or 
```
python 3_analysis/label_analysis.py -i results -s gc1
```

**8) Validation: comparison to raw features**

Run
```
python 3_analysis/analyze_graph_vs_raw.py -o results_quantile -i out_features/final_7_n0_quantile_cleaned -s all_datasets
```

