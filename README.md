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
python 3_analysis/analyze_across_datasets.py --inp_dir='out_features/final_1_n0' --out_dir='results'
```
This will save a csv file with all graph features combined for all datasets (saved to the same folder as the input graph feature csvs), secondly a csv file with the averages per feature, and third a plot for the correlation matrix.
The latter two will be saved to the output directory called `results`.






