# Mobility Profiling Based on Graph Representations

This is the code for the paper _Graph Based Mobility Profiling_ available [here](https://www.sciencedirect.com/science/article/pii/S0198971522001545) (open access)
## Import and preprocessing

### Import data

To reproduce the results of the paper, all five datasets would be needed. However, only the Foursquare and the Geolife datasets are publicly available. For the work in the paper, all datasets were imported into a Postgresql/PostGIS database. To allows the reproduction of the results to some extent, we have modified the pipeline so that it can be run using only the publicly available datasets (Geolife + Foursquare) using .csv files. The first step is to download and read the raw data. 
Geolife can be downloaded [here](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) and Foursquare [here](https://drive.google.com/file/d/1PNk3zY8NjLcDiAbzjABzY5FiPAFHq6T8/view?usp=sharing).

Read and preprocess Geolife:
```
python 1_import_csv/import_geolife_csv.py -d path/to/geolife_data_folder
```

Read Foursquare data and preprocess:
```
python 1_import_csv/import_foursquare_csv.py -c path/to/checkins -p path/to/pois
```

The preprocessed data is saved in a new folder `data/raw`.

### Generate graphs

Generate graphs for Geolife and Foursquare data
```
python 2_preprocessing_csv/generate_graphs_csv.py -i data/raw
```
The graphs are saved as pickle files in the folder `data/graph_data`

## Analysis

Given the mobility graphs, our analysis is grouped into scripts in the folder `3_analysis`. The workflow is the following:

**1) Extract features**

First, all graph (and raw) mobility features are extracted from the graphs. Run
```
python 3_analysis/get_all_features.py --in_path data/graph_data --out_dir='out_features/final_1_n0'
```
NOTE: this can take up to half an hour. It is computing all features for all graphs.

If you also want to experiment with the basic non-graph features, run
```
python 3_analysis/get_all_features.py --in_path data/raw --out_dir='out_features/final_1_n0' --f raw
```

The features are dumped in csv Files into a folder called `final_1_n0`, and then cleaned (outlier removal) and saved to a folder `final_1_n0_cleaned`.

**2) Merge datasets, compute correlation matrix and feature characteristics:**

Adjust the parameters that are hard-coded in the beginning of the file (input directory, studies to merge, etc), and run 
```
python 3_analysis/merge_datasets.py --inp_dir='out_features/final_1_n0'
```
This will remove outliers and save the csvs with all graph features combined for all datasets in a new folder, which is named the same as the `inp_dir` folder but with the suffiz `_cleaned`. 

(To merge the datasets of the computed basic features, add the flag `--feat_type=raw`)

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

**5) Validation: comparison to raw features**

Run
```
python 3_analysis/analyze_graph_vs_raw.py -o results -i out_features/final_1_n0_cleaned -s all_datasets
```

## MaaS Impact Analysis

**All further steps can only be reproduced with full data access, as they rely on the Green Class and Yumuv data**

**6) Transform other features to the identified user groups**

After step 4, all users in the five main datasets have their (most consistent) group assigned. In step 4, we also saved one specific clustering C with the k that had the highest correspondence with the consistent user groups. Now, for the MAAS applictations, we need to transform the features of control group / test group to the clustering C.

Run
```
python 3_analysis/transform_new_features.py -i out_features/final_1_n0_cleaned -o results
```
This will output files `long_yumuv_clustering.csv` and the same for gc1 and gc2 into the `results` folder. 

**7) Cross sectional study with GC and YUMUV**

For the cross secional study, we use the assigned groups from above (`results/all_datasets_clustering.csv`). In this script we simply compare the assigned groups between control group and test group. Run
```
python 3_analysis/cross_sectional.py -i results
```

**8) Longitudinal study with GC1 and YUMUV**

Run the following to save all longitudinal plots into the results folder:
```
python 3_analysis/longitudinal.py -i results
```

**9) Label analysis**

For GC and YUMUV, the results of a user survey are also available, with questions about demographics and mobility behavior. We compare the replies of each user group vs the other user groups and save the results in a csv file (and plot significant ones). This is done by running
```
python 3_analysis/label_analysis.py -i results -s yumuv_graph_rep
```
or 
```
python 3_analysis/label_analysis.py -i results -s gc1
```
