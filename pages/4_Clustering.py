import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Get the path of the current file (file1.py)
current_file_path = Path(__file__).resolve()
# Get the parent directory (folder1)
parent_dir = current_file_path.parent
# Get the path to the other folder (folder2)
other_folder_path = parent_dir.parent / "lib"
# Add the other folder to sys.path so Python can find the module
sys.path.append(str(other_folder_path))
# Now you can import from file2.py
from utils import parse_text_entry
from clustering import *

def add_to_clust_queue(name,model):
    st.session_state['clust_name_queue'].append(name)
    st.session_state['clust_model_queue'].append(model)
    print("Queue: ", st.session_state['clust_name_queue'])

def execute_clust_gridsearch():
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    gridsearch_clustering(names=st.session_state['clust_name_queue'],pipes=st.session_state['clust_model_queue'],
                          X=X_test,y=y_test,plot_number=3,save_best=True,log=True,stream=True)

st.sidebar.title("Clustering")

clust_model_tuple = ('affinity propagation', 'dbscan', 'optics', 'mean shift', 'k means',
               'bisecting k means', 'mini-batch k means', 'time series k means',
               'spectral clustering')

clust_model_box = st.sidebar.selectbox('**Select the model(s) you like to use.**', clust_model_tuple)

########################################################
# Model Variable Entries
########################################################

# Affinity Propagation
if clust_model_box == 'affinity propagation':
    ap_damp = st.sidebar.text_input("Dampening: floats", value='0.5', key="aff_prop_dampening")
    ap_max_it = st.sidebar.text_input("Max Iterations: None or integers", value='200', key="aff_prop_max_iterations")
    ap_verb = st.sidebar.text_input("Verbose: True, False", value='False', key="aff_prop_verbose")
    ap_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="aff_prop_random_state")


if clust_model_box == 'affinity propagation':
    if st.sidebar.button("Add Affinity Propagation to Clustering Queue"):    
        ap_dampening_list = parse_text_entry(ap_damp,'float')
        ap_max_iter_list = parse_text_entry(ap_max_it,'int')
        ap_verbose_list = parse_text_entry(ap_verb,'bool')
        ap_random_state_list = parse_text_entry(ap_rand,'int')
        name = "Affinity Propagation"
        aff_prop = pipeBuild_AffinityPropagation(damping=ap_dampening_list, max_iter=ap_max_iter_list, 
                                                 verbose=ap_verbose_list, random_state=ap_random_state_list[0])
        add_to_clust_queue(name,aff_prop)

# DBSCAN
if clust_model_box == 'dbscan':
    dbscan_eps = st.sidebar.text_input("Eps: floats", value='0.5', key="dbscan_eps")
    dbscan_min_samp = st.sidebar.text_input("Minimum Samples: integers", value='5', key="dbscan_min_samples")
    dbscan_metric = st.sidebar.text_input("Distance Metric: euclidean, manhattan, chebyshev, minkowski", value='euclidean', key="dbscan_metric")
    dbscan_algo = st.sidebar.text_input("Algorithm: auto, ball_tree, kd_tree, brute", value='auto', key="dbscan_algorithm")
    dbscan_ls = st.sidebar.text_input("Leaf Size: integers", value='30', key="dbscan_leaf_size")
    dbscan_p = st.sidebar.text_input("Power: floats", value='None', key="dbscan_p")


if clust_model_box == 'dbscan':
    if st.sidebar.button("Add DBSCAN to Clustering Queue"):    
        dbscan_eps_list = parse_text_entry(dbscan_eps,'float')
        dbscan_min_samp_list = parse_text_entry(dbscan_min_samp,'int')
        dbscan_metric_list = parse_text_entry(dbscan_metric,'string')
        dbscan_algo_list = parse_text_entry(dbscan_algo,'string')
        dbscan_ls_list = parse_text_entry(dbscan_ls,'int')
        dbscan_p_list = parse_text_entry(dbscan_p,'float')
        
        name = "DBSCAN"
        dbscan = pipeBuild_DBSCAN(eps=dbscan_eps_list, min_samples=dbscan_min_samp_list, metric=dbscan_metric_list, 
                                           algorithm=dbscan_algo_list, leaf_size=dbscan_ls_list, p=dbscan_p_list)
        add_to_clust_queue(name,dbscan)

# OPTICS
if clust_model_box == 'optics':
    optics_method = st.sidebar.text_input("Cluster Method: dbscan, xi", value='xi', key="optics_method")
    optics_xi = st.sidebar.text_input("Xi: floats between 0 and 1", value='0.05', key="optics_xi")
    optics_eps = st.sidebar.text_input("Eps: floats", value='0.5', key="optics_eps")
    optics_min_samp = st.sidebar.text_input("Minimum Samples: integers", value='5', key="optics_min_samples")
    optics_metric = st.sidebar.text_input("Distance Metric: euclidean, manhattan, chebyshev, minkowski", value='euclidean', key="optics_metric")
    optics_algo = st.sidebar.text_input("Algorithm: auto, ball_tree, kd_tree, brute", value='auto', key="optics_algorithm")
    optics_ls = st.sidebar.text_input("Leaf Size: integers", value='30', key="optics_leaf_size")
    optics_p = st.sidebar.text_input("Minkowski Parameter: 1 or 2 only", value='2', key="optics_p")


if clust_model_box == 'optics':
    if st.sidebar.button("Add OPTICS to Clustering Queue"):
        optics_method_list = parse_text_entry(optics_method,'string')
        optics_xi_list = parse_text_entry(optics_xi,'float')    
        optics_eps_list = parse_text_entry(optics_eps,'float')
        optics_min_samp_list = parse_text_entry(optics_min_samp,'int')
        optics_metric_list = parse_text_entry(optics_metric,'string')
        optics_algo_list = parse_text_entry(optics_algo,'string')
        optics_ls_list = parse_text_entry(optics_ls,'int')
        optics_p_list = parse_text_entry(optics_p,'float')
        
        name = "OPTICS"
        optics = pipeBuild_OPTICS(cluster_method=optics_method_list, xi=optics_xi_list, eps=optics_eps_list, 
                                  min_samples=optics_min_samp_list, metric=optics_metric_list, 
                                  algorithm=optics_algo_list, leaf_size=optics_ls_list, p=optics_p_list)
        add_to_clust_queue(name,optics)


# Mean Shift
if clust_model_box == 'mean shift':
    ms_bw = st.sidebar.text_input("Bandwidth: None or floats", value='None', key="mean_shift_bandwidth")
    ms_bs = st.sidebar.text_input("Bin Seeding: True, False", value='True', key="mean_shift_bin_seeding")
    ms_mbf = st.sidebar.text_input("Minimum Bin Frequency: integers", value='1', key="mean_shift_min_bin_freq")
    ms_ca = st.sidebar.text_input("Cluster All: True, False", value='True', key="mean_shift_cluster_all")
    ms_max_it = st.sidebar.text_input("Max Iterations: integers", value='300', key="mean_shift_max_iter")


if clust_model_box == 'mean shift':
    if st.sidebar.button("Add Mean Shift to Clustering Queue"):    
        ms_bandwidth_list = parse_text_entry(ms_bw,'float')
        ms_bin_list = parse_text_entry(ms_bs,'bool')
        ms_min_freq_list = parse_text_entry(ms_mbf,'int')
        ms_cluster_list = parse_text_entry(ms_ca,'bool')
        ms_max_iter_list = parse_text_entry(ms_max_it,'int')

        name = "Mean Shift"
        mean_shift = pipeBuild_MeanShift(bandwidth=ms_bandwidth_list, bin_seeding=ms_bin_list, 
                                         min_bin_freq=ms_min_freq_list, cluster_all=ms_cluster_list, 
                                         max_iter=ms_max_iter_list)
        add_to_clust_queue(name,mean_shift)


# K Means
if clust_model_box == 'k means':
    km_nc = st.sidebar.text_input("Number of Clusters: integers", value='8', key="kmeans_n_clusters")
    km_init = st.sidebar.text_input("Initial Cluster Centroids: k-means++ , random", value='k-means++', key="kmeans_init")
    km_n_init = st.sidebar.text_input("Number of Initializations: auto or integers", value='10', key="kmeans_n_init")
    km_mi = st.sidebar.text_input("Max Iterations: integers", value='300', key="kmeans_max_iter")
    km_tol = st.sidebar.text_input("Tolerance: floats only", value='0.0001', key="kmeans_tol")
    km_algo = st.sidebar.text_input("Algorithm: auto, lloyd, elkan, full", value='lloyd', key="kmeans_algorithm")
    km_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="kmeans_random_state")

if clust_model_box == 'k means':
    if st.sidebar.button("Add K Means to Clustering Queue"):
        km_n_clusters_list = parse_text_entry(km_nc,'int')
        km_init_list = parse_text_entry(km_init,'string')

        new_string = km_n_init.replace("auto", "None") # replace 'auto' in string with 'None' before using parse   
        km_n_init_list = parse_text_entry(km_n_init,'int') # parse will not work with 'auto', but will work with 'None'
        km_n_init_list = ['auto' if item == None else item for item in km_n_init_list] # replace None in list with 'auto' after parse

        km_max_iter_list = parse_text_entry(km_mi,'int')
        km_tol_list = parse_text_entry(km_tol,'float')        
        km_algo_list = parse_text_entry(km_algo,'string')
        km_rand_state_list = parse_text_entry(km_rand,'int')
                
        name = "K Means"
        k_means = pipeBuild_KMeans(n_clusters=km_n_clusters_list, init=km_init_list, n_init=km_n_init_list,
                                   max_iter=km_max_iter_list, tol=km_tol_list, 
                                   random_state=km_rand_state_list[0],algorithm=km_algo_list)
        add_to_clust_queue(name,k_means)


# Bisecting K Means
if clust_model_box == 'bisecting k means':
    bskm_nc = st.sidebar.text_input("Number of Clusters: integers", value='8', key="bi_kmeans_n_clusters")
    bskm_init = st.sidebar.text_input("Initial Cluster Centroids: k-means++ , random", value='k-means++', key="bi_kmeans_init")
    bskm_n_init = st.sidebar.text_input("Number of Initializations: integers", value='1', key="bi_kmeans_n_init")
    bskm_mi = st.sidebar.text_input("Max Iterations: integers", value='300', key="bi_kmeans_max_iter")
    bskm_tol = st.sidebar.text_input("Tolerance: floats only", value='0.0001', key="bi_kmeans_tol")
    bskm_algo = st.sidebar.text_input("Algorithm: auto, lloyd, elkan, full", value='lloyd', key="bi_kmeans_algorithm")
    bskm_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="bi_kmeans_random_state")
    bskm_strat = st.sidebar.text_input("Bisecting Strategy: biggest_inertia, largest_cluster", value='biggest_inertia', key="bi_kmeans_bisecting_strategy")

if clust_model_box == 'bisecting k means':
    if st.sidebar.button("Add Bisecting K Means to Clustering Queue"):
        bskm_n_clusters_list = parse_text_entry(bskm_nc,'int')
        bskm_init_list = parse_text_entry(bskm_init,'string')    
        bskm_n_init_list = parse_text_entry(bskm_n_init,'int')
        bskm_max_iter_list = parse_text_entry(bskm_mi,'int')
        bskm_tol_list = parse_text_entry(bskm_tol,'float')        
        bskm_algo_list = parse_text_entry(bskm_algo,'string')
        bskm_rand_state_list = parse_text_entry(bskm_rand,'int')
        bskm_strategy_list = parse_text_entry(bskm_strat,'string')
                
        name = "Bisecting K Means"
        bsk_means = pipeBuild_BisectingKMeans(n_clusters=bskm_n_clusters_list, init=bskm_init_list, 
                                              n_init=bskm_n_init_list, max_iter=bskm_max_iter_list, tol=bskm_tol_list,
                                              random_state=bskm_rand_state_list[0], algorithm=bskm_algo_list,
                                              bisecting_strategy=bskm_strategy_list)
        add_to_clust_queue(name,bsk_means)


# Mini-Batch K Means
if clust_model_box == 'mini-batch k means':
    mbkm_nc = st.sidebar.text_input("Number of Clusters: integers", value='8', key="mini_kmeans_n_clusters")
    mbkm_init = st.sidebar.text_input("Initial Cluster Centroids: k-means++ , random", value='k-means++', key="mini_kmeans_init")
    mbkm_n_init = st.sidebar.text_input("Number of Initializations: integers", value='10', key="mini_kmeans_n_init")
    mbkm_mi = st.sidebar.text_input("Max Iterations: integers", value='100', key="mini_kmeans_max_iter")
    mbkm_tol = st.sidebar.text_input("Tolerance: floats only", value='0.0', key="mini_kmeans_tol")
    mbkm_bs = st.sidebar.text_input("Batch Size: integers", value='1024', key="mini_kmeans_batch_size")
    mbkm_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="mini_kmeans_random_state")    

if clust_model_box == 'mini-batch k means':
    if st.sidebar.button("Add Mini-Batch K Means to Clustering Queue"):
        mbkm_n_clusters_list = parse_text_entry(mbkm_nc,'int')
        mbkm_init_list = parse_text_entry(mbkm_init,'string')

        new_string = mbkm_n_init.replace("auto", "None") # replace 'auto' in string with 'None' before using parse   
        mbkm_n_init_list = parse_text_entry(mbkm_n_init,'int') # parse will not work with 'auto', but will work with 'None'
        mbkm_n_init_list = ['auto' if item == None else item for item in mbkm_n_init_list] # replace None in list with 'auto' after parse

        mbkm_max_iter_list = parse_text_entry(mbkm_mi,'int')
        mbkm_tol_list = parse_text_entry(mbkm_tol,'float')
        mbkm_rand_state_list = parse_text_entry(mbkm_rand,'int')
        mbkm_batch_size_list = parse_text_entry(mbkm_bs,'int')
                
        name = "Mini-Batch K Means"
        mbk_means = pipeBuild_MiniBatchKMeans(n_clusters=mbkm_n_clusters_list, init=mbkm_init_list, 
                                              n_init=mbkm_n_init_list, max_iter=mbkm_max_iter_list, tol=mbkm_tol_list,
                                              random_state=mbkm_rand_state_list[0], batch_size=mbkm_batch_size_list)
        add_to_clust_queue(name,mbk_means)



# Time Series K Means
if clust_model_box == 'time series k means':
    tskm_nc = st.sidebar.text_input("Number of Clusters: integers", value='3', key="ts_kmeans_n_clusters")
    tskm_init = st.sidebar.text_input("Initial Cluster Centroids: k-means++ , random", value='k-means++', key="ts_kmeans_init")
    tskm_n_init = st.sidebar.text_input("Number of Initializations: integers", value='1', key="ts_kmeans_n_init")
    tskm_mi = st.sidebar.text_input("Max Iterations: integers", value='50', key="ts_kmeans_max_iter")
    tskm_tol = st.sidebar.text_input("Tolerance: floats only", value='0.000001', key="ts_kmeans_tol")
    tskm_metric = st.sidebar.text_input("Distance Metric: euclidean, dtw, softdtw", value='dtw', key="ts_kmeans_metric")
    tskm_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="ts_kmeans_random_state")

if clust_model_box == 'time series k means':
    if st.sidebar.button("Add Time Series K Means to Clustering Queue"):
        tskm_n_clusters_list = parse_text_entry(tskm_nc,'int')
        tskm_init_list = parse_text_entry(tskm_init,'string')
        tskm_n_init_list = parse_text_entry(tskm_n_init,'int')
        tskm_max_iter_list = parse_text_entry(tskm_mi,'int')
        tskm_tol_list = parse_text_entry(tskm_tol,'float')        
        tskm_metric_list = parse_text_entry(tskm_metric,'string')
        tskm_rand_state_list = parse_text_entry(tskm_rand,'int')
                
        name = "Time Series K Means"
        tsk_means = pipeBuild_TimeSeriesKMeans(n_clusters=tskm_n_clusters_list, init=tskm_init_list, n_init=tskm_n_init_list,
                                               max_iter=tskm_max_iter_list, tol=tskm_tol_list,
                                               random_state=tskm_rand_state_list[0], metric=tskm_metric_list)
        add_to_clust_queue(name,tsk_means)


# Spectral Clustering
if clust_model_box == 'spectral clustering':
    spec_n_c = st.sidebar.text_input("Number of Clusters: integers", value='8', key="spectral_n_clusters")
    spec_es = st.sidebar.text_input("Eigen Solver: None, arpack, lobpcg, amg", value='None', key="spectral_eigen_solver")
    spec_ni = st.sidebar.text_input("Number of Initializations: integers", value='10', key="spectral_n_init")
    spec_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="spectral_random_state")


if clust_model_box == 'spectral clustering':
    if st.sidebar.button("Add Spectral Clustering to Clustering Queue"):    
        spec_n_clusters_list = parse_text_entry(spec_n_c,'int')
        spec_eigen_solver_list = parse_text_entry(spec_es,'string')
        spec_n_init_list = parse_text_entry(spec_ni,'int')
        spec_rand_state_list = parse_text_entry(spec_rand,'int')

        name = "Spectral Clustering"
        spectral = pipeBuild_SpectralClustering(n_clusters=spec_n_clusters_list, eigen_solver=spec_eigen_solver_list, 
                                                random_state=spec_rand_state_list[0], n_init=spec_n_init_list)
        add_to_clust_queue(name,spectral)


########################################################
# End Model Variable Entries
########################################################

if st.sidebar.button("Show Clustering Model Queue"):
    print(st.session_state['clust_name_queue'])
    st.write(str(st.session_state['clust_name_queue']))

if st.sidebar.button("Clear Clustering Model Queue"):
    st.session_state['clust_name_queue'] = []
    st.session_state['clust_model_queue'] = []
    st.session_state["clust log loader"] = None
    st.session_state['show_clust_log'] = False

if st.sidebar.button("Run Clustering Grid Search"):
    print("Clustering gridsearch started")
    st.session_state['show_clust_log'] = True
    execute_clust_gridsearch()

# if st.session_state['show_clust_log'] == True:
#     log_file = st.file_uploader("Choose a txt file", type="txt",key="clust log loader")
#     if log_file is not None:
#         root, extension = os.path.splitext(log_file.name)
#         if extension.lower() == ".txt":
#             bytes_data = log_file.getvalue()  
#             string_data = bytes_data.decode('utf-8') 
#             # Display the content
#             st.write("File Content:")
#             st.code(string_data, language="text") # Use st.code for displaying raw text            
#         else:
#             st.write("You have selected an incorrect file type")