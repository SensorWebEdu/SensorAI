import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import time

# Get the path of the current file (file1.py)
current_file_path = Path(__file__).resolve()
# Get the parent directory (folder1)
parent_dir = current_file_path.parent
# Get the path to the other folder (folder2)
other_folder_path = parent_dir.parent / "lib"
# Add the other folder to sys.path so Python can find the module
sys.path.append(str(other_folder_path))
# Now you can import from file2.py
from utils import parse_text_entry, display_log_updates
from classification import *

def add_to_class_queue(name,model):
    st.session_state['class_name_queue'].append(name)
    st.session_state['class_model_queue'].append(model)
    print("Queue: ", st.session_state['class_name_queue'])

def execute_class_gridsearch():
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    gridsearch_classifier(names=st.session_state['class_name_queue'],pipes=st.session_state['class_model_queue'],
                              X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,
                              plot_number=3,scoring="neg_mean_squared_error",save_best=True,log=True,stream=True)
    


st.sidebar.title("Classification")

model_tuple = ('decision tree', 'extra trees', 'random forest', 'gradient boosting',
               'k nearest neighbors', 'nearest centroid', 'radius nearest neighbors',
               'time series knn','support vector', 'nu support vector', 'time series svc')

model_box = st.sidebar.selectbox('**Select the model(s) you like to use.**', model_tuple)

########################################################
# Model Variable Entries
########################################################

# Decision Tree
if model_box == 'decision tree':
    dec_crit = st.sidebar.text_input("Criterion: gini, entropy, or log_loss", value='gini', key="dec crit")
    dec_split = st.sidebar.text_input("Splitter: best, random", value='best', key="dec split")
    dec_max_d = st.sidebar.text_input("Maximum Tree Depth: integers", value='None', key="dec max depth")
    dec_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="dec random state")

if model_box == 'decision tree':
    if st.sidebar.button("Add Decision Tree to Classifier Queue"):    
        criterion_list = parse_text_entry(dec_crit,'string')
        splitter_list = parse_text_entry(dec_split,'string')
        max_depth_list = parse_text_entry(dec_max_d,'int')
        random_state_list = parse_text_entry(dec_rand,'int')
        name = "Decision Tree"
        decision_tree = pipeBuild_DecisionTreeClassifier(criterion=criterion_list,splitter=splitter_list,max_depth=max_depth_list,random_state=random_state_list[0])
        add_to_class_queue(name,decision_tree)


# Extra Trees
if model_box == 'extra trees':
    extra_ne = st.sidebar.text_input("Number of Estimators: integers", value='100', key="extra_n_estimators")
    extra_crit = st.sidebar.text_input("Criterion: gini, entropy, or log_loss", value='gini', key="extra_criterion")
    extra_max_d = st.sidebar.text_input("Maximum Tree Depth: integers", value='None', key="extra_max_depth")
    extra_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="extra_random_state")
    extra_ms_split = st.sidebar.text_input("Minimum Sample # for Node Splits: integers", value='2', key="extra_min_samples_split")
    extra_ms_leaf = st.sidebar.text_input("Minimum Sample # Required for a Leaf: integers", value='1', key="extra_min_samples_leaf")

if model_box == 'extra trees':
    if st.sidebar.button("Add Extra Trees to Classifier Queue"): 
        extra_ne_list = parse_text_entry(extra_ne,'int')   
        extra_crit_list = parse_text_entry(extra_crit,'string')
        extra_max_d_list = parse_text_entry(extra_max_d,'int')
        extra_rand_list = parse_text_entry(extra_rand,'int')
        extra_ms_split_list = parse_text_entry(extra_ms_split,'int')
        extra_ms_leaf_list = parse_text_entry(extra_ms_leaf,'int')
        name = "Extra Trees"
        extra_trees = pipeBuild_ExtraTreesClassifier(n_estimators=extra_ne_list,criterion=extra_crit_list,max_depth=extra_max_d_list,min_samples_split=extra_ms_split_list,min_samples_leaf=extra_ms_leaf_list,random_state=extra_rand_list[0])
        add_to_class_queue(name,extra_trees)

# Random Forest
if model_box == 'random forest':
    rfor_ne = st.sidebar.text_input("Number of Estimators: integers", value='100', key="forest_n_estimators")
    rfor_crit = st.sidebar.text_input("Criterion: gini, entropy, or log_loss", value='gini', key="forest_criterion")
    rfor_max_d = st.sidebar.text_input("Maximum Tree Depth: integers", value='None', key="forest_max_depth")
    rfor_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="forest_random_state")
    

if model_box == 'random forest':
    if st.sidebar.button("Add Random Forest to Classifier Queue"):
        rfor_ne_list = parse_text_entry(rfor_ne,'int')    
        rfor_crit_list = parse_text_entry(rfor_crit,'string')
        rfor_max_d_list = parse_text_entry(rfor_max_d,'int')
        rfor_rand_list = parse_text_entry(rfor_rand,'int')        
        name = "Random Forest"
        random_forest = pipeBuild_RandomForestClassifier(n_estimators=rfor_ne_list,criterion=rfor_crit_list,max_depth=rfor_max_d_list,random_state=rfor_rand_list[0])
        add_to_class_queue(name,random_forest)

# Gradient Boosting
if model_box == 'gradient boosting':
    gb_ne = st.sidebar.text_input("Number of Estimators: integers", value='100', key="gradient_n_estimators")
    gb_crit = st.sidebar.text_input("Criterion: friedman_mse, squared_error", value='friedman_mse', key="gradient_criterion")
    gb_max_d = st.sidebar.text_input("Maximum Tree Depth: integers", value='3', key="gradient_max_depth")
    gb_rand = st.sidebar.text_input("Random State: None or a single integer", value='None', key="gradient_random_state")
    gb_ms_split = st.sidebar.text_input("Minimum Sample # for Node Splits: integers", value='2', key="gradient_min_samples_split")
    gb_ms_leaf = st.sidebar.text_input("Minimum Sample # Required for a Leaf: integers", value='1', key="gradient_min_samples_leaf")

if model_box == 'gradient boosting':
    if st.sidebar.button("Add Gradient Boosting to Classifier Queue"):
        gb_ne_list = parse_text_entry(gb_ne,'int')    
        gb_crit_list = parse_text_entry(gb_crit,'string')
        gb_max_d_list = parse_text_entry(gb_max_d,'int')
        gb_rand_list = parse_text_entry(gb_rand,'int')
        gb_ms_split_list = parse_text_entry(gb_ms_split,'int')
        gb_ms_leaf_list = parse_text_entry(gb_ms_leaf,'int')
        name = "Gradient Boosting"
        gradient_boosting = pipeBuild_GradientBoostingClassifier(n_estimators=gb_ne_list, criterion=gb_crit_list,
                                                                 min_samples_split=gb_ms_split_list, 
                                                                 min_samples_leaf=gb_ms_leaf_list, max_depth=gb_max_d_list,
                                                                 random_state=gb_rand_list[0])
        add_to_class_queue(name,gradient_boosting)

# K Nearest Neighbors
if model_box == 'k nearest neighbors':
    knn_nn = st.sidebar.text_input("Number of Neighbors: integers only", value='5', key="knn_n_neighbors")
    knn_weight = st.sidebar.text_input("Weights: uniform, distance", value='uniform', key="knn_weights")
    knn_algo = st.sidebar.text_input("Algorithm: auto, ball_tree, kd_tree, brute", value='auto', key="knn_algoritm")
    knn_ls = st.sidebar.text_input("Leaf Size: integers only", value='30', key="knn_leaf_size")

if model_box == 'k nearest neighbors':
    if st.sidebar.button("Add K Nearest Neighbors to Classifier Queue"):
        knn_nn_list = parse_text_entry(knn_nn,'int')    
        knn_weight_list = parse_text_entry(knn_weight,'string')
        knn_algo_list = parse_text_entry(knn_algo,'string')
        knn_ls_list = parse_text_entry(knn_ls,'int')
        name = "K Nearest Neighbors"
        knn = pipeBuild_KNeighborsClassifier(n_neighbors=knn_nn_list, weights=knn_weight_list, 
                                             algorithm=knn_algo_list, leaf_size=knn_ls_list)
        add_to_class_queue(name,knn)

# Nearest Centroid
if model_box == 'nearest centroid':
    nc_metric = st.sidebar.text_input("Distance Metric: euclidean, manhattan, chebyshev, minkowski", value='euclidean', key="ncent_metric")
    nc_shrink = st.sidebar.text_input("Shrink Threshold: None and floaty", value='None', key="ncent_shrink")

if model_box == 'nearest centroid':
    if st.sidebar.button("Add Nearest Centroid to Classifier Queue"):
        nc_metric_list = parse_text_entry(nc_metric,'string')
        nc_shrink_list = parse_text_entry(nc_shrink,'float')
        name = "Nearest Centroid"
        nearest_centroid = pipeBuild_NearestCentroid(metric=nc_metric_list, shrink_threshold=nc_shrink_list)
        add_to_class_queue(name,nearest_centroid)


# Radiaus Nearest Neighbors
if model_box == 'radius nearest neighbors':
    rnn_rad = st.sidebar.text_input("Radius: floats only", value='1.0', key="radius_nn_n_neighbors")
    rnn_weight = st.sidebar.text_input("Weights: uniform, distance", value='uniform', key="radius_nn_weights")
    rnn_algo = st.sidebar.text_input("Algorithm: auto, ball_tree, kd_tree, brute", value='auto', key="radius_nn_algoritm")
    rnn_ls = st.sidebar.text_input("Leaf Size: integers only", value='30', key="radius_nn_leaf_size")
    rnn_p = st.sidebar.text_input("Power: integers only", value='2', key="radius_nn_p")
    rnn_metric = st.sidebar.text_input("Distance Metric: euclidean, manhattan, chebyshev, minkowski", value='minkowski', key="radius_nn_metric")

if model_box == 'radius nearest neighbors':
    if st.sidebar.button("Add Radius Nearest Neighbors to Classifier Queue"):
        rnn_rad_list = parse_text_entry(rnn_rad,'float')    
        rnn_weight_list = parse_text_entry(rnn_weight,'string')
        rnn_algo_list = parse_text_entry(rnn_algo,'string')
        rnn_ls_list = parse_text_entry(rnn_ls,'int')
        rnn_p_list = parse_text_entry(rnn_p,'int')
        rnn_metric_list = parse_text_entry(rnn_metric,'string')
        name = "Radius Nearest Neighbors"
        radius_nn = pipeBuild_RadiusNeighborsClassifier(radius=rnn_rad_list, weights=rnn_weight_list,
                                                        algorithm=rnn_algo_list, leaf_size=rnn_ls_list,
                                                        p=rnn_p_list, metric=rnn_metric_list)
        add_to_class_queue(name,radius_nn)

# Time Series K Nearest Neighbors
if model_box == 'time series knn':
    tsknn_nn = st.sidebar.text_input("Number of Neighbors: integers only", value='5', key="tsknn_n_neighbors")
    tsknn_weight = st.sidebar.text_input("Weights: uniform, distance", value='uniform', key="tsknn_weights")
    tsknn_metric = st.sidebar.text_input("Distance Metric: dtw, softdtw, ctw, sqeuclidean, sax", value='dtw', key="tsknn_metric")

if model_box == 'time series knn':
    if st.sidebar.button("Add Time Series KNN to Classifier Queue"):
        tsknn_nn_list = parse_text_entry(tsknn_nn,'int')    
        tsknn_weight_list = parse_text_entry(tsknn_weight,'string')
        tsknn_metric_list = parse_text_entry(tsknn_metric,'string')
        name = "Time Series KNN"
        tsknn = pipeBuild_KNeighborsTimeSeriesClassifier(n_neighbors=tsknn_nn_list, weights=tsknn_weight_list, 
                                                         metric=tsknn_metric_list)
        add_to_class_queue(name,tsknn)

# Support Vector Classifier
if model_box == 'support vector':
    svc_c = st.sidebar.text_input("Regularization Parameter: floats only", value='1.0', key="svc_c")
    svc_kernel = st.sidebar.text_input("Kernel: linear, poly, rbf, sigmoid", value='rbf', key="svc_kernel")
    svc_degree = st.sidebar.text_input("Degree (for poly kernel): integers only", value='3', key="svc_degree")
    svc_gamma = st.sidebar.text_input("Gamma: scale, auto", value='scale', key="svc_gamma")
    svc_tol = st.sidebar.text_input("Tolerance: floats only", value='0.001', key="svc_tol")


if model_box == 'support vector':
    if st.sidebar.button("Add Support Vector to Classifier Queue"):
        svc_c_list = parse_text_entry(svc_c,'float')    
        svc_kernel_list = parse_text_entry(svc_kernel,'string')
        svc_degree_list = parse_text_entry(svc_degree,'int')
        svc_gamma_list = parse_text_entry(svc_gamma,'string')
        svc_tol_list = parse_text_entry(svc_tol,'float')

        name = "Support Vector Classifier"
        svc = pipeBuild_SVC(C=svc_c_list, kernel=svc_kernel_list, degree=svc_degree_list,
                                  gamma=svc_gamma_list, tol=svc_tol_list)
        add_to_class_queue(name,svc)

# Nu Support Vector Classifier
if model_box == 'nu support vector':
    nusvc_nu = st.sidebar.text_input("Regularization Parameter: floats only", value='0.5', key="nusvc_nu")
    nusvc_kernel = st.sidebar.text_input("Kernel: linear, poly, rbf, sigmoid", value='rbf', key="nusvc_kernel")
    nusvc_degree = st.sidebar.text_input("Degree (for poly kernel): integers only", value='3', key="nusvc_degree")
    nusvc_gamma = st.sidebar.text_input("Gamma: scale, auto", value='scale', key="nusvc_gamma")
    nusvc_tol = st.sidebar.text_input("Tolerance: floats only", value='0.001', key="nusvc_tol")


if model_box == 'nu support vector':
    if st.sidebar.button("Add Nu Support Vector to Classifier Queue"):
        nusvc_nu_list = parse_text_entry(nusvc_nu,'float')    
        nusvc_kernel_list = parse_text_entry(nusvc_kernel,'string')
        nusvc_degree_list = parse_text_entry(nusvc_degree,'int')
        nusvc_gamma_list = parse_text_entry(nusvc_gamma,'string')
        nusvc_tol_list = parse_text_entry(nusvc_tol,'float')

        name = "Nu Support Vector Classifier"
        nusvc = pipeBuild_NuSVC(nu=nusvc_nu_list, kernel=nusvc_kernel_list, degree=nusvc_degree_list, 
                                gamma=nusvc_gamma_list, tol=nusvc_tol_list)
        add_to_class_queue(name,nusvc)


# Time Series Support Vector Classifier
if model_box == 'time series svc':
    tssvc_c = st.sidebar.text_input("Regularization Parameter: floats only", value='1.0', key="tssvc_c")
    tssvc_kernel = st.sidebar.text_input("Kernel: gak, linear, poly, rbf, sigmoid", value='gak', key="tssvc_kernel")
    tssvc_degree = st.sidebar.text_input("Degree (for poly kernel): integers only", value='3', key="tssvc_degree")
    tssvc_gamma = st.sidebar.text_input("Gamma: auto or floats", value='None', key="tssvc_gamma")
    tssvc_tol = st.sidebar.text_input("Tolerance: floats only", value='0.001', key="tssvc_tol")


if model_box == 'time series svc':
    if st.sidebar.button("Add Time Series SVC to Classifier Queue"):
        tssvc_c_list = parse_text_entry(tssvc_c,'float')    
        tssvc_kernel_list = parse_text_entry(tssvc_kernel,'string')
        tssvc_degree_list = parse_text_entry(tssvc_degree,'int')
        new_string = tssvc_gamma.replace("auto", "None") # replace 'auto' in stringwith 'None' before using parse
        tssvc_gamma_list = parse_text_entry(new_string,'float') # parse will not work with 'auto', but will work with 'None'
        tssvc_gamma_list = ['auto' if item == None else item for item in tssvc_gamma_list] # replace None in list with 'auto' after parse
        tssvc_tol_list = parse_text_entry(tssvc_tol,'float')

        name = "Time Series SVC"
        tssvc = pipeBuild_TimeSeriesSVC(C=tssvc_c_list, kernel=tssvc_kernel_list, degree=tssvc_degree_list,
                                  gamma=tssvc_gamma_list, tol=tssvc_tol_list)
        add_to_class_queue(name,tssvc)

########################################################
# End Model Variable Entries
########################################################

if st.sidebar.button("Show Classifier Model Queue"):
    print(st.session_state['class_name_queue'])
    st.write(str(st.session_state['class_name_queue']))

if st.sidebar.button("Clear Classifier Model Queue"):
    st.session_state['class_name_queue'] = []
    st.session_state['class_model_queue'] = []
    st.session_state["class log loader"] = None
    st.session_state['show_class_log'] = False

if st.sidebar.button("Run Classifier Grid Search"):
    print("Classifier gridsearch started")
    st.session_state['show_class_log'] = True
    execute_class_gridsearch()


# if st.session_state['show_class_log'] == True:
#     log_file = st.file_uploader("Choose a txt file", type="txt",key="class log loader")
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

