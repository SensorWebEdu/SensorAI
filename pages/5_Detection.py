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
from detection import *

def add_to_detect_queue(name,model):
    st.session_state['detect_name_queue'].append(name)
    st.session_state['detect_model_queue'].append(model)
    print("Queue: ", st.session_state['detect_name_queue'])

def execute_detect_gridsearch():
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    gridsearch_outlier(names=st.session_state['detect_name_queue'],pipes=st.session_state['detect_model_queue'],
                          X=X_test,y=y_test, plot_number=3,save_best=True,log=True,stream=True)

st.sidebar.title("Novelty Detection")

detect_model_tuple = ('local outlier factor', 'isolation forest')

detect_model_box = st.sidebar.selectbox('**Select the model(s) you like to use.**', detect_model_tuple)

########################################################
# Model Variable Entries
########################################################

# Local Outlier Factor
if detect_model_box == 'local outlier factor':
    lof_nn = st.sidebar.text_input("Number of Neighbors: integers only", value='20', key="lof_n_neighbors")
    lof_algo = st.sidebar.text_input("Algorithm: auto, ball_tree, kd_tree, brute", value='auto', key="lof_algorithm")
    lof_leaf = st.sidebar.text_input("Leaf Size: integers only", value='30', key="lof_leaf_size")
    lof_power = st.sidebar.text_input("Power: intergers only", value='2', key="lof_metri")
    lof_met = st.sidebar.text_input("Distance Metric: euclidean, manhattan, chebyshev, minkowski", value='minkowski', key="low_p")
    #lof_nov = st.text_input("Novelty: True or False", value='False', key="lof_novelty")

if detect_model_box == 'local outlier factor':
    if st.sidebar.button("Add LOF to Detection Queue"):    
        nn_list = parse_text_entry(lof_nn,'int')
        algo_list = parse_text_entry(lof_algo,'string')
        leaf_list = parse_text_entry(lof_leaf,'int')
        power_list = parse_text_entry(lof_power,'int')
        metric_list = parse_text_entry(lof_met,'string')
        #novelty_list = parse_text_entry(lof_nov,'bool')
        
        name = "Local Outlier Factor"
        local_outlier_factor = pipeBuild_LocalOutlierFactor(n_neighbors=nn_list,algorithm=algo_list,leaf_size=leaf_list,p=power_list,metric=metric_list,novelty=[True])
        add_to_detect_queue(name,local_outlier_factor)

# Isolation Forest
if detect_model_box == 'isolation forest':
    iso_ne = st.sidebar.text_input("Number of Estimators: integers", value='100', key="iso_n_estimators")
    iso_ms = st.sidebar.text_input("Maximum Samples: auto or integers", value='auto', key="iso_max_samples")


if detect_model_box == 'isolation forest':
    if st.sidebar.button("Add Iso Forest to Detection Queue"):    
        ne_list = parse_text_entry(iso_ne,'int')
        new_string = iso_ms.replace("auto", "None") # replace 'auto' in string with 'None' before using parse
        ms_list = parse_text_entry(new_string,'float') # parse will not work with 'auto', but will work with 'None'
        ms_list = ['auto' if item == None else item for item in ms_list] # replace None in list with 'auto' after parse
        
        name = "Isolation Forest"
        isolation_forest = pipeBuild_IsolationForest(n_estimators=ne_list,max_samples=ms_list)
        add_to_detect_queue(name,isolation_forest)

########################################################
# End Model Variable Entries
########################################################

if st.sidebar.button("Show Detection Model Queue"):
    print(st.session_state['detect_name_queue'])
    st.write(str(st.session_state['detect_name_queue']))

if st.sidebar.button("Clear Detection Model Queue"):
    st.session_state['detect_name_queue'] = []
    st.session_state['detect_model_queue'] = []
    st.session_state["detect log loader"] = None
    st.session_state['show_detect_log'] = False

if st.sidebar.button("Run Detection Grid Search"):
    print("Detection gridsearch started")
    st.session_state['show_detect_log'] = True
    execute_detect_gridsearch()

if st.session_state['show_detect_log'] == True:
    log_file = st.file_uploader("Choose a txt file", type="txt",key="detect log loader")
    if log_file is not None:
        root, extension = os.path.splitext(log_file.name)
        if extension.lower() == ".txt":
            bytes_data = log_file.getvalue()  
            string_data = bytes_data.decode('utf-8') 
            # Display the content
            st.write("File Content:")
            st.code(string_data, language="text") # Use st.code for displaying raw text            
        else:
            st.write("You have selected an incorrect file type")