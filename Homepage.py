import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sensor AI Framework",
)

st.title("Sensor AI Framework")
st.write("Welcome to the Center for Cyber-Physical Systems' artificial intelligence framework for sensor data")
st.sidebar.success("Select a page above")
#st.sidebar.title("Test")


########################################
# Session Variables Initializations
########################################

if "active_dataset" not in st.session_state:
    st.session_state["active_dataset"] = ""

if "active_labels" not in st.session_state:
    st.session_state["active_labels"] = ""

if "X_train" not in st.session_state:
    st.session_state["X_train"] = ""

if "y_train" not in st.session_state:
    st.session_state["y_train"] = ""

if "X_test" not in st.session_state:
    st.session_state["X_test"] = ""

if "y_test" not in st.session_state:
    st.session_state["y_test"] = ""

if "class_name_queue" not in st.session_state:
    st.session_state["class_name_queue"] = []

if "class_model_queue" not in st.session_state:
    st.session_state["class_model_queue"] = []

if "clust_name_queue" not in st.session_state:
    st.session_state["clust_name_queue"] = []

if "clust_model_queue" not in st.session_state:
    st.session_state["clust_model_queue"] = []

if "detect_name_queue" not in st.session_state:
    st.session_state["detect_name_queue"] = []

if "detect_model_queue" not in st.session_state:
    st.session_state["detect_model_queue"] = []

if "regress_name_queue" not in st.session_state:
    st.session_state["regress_name_queue"] = []

if "regress_model_queue" not in st.session_state:
    st.session_state["regress_model_queue"] = []

if 'show_class_log' not in st.session_state:
    st.session_state['show_class_log'] = False

if 'show_clust_log' not in st.session_state:
    st.session_state['show_clust_log'] = False

if 'show_detect_log' not in st.session_state:
    st.session_state['show_detect_log'] = False

if 'show_regress_log' not in st.session_state:
    st.session_state['show_regress_log'] = False