import streamlit as st
import yaml
import numpy as np
import os
import sys
import threading
import tempfile
from pathlib import Path

# Get the path of the current file
current_file_path = Path(__file__).resolve()
# Get the parent directory
parent_dir = current_file_path.parent
# Get the path to the other folder
other_folder_path = parent_dir.parent / "lib"
# Add the other folder to sys.path so Python can find the module
sys.path.append(str(other_folder_path))


from utils import run_script, data_loader_confg, convert_list_to_string
from dsp import *


current_dir = Path.cwd() # assumes your working directory is in "SensorwebAiFramework"

temp_dir = tempfile.mkdtemp()

if 'device_name' not in st.session_state:
    st.session_state['device_name'] = ""

if 'model_name' not in st.session_state:
    st.session_state['model_name'] = ""

if 'conn_script' not in st.session_state:
    st.session_state['conn_script'] = ""

if 'stop_event' not in st.session_state:
    st.session_state['stop_event'] = ""

st.title("Historical Data Connection")


# Method to connect to db
def connect_model_to_db(device_name,model_name):
    print("Connecting - model: ",model_name,"To database: ",device_name)

    st.session_state["stop_event"] = threading.Event()
    script_path = current_dir / "mqtt/main_ai_influx.py"
    arguments = [device_name, model_name]

    conn_script = threading.Thread(target=run_script, args=(script_path, arguments, st.session_state["stop_event"]))    
    conn_script.start()
    st.session_state['conn_script'] = conn_script

# Method to disconnect from db
def disconnect_model_from_db():
    st.session_state["stop_event"].set()
    st.session_state['conn_script'].join()
    st.session_state['device_name'] = ""
    st.session_state['model_name'] = ""
    #st.session_state['database yaml loader'] = ""
    #st.session_state['model yaml loader'] = ""
    st.session_state["stop_event"] = ''
    st.session_state['conn_script'] = ''
    print("Thread terminated.")

db_yaml = st.file_uploader("Choose a yaml file for a database", type="yaml",key="database yaml loader")
if db_yaml is not None:
    root, extension = os.path.splitext(db_yaml.name)
    if extension.lower() == ".yaml":
        db_path = os.path.join(temp_dir, db_yaml.name)
        with open(db_path, "wb") as f:
            f.write(db_yaml.getvalue())
        st.session_state['device_name'] = db_path
                  
    else:
        st.write("You have selected an incorrect file type")


#save_data_check = st.checkbox("Select if you want to save downloaded data to a .npy file", key = "hist download save")

h_model_yaml = st.file_uploader("Choose a yaml file for your model", type="yaml",key="hist model yaml loader")
if h_model_yaml is not None:
    root, extension = os.path.splitext(h_model_yaml.name)
    if extension.lower() == ".yaml":
        h_model_path = os.path.join(temp_dir, h_model_yaml.name)
        with open(h_model_path, "wb") as f:
            f.write(h_model_yaml.getvalue())
        st.session_state['model_name'] = h_model_path  
                  
    else:
        st.write("You have selected an incorrect file type")

if st.session_state['model_name'] != "" and st.session_state['device_name'] != "":
    if st.button("connect mode to device"):
        connect_model_to_db(device_name=st.session_state['device_name'],model_name=st.session_state['model_name'])
    if st.button("disconnect model from device"):
        disconnect_model_from_db()