import streamlit as st
import yaml
import numpy as np
import os
import sys
import threading
import tempfile
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

st.title("Smart Device Connection")

def connect_model_to_device(device_name,model_name):    
    print("Connecting - model: ",model_name,"To device: ",device_name)

    st.session_state["stop_event"] = threading.Event()
    script_path = current_dir / "mqtt/main_ai_mqtt.py"
    arguments = [device_name, model_name]

    conn_script = threading.Thread(target=run_script, args=(script_path, arguments, st.session_state["stop_event"]))    
    conn_script.start()
    st.session_state['conn_script'] = conn_script

def disconnect_model_from_device():
    st.session_state["stop_event"].set()
    st.session_state['conn_script'].join()
    st.session_state['device_name'] = ""
    st.session_state['model_name'] = ""
    #st.session_state['device yaml loader'] = ""
    #st.session_state['model yaml loader'] = ""
    st.session_state["stop_event"] = ''
    st.session_state['conn_script'] = ''
    print("Thread terminated.")


device_yaml = st.file_uploader("Choose a yaml file for a smart device", type="yaml",key="device yaml loader")
if device_yaml is not None:
    root, extension = os.path.splitext(device_yaml.name)
    if extension.lower() == ".yaml":
        dev_path = os.path.join(temp_dir, device_yaml.name)
        with open(dev_path, "wb") as f:
            f.write(device_yaml.getvalue())
        st.session_state['device_name'] = dev_path
                  
    else:
        st.write("You have selected an incorrect file type")

model_yaml = st.file_uploader("Choose a yaml file for a model", type="yaml",key="model yaml loader")
if model_yaml is not None:
    root, extension = os.path.splitext(model_yaml.name)
    if extension.lower() == ".yaml":
        model_path = os.path.join(temp_dir, model_yaml.name)
        with open(model_path, "wb") as f:
            f.write(model_yaml.getvalue())
        st.session_state['model_name'] = model_path  
                  
    else:
        st.write("You have selected an incorrect file type")

if st.session_state['model_name'] != "" and st.session_state['device_name'] != "":
    if st.button("connect mode to device"):
        connect_model_to_device(device_name=st.session_state['device_name'],model_name=st.session_state['model_name'])
    if st.button("disconnect model from device"):
        disconnect_model_from_device()