import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from pathlib import Path
from datetime import datetime
import yaml
import re
import subprocess
import time
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tkinter import filedialog

current_dir = Path.cwd()
config_path = current_dir / "config/current_data.yaml"


# 此代码需要大改，但暂时可以用
def calc_mae(gt, pred):
    return np.mean(abs(np.array(gt) - np.array(pred)))

def plot_2vectors_old(label, pred, save=False, name=None, path=None, size=1):
    """lsit1: label, list2: prediction"""

    list1 = label
    list2 = np.array(pred)
    if list2.ndim == 2:
        mae = calc_mae(list1, list2[:, 0])
    else:
        mae = calc_mae(list1, list2)

    sorted_id = sorted(range(len(list1)), key=lambda k: list1[k])

    plt.clf()
    plt.text(0, np.min(list2), f'MAE={mae}')

    plt.scatter(np.arange(list2.shape[0]), list2[sorted_id], s=size, alpha=0.5, label=f'{name} prediction', color='blue')
    plt.scatter(np.arange(list1.shape[0]), list1[sorted_id], s=size, alpha=0.5, label=f'{name} label', color='red')
    plt.legend(loc='lower right')

    if save:
        if path is None:
            raise ValueError("If save is True, 'path' argument must be provided.")
        plt.savefig(f'{path}.jpg', dpi=300)
        print(f'Saved plot to {path}.jpg')
    #plt.show()
    return plt


def plot_2vectors(label, pred, save=False, name="", path=None, size=5):
    """label: ground truth values, pred: predicted values"""

    list1 = np.array(label)
    list2 = np.array(pred)

    # Compute MAE
    if list2.ndim == 2:
        mae = calc_mae(list1, list2[:, 0])
        list2 = list2[:, 0]
    else:
        mae = calc_mae(list1, list2)

    # Sort by label values
    sorted_id = sorted(range(len(list1)), key=lambda k: list1[k])
    sorted_x = np.arange(len(list1))

    # Create DataFrame for Plotly Express
    df = pd.DataFrame({
        'Index': list(sorted_x) * 2,
        'Value': np.concatenate([list2[sorted_id], list1[sorted_id]]),
        'Type': [f'{name} prediction'] * len(list1) + [f'{name} label'] * len(list1)
    })

    trace_colors = ['blue', 'red', 'green', 'purple', 'orange'] 

    # Plot
    fig = px.scatter(
        df,
        x='Index',
        y='Value',
        color='Type',
        opacity=0.5,
        size_max=size,
        title=f"Label vs Prediction (MAE={mae:.4f})"
    )

    # Update layout
    fig.update_layout(
        legend=dict(x=0.85, y=0.05),
        template='simple_white'
    )

    # Iterate through the traces and update their marker color based on the index
    for i, trace in enumerate(fig.data):
        if i < len(trace_colors): # Ensure there's a color for the current trace
            fig.update_traces(marker_color=trace_colors[i], selector=dict(name=trace.name))

    # Save plot if requested
    if save:
        if path is None:
            raise ValueError("If save is True, 'path' argument must be provided.")
        fig.write_image(f"{path}.jpg", scale=3)
        print(f"Saved plot to {path}.jpg")

    return fig

def ls2pkl(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def pkl2ls(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def dic2pkl(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def dicl2ls(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_noise_signal(original_signal, noisy_signal, title_name):
    plt.figure()
    plt.plot(noisy_signal, label='Noisy Signal')
    plt.plot(original_signal, label='Original Signal')
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.title(title_name)
    plt.legend()
    plt.show()

def plot_decomposed_components(signal, components, title_name):
    n_components = len(components)

    plt.subplots(n_components+1, 1)
    plt.subplot(n_components+1, 1, 1)
    plt.title(title_name)

    plt.plot(signal, label='Original Signal', color='r')

    for cnt, component in enumerate(components):
        # print(cnt+1, n_components)
        plt.subplot(n_components+1, 1, cnt+2)
        plt.plot(component, label='Component'+str(cnt+1))
        plt.legend()
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.show()

def plot_filtered_signal(filtered_signal, signal, title_name):
    plt.figure()
    plt.plot(signal, label='Original Signal', alpha=0.6)
    plt.plot(filtered_signal, label='Filtered Signal')
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.title(title_name)
    plt.legend()
    plt.show()

def plot_sim_waves(signal, wave_name):
    plt.figure()
    plt.plot(signal, label=wave_name)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.title('Generated Wave')
    plt.legend()
    plt.show()

def plot_adp_filtered_signal(y, d_signal, error):
    plt.figure()

    plt.subplot(211)
    plt.title("Adaptation")
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.plot(d_signal, "b", label="d_signal - target")
    plt.plot(y, "g", label="output")
    plt.legend()

    plt.subplot(212)
    plt.title("Filter error")
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.plot(10 * np.log10(error ** 2), "r", label="error [dB]")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_averaging_center(center, pieces):
    plt.figure()
    plt.title("Center of Signal Pieces")
    for piece in pieces:
        plt.plot(piece, alpha=0.35)
    plt.plot(center, "r", linewidth=2, label="Center")
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

def plot_confusion_matrix_old(y_true, y_pred, classes, title='Confusion Matrix', cmap='Blues'):
    """
    Plots the confusion matrix.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classes (list): List of class names.
        title (str, optional): Title for the plot. Defaults to 'Confusion Matrix'.
        cmap (str, optional): Colormap for the heatmap. Defaults to 'Blues'.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix",stream=False):
    """
    Plots an interactive confusion matrix using Plotly.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of class labels for display. Defaults to unique values in y_true.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".
        stream (bool): True will plot in a streamlit webpage, False will display in browser
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if labels is None:
        labels = np.unique(y_true).astype(str) # Convert to string for display

    fig = px.imshow(cm,
                    x=labels,
                    y=labels,
                    color_continuous_scale='Blues', # Choose a color scale
                    aspect="auto",
                    text_auto=True) # Automatically add text annotations

    fig.update_layout(title=title,
                      xaxis_title='Predicted Label',
                      yaxis_title='True Label',
                      xaxis_nticks=len(labels), # Ensure all x-axis labels are shown
                      yaxis_nticks=len(labels)) # Ensure all y-axis labels are shown

    if stream == True:
        st.plotly_chart(fig)
    else:
        fig.show()

# Load the model from a pickle file
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Save model to a pickle file
def save_model(model,filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return

# returns current time stamp string in yyyymmddhhmmss format
def get_timestamp_string():
    now = datetime.now()
    timestamp_string = now.strftime("%Y%m%d%H%M%S")
    return timestamp_string


# create a new directory based on name string and return the path
def create_directory(directory_name):
    directory_path = Path(directory_name)
    try:
        # Create the directory
        directory_path.mkdir()
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return directory_path

def create_model_yaml(yaml_name,model_name,model_path,model_type,n_inputs,n_outputs):
    # Data to write to the YAML file (Python dictionary)
    data = {
        'name': model_name,
        'model_path': model_path, 
        'model_type': model_type,
        'inputs': n_inputs,
        'outputs': n_outputs
        }

    file = model_path + '/' + yaml_name

    # Open the YAML file in write mode
    with open(model_path + '/' + yaml_name, "w") as file:
        # Write the data to the YAML file
        yaml.dump(data, file, default_flow_style=False)
    return

# GUI RELATED METHODS

def run_script(script_path, args, stop_flag):
    """
    Runs a python script in a new console window with given arguments.

    Args:
        script_path (str): Path to the python script to execute.
        args (list): List of arguments to pass to the script.
        stop_flag (threading.Event): Event to signal the thread to stop.
    """
    command = ["start", "cmd", "/k", "python", script_path] + args
    process = subprocess.Popen(command, shell=True)
    while not stop_flag.is_set():
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if not stop_flag.is_set():
       process.terminate()
       process.wait()

# Change tkinter entry strings to appropriate list formats
def clean_list(input_list,type='string'):
    new_list = []
    for item in input_list:
        if item == 'None':
            new_list.append(None)
        else:
            if type == 'string':
                new_list.append(item)
            elif type == 'int':
                new_item = int(item)
                new_list.append(new_item)
            elif type == 'float':
                new_item = float(item)
                new_list.append(new_item)
            elif type == 'bool':
                new_item = item.lower()
                if new_item == 'true' or new_item == 't' or new_item == '1':
                    truth_item = True
                    new_list.append(truth_item)
                else:
                    truth_item = False
                    new_list.append(truth_item)            
    return new_list

def parse_text_entry(entry,text_type='string'):
    parsed_entry = re.split(r'[,;]+', entry)    
    if text_type == 'string':
        cleaned_entry = clean_list(parsed_entry,type='string')
        text_list = cleaned_entry
    elif text_type =='int':
        cleaned_entry = clean_list(parsed_entry,type='int')
        text_list = cleaned_entry
    elif text_type == 'float':
        cleaned_entry = clean_list(parsed_entry,type='float')
        text_list = cleaned_entry
    elif text_type == 'bool':
        cleaned_entry = clean_list(parsed_entry,type='bool')
        text_list = cleaned_entry
    else:
        print("Incorrect Text Type Entered")
        text_list = entry    
    return text_list

# Updates a specific variable of a yaml file
def update_yaml_variable(file_path, variable_path, new_value):
    """
    Updates a specific variable in a YAML file.

    Args:
        file_path (str): Path to the YAML file.
        variable_path (str): Path to the variable, separated by dots (e.g., "section1.subsection2.variable").
        new_value: The new value for the variable.
    """
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    if yaml_data is None:
        yaml_data = {}

    keys = variable_path.split('.')
    current = yaml_data

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = new_value

    with open(file_path, 'w') as file:
        yaml.dump(yaml_data, file, sort_keys=False)

# Configure/Reset data file information
def data_loader_confg():
    source_file = current_dir / "config/data_template.yaml"
    destination_file = current_dir / "config/current_data.yaml"
    shutil.copy(source_file, destination_file)

def load_gui_data():
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    label_bool = yaml_data['labeled_data']
    sep_data_bool = yaml_data['seperate_labels']
    split_bool = yaml_data['test_train_split']
    split_value = yaml_data['split_value']
    data_file = yaml_data['current_data_file']
    label_file = yaml_data['current_label_file']

    data = np.load(data_file)

    if label_bool == True and sep_data_bool == False:
        y = data[:, -1]
        x = data[:, :-1]
    elif label_bool == True and sep_data_bool == True:
        x = data
        y = np.load(label_file)
    else:
        x = data
        y = None
    
    if split_bool == True:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=float(split_value), random_state=None)
    else:
        X_train = None
        y_train = None
        X_test = x
        y_test = y

    return X_train, y_train, X_test, y_test

def convert_list_to_string(list):
    """
    converst a list to a string
    """
    formatted_string = "\n".join(str(item) for item in list)
    return formatted_string

# Open Dialog to Save an .npy file.
def save_numpy_array(array_to_save):
    """Opens a file dialog to save a NumPy array as a .npy file."""
    filename = filedialog.asksaveasfilename(
        defaultextension=".npy",  # Set default file extension
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]  # Specify file types
    )
    if filename:  # Check if a filename was selected (dialog was not canceled)
        try:
            np.save(filename, array_to_save)
            print(f"NumPy array saved successfully to: {filename}")
        except Exception as e:
            print(f"Error saving NumPy array: {e}")

def data_setup(label_bool,sep_data_bool,split_bool,data_file,split_value=None,label_file=None):    
    
    data =  data_file

    if label_bool == True and sep_data_bool == False:
        y = data[:, -1]
        x = data[:, :-1]
    elif label_bool == True and sep_data_bool == True:
        x = data
        y = label_file
    else:
        x = data
        y = None
    
    if split_bool == True:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=float(split_value), random_state=None)
    else:
        X_train = None
        y_train = None
        X_test = x
        y_test = y

    X_full = x
    y_full = y
    
    return X_full, y_full, X_train, y_train, X_test, y_test

def display_log_updates(filename,log_container):
    """Reads the log file and displays its content."""
    try:
        with open(filename, 'r') as f:
            log_content = f.read()
            log_container.code(log_content, language='text') # Display as code block for formatting
    except FileNotFoundError:
        log_container.warning("Log file " + filename + " not found.")


def plot_single_row(data):
    # Check if the data is multi-dimensional and has at least one row
    if data.ndim > 0 and data.shape[0] > 0:
        # Extract the first row
        first_row = data[0]

        # Create a Matplotlib figure and plot the first row
        fig, ax = plt.subplots()
        ax.plot(first_row)
        ax.set_title("Plot of the First Row")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")

        # Display the plot in Streamlit
        #st.pyplot(fig)
    else:
        st.warning("The uploaded NPY file does not contain a valid first row to plot.")
    return fig