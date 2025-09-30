import paho.mqtt.client as mqtt
import numpy as np
import yaml
from msg_proc import parse_beddot_data
from pathlib import Path
import sys
from influxdb import InfluxDBClient
from tkinter import filedialog as fd
import warnings
#import datetime
from datetime import datetime, timezone
import pandas as pd
import re

# Get the path of the current file (file1.py)
current_file_path = Path(__file__).resolve()
# Get the parent directory (folder1)
parent_dir = current_file_path.parent
# Get the path to the other folder (folder2)
other_folder_path = parent_dir.parent / "lib"
# Add the other folder to sys.path so Python can find the module
sys.path.append(str(other_folder_path))
# Now you can import from file2.py
import utils

# Ignore warnings
warnings.filterwarnings("ignore")


def nanosecond_to_datetime(nanosecond_timestamp):
    """Converts a nanosecond integer timestamp to a datetime object.

    Args:
        nanosecond_timestamp: An integer representing nanoseconds since the epoch.

    Returns:
        A tuple containing:
            - A datetime object representing the time.
            - An integer representing the remaining nanoseconds (beyond microseconds).
    """
    seconds = nanosecond_timestamp / 1e9
    remaining_nanoseconds = int(nanosecond_timestamp % 1e9)
    dt_object = datetime.fromtimestamp(seconds)

    return dt_object, remaining_nanoseconds

# Convert date string to timestamp
def date_to_timestamp(date_string):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    datetime_object = datetime.strptime(date_string, date_format)
    timestamp = datetime_object.timestamp()
    nanoseconds = int(timestamp * 1000000000)
    return nanoseconds


# Define Device Setup
def device_setup(device_path,model_path):
    dev_file =  device_path #open_yaml_file()
    # dc:da:0c:3c:6d:40
    # Load the Model YAML file
    with open(dev_file, "r") as file:
        device = yaml.safe_load(file)
    
    if device["device"]["type"] == "smartplug":
        global start_time
        global end_time 
        global topics
        global window_size
        # Extract smartplug yaml data
        start_time = device["device"]["start_time"]
        end_time = device["device"]["end_time"]
        topics = device["device"]["topics"]
        window_size = device["device"]["window_size"]
    
    # Set up the Topics dictionary
    global combined_data
    combined_data = {"time": None}
    for top in topics:
        combined_data[f"{top}"] = None
    #print(combined_data)
        
    # Setup window for windowed univariate data input 
    global window_data
    window_data = []

    # InfluxDB Configuration
    global INFLUXDB_DATABASE
    global prediction_location
    global device_location
    INFLUXDB_HOST = device["db_server"]["host"]
    INFLUXDB_PORT = device["db_server"]["port"]
    INFLUXDB_DATABASE = device["db_server"]["database"]
    INFLUXDB_USER = device["db_server"]["user"]
    INFLUXDB_PASS = device["db_server"]["password"]
    isSSL = device["db_server"]["ssl"]
    prediction_location = device["db_server"]["prediction-location"]
    device_location = device["device"]["location"]
    
    global influx_client
    # Connect to InfluxDB
    influx_client = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT,username=INFLUXDB_USER,password=INFLUXDB_PASS,database=INFLUXDB_DATABASE,ssl=isSSL)

    influx_client.switch_database(INFLUXDB_DATABASE)

    model_file =  model_path

    with open(model_file, "r") as file:
        best_model = yaml.safe_load(file)

    return best_model

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print("NO ARGUMENTS GIVEN")
        sys.exit()
    elif len(sys.argv) > 1:
        arguments = sys.argv[1:]
        print("Command-line arguments:", arguments)
        if len(arguments) != 2:
            print("INCORRECT ARGUMENTS")
            sys.exit()

    device_arg = arguments[0]
    model_arg = arguments[1]

    print("Device: ", arguments[0])
    print("Model: ", arguments[1])

    best_model = device_setup(device_path=device_arg,model_path=model_arg)
    file_name = best_model['model_path'] + '/' + best_model['name']
    model = utils.load_model(file_name)

    #query = f"SELECT * FROM {', '.join(topics)} WHERE time >= '{start_time}' AND time <= '{end_time}'"
    query = f"SELECT * FROM {', '.join(topics)} WHERE location = '{device_location}' AND time >= '{start_time}' AND time <= '{end_time}'"

    
    results = influx_client.query(query)

    if window_size == 1:
        results_list = []
        time_list = []
        t=0
        for top in topics:        
            top_data = list(results.get_points(measurement=top))        
            top_frame = pd.DataFrame(top_data)
            if t == 0:
                time_list = top_frame['time'].tolist()
                #("TIME LIST: ",time_list)
                #print(type(time_list[0]))
                for l in range(len(time_list)):
                    time_list[l] = date_to_timestamp(time_list[l])
            top_list = top_frame['value'].tolist()
            results_list.append(top_list)
            t += 1
        results_array = np.array(results_list)
        results_array = results_array.transpose()
        #print("RESULTS TYPE IS: ",type(results_array[0][0]))

        #print("TIME LIST: ",time_list)
        #print("results array: ",results_array)

        for i in range(len(time_list)):
            data = results_array[i].reshape(1, -1) #(-1, 1)
            prediction = model.predict(data)
            #print("Predition Type is: ",type(prediction))
            print("Model Prediction: ",prediction[0])

            #print("TIME LIST OBJECT IS: ", type(time_list[i]))
            line_data = f"prediction,location={prediction_location} value={prediction[0]} {time_list[i]}"
            print(line_data)

            temp_time = nanosecond_to_datetime(time_list[i])
            print("Converted Time Stamp: ", temp_time[0])

            # write to influxdb
            influx_client.write([line_data],params={'db':INFLUXDB_DATABASE},protocol='line')
    else:
        for top in topics:        
            top_data = list(results.get_points(measurement=top)) 
            top_frame = pd.DataFrame(top_data)
            value_list = top_frame["value"].tolist()
            time_list = top_frame["time"].tolist()
            for l in range(len(time_list)):
                    time_list[l] = date_to_timestamp(time_list[l])

            for n in range(len(value_list)-(window_size-1)):
                window_list = value_list[n:n+window_size]
                window_array = np.array(window_list)
                data = window_array.reshape(1, -1) #(-1, 1)
                prediction = model.predict(data)
                #print("Predition Type is: ",type(prediction))
                print("Model Prediction: ",prediction[0])

                #print("TIME LIST OBJECT IS: ", type(time_list[i]))
                line_data = f"prediction,location={prediction_location} value={prediction[0]} {time_list[n]}"
                print(line_data)

                temp_time = nanosecond_to_datetime(time_list[n])
                print("Converted Time Stamp: ", temp_time[0])

                # write to influxdb
                influx_client.write([line_data],params={'db':INFLUXDB_DATABASE},protocol='line')
