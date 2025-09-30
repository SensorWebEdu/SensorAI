import paho.mqtt.client as mqtt
import numpy as np
import json
import yaml
from msg_proc import parse_beddot_data
from pathlib import Path
import sys
from influxdb import InfluxDBClient
import json
from tkinter import filedialog as fd
import warnings
import time


# Ignore warnings
warnings.filterwarnings("ignore")

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


# Define open file
def open_yaml_file():
    filetypes = (
        ('yaml files', '*.yaml'),
        ('pickle files', '*.pkl')
                )
    file = fd.askopenfilename(filetypes=filetypes)    
    return file


# Define Device Setup
def device_setup(device_path,model_path):
    dev_file =  device_path #open_yaml_file()
    # dc:da:0c:3c:6d:40
    # Load the Model YAML file
    with open(dev_file, "r") as file:
        device = yaml.safe_load(file)
    
    if device["device"]["type"] == "smartplug":
        global org
        global mac
        global topics
        global window_size
        # Extract smartplug yaml data
        org = device["device"]["organization"]
        mac = device["device"]["mac"]
        topics = device["device"]["topics"]
        window_size = device['device']['window_size']
    
    # Set up the Topics dictionary
    global combined_data
    combined_data = {"time": None}
    for top in topics:
        combined_data[f"{top}"] = None
    print(combined_data)

    # Setup window for windowed univariate data input 
    global window_data
    window_data = []
    
    # MQTT Configuration
    MQTT_BROKER = device["device"]["broker"] #"sensorserver2.engr.uga.edu"
    MQTT_PORT = device["device"]["port"] #1883

    # InfluxDB Configuration
    global INFLUXDB_DATABASE
    global location
    INFLUXDB_HOST = device["db_server"]["host"]
    INFLUXDB_PORT = device["db_server"]["port"]
    INFLUXDB_DATABASE = device["db_server"]["database"]
    INFLUXDB_USER = device["db_server"]["user"]
    INFLUXDB_PASS = device["db_server"]["password"]
    isSSL = device["db_server"]["ssl"]
    location = device["db_server"]["prediction-location"]
    
    global influx_client
    # Connect to InfluxDB
    influx_client = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT,username=INFLUXDB_USER,password=INFLUXDB_PASS,database=INFLUXDB_DATABASE,ssl=isSSL)
    #influx_client.create_database(INFLUXDB_DATABASE)
    influx_client.switch_database(INFLUXDB_DATABASE)
    #write_api = influx_client.write_api(write_options=WriteOptions(batch_size=1))

    model_file =  model_path #open_yaml_file()

    with open(model_file, "r") as file:
        best_model = yaml.safe_load(file)

    # Create a new MQTT client instance
    #client = mqtt.Client()

    #topic_subscriber(org, mac, topics, best_model, client)

    return MQTT_BROKER, MQTT_PORT, org, mac, best_model

def topic_subscriber(org,mac,topics,best_model,client):
    global model
    for top in topics:
        TOPIC = "/" + org + "/" + mac + "/" + top
        client.subscribe(TOPIC)
    file_name = best_model['model_path'] + '/' + best_model['name']
    model = utils.load_model(file_name)
    print(best_model['name']," was loaded successfully")
    return model

def topic_unsubscriber(org,mac,topics,client):
    for top in topics:
        TOPIC = "/" + org + "/" + mac + "/" + top
        client.unsubscribe(TOPIC)
        print("Unsubscribed from: ",TOPIC)
    return

# Define what happens when connecting to the smart device
def on_connect(client, userdata, flags, rc):
    global model
    for top in topics:
        TOPIC = "/" + org + "/" + mac + "/" + top
        client.subscribe(TOPIC)
    file_name = best_model['model_path'] + '/' + best_model['name']
    model = utils.load_model(file_name)
    print(best_model['name']," was loaded successfully")
    #print(f"Connected with result code {rc}")    



# Define what happens when a message is received
def on_message(client, userdata, msg):
    global window_size
    global window_data
    top = shorten_topic(msg.topic)
    try:   
        mac_addr, timestamp, data_interval, data =  parse_beddot_data(msg)
        if combined_data["time"] == None:
            combined_data["time"] = timestamp
            combined_data[f"{top}"] = data
        elif combined_data["time"] != None and combined_data["time"] == timestamp and combined_data[f"{top}"] == None:
            combined_data[f"{top}"] = data
        
        # Combine or process the data (here we print it as an example)
        if window_size == 1:
            combine_and_process_data()
        else:
            if len(window_data) < window_size:
                window_data.append(data[0])
                #print(window_data," window data")
                if len(window_data) == window_size:
                    process_window_data()
            else:
                del window_data[0]
                window_data.append(data[0])
                process_window_data()
            #print("on connect ",window_data)
    
    except json.JSONDecodeError:
        print(f"Failed to decode message on {msg.topic}")

# Function to combine and process the data from all topics
def combine_and_process_data():
    global location
    # Check if all data is available (you can also do other checks here)
    if all(combined_data.values()):
        print("Combined Data:", combined_data)
        # Write code to preprocess and send data to AI model        
        data_list = list(combined_data.values())
        del data_list[0]
        data = np.array(data_list)
        data = data.reshape(1, -1) #(-1, 1)

        prediction = model.predict(data)
        #print("Predition Type is: ",type(prediction))
        print("Model Prediction: ",prediction)

        line_data = f"prediction,location={location} value={prediction[0]} {combined_data['time']}"
        print("TIME IS: ", combined_data["time"])
        print(line_data)

        # write to influxdb
        influx_client.write([line_data],params={'db':INFLUXDB_DATABASE},protocol='line')
        
        # Reset data for next cycle if required
        reset_combined_data()

def process_window_data():
    global window_data
    list_data = window_data
    #print("in process ",list_data)
    data = np.array(list_data)
    data = data.reshape(1, -1) #(-1, 1)

    prediction = model.predict(data)
    print("Model Prediction: ",prediction)

    # Get the current time in nanoseconds since the epoch
    nanosecond_timestamp = time.time_ns()

    line_data = f"prediction,location={location} value={prediction[0]} {nanosecond_timestamp}"
    print("TIME IS: ", nanosecond_timestamp)
    print(line_data)

    # write to influxdb
    influx_client.write([line_data],params={'db':INFLUXDB_DATABASE},protocol='line')

# Function to reset combined data after processing (if necessary)
def reset_combined_data():
    global combined_data
    combined_data = {key: None for key in combined_data}

# Function to shorten the smart device topic name.  Drops org name and mac address.
def shorten_topic (topic):
    #global topics
    for top in topics:
        if top in topic:
            short_topic = top
    if short_topic == None:
        print("topic error")
    return short_topic

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

    MQTT_BROKER, MQTT_PORT, org, mac, best_model = device_setup(device_path=device_arg,model_path=model_arg)
    

    # Create a new MQTT client instance
    client = mqtt.Client()

    # Attach the callback functions
    client.on_connect = on_connect
    #topic_subscriber(org,mac,topics,best_model,client)
    client.on_message = on_message    

    # Connect to the broker
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # Loop to process network traffic, dispatch callbacks, etc.
    client.loop_forever()

    # Keep the script running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        topic_unsubscriber(org,mac,topics,client)
        print("Disconnected")        
        client.loop_stop()  # Stop the loop when exiting