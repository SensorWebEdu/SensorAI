import re
import struct
import os

debug=True

def format_influxdb_data(tag_mac_addr,field_name, field_value, timestamp, measurement="vitals"):
    data_point = [
        {
            "measurement": measurement,
            "tags": {"location": tag_mac_addr},
            "time": timestamp,
            "fields": {field_name: field_value}
        }
    ]
    return data_point

def pack_device_data_for_influx(topic, mac_addr, timestamp, data_interval, data, db_measurement,db_field='value'):
    data_pionts=[]

    for value in data:
        point=format_influxdb_data(mac_addr,db_field,float(value),timestamp, db_measurement)
        data_pionts +=point
        timestamp +=data_interval
    return data_pionts

def is_mac_address(mac):
    """
    Check if the given string is a valid MAC address.
    
    Parameters:
    mac (str): The string to check.
    
    Returns:
    bool: True if the string is a valid MAC address, False otherwise.
    """
    # Regular expression for validating MAC address
    mac_regex = re.compile(r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$')
    
    # Check if the string matches the regex
    return bool(mac_regex.match(mac))

#extract mac_address from topic 
def get_mac_from_topic(topic):
    mac=""
    sub_strs=topic.split("/")
    if len(sub_strs)>2: 
        mac_field=sub_strs[2]
        # is the format of a MAC address?
        if len(mac_field) ==17:   # 11:22:33:44:55:66
            mac=mac_field
        elif len(mac_field) == 12 :   # 112233445566 
            # Split into chunks of two characters
            chunks = [mac_field[i:i+2] for i in range(0, len(mac_field), 2)]
            # Join chunks with ":"
            mac = ":".join(chunks)               
    return mac

def get_mac_from_payload_lagacy(payload):
    mac_addr=""
    payload_bytes = payload
    if len(payload_bytes) >=12:
        mac_field = ":".join(f"{x:02x}" for x in struct.unpack("!BBBBBB", payload_bytes[0:6]))
    if is_mac_address(mac_field):
        mac_addr=mac_field
    return mac_addr


def is_binary_message(topic):
    # either "/+/112233445566/measurement" or "measurement", is binary format message
    #  "/+/11:22:33:44:55:66/measurement" is text-based message
    sub_strs=topic.split("/")
    if len(sub_strs)>2: 
        mac_field=sub_strs[2]
        # is the format of a MAC address?
        if len(mac_field) ==17:   # 11:22:33:44:55:66
            return False           
    return True


def get_measurement_from_topic(topic):
    db_measurement=""
    # Obtain measurement name for database: the last field in the topic treat as measusrement by default
    sub_strs=topic.split("/")
    db_measurement=sub_strs[-1]

    # exception process for lagacy data format
    if db_measurement == "geophone": db_measurement="Z"
    if db_measurement == "battery_volt" or db_measurement == "batter_volt": db_measurement="batvol"
    if db_measurement == "usb_volt": db_measurement="usbvol"
    return db_measurement

### add topic extraction?
def parse_beddot_data(msg):
    print("========================",msg.topic,)
    bytedata=msg.payload
    mac_addr = ":".join(f"{x:02x}" for x in struct.unpack("!BBBBBB", bytedata[0:6]))
    data_len =struct.unpack("H",bytedata[6:8])[0]
    if os.name == 'nt':
        timestamp=struct.unpack("Q",bytedata[8:16])[0]  # in micro second
    else:
        timestamp=struct.unpack("L",bytedata[8:16])[0]  # in micro second          
    data_interval=struct.unpack("I",bytedata[16:20])[0]  # in micro second
    #topic=struct.unpack("I",bytedata[16:20])[0]  # in micro second

    timestamp -=data_interval
    # data=[0]*int((len(bytedata)-20)/4)
    data=[0]*data_len
    index=20
    for i in range(data_len):
        if len(bytedata[index:index+4]) == 4:
            data[i] = struct.unpack("i", bytedata[index:index+4])[0]
            index +=4
            # timestamp +=data_interval
    # print("mac_addr,",mac_addr,timestamp,data_interval)
    if (data_interval < 10**6):     # less than 1 second
        timestamp = (timestamp // data_interval)*data_interval*1000 #align timestamp and convert to nano second.
    else:
        timestamp = (timestamp // 1000000)* 10**9   # align to second
    # timestamp = (timestamp // 10000)* 10**7 #convert to nano second. resolution=10ms
    data_interval *=1000 #convert to nano second

    return  mac_addr, timestamp, data_interval, data

def parse_text_msg(msg, mac):
    data_pionts=[]
    #extract mac_address from topic 
    sub_strs=msg.topic.split("/")

    # extract "measurement" from the topic
    measurement="vitals"
    if len(sub_strs)>3 and (sub_strs[3] != "vital"): 
        measurement=sub_strs[3]
    # print(msg.topic, measurement)

    #extract payload 
    payload_str = msg.payload.decode()
    data_items=payload_str.split("; ")
    b_time_valid=False
    for item in data_items:
        key, val=item.split("=")
        if key == "timestamp":      #extract timestamp
            timestamp=int(val)
            b_time_valid=True
        elif b_time_valid:    #extract data point
            data_pionts +=format_influxdb_data(mac,key,float(val), timestamp,measurement)

    # print(data_pionts)
    return data_pionts

def unpack_msg(msg):  ### 

    data_pionts=[]

    mac =  get_mac_from_topic(msg.topic)

    if is_binary_message(msg.topic):
        # this is a binary-based messeg;  
        # Reserve a more standardized binary message format for future expansion. 
        if mac=="":
            mac=get_mac_from_payload_lagacy(msg.payload)
        mac_addr, timestamp, data_interval, data=parse_beddot_data(msg) ###
        db_measurement=get_measurement_from_topic(msg.topic)  ### use this to get measurement from topic
        data_pionts=pack_device_data_for_influx(msg.topic, mac,timestamp, data_interval, data, db_measurement)
    else:
        # this is a text-based message, use {key:value} pairs as payload
        data_pionts=parse_text_msg(msg,mac)
        
    return data_pionts
