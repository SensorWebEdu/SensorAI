import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder
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
from utils import data_setup, parse_text_entry, save_numpy_array, plot_single_row
from dsp import generate_class_data, generate_anomaly_data, generate_regression_data, scg_simulate

st.sidebar.title("Data")

sep_labels = False
split = 0.00

if not str(st.session_state["active_dataset"]) == "":
    data = st.session_state["active_dataset"]
    df = pd.DataFrame(data)

    st.write("Active Dataset ...")

    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Data Summary") 
    st.write(df.describe())

    temp = data[:, :-1]
        
    fig = plot_single_row(temp)
    st.pyplot(fig)  


option = st.sidebar.selectbox('What data would you like to use?',
    ('.csv or .npy file', 'generate waveforms', 'generate scg signals'))

if option == '.csv or .npy file':
    if st.sidebar.button("Clear All Data"):
        sep_labels = False
        split = 0.00
        data = None
        labels = None
        st.session_state["X_train"] = ""
        st.session_state["y_train"] = ""
        st.session_state["X_test"] = ""
        st.session_state["y_test"] = ""
        st.session_state["active_dataset"] = ""
        st.session_state["active_labels"] = ""
        st.session_state["header check"] = False
        st.session_state["label check"] = False
        st.session_state["seperate labels"] = False
        st.session_state["split data"] = False
        st.session_state["split value"] = 0.00
        st.session_state["ready to load"] = False
        st.session_state["data loader"] = None
        st.session_state["label loader"] = None

# .CSV or .NPY File Load Options
if option == '.csv or .npy file':
    header_check = st.sidebar.checkbox("Does your data/label file(s) have headers? Check for yes",key="header check")

if option == '.csv or .npy file':
    label_check = st.sidebar.checkbox("Does your data file contain labels? Check for yes",key="label check")

if option == '.csv or .npy file' and label_check == False:
    sep_label_check = st.sidebar.checkbox("Do you have labels in a seperate file? Check for yes",key="seperate labels")
    if sep_label_check == True:
        sep_labels = sep_label_check

if option == '.csv or .npy file':
    split_check = st.sidebar.checkbox("Select if you will require a train/test split on your data", key = "split data")

    if split_check == True:
        split_value = st.sidebar.number_input("Enter a float value for you data split:", value=0.0, step=0.01, format="%.2f",key="split value")
        if split_value != 0.00:
            split = split_value

if option == '.csv or .npy file':
    ready = st.sidebar.checkbox("Select when ready to choose your file(s)",key = "ready to load")

if option == '.csv or .npy file':
    if ready == True:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV or NPY data file", type=["csv","npy"],key="data loader")

        if sep_labels == True:
            uploaded_labels = st.sidebar.file_uploader("Choose a CSV or NPY label file", type=["csv","npy"],key="label loader")
            if uploaded_labels is not None:
                root, extension = os.path.splitext(uploaded_labels.name)
                if extension.lower() == ".npy":
                    labels = np.load(uploaded_labels)
                    le = LabelEncoder()
                    labels = le.fit_transform(labels) #labels.astype(float)                
                else:
                    if header_check == False:
                        df = pd.read_csv(uploaded_labels,header=None)
                    else:
                        df = pd.read_csv(uploaded_labels)
                    labels = df.to_numpy
        else:
            labels = None


        if uploaded_file is not None:
            root, extension = os.path.splitext(uploaded_file.name)
            if extension.lower() == ".npy":
                data = np.load(uploaded_file)
                df = pd.DataFrame(data)                
            else:
                if header_check == False:
                    df = pd.read_csv(uploaded_file,header=None)
                else:
                    df = pd.read_csv(uploaded_file)
                data = df.to_numpy 

            st.write("File uploaded ...")

            st.subheader("Data Preview")
            st.write(df.head())

            st.subheader("Data Summary")
            st.write(df.describe())

            if st.session_state["label check"] == True:
                temp = df.to_numpy()
                temp = temp[:, :-1]
            else:
                temp = df.to_numpy()
                
            fig = plot_single_row(temp)
            st.pyplot(fig)
                
        
        else:
            st.write("Waiting on file upload")

        if (uploaded_file is not None) and (sep_labels == True):
            X_full, y_full, X_train, y_train, X_test, y_test = data_setup(label_bool=label_check,
                                                            sep_data_bool=sep_labels,
                                                            split_bool=split_check,
                                                            data_file=data,
                                                            split_value=split,
                                                            label_file=labels)
            st.session_state["X_train"] = X_train
            st.session_state["y_train"] = y_train
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["active_dataset"] = X_full
            st.session_state["active_labels"] = y_full
        elif (uploaded_file is not None) and (sep_labels == False):
            X_full, y_full, X_train, y_train, X_test, y_test = data_setup(label_bool=label_check,
                                                            sep_data_bool=sep_labels,
                                                            split_bool=split_check,
                                                            data_file=data,
                                                            split_value=split,
                                                            label_file=labels)
            st.session_state["X_train"] = X_train
            st.session_state["y_train"] = y_train
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["active_dataset"] = X_full
            st.session_state["active_labels"] = y_full

# Generate Wavefor Data Options
elif option == 'generate waveforms':

    if st.sidebar.button("Clear All Waveform Inputs"):
        st.session_state["category"] = 'classification'
        st.session_state.wave_amp = 0.00
        st.session_state.wave_freq = 0.00
        st.session_state.wave_noise_box = False
        st.session_state.wave_num = 10
        st.session_state["wav_labels"] = 'frequency'
        st.rerun()

    wav_cat = st.sidebar.selectbox('Category:', ('classification', 'detection', 'regression'),key="category")
    wav_amp = st.sidebar.number_input("Amplitude: float, 0.00 for None", value=0.00, step=0.01, format="%.2f", key="wave_amp")
    wav_freq = st.sidebar.number_input("Frequency: float, 0.00 for None", value=0.00, step=0.01, format="%.2f", key="wave_freq")
    wav_noise = st.sidebar.checkbox("Check to add Noise", key="wave_noise_box")
    wav_num = st.sidebar.number_input("Wave Number: integer", value=10, step=1, key="wave_num")
    wav_label = st.sidebar.selectbox('Labels:', ('frequency', 'amplitude'),key='wav_labels')

   
    if st.sidebar.button("Generate the Wave Data"):
        if wav_amp == 0.0:
            new_amp = None
        else:
            new_amp = wav_amp

        if wav_freq == 0.0:
            new_freq = None
        else:
            new_freq = wav_freq
        
        if wav_cat == 'classification':
            data, labels = generate_class_data(amplitude=new_amp,frequency=new_freq,noise=wav_noise,wave_number=wav_num,show=True)
        elif wav_cat == 'detection':
            data, labels = generate_anomaly_data(amplitude=new_amp,frequency=new_freq,noise=wav_noise,wave_number=wav_num,show=True)
        else:
            data, labels = generate_regression_data(amplitude=new_amp,frequency=new_freq,noise=wav_noise,wave_number=wav_num,label_type=wav_label,show=True)

        labels = labels.reshape(-1, 1)
        data_file = np.concatenate((data,labels),axis=1)

        fig = plot_single_row(data_file)
        st.pyplot(fig)
        save_numpy_array(data_file)

elif option == 'generate scg signals':
    if st.sidebar.button("Clear All Wavefor Inputs"):
        st.session_state.num_rows = 1
        st.session_state.duration = 10
        st.session_state.sampling_rate = 100
        st.session_state["add_respiratory"] = True
        st.session_state['repiratory_rate'] = '10,30'
        st.session_state["systolic"] = '90,140'
        st.session_state["diastolic"] = '80,100'
        st.session_state["pulse_type"] = 'db'
        st.session_state["noise_type"] = 'basic'
        st.session_state["noise_shape"] = 'laplace'
        st.session_state.noise_amplitude = 0.1
        st.session_state["noise_frequency"] = '5,10,100'
        st.session_state.power_line_amplitude = 0.0
        st.session_state.power_line_frequency = 100
        st.session_state.artifacts_amplitude = 0.0
        st.session_state.artifacts_frequency = 100.0
        st.session_state.artifacts_number = 5
        st.session_state["artifacts_shape"] = 'laplace'
        st.session_state.n_echo = 3
        st.session_state["attenuation_factor"] = '0.1,0.05,0.02'
        st.session_state.delay_factor = 15
        st.session_state["random_state"] = 'None'
        st.session_state.silent = False
        st.session_state["scg_labels"] = 'heart rate'
        st.rerun()

    n_row = st.sidebar.number_input("Wave Number: integer", value=1, step=1, key="num_rows")
    
    dur = st.sidebar.number_input("Wave Duration: integer", value=10, step=1, key="duration")

    s_rate = st.sidebar.number_input("Wave Sampling Rate: integer", value=100, step=1, key="sampling_rate")

    add_resp = st.sidebar.checkbox("Check to add Noise", value=True, key="add_respiratory")

    resp_rate = st.sidebar.text_input("Respiratory Rate: comma seperated integers", value='10,30',key="respiratory_rate")
    resp_rate_list = parse_text_entry(resp_rate,'int')

    syst = st.sidebar.text_input("Systolic Pressure: comma seperated integers", value='90,140',key="systolic")
    syst_list = parse_text_entry(syst,'int')

    dias = st.sidebar.text_input("Diastolic Pressure: comma seperated integers", value='90,140',key="diastolic")
    dias_list = parse_text_entry(dias,'int')

    pulse_t = st.sidebar.selectbox('Pulse Type:', ('db','mor','ricker','sym','coif'),key="pulse_type")

    noise_t = st.sidebar.selectbox('Noise Type:', ('basic','resonance','powerline','artifacts','linear_drift'),key="noise_type")

    noise_s = st.sidebar.selectbox('Noise Shape:', ('laplace', 'gaussian'),key="noise_shape")

    noise_a = st.sidebar.number_input("Noise Amplitude: float", value=0.10, step=0.01, format="%.2f", key="noise_amplitude")

    noise_f = st.sidebar.text_input("Noise Frequency: comma seperated floats", value='5,10,100',key="noise_frequency")
    noise_f_list = parse_text_entry(noise_f,'float')

    pl_a = st.sidebar.number_input("Power Line Amplitude: float", value=0.00, step=0.01, format="%.2f", key="power_line_amplitude")

    pl_f = st.sidebar.number_input("Power Line Frequency: float", value=50.00, step=0.01, format="%.2f", key="power_line_frequency")

    art_a = st.sidebar.number_input("Artifacts Amplitude: float", value=0.00, step=0.01, format="%.2f", key="artifacts_amplitude")

    art_f = st.sidebar.number_input("Artifacts Frequency: float", value=100.00, step=0.01, format="%.2f", key="artifacts_frequency")

    art_n = st.sidebar.number_input("Artifacts Number: integer", value=5, step=1, key="art_number")

    art_s = st.sidebar.selectbox('Artifacts Shape:', ('laplace', 'gaussian'),key="artifacts_shape")

    n_e = st.sidebar.number_input("Echo Number: integer", value=3, step=1, key="n_echo")

    att_f = st.sidebar.text_input("Attenuation Factor: comma seperated floats", value='0.1,0.05,0.02',key="attenuation_factor")
    att_f_list = parse_text_entry(noise_f,'float')

    del_f = st.sidebar.number_input("Delay Factor: float", value=15.00, step=0.01, format="%.2f", key="delay_factor")

    rs = st.sidebar.text_input("Random State: a single float or None", value='None',key="random_state")
    rs_list = parse_text_entry(noise_f,'int')

    sil = st.sidebar.checkbox("Silent", key="silent")

    lab = st.sidebar.selectbox('SCG Labels:', ("heart rate", "respiratory rate", "systolic pressure", "diastolic pressue"),key="scg_labels")

    if st.sidebar.button("Generate the SCG Data"):
        scg_data = scg_simulate(num_rows=n_row, duration=dur, sampling_rate=s_rate, add_respritory=add_resp, respiratory_rate=resp_rate_list,
                                systolic=syst_list, diastolic=dias_list, pulse_type=pulse_t, noise_type=noise_t, noise_shape=noise_s,
                                noise_amplitude=noise_a, noise_frequency=noise_f_list, power_line_amplitude=pl_a, power_line_frequency=pl_f,
                                artifacts_amplitude=art_a, artifacts_frequency=art_f, artifacts_number=art_n, artifacts_shape=art_s,
                                n_echo=n_e, attenuation_factor=att_f_list, delay_factor=del_f, random_state=rs_list[0], silent=sil,
                                label_data=lab,save_data=True)
        
        temp = scg_data.copy()
        fig = plot_single_row(temp[:, :-1])
        st.pyplot(fig)
        print("SCG Data Created")