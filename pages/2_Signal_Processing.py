import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import kurtosis, skew
import pywt
from pathlib import Path
import time
import matplotlib.pyplot as plt

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
from dsp import *


def active_save_verification(signal):
    if make_active_check == True:
        st.session_state["active_dataset"] = signal
        st.write(st.session_state["active_dataset"])
    if save_output_check == True:
        save_numpy_array(signal)
        st.write("Noisy Data saved to .npy file")

st.sidebar.title("Digital Signal Processing")

dsp_tuple = ('noise generation', 'filtering', 'time & frequency domain features', 'signal decomposition',
               'wavelet analysis', 'transforms')

##########################################################
# NOISE GENERATION
##########################################################

noise_tuple = ('white', 'impulse', 'burst', 'colored', 
               'echo', 'flicker', 'powerline')

white_noise_tuple = ('gaussian', 'laplacian', 'band-limited')

color_noise_tuple = ('blue', 'brown', 'pink')

make_active_check = st.sidebar.checkbox("Would you like to make the results your active data file?",key="make active check")
save_output_check = st.sidebar.checkbox("Would you like to save the output as a .npy file?",key="save output check")

dsp_box = st.sidebar.selectbox('**Select the type of processing to perform.**', dsp_tuple)

if dsp_box == 'noise generation':    
    noise_box = st.sidebar.selectbox('**Select the type of noise**', noise_tuple)

if dsp_box == 'noise generation':
    if noise_box == 'white':
        white_type_box = st.sidebar.selectbox('**Select the type of white noise**', white_noise_tuple)

if dsp_box == 'noise generation':
    if noise_box == 'colored':
        color_box = st.sidebar.selectbox('**Select the color of noise**', color_noise_tuple,key='color_noise_box')

if dsp_box == 'noise generation':
    if noise_box != 'echo':
        noise_amp = st.sidebar.number_input(label='noise amplitude',step=0.01,format="%.02f",value=0.3,key='noise_amplitude_box')

if  dsp_box == 'noise generation':
    if noise_box == 'white' and white_type_box == 'band-limited':
        blw_lc = st.sidebar.number_input(label='noise low cutoff frequency',step=0.01,format="%.02f",value=0.01)
        blw_hc = st.sidebar.number_input(label='noise high cutoff frequency',step=0.01,format="%.02f",value=10.0)
        blw_sr = st.sidebar.number_input(label='band-limited sampling rate',min_value=0,step=1,value=100)
        blw_or = st.sidebar.number_input(label='band-limited order',min_value=0,step=1,value=3)

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'noise generation':
    if noise_box == 'white':
        if st.sidebar.button("Add White Noise to Signal"):
            noisy_signal = st.session_state["active_dataset"].copy()
            wr_count = 0
            for row in noisy_signal:
                if wr_count == 0:
                    show = True
                else:
                    show = False

                if white_type_box == 'gaussian':
                    noisy_row = add_white_noise(signal=row, noise_amplitude=noise_amp, model=0, show=show, stream=show)
                    noisy_signal[wr_count] = noisy_row
                elif white_type_box == 'laplacian':
                    noisy_row = add_white_noise(signal=row, noise_amplitude=noise_amp, model=1, show=show, stream=show)
                    noisy_signal[wr_count] = noisy_row
                elif white_type_box == 'band-limited':
                    noisy_row = add_band_limited_white_noise(signal=row, noise_amplitude=noise_amp, lowcut=blw_lc, highcut=blw_hc, 
                                                sampling_rate=blw_sr, order=blw_or, show=show, stream=show)
                    noisy_signal[wr_count] = noisy_row

                wr_count += 1
            
            active_save_verification(noisy_signal)

if dsp_box == 'noise generation':
    if noise_box == 'impulse':
        add_imp_rate = st.sidebar.checkbox("Add an impulse rate", value=False, key="add_impulse_rate")
        if add_imp_rate == True:
            imp_rt = st.sidebar.number_input(label='impulse rate',step=0.01,format="%.02f",value=0.5)
        else:
            imp_rt = None

        add_imp_num = st.sidebar.checkbox("Add more than one impulse", value=False, key="add_impulse_number")
        if add_imp_num == True:
            imp_num = st.sidebar.number_input(label='number of impulses',min_value=0,step=1,value=2)
        else:
            imp_num = None

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'noise generation':
    if noise_box == 'impulse':
        if st.sidebar.button("Add Impulse Noise to Signal"):
            noisy_signal = st.session_state["active_dataset"].copy()
            ir_count = 0
            for row in noisy_signal:           
                if ir_count == 0:
                    show = True
                else:
                    show = False
                noisy_row = add_impulsive_noise(signal=row, noise_amplitude=noise_amp, rate=imp_rt, 
                                                number=imp_num, show=show, stream=show)
                noisy_signal[ir_count] = noisy_row
                ir_count += 1
    
            active_save_verification(noisy_signal)


if dsp_box == 'noise generation':
    if noise_box == 'burst':
        brst_num_max = st.sidebar.number_input(label='max number of burst noise events to add',min_value=1,step=1,value=1, key="burst_number_maximum")
        brst_dur = st.sidebar.text_input(label='burst duration: list minimum and maximum, integers',value='1,2', key="burst_duration")    

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'noise generation':
    if noise_box == 'burst':
        if st.sidebar.button("Add Impulse Noise to Signal"):
            brst_dur_list = parse_text_entry(brst_dur,'int') 
            noisy_signal = st.session_state["active_dataset"].copy()
            br_count = 0
            for row in noisy_signal:
                if br_count == 0:
                    show = True
                else:
                    show = False
                noisy_row = add_burst_noise(signal=row, noise_amplitude=noise_amp, 
                                                burst_num_max=brst_num_max, burst_durations=brst_dur_list, 
                                                show=show, stream=show)
                noisy_signal[br_count] = noisy_row
                br_count += 1
    
            active_save_verification(noisy_signal)
    
if  dsp_box == 'noise generation':
    if noise_box == 'colored':
        clr_sam_rt = st.sidebar.number_input(label='sampling rate',min_value=1,step=1,value=100,key='color_noise_sampling_rate')
        clr_dur = st.sidebar.number_input(label='duration',min_value=1,step=1,value=10,key='color_noise_duration')

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'noise generation':
    if noise_box == 'colored':
        if st.sidebar.button("Add Colored Noise to Signal"):
            noisy_signal = st.session_state["active_dataset"].copy()
            cr_count = 0
            for row in noisy_signal:
                if cr_count == 0:
                    show = True
                else:
                    show = False

                noisy_row = add_colored_noise(signal=row, noise_amplitude=noise_amp, model=str(color_box), sampling_rate=int(clr_dur), 
                                            duration=int(clr_dur), show=show, stream=show)
                noisy_signal[cr_count] = noisy_row
                cr_count += 1
            
            active_save_verification(noisy_signal)


if dsp_box == 'noise generation':
    if noise_box == 'echo':
        echo_num = st.sidebar.number_input(label='number of echos: single integer',min_value=1,step=1,value=2, key="echo_number")
        echo_att = st.sidebar.text_input(label='echo attenuation factors: must have an entry for each echo number',value='0.5,0.4', key="echo_attenuation_factor")
        echo_del = st.sidebar.text_input(label='echo delay factor: delay for each echoe, must have same number of entries as echo numers',value='5,5', key="echo_delay_factor")  

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'noise generation':
    if noise_box == 'echo':
        if st.sidebar.button("Add Echo Noise to Signal"):
            echo_att_list = parse_text_entry(echo_att,'float')
            echo_del_list = parse_text_entry(echo_del,'int') 
            noisy_signal = st.session_state["active_dataset"].copy()
            er_count = 0
            for row in noisy_signal:
                if er_count == 0:
                    show = True
                else:
                    show = False
                noisy_row = add_echo_noise(signal=row, n_echo=echo_num, 
                                           attenuation_factor=echo_att_list, delay_factor=echo_del_list,
                                           show=show, stream=show)
                noisy_signal[er_count] = noisy_row
                er_count += 1
    
            active_save_verification(noisy_signal)

if dsp_box == 'noise generation':
    if noise_box == 'flicker':
        flk_sr = st.sidebar.number_input(label='flicker sampling rate: float',min_value=1,step=1,value=10, key="flicker_sampling_rate")
        flk_dur = st.sidebar.number_input(label='flicker duration: float',min_value=1,step=1,value=2, key="flicker_duration")

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'noise generation':
    if noise_box == 'flicker':
        if st.sidebar.button("Add Flicker Noise to Signal"):
            noisy_signal = st.session_state["active_dataset"].copy()
            fr_count = 0
            for row in noisy_signal:
                if fr_count == 0:
                    show = True
                else:
                    show = False
                noisy_row = add_flicker_noise(signal=row, noise_amplitude=noise_amp, 
                                              sampling_rate=flk_sr, duration=flk_dur,
                                              show=show, stream=show)
                noisy_signal[fr_count] = noisy_row
                fr_count += 1
    
            active_save_verification(noisy_signal)


if dsp_box == 'noise generation':
    if noise_box == 'powerline':
        pow_sr = st.sidebar.number_input(label='sampling rate',min_value=1,step=1,value=2, key="powerline_sampling_rate")
        pow_dur = st.sidebar.number_input(label='powerline duration',min_value=1,step=1,value=2, key="powerline_duration")
        pow_frq = st.sidebar.number_input(label='powerline frequency',min_value=1,step=1,value=50, key="powerline_frequency")

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'noise generation':
    if noise_box == 'powerline':
        if st.sidebar.button("Add Powerline Noise to Signal"):
            noisy_signal = st.session_state["active_dataset"].copy()
            pr_count = 0
            for row in noisy_signal:
                if pr_count == 0:
                    show = True
                else:
                    show = False
                noisy_row = add_powerline_noise(signal=row, sampling_rate=pow_sr, duration=pow_dur,
                                                powerline_frequency=pow_frq, powerline_amplitude=noise_amp,
                                                show=show, stream=show)
                noisy_signal[pr_count] = noisy_row
                pr_count += 1
    
            active_save_verification(noisy_signal)


##########################################################
# FILTERING
##########################################################

filter_tuple = ('butterworth', 'adaptive', 'denoising', 'kalman', 'matched', 'notch', 'sav-gol', 'wiener')

butter_tuple = ('high-pass','low-pass','band-pass','band-stop')

denoise_tuple = ('fft','wavelet')

if dsp_box == 'filtering':
    filter_box = st.sidebar.selectbox('**Select the type of filtering.**', filter_tuple)
            
if dsp_box == 'filtering':
    if filter_box == 'butterworth':
        btt_type = st.sidebar.selectbox('**Select the type of filtering.**', butter_tuple)
        if btt_type == 'high-pass' or btt_type == 'low-pass':
            btt_co = st.sidebar.number_input(label='cutoff frequency',min_value=1,step=1,value=10, key="butterworth_cutoff")
        elif btt_type == 'band-pass' or btt_type == 'band-stop':
            btt_lc = st.sidebar.number_input(label='low cutoff frequency',min_value=1,step=1,value=10, key="butterworth_low_cutoff")
            btt_hc = st.sidebar.number_input(label='high cutoff frequency',min_value=1,step=1,value=10, key="butterworth_high_cutoff")

        btt_sf = st.sidebar.number_input(label='sampling frequency',min_value=1,step=1,value=100, key="butterworth_sampling_frequency")
        btt_or = st.sidebar.number_input(label='order',min_value=1,step=1,value=5, key="butterworth_order")
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'filtering':
    if filter_box == 'butterworth':
        if st.sidebar.button("Filter Signal with Butterworth"):
            filtered_signal = st.session_state["active_dataset"].copy()
            pr_count = 0
            for row in filtered_signal:
                if pr_count == 0:
                    show = True
                else:
                    show = False
                if btt_type == 'high-pass':
                    filtered_row = butter_highpass_filter(signal=row, cutoff=btt_co, fs=btt_sf, 
                                                       order=btt_or, show=show, stream=show)
                elif btt_type == 'low-pass':
                    filtered_row =  butter_lowpass_filter(signal=row, cutoff=btt_co, fs=btt_sf, 
                                                       order=btt_or, show=show, stream=show)
                elif btt_type == 'band-pass':
                    filtered_row = butter_bandpass_filter(signal=row, lowcut=btt_lc, highcut=btt_hc,
                                                       fs=btt_sf, order=btt_or, show=show, stream=show)
                elif btt_type == 'band-stop':
                    filtered_row = butter_bandstop_filter(signal=row, lowcut=btt_lc, highcut=btt_hc,
                                                       fs=btt_sf, order=btt_or, show=show, stream=show)
                #my_stft(row, plot=show, fs=2, stream=show)
                #my_stft(filtered_row, plot=show, fs=2, stream=show)
                filtered_signal[pr_count] = filtered_row
                pr_count += 1
    
            active_save_verification(filtered_signal)

if dsp_box == 'filtering':
    if filter_box == 'adaptive':
        adpt_n = st.sidebar.number_input(label='order',min_value=1,step=1,value=100, key="adaptive_order")
        adpt_mu = st.sidebar.number_input(label='convergence factor',min_value=1,step=1,value=5, key="adaptive_convergence_factor")
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'filtering':
    if filter_box == 'adaptive':
        if st.sidebar.button("Filter Signal with Adaptive"):
            filtered_signal = st.session_state["active_dataset"].copy()
            ad_count = 0
            for row in filtered_signal:
                if ad_count == 0:
                    show = True
                else:
                    show = False
                filtered_row, error, weight = lms_filter(x=noise_template, d=row, n=adpt_n, mu=adpt_mu, show=show, stream=show)
                filtered_new = row - filtered_row
                filtered_signal[ad_count] = filtered_new
                ad_count += 1
    
            active_save_verification(filtered_signal)



wavelet_tuple = ('haar', 'dmey')

if dsp_box == 'filtering':
    if filter_box == 'denoising':
        dns_type = st.sidebar.selectbox('**Select the type of filtering.**', denoise_tuple)
        
        dns_th = st.sidebar.number_input(label='threshold',min_value=1,step=1,value=5, key="denoising_threshold")

        if dns_type == 'wavelet':
            dns_mt = st.sidebar.selectbox('Select Wavelet Method.', wavelet_tuple)
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'filtering':
    if filter_box == 'denoising':
        if st.sidebar.button("Denoise Signal"):
            filtered_signal = st.session_state["active_dataset"].copy()
            pr_count = 0
            for row in filtered_signal:
                if pr_count == 0:
                    show = True
                else:
                    show = False
                if dns_type == 'fft':
                    filtered_row = fft_denoise(signal=row, threshold=dns_th, show=show, stream=show)
                elif dns_type == 'wavelet':
                    filtered_row =  wavelet_denoise(data=row, method=dns_mt, threshold=dns_th, show=show, 
                                                    title="Wavelet Denoising", stream=show)
                filtered_signal[pr_count] = filtered_row
                pr_count += 1
    
            active_save_verification(filtered_signal)


if dsp_box == 'filtering':
    if filter_box == 'kalman':
      
        kal_xl = st.sidebar.number_input(label='x_last',min_value=0,step=1,value=0, key="kalman_x_last")
        kal_pl = st.sidebar.number_input(label='p_last',min_value=0,step=1,value=0, key="kalman_p_last")
        kal_q = st.sidebar.number_input(label='Q',step=0.01,format="%.02f",value=0.1, key="kalman_q")
        kal_r = st.sidebar.number_input(label='R',step=0.01,format="%.02f",value=0.1, key='kalman_r')
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'filtering':
    if filter_box == 'kalman':
        if st.sidebar.button("Filter Signal with Kalman"):
            filtered_signal = st.session_state["active_dataset"].copy()
            pr_count = 0
            for row in filtered_signal:
                if pr_count == 0:
                    show = True
                else:
                    show = False
                filtered_row =  kalman_filter(x=row, x_last=kal_xl, p_last=kal_pl, Q=kal_q, R=kal_r, show=show, stream=show)
                filtered_signal[pr_count] = filtered_row
                pr_count += 1
    
            active_save_verification(filtered_signal)

##########################################################
# TIME & FREQUENCY DOMAIN FEATURES
##########################################################
            
time_freq_tuple = ('time domain features', 'peaks & envelopes', 'power spectral density') #, 'transforms', 'misc')

#td_tuple = ('statistical moments', 'petrosian fractal dimension')
if dsp_box == 'time & frequency domain features':
    time_freq_box = st.sidebar.selectbox('**Select the features to extract.**', time_freq_tuple)

any_checked = False

if dsp_box == 'time & frequency domain features':
    if time_freq_box == 'time domain features':
      
        pmu_check = st.sidebar.checkbox("PMU (frequency, amplitude, phase angle)",key="pmu check")
        if pmu_check == True:
            pmu_sf = st.sidebar.number_input(label='pmu sampling frequency',min_value=1,step=1,value=1, key="pmu_sampline_frequency")
        mean_check = st.sidebar.checkbox("Mean",key="mean check")
        var_check = st.sidebar.checkbox("Variance",key="variance check")
        skew_check = st.sidebar.checkbox("Skewness",key="skewness check")
        kurt_check = st.sidebar.checkbox("Kurtosis",key="kurtosis check")
        pfd_check = st.sidebar.checkbox("Petrosian Fractal Dimension",key="pfd check")
        thd_check = st.sidebar.checkbox("Total Harmonic Distortion",key="thd check")
        if thd_check == True:
            thd_fund = st.sidebar.number_input(label='fundamental frequency',min_value=1,step=1,value=1, key="thd_fundamental_frequency")
            thd_sf = st.sidebar.number_input(label='thd sampling frequency',min_value=2,step=1,value=2, key="thd_sampling_frequency")
            thd_har = st.sidebar.number_input(label='number of harmonics',min_value=1,step=1,value=5, key="thd_number_harmonics")
        
        if pmu_check == True or mean_check == True or var_check == True or skew_check == True or kurt_check == True or pfd_check == True or thd_check == True:
            any_checked = True
        else:
            any_checked = False

        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'time & frequency domain features' and any_checked == True:
    if time_freq_box == 'time domain features':
        if st.sidebar.button("Extract Features"):
            current_signal = st.session_state["active_dataset"].copy()
            ex_count = 0
            extracted_list = []            
            first_row_info = []
            for row in current_signal:  
                current_list = []              
                if pmu_check == True:
                    pmu_tuple = extract_pmu(row, pmu_sf)
                    current_list.append(pmu_tuple[0])
                    current_list.append(pmu_tuple[1])
                    current_list.append(pmu_tuple[2])
                    if ex_count == 0:
                        freq_txt = "The Frequency is " + str(pmu_tuple[0])
                        first_row_info.append(freq_txt)
                        amp_txt = "The Amplitude is " + str(pmu_tuple[1])
                        first_row_info.append(amp_txt)
                        phas_txt = "The Phase Angle is " + str(pmu_tuple[2])
                        first_row_info.append(phas_txt)
                if mean_check == True:
                    cur_mean = np.mean(row)
                    current_list.append(cur_mean)
                    if ex_count == 0:
                        mean_txt = "The Mean is " + str(cur_mean)
                        first_row_info.append(mean_txt)
                if var_check == True:
                    cur_var = np.var(row)
                    current_list.append(cur_var)
                    if ex_count == 0:
                        var_txt = "The Variance is " + str(cur_var)
                        first_row_info.append(var_txt)
                if skew_check == True:
                    cur_skew = skew(row)
                    current_list.append(cur_skew)
                    if ex_count == 0:
                        skew_txt = "The Skewness is " + str(cur_skew)
                        first_row_info.append(skew_txt)
                if kurt_check == True:
                    cur_kurt = kurtosis(row)
                    current_list.append(cur_kurt)
                    if ex_count == 0:
                        kurt_txt = "The Kurtosis is " + str(cur_kurt)
                        first_row_info.append(kurt_txt)
                if pfd_check == True:
                    cur_pfd = pfd(row)
                    current_list.append(cur_pfd)
                    if ex_count == 0:
                        pfd_txt = "Petrosian Fractal Dimension " + str(cur_pfd)
                        first_row_info.append(pfd_txt)
                if thd_check == True:
                    cur_thd = calculate_thd(row, thd_fund, thd_sf, thd_har)
                    current_list.append(cur_thd)
                    if ex_count == 0:
                        thd_txt = "The Total Harmonic Distortion is " + str(cur_thd)
                        first_row_info.append(thd_txt)
                
                #current_array = np.array(current_list)
                if ex_count == 0:
                    st.write("THE EXTRACTED FEATURES FOR THE FIRST INPUT SIGNAL ARE:")
                    for item in first_row_info:
                        st.write(item)

                extracted_list.append(current_list)

                ex_count += 1

            extracted_signal = np.array(extracted_list)

            df = pd.DataFrame(extracted_signal)
            st.title("Feature Preview")
            st.dataframe(df.head(5))
            active_save_verification(extracted_signal)

### Peak Detection Algorithms

peak_tuple = ('envelope from peaks', 'average envelope', 'hilbert envelope & phase') # 'peak detection',

# Peak Detection

if dsp_box == 'time & frequency domain features':
    if time_freq_box == 'peaks & envelopes':
        peak_box = st.sidebar.selectbox('**Select the peak extraction algorithm.**', peak_tuple)

        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'time & frequency domain features':
    if time_freq_box == 'peaks & envelopes':
        if peak_box == 'peak detection':
            if st.sidebar.button("Get Peaks"):
                filtered_signal = st.session_state["active_dataset"].copy()
                pr_count = 0
                print("Data Shape is: ",filtered_signal.shape)
                for row in filtered_signal:
                    peak =  get_peaks(signal=row)                    

                    if pr_count == 0:
                        peaks = peak
                        plt.plot(row, label="Original signal")
                        plt.scatter(peak,row[peak],c="red", label="Peak of the signal")
                        plt.xlabel("Time")
                        plt.ylabel("Amplitude")
                        plt.legend()
                        st.pyplot(plt)
                    else:
                        peaks = np.vstack((peaks,peak))

                    if pr_count == 485 or pr_count == 486:
                        print(row)
                        print(peak)
                        
                    pr_count += 1

                active_save_verification(peaks)


# Envelope from Peaks
                       
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'time & frequency domain features':
    if time_freq_box == 'peaks & envelopes':
        if peak_box == 'envelope from peaks':
            if st.sidebar.button("Get Envelope"):
                filtered_signal = st.session_state["active_dataset"].copy()
                pr_count = 0
                envelope_list=[]
                for row in filtered_signal:
                    envelope =  envelope_from_peaks(signal=row)                    

                    if pr_count == 0:
                        plt.plot(row,label="Original signal")
                        plt.plot(envelope,c="red", label="Envelope of signal")
                        plt.xlabel("Time")
                        plt.ylabel("Amplitude")
                        plt.legend()
                        st.pyplot(plt)

                    pr_count += 1

                envelopes = np.array(envelope_list)
                active_save_verification(envelopes)

# Average Envelope

if dsp_box == 'time & frequency domain features':
    if time_freq_box == 'peaks & envelopes':
        if peak_box == 'average envelope':

            env_win = st.sidebar.number_input(label='window size',min_value=0,step=1,value=10, key="envelope_window_size")
      

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'time & frequency domain features':
    if time_freq_box == 'peaks & envelopes':
        if peak_box == 'average envelope':
            if st.sidebar.button("Get Average Envelope"):
                filtered_signal = st.session_state["active_dataset"].copy()
                pr_count = 0
                avg_envelope_list=[]
                for row in filtered_signal:
                    avg_envelope =  average_envelope(signal=row, window_length=env_win)                  

                    if pr_count == 0:
                        plt.plot(row,label="Original signal")
                        plt.plot(avg_envelope,c="red", label="Envelope of signal")
                        plt.xlabel("Time")
                        plt.ylabel("Amplitude")
                        plt.legend()
                        st.pyplot(plt)

                    pr_count += 1

                avg_envelopes = np.array(avg_envelope_list)
                active_save_verification(avg_envelopes)

# Envelope for the Hilbert Transform Old   

# if dsp_box == 'time & frequency domain features':
#     if time_freq_box == 'peaks & envelopes':
#         if peak_box == 'hilbert envelope & phase':

#             hil_win = st.sidebar.number_input(label='window size',min_value=2,step=1,value=20, key="hilbert_env_window_size")
#             hil_lag = st.sidebar.number_input(label='lag',min_value=1,step=1,value=1, key="hilbert_env_lag")
      

# if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'time & frequency domain features':
#     if time_freq_box == 'peaks & envelopes':
#         if peak_box == 'hilbert envelope & phase':
#             if st.sidebar.button("Get Hilbert Envelope"):
#                 filtered_signal = st.session_state["active_dataset"].copy()
#                 he_count = 0
#                 he_envelope_list=[]
#                 L = hil_win  # window size
#                 d = hil_lag   # lag
#                 for row in filtered_signal:
#                     # Step 1: Create the trajectory matrix
                    
#                     X = np.array([row[i:i+L] for i in range(len(row) - L)])

#                     # Step 2: Perform Singular Value Decomposition (SVD)
#                     U, S, Vt = np.linalg.svd(X, full_matrices=False)

#                     # Step 3: Reconstruct the signal by keeping the largest singular value
#                     reconstructed_signal = U[:, 0] @ S[0]  # Using the first singular value for reconstruction

#                     he_envelope_list.append(reconstructed_signal)

#                     if he_count == 0:
#                         # Step 4: Plot the original and reconstructed signal
#                         plt.figure(figsize=(10, 6))
#                         plt.plot(time[L:], row[L:], label="Original Signal")
#                         plt.plot(time[L:], reconstructed_signal, label="Reconstructed Signal", linestyle='--')
#                         plt.legend()
#                         plt.xlabel('Time')
#                         plt.ylabel('Amplitude')
#                         plt.title('Singular Spectrum Transform - Signal Reconstruction')
#                         plt.show()

#                     he_count += 1

#                 he_envelopes = np.array(he_envelope_list)
#                 active_save_verification(he_envelopes)

# Envelope for the Hilbert Transform

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'time & frequency domain features':
    if time_freq_box == 'peaks & envelopes':
        if peak_box == 'hilbert envelope & phase':
            if st.sidebar.button("Get Hilbert Envelope"):
                filtered_signal = st.session_state["active_dataset"].copy()
                pr_count = 0
                hilbert_envelope_list=[]
                for row in filtered_signal:
                    analytic =  analytic_signal(x=row)
                    hilbert_envelope = hilbert_transform(x=analytic)                 

                    if pr_count == 0:
                        plt.plot(row,label="Original signal")
                        plt.plot(hilbert_envelope,c="red", label="Envelope of signal")
                        plt.xlabel("Time")
                        plt.ylabel("Amplitude")
                        plt.legend()
                        st.pyplot(plt)

                    pr_count += 1

                hilbert_envelopes = np.array(hilbert_envelope_list)
                active_save_verification(hilbert_envelopes)                


# Power Spectral Density

if dsp_box == 'time & frequency domain features':
    if time_freq_box == 'power spectral density':

        psd_sf = st.sidebar.number_input(label='sampling frequency',min_value=2,step=1,value=10, key="psd_sampling_frequency")
      

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'time & frequency domain features':
    if time_freq_box == 'power spectral density':

        if st.sidebar.button("Get the PSD"):
            filtered_signal = st.session_state["active_dataset"].copy()
            pr_count = 0
            psd_list=[]
            for row in filtered_signal:
                f,p =  psd(signal=row, fs=psd_sf)                  
                psd_list.append(p)

                if pr_count == 0:

                    fig, axis = plt.subplots(2,1,figsize=(8,6))
                    axis[0].set_title("Input signal of first row of data")
                    axis[0].plot(row)
                    axis[0].set_xlabel("Time")
                    axis[0].set_ylabel("Amplitude")


                    axis[1].set_title("PSD of input signal of first row of data")
                    axis[1].plot(f,p)
                    axis[1].set_xlabel("Frequency")
                    axis[1].set_ylabel("Power")
                    plt.tight_layout()
                    st.pyplot(plt)

                pr_count += 1

            psds = np.array(psd_list)
            active_save_verification(psds)


##########################################################
# SIGNAL DECOMPOSITION
##########################################################
            
decomp_tuple = ('empirical mode decomposition', 'ensemble emd', 'complete eemd', 'variational mode', 
                'singular spectrum analysis')#, 'principal component analysis', 'independent component analysis')


# EMD

if dsp_box == 'signal decomposition':
    decomp_box = st.sidebar.selectbox('**Select the features to extract.**', decomp_tuple)
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'signal decomposition':
    if decomp_box == 'empirical mode decomposition':
        if st.sidebar.button("Decompose signal with EMD"):
            filtered_signal = st.session_state["active_dataset"].copy()
            emd_count = 0
            for row in filtered_signal:
                if emd_count == 0:
                    show = True
                else:
                    show = False

                filtered_row = emd_decomposition(signal=row,show=show, stream=show)

                if emd_count == 0:
                    emd_array = filtered_row
                else:
                    np.vstack((emd_array,filtered_row))
                emd_count += 1

            active_save_verification(emd_array)

# EEMD

if dsp_box == 'signal decomposition':
    if decomp_box == 'ensemble emd':
        eemd_n = st.sidebar.number_input(label='noise width',step=0.01,format="%.02f",value=0.05, key="eemd_noise_width")
        eemd_s = st.sidebar.number_input(label='ensemble size',min_value=2,step=1,value=100, key="eemd_ensemble_size")
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'signal decomposition':
    if decomp_box == 'ensemble emd':
        if st.sidebar.button("Decompose signal with EEMD"):
            filtered_signal = st.session_state["active_dataset"].copy()
            emd_count = 0
            for row in filtered_signal:
                if emd_count == 0:
                    show = True
                else:
                    show = False

                filtered_row = eemd_decomposition(signal=row, noise_width=eemd_n, ensemble_size=eemd_s, show=show, stream=show)

                if emd_count == 0:
                    eemd_array = filtered_row
                else:
                    np.vstack((eemd_array,filtered_row))
                emd_count += 1

            active_save_verification(eemd_array)


# CEEMD
            
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'signal decomposition':
    if decomp_box == 'complete eemd':
        if st.sidebar.button("Decompose signal with CEEMD"):
            filtered_signal = st.session_state["active_dataset"].copy()
            emd_count = 0
            for row in filtered_signal:
                if emd_count == 0:
                    show = True
                else:
                    show = False

                filtered_row = ceemd_decomposition(signal=row, show=show, stream=show)

                if emd_count == 0:
                    ceemd_array = filtered_row
                else:
                    np.vstack((ceemd_array,filtered_row))
                emd_count += 1

            active_save_verification(ceemd_array)


# VMD

if dsp_box == 'signal decomposition':
    if decomp_box == 'variational mode':
        vmd_k = st.sidebar.number_input(label='K',min_value=1,step=1,value=5, key="vmd_k")
        vmd_a = st.sidebar.number_input(label='alpha',step=0.01,format="%.02f",value=2000.00, key="vmd_alpha")
        vmd_t = st.sidebar.number_input(label='tau',step=0.01,format="%.02f",value=0.00, key="vmd_tau")
        vmd_d = st.sidebar.number_input(label='DC',min_value=0, max_value=1, step=1,value=1, key="vmd_dc")
        
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'signal decomposition':
    if decomp_box == 'variational mode':
        if st.sidebar.button("Decompose signal with VMD"):
            filtered_signal = st.session_state["active_dataset"].copy()
            emd_count = 0
            for row in filtered_signal:
                if emd_count == 0:
                    show = True
                else:
                    show = False

                filtered_row = vmd_decomposition(signal=row, K=vmd_k, alpha=vmd_a, tau=vmd_t, DC=vmd_d, show=show, stream=show)

                if emd_count == 0:
                    vmd_array = filtered_row                    
                else:
                    np.vstack((vmd_array,filtered_row))
                emd_count += 1

            active_save_verification(vmd_array)

# Singular Spectrum Analysis

if dsp_box == 'signal decomposition':
    if decomp_box == 'singular spectrum analysis':
        ssa_l = st.sidebar.number_input(label='window length',min_value=2,step=1,value=100, key="ssa_window_length")
        
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'signal decomposition':
    if decomp_box == 'singular spectrum analysis':
        if st.sidebar.button("Decompose signal with SSA"):
            filtered_signal = st.session_state["active_dataset"].copy()
            emd_count = 0
            for row in filtered_signal:

                ssa_signal = SSA(tseries=row, L=ssa_l)
                ssa_signal.calc_wcorr()
                wcorr_matrix = ssa_signal.Wcorr

                if emd_count == 0:
                    ssa_array = wcorr_matrix
                    # Plot the within-correlation matrix
                    ssa_signal.plot_wcorr()
                    plt.title(r"W-Correlation for Components 0–" + str(ssa_l))
                    st.pyplot(plt)
                else:
                    np.vstack((ssa_array,wcorr_matrix))
                emd_count += 1

            active_save_verification(ssa_array)            

# PCA

if dsp_box == 'signal decomposition':
    if decomp_box == 'principal component analysis':
        pca_n = st.sidebar.number_input(label='number of components',min_value=2,step=1,value=3, key="pca_n_components")
        
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'signal decomposition':
    if decomp_box == 'principal component analysis':
        if st.sidebar.button("Decompose signal with PCA"):
            filtered_signal = st.session_state["active_dataset"].copy()
            emd_count = 0
            for row in filtered_signal:
                print(row.shape," is the shape of a row")
                filtered_signal = bss_pca(X=row, n_components=pca_n)
                
                #filtered_signal.T

                if emd_count == 0:
                    pca_array = filtered_row
                    plot_decomposed_components(signal=row, components=filtered_row, title='PCA', stream=True)
                else:
                    np.vstack((pca_array,filtered_row))
                emd_count += 1

            active_save_verification(pca_array)


# ICA

if dsp_box == 'signal decomposition':
    if decomp_box == 'independent component analysis':
        ica_n = st.sidebar.number_input(label='number of components',min_value=2,step=1,value=3, key="ica_n_components")
        
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'signal decomposition':
    if decomp_box == 'independent component analysis':
        if st.sidebar.button("Decompose signal with ICA"):
            filtered_signal = st.session_state["active_dataset"].copy()
            emd_count = 0
            for row in filtered_signal:

                filtered_row = bss_ica(X=row, n_components=ica_n)

                if emd_count == 0:
                    ica_array = filtered_row
                    plot_decomposed_components(signal=row, components=filtered_row, title='ICA', stream=True)
                else:
                    np.vstack((ica_array,filtered_row))
                emd_count += 1

            active_save_verification(ica_array)


##########################################################
# TRANSFORMS
##########################################################

transform_tuple = ('fast fourier', 'short time fourier', 'singular spectrum')

if dsp_box == 'transforms':
    transform_box = st.sidebar.selectbox('**Select a Transform.**', transform_tuple)

# FFT

if dsp_box == 'transforms':
    if transform_box == 'fast fourier':
        fft_f = st.sidebar.number_input(label='sampling rate',min_value=1,step=1,value=100,key='fft_sampling_rate')
        
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'transforms':
    if transform_box == 'fast fourier':
        if st.sidebar.button("Perform FFT"):
            filtered_signal = st.session_state["active_dataset"].copy()
            fft_count = 0
            for row in filtered_signal:

                freq, mag = my_fft(signal=row, fs=fft_f)
                index = np.argsort(freq)
                freq = freq[index]
                mag = mag[index]

                fft_sig = np.vstack((freq,mag))

                signal_recovered = my_ifft(mag)

                if fft_count == 0:
                    fft_array = fft_sig
                    fig, axis = plt.subplots(3,1,figsize=(8,9))
                    axis[0].set_title("FFT result")
                    axis[0].set_xlabel("Frequency")
                    axis[0].set_ylabel("Magnitude")
                    axis[0].plot(freq, np.abs(mag))

                    axis[1].set_title("Original signal")
                    axis[1].set_xlabel("Time")
                    axis[1].set_ylabel("Amplitude")
                    axis[1].plot(row)

                    axis[2].set_title("Signal recovered from Spectrum")
                    axis[2].plot(signal_recovered.real)
                    axis[2].set_xlabel("Time")
                    axis[2].set_ylabel("Amplitude")
                    plt.tight_layout()
                    st.pyplot(plt)                   
                else:
                    np.vstack((fft_array,fft_sig))
                fft_count += 1

            active_save_verification(fft_array)


# STFT

if dsp_box == 'transforms':
    if transform_box == 'short time fourier':
        stft_f = st.sidebar.number_input(label='sampling rate',min_value=1,step=1,value=100,key='stft_sampling_rate')
        stft_n = st.sidebar.number_input(label='nperseg',min_value=1,step=1,value=256,key='stft_nperseg')
        
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'transforms':
    if transform_box == 'short time fourier':
        if st.sidebar.button("Perform STFT"):
            filtered_signal = st.session_state["active_dataset"].copy()
            fft_count = 0
            for row in filtered_signal:
                if fft_count == 0:
                    show = True
                else:
                    show = False
                
                f, t, Z = my_stft(signal=row, fs=stft_f, nperseg=stft_n, plot=show, stream=show)                

                if fft_count == 0:
                    stft_array = Z                  
                else:
                    np.vstack((stft_array,Z))
                fft_count += 1

            active_save_verification(stft_array)

# Singular Spectrum Transform (SST)

if dsp_box == 'transforms':
    if transform_box == 'singular spectrum':
        sst_w = st.sidebar.number_input(label='window length',min_value=1,step=1,value=100,key='sst_window_length')
        
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'transforms':
    if transform_box == 'singular spectrum':
        if st.sidebar.button("Perform SST"):
            filtered_signal = st.session_state["active_dataset"].copy()
            fft_count = 0
            #sst_scores = []
            for row in filtered_signal:

                scores = sst(signal=row, win_length=sst_w)

                if fft_count == 0:
                    sst_array = scores
                    plt.figure(figsize=(8,6))
                    plt.subplot(2,1,1)
                    plt.title("Original signal")
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.plot(row)

                    plt.subplot(2,1,2)
                    plt.title("Result of SST")
                    plt.plot(scores)
                    plt.xlabel("Time")
                    plt.ylabel("Score of changing")
                    plt.tight_layout()
                    st.pyplot(plt)
                else:
                    np.vstack((sst_array,scores))

                fft_count += 1

            active_save_verification(sst_array)

##########################################################
# WAVELET ANALYSIS
##########################################################
            
wavelet_tuple = ('chirplet', 'wavelet', 'synchro-squeezing', 'wigner ville distribution')

transform_tuple = ('fast fourier', 'short time fourier', 'singular spectrum')

wavelet_list = pywt.wavelist(kind='continuous')

if dsp_box == 'wavelet analysis':
    wave_box = st.sidebar.selectbox('**Select Wavelet Analysis.**', wavelet_tuple)

# Chirplet     
        
if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'wavelet analysis':
    if wave_box == 'chirplet':
        if st.sidebar.button("Apply Chirplet Transform"):
            filtered_signal = st.session_state["active_dataset"].copy()
            fft_count = 0
            for row in filtered_signal:
                if fft_count == 0:
                    show = True
                else:
                    show = False

                ct_matrix = chirplet_transform(signal=row, show=show, stream=show)

                if fft_count == 0:
                    chirp_array = ct_matrix          
                else:
                    np.vstack((chirp_array,ct_matrix))
                fft_count += 1

            active_save_verification(chirp_array)

# Wavelet Transform  
             
if dsp_box == 'wavelet analysis':
    if wave_box == 'wavelet':
        wav_s = st.sidebar.text_input(label='scales: comma seperated list of floats',value='2,4,8,16', key="wavelet_scales")
        wav_w = st.sidebar.selectbox("Choose a Wavelet:", wavelet_list)
        wav_f = st.sidebar.number_input(label='sampling frequency',min_value=2,step=1,value=100,key='wavelet_sampling_frequency')

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'wavelet analysis':
    if wave_box == 'wavelet':
        if st.sidebar.button("Apply Wavelet Transform"):
            scales_list = parse_text_entry(wav_s,'float')
            filtered_signal = st.session_state["active_dataset"].copy()
            fft_count = 0
            for row in filtered_signal:
                if fft_count == 0:
                    show = True
                else:
                    show = False

                coefficients, frequencies = my_cwt(signal=row, scales=scales_list, wavelet=wav_w, fs=wav_f, show=show, stream=show)

                if fft_count == 0:
                    wave_array = coefficients         
                else:
                    np.vstack((wave_array,coefficients))
                fft_count += 1

            active_save_verification(wave_array)

# SynchroSqueezing Transform (SST)

if dsp_box == 'wavelet analysis':
    if wave_box == 'synchro-squeezing':
        syn_f = st.sidebar.number_input(label='sampling frequency',min_value=2,step=1,value=100,key='synchro_squeeze_sampling_frequency')
        syn_w = st.sidebar.text_input(label='window type',value='ham', key="synchro_squeeze_window_type")
        syn_n = st.sidebar.number_input(label='nperseg',min_value=1,step=1,value=256,key='synchro_squeeze_nperseg')
 

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'wavelet analysis':
    if wave_box == 'synchro-squeezing':
        if st.sidebar.button("Apply SynchroSqueeze"):
            filtered_signal = st.session_state["active_dataset"].copy()
            fft_count = 0
            for row in filtered_signal:
                if fft_count == 0:
                    show = True
                else:
                    show = False

                Tx, Sx, ssq_freqs, Sfs= sst_stft(signal=row, window=syn_w, nperseg=syn_n, fs=syn_f, show=show, stream=show)

                if fft_count == 0:
                    sq_array = Tx         
                else:
                    np.vstack((sq_array,Tx))
                fft_count += 1

            active_save_verification(sq_array)


# Wigner Ville Distribution (WVD)

if not str(st.session_state["active_dataset"]) == "" and dsp_box == 'wavelet analysis':
    if wave_box == 'wigner ville distribution':
        if st.sidebar.button("Apply WVD"):
            filtered_signal = st.session_state["active_dataset"].copy()
            fft_count = 0
            for row in filtered_signal:
                if fft_count == 0:
                    show = True
                else:
                    show = False

                matrix, t, f = my_wvd(signal=row, show=show, stream=show)

                if fft_count == 0:
                    wvd_array = matrix         
                else:
                    np.vstack((wvd_array,matrix))
                fft_count += 1

            active_save_verification(wvd_array)            