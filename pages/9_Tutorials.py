import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import numpy as np
import os
import sys
import threading
import tempfile
from pathlib import Path

current_directory = Path.cwd()
tutorial_directory = current_directory / 'tutorials'

# Get the path of the current file
current_file_path = Path(__file__).resolve()
# Get the parent directory
parent_dir = current_file_path.parent
# Get the path to the other folder
other_folder_path = parent_dir.parent / "lib"
# Add the other folder to sys.path so Python can find the module
sys.path.append(str(other_folder_path))

st.sidebar.title("Tutorials")

tutorial = st.sidebar.selectbox('Select a Tutorial',
    ('','digital signal processing','classification','regression','unsupervised'))

if tutorial == 'digital signal processing':
    pdf_file = tutorial_directory / 'dsp_slides.pdf'
    tutorial_notebook = 'https://github.com/wsonguga/SensorAI/blob/main/tutorials/dsp_tutorial.ipynb'
    pdf_viewer(input=pdf_file,show_page_separator=True)
elif tutorial == 'classification':
    pdf_file = tutorial_directory / 'classification_slides.pdf'
    tutorial_notebook = 'https://github.com/wsonguga/SensorAI/blob/main/tutorials/classification_tutorial.ipynb'
    pdf_viewer(input=pdf_file,show_page_separator=True)
elif tutorial == 'regression':
    pdf_file = tutorial_directory / 'regression_slides.pdf'
    tutorial_notebook = 'https://github.com/wsonguga/SensorAI/blob/main/tutorials/regression_tutorial.ipynb'
    pdf_viewer(input=pdf_file,show_page_separator=True)
elif tutorial == 'unsupervised':
    pdf_file = tutorial_directory / 'unsupervised_slides.pdf'
    tutorial_notebook = 'https://github.com/wsonguga/SensorAI/blob/main/tutorials/unsupervised_tutorial.ipynb'
    pdf_viewer(input=pdf_file,show_page_separator=True)

if tutorial != '':
    st.sidebar.link_button("Open Tutorial Notebook a New Tab",tutorial_notebook)
