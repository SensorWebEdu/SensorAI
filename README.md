# SensorAI
Sensor data science and AI tutorial


## Citation 

Bib: 

@misc{sensor2024ai,
author = "Song, WenZhan and Coshatt, Stephen and Zhang, Yida and Chen, Jiayu",
year = "2024",
title = "Sensor Data Science and AI Tutorial",
url = "https://github.com/wsonguga/SensorAI",
institution = "SensorWeb Research Laboratory, University of Georgia" }

MLA:

Song, WenZhan, Stephen Coshatt, Yida Zhang, and Jiayu Chen. "Sensor Data Science and AI Tutorial: https://github.com/wsonguga/SensorAI", SensorWeb Research Laboratory, University of Georgia, 2024.

## Outline

1. Advanced signal processing: tutorial_dsp.ipynb
2. Classification: tutorial_classification.ipynb
3. Regression: tutorial_regression.ipynb
4. Clustering: tutorial_clustering.ipynb
5. Deep Learning: https://github.com/timeseriesAI/tsai or https://github.com/thuml/Time-Series-Library

## Package Installation
Create and activate a Python 3.11 environment, then install the dependencies:

```bash
conda create -n sensorai python=3.11 pip -y
conda activate sensorai
bash install.sh
python -m notebook
```

If `sensorai` already exists, skip the environment creation command. Select that
environment as the notebook kernel and restart the kernel after updating packages.
The tutorials use the current kernel's packages; do not add another virtual
environment's `site-packages` directory to `sys.path`.

`pandas` is pinned for compatibility with `statsmodels==0.14.1`, and `pyarrow` is
pinned for compatibility with NumPy 1.26. `streamlit` is required by the shared
helper modules, and `pyts` is used in the unsupervised tutorial.
The installer adds `fastsst` separately with `--no-deps` because its
upstream dependency list uses the deprecated `sklearn` package name; the actual
`scikit-learn` package is already included in `requirements.txt`.

## General Workflow of Sensor AI

        +----------------------------------+
        |          Raw Signal/Data         |
        +----------------------------------+
                     |
                     v
        +----------------------------------+
        |       Signal Preprocessing       |
        | (signal denoising, recovery, etc)|
        +----------------------------------+
                     |
                     v
        +----------------------------------+
        |       Feature Extraction         |
        +----------------------------------+
                     |
                     v
        +----------------------------------+
        |   Feature Selection/Synthesis    |
        +----------------------------------+
                     |
                     v
        +----------------------------------+
        |      Machine/Deep Learning       |
        |            Algorithm             |
        +----------------------------------+
                     |
                     v
        +----------------------------------+
        |         Model Evaluation         |
        +----------------------------------+
                     |
                     v
        +----------------------------------+
        |        Deployment/Use            |
        +----------------------------------+
