import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
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
from utils import parse_text_entry
from regression import *

def add_to_regress_queue(name,model):
    st.session_state['regress_name_queue'].append(name)
    st.session_state['regress_model_queue'].append(model)
    print("Queue: ", st.session_state['regress_name_queue'])

def execute_regress_gridsearch():
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    gridsearch_regressor(names=st.session_state['regress_name_queue'],pipes=st.session_state['regress_model_queue'],
                              X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,
                              scoring="neg_mean_squared_error",save_best=True,log=True,stream=True)

st.sidebar.title("Regression")

regress_model_tuple = ('linear', 'gamma', 'poisson', 'tweedie',
               'lars', 'lasso', 'lasso-lars', 'lasso-lars ic',
               'ridge', 'bayesian ridge', 'elastic net', 'quantile',
               'support vector', 'linear svr', 'nu svr', 'time series svr')

regress_model_box = st.sidebar.selectbox('**Select the model(s) you like to use.**', regress_model_tuple)

########################################################
# Model Variable Entries
########################################################

# Linear Regression
if regress_model_box == 'linear':
    lin_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="linear_fit_intercept")

if regress_model_box == 'linear':
    if st.sidebar.button("Add Linear to Regression Queue"):    
        lin_fit_list = parse_text_entry(lin_fi,'bool')

        name = "Linear"
        linear = pipeBuild_LinearRegression(fit_intercept=lin_fit_list)
        add_to_regress_queue(name,linear)


# Gamma Regression
if regress_model_box == 'gamma':
    gam_a = st.sidebar.text_input("Alpha: positive integers", value='1', key="gamma_alpha")
    gam_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="gamma_fit_intercept")
    gam_s = st.sidebar.text_input("Solver: lbfgs, newton-cholesky", value='lbfgs', key="gamma_solver")
    gam_mi = st.sidebar.text_input("Max Iterations: integers", value='100', key="gamma_max_iter")
    gam_t = st.sidebar.text_input("Tolerance: floats only", value='0.0001', key="gamma_fit_tol")

if regress_model_box == 'gamma':
    if st.sidebar.button("Add Gamma to Regression Queue"):    
        gam_alpha_list = parse_text_entry(gam_a,'int')
        gam_fit_list = parse_text_entry(gam_fi,'bool')
        gam_solver_list = parse_text_entry(gam_s,'string')
        gam_max_iter_list = parse_text_entry(gam_mi,'int')
        gam_tol_list = parse_text_entry(gam_t,'float')

        name = "Gamma"
        gamma = pipeBuild_GammaRegressor(alpha=gam_alpha_list, fit_intercept=gam_fit_list, 
                                         solver=gam_solver_list, max_iter=gam_max_iter_list, tol=gam_tol_list)
        add_to_regress_queue(name,gamma)

# Poisson Regression
if regress_model_box == 'poisson':
    poi_a = st.sidebar.text_input("Alpha: positive integers", value='1', key="poisson_alpha")
    poi_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="poisson_fit_intercept")
    poi_s = st.sidebar.text_input("Solver: lbfgs, newton-cholesky", value='lbfgs', key="poisson_solver")
    poi_mi = st.sidebar.text_input("Max Iterations: integers", value='100', key="poisson_max_iter")
    poi_t = st.sidebar.text_input("Tolerance: floats only", value='0.0001', key="poisson_fit_tol")

if regress_model_box == 'poisson':
    if st.sidebar.button("Add Poisson to Regression Queue"):    
        poi_alpha_list = parse_text_entry(poi_a,'int')
        poi_fit_list = parse_text_entry(poi_fi,'bool')
        poi_solver_list = parse_text_entry(poi_s,'string')
        poi_max_iter_list = parse_text_entry(poi_mi,'int')
        poi_tol_list = parse_text_entry(poi_t,'float')

        name = "Poisson"
        poisson = pipeBuild_PoissonRegressor(alpha=poi_alpha_list, fit_intercept=poi_fit_list, 
                                         solver=poi_solver_list, max_iter=poi_max_iter_list, tol=poi_tol_list)
        add_to_regress_queue(name,poisson)

# Tweedie Regression
if regress_model_box == 'tweedie':
    twe_p = st.sidebar.text_input("Power: 0, 1, 2, 3, or float between 1 and 2", value='0', key="tweedie_power")
    twe_a = st.sidebar.text_input("Alpha: positive integers", value='1', key="tweedie_alpha")
    twe_fi = st.sidebar.text_input("Link Function: auto, identity, log", value='True', key="tweedie_fit_intercept")
    twe_l = st.sidebar.text_input("Link: True, False", value='auto', key="tweedie_link")
    twe_s = st.sidebar.text_input("Solver: lbfgs, newton-cholesky", value='lbfgs', key="tweedie_solver")
    twe_mi = st.sidebar.text_input("Max Iterations: integers", value='100', key="tweedie_max_iter")
    twe_t = st.sidebar.text_input("Tolerance: floats only", value='0.0001', key="tweedie_fit_tol")

if regress_model_box == 'tweedie':
    if st.sidebar.button("Add Tweedie to Regression Queue"):
        twe_power_list = parse_text_entry(twe_p,'float')
        twe_alpha_list = parse_text_entry(twe_a,'int')
        twe_fit_list = parse_text_entry(twe_fi,'bool')
        twe_link_list = parse_text_entry(twe_l,'string')
        twe_solver_list = parse_text_entry(twe_s,'string')
        twe_max_iter_list = parse_text_entry(twe_mi,'int')
        twe_tol_list = parse_text_entry(twe_t,'float')

        name = "Tweedie"
        tweedie = pipeBuild_TweedieRegressor(alpha=twe_alpha_list, fit_intercept=twe_fit_list, 
                                         solver=twe_solver_list, max_iter=twe_max_iter_list, tol=twe_tol_list,
                                         power=twe_power_list, link=twe_link_list)
        add_to_regress_queue(name,tweedie)


# LARS Regression
if regress_model_box == 'lars':
    lar_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="lars_fit_intercept")
    lar_mi = st.sidebar.text_input("Max Iterations: integers", value='100', key="lars_max_iter")
    lar_cv = st.sidebar.text_input("Cross Validation: None or integers", value='5', key="lars_cv")

if regress_model_box == 'lars':
    if st.sidebar.button("Add LARS to Regression Queue"):    
        lar_fit_list = parse_text_entry(lar_fi,'bool')
        lar_max_iter_list = parse_text_entry(lar_mi,'int')
        lar_cv_list = parse_text_entry(lar_cv,'int')

        name = "LARS"
        lars = pipeBuild_LarsCV(fit_intercept=lar_fit_list, max_iter=lar_max_iter_list, cv=lar_cv_list)
        add_to_regress_queue(name,lars)

# LASSO Regression
if regress_model_box == 'lasso':
    las_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="lasso_fit_intercept")
    las_mi = st.sidebar.text_input("Max Iterations: integers", value='100', key="lasso_max_iter")
    las_cv = st.sidebar.text_input("Cross Validation: None or integers", value='5', key="lasso_cv")

if regress_model_box == 'lasso':
    if st.sidebar.button("Add LASSO to Regression Queue"):    
        las_fit_list = parse_text_entry(las_fi,'bool')
        las_max_iter_list = parse_text_entry(las_mi,'int')
        las_cv_list = parse_text_entry(las_cv,'int')

        name = "LASSO"
        lasso = pipeBuild_LassoCV(fit_intercept=las_fit_list, max_iter=las_max_iter_list, cv=las_cv_list)
        add_to_regress_queue(name,lasso)


# LASSO-LARS Regression
if regress_model_box == 'lasso-lars':
    ll_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="lasso-lars_fit_intercept")
    ll_mi = st.sidebar.text_input("Max Iterations: integers", value='100', key="lasso-lars_max_iter")
    ll_cv = st.sidebar.text_input("Cross Validation: None or integers", value='5', key="lasso-lars_cv")

if regress_model_box == 'lasso-lars':
    if st.sidebar.button("Add LASSO-LARS to Regression Queue"):    
        ll_fit_list = parse_text_entry(ll_fi,'bool')
        ll_max_iter_list = parse_text_entry(ll_mi,'int')
        ll_cv_list = parse_text_entry(ll_cv,'int')

        name = "LASSO-LARS"
        lasso_lars = pipeBuild_LassoLarsCV(fit_intercept=ll_fit_list, max_iter=ll_max_iter_list, cv=ll_cv_list)
        add_to_regress_queue(name,lasso_lars)


# LASSO-LARS with Information Criteria Regression
if regress_model_box == 'lasso-lars ic':
    llic_c = st.sidebar.text_input("Criterion: aic, bic", value='aic', key="lasso-lars-ic_criterion")
    llic_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="lasso-lars-ic_fit_intercept")
    llic_mi = st.sidebar.text_input("Max Iterations: integers", value='100', key="lasso-lars-ic_max_iter")

if regress_model_box == 'lasso-lars ic':
    if st.sidebar.button("Add LASSO-LARS IC to Regression Queue"):
        llic_crit_list = parse_text_entry(llic_c,'string')  
        llic_fit_list = parse_text_entry(llic_fi,'bool')
        llic_max_iter_list = parse_text_entry(llic_mi,'int')
        
        name = "LASSO-LARS IC"
        lasso_lars_ic = pipeBuild_LassoLarsIC(criterion= llic_crit_list,fit_intercept= llic_fit_list, max_iter= llic_max_iter_list)
        add_to_regress_queue(name,lasso_lars_ic)


# Ridge Regression
if regress_model_box == 'ridge':
    rid_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="ridge_fit_intercept")
    rid_cv = st.sidebar.text_input("Cross Validation: None or integers", value='None', key="ridge_cv")

if regress_model_box == 'ridge':
    if st.sidebar.button("Add Ridge to Regression Queue"):    
        rid_fit_list = parse_text_entry(rid_fi,'bool')
        rid_cv_list = parse_text_entry(rid_cv,'int')

        name = "Ridge"
        ridge = pipeBuild_RidgeCV(cv=rid_cv_list,fit_intercept=rid_fit_list)
        add_to_regress_queue(name,ridge)


# Bayesian Ridge Regression
if regress_model_box == 'bayesian ridge':
    byr_a = st.sidebar.text_input("Alpha Value: None or integerss", value='None', key="bay_rid_alpha")
    byr_l = st.sidebar.text_input("Initial Lambda Value: None or integers", value='None', key="bay_rid_lambda")
    byr_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="bay_rid_fit_intercept")
    
if regress_model_box == 'bayesian ridge':
    if st.sidebar.button("Add Bayesian Ridge to Regression Queue"):
        byr_alpha_list = parse_text_entry(byr_a,'int')
        byr_lambda_list = parse_text_entry(byr_l,'int')   
        byr_fit_list = parse_text_entry(byr_fi,'bool')
        
        name = "Bayesian Ridge"
        bayes_ridge = pipeBuild_BayesianRidge(alpha_init=byr_alpha_list,fit_intercept=byr_fit_list, lambda_init=byr_lambda_list)
        add_to_regress_queue(name,bayes_ridge)


# Elastic Net Regression
if regress_model_box == 'elastic net':
    enet_c = st.sidebar.text_input("Selection: cyclic, random", value='cyclic', key="elastic_net_cyclic")
    enet_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="elastic_net_intercept")

if regress_model_box == 'elastic net':
    if st.sidebar.button("Add Elastic Net to Regression Queue"):
        enet_selection_list = parse_text_entry(enet_c,'string')
        enet_fit_list = parse_text_entry(enet_fi,'bool')        

        name = "Elastic Net"
        elastic_net = pipeBuild_ElasticNetCV(selection=enet_selection_list,fit_intercept=enet_fit_list)
        add_to_regress_queue(name,elastic_net)


# Quantile Regression
if regress_model_box == 'quantile':
    qnt_a = st.sidebar.text_input("Alpha: floats", value='0.5', key="quantile_alpha")
    qnt_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="quantile_intercept")

if regress_model_box == 'quantile':
    if st.sidebar.button("Add Quantile to Regression Queue"):
        qnt_alpha_list = parse_text_entry(qnt_a,'float')
        qnt_fit_list = parse_text_entry(qnt_fi,'bool')        

        name = "Quantile"
        quantile = pipeBuild_QuantileRegressor(alpha=qnt_alpha_list,fit_intercept=qnt_fit_list)
        add_to_regress_queue(name,quantile)


# Support Vector Regressor
if regress_model_box == 'support vector':
    svr_c = st.sidebar.text_input("Regularization Parameter: floats only", value='1.0', key="svr_c")
    svr_kernel = st.sidebar.text_input("Kernel: linear, poly, rbf, sigmoid", value='rbf', key="svr_kernel")
    svr_degree = st.sidebar.text_input("Degree (for poly kernel): integers only", value='3', key="svr_degree")
    svr_gamma = st.sidebar.text_input("Gamma: scale, auto", value='scale', key="svr_gamma")
    svr_tol = st.sidebar.text_input("Tolerance: floats only", value='0.001', key="svr_tol")

if regress_model_box == 'support vector':
    if st.sidebar.button("Add Support Vector to Regressor Queue"):
        svr_c_list = parse_text_entry(svr_c,'float')    
        svr_kernel_list = parse_text_entry(svr_kernel,'string')
        svr_degree_list = parse_text_entry(svr_degree,'int')
        svr_gamma_list = parse_text_entry(svr_gamma,'string')
        svr_tol_list = parse_text_entry(svr_tol,'float')

        name = "Support Vector Regressor"
        svr = pipeBuild_SVR(C=svr_c_list, kernel=svr_kernel_list, degree=svr_degree_list,
                                  gamma=svr_gamma_list, tol=svr_tol_list)
        add_to_regress_queue(name,svr)


# Linear Support Vector Regressor
if regress_model_box == 'linear svr':
    lsvr_c = st.sidebar.text_input("Regularization Parameter: floats only", value='1.0', key="lsvr_c")
    lsvr_loss = st.sidebar.text_input("Loss: epsilon_insensitive, squared_epsilon_insensitive", value='epsilon_insensitive', key="lsvr_loss")
    lsvr_fi = st.sidebar.text_input("Fit Intercept: True, False", value='True', key="lsvr_fit_inetercept")
    lsvr_tol = st.sidebar.text_input("Tolerance: floats only", value='0.001', key="lsvr_tol")

if regress_model_box == 'linear svr':
    if st.sidebar.button("Add Linear SVR to Regressor Queue"):
        lsvr_c_list = parse_text_entry(lsvr_c,'float')    
        lsvr_loss_list = parse_text_entry(lsvr_loss,'string')
        lsvr_fit_list = parse_text_entry(lsvr_fi,'bool')
        lsvr_tol_list = parse_text_entry(lsvr_tol,'float')

        name = "Linear SVR"
        linear_svr = pipeBuild_LinearSVR(C=lsvr_c_list,loss=lsvr_loss_list,fit_intercept=lsvr_fit_list,tol=lsvr_tol_list)
        add_to_regress_queue(name,linear_svr)


# Nu Support Vector Regressor
if regress_model_box == 'nu svr':
    nsvr_nu = st.sidebar.text_input("Regularization Parameter: floats only", value='1.0', key="nsvr_nu")
    nsvr_kernel = st.sidebar.text_input("Kernel: linear, poly, rbf, sigmoid", value='rbf', key="nsvr_kernel")
    nsvr_degree = st.sidebar.text_input("Degree (for poly kernel): integers only", value='3', key="nsvr_degree")
    nsvr_gamma = st.sidebar.text_input("Gamma: scale, auto", value='scale', key="nsvr_gamma")
    nsvr_tol = st.sidebar.text_input("Tolerance: floats only", value='0.001', key="nsvr_tol")

if regress_model_box == 'nu svr':
    if st.sidebar.button("Add Nu SVR to Regressor Queue"):
        nsvr_nu_list = parse_text_entry(nsvr_nu,'float')    
        nsvr_kernel_list = parse_text_entry(nsvr_kernel,'string')
        nsvr_degree_list = parse_text_entry(nsvr_degree,'int')
        nsvr_gamma_list = parse_text_entry(nsvr_gamma,'string')
        nsvr_tol_list = parse_text_entry(nsvr_tol,'float')

        name = "Nu SVR"
        nu_svr = pipeBuild_NuSVR(nu=nsvr_nu_list, kernel=nsvr_kernel_list, 
                                 degree=nsvr_degree_list, gamma=nsvr_gamma_list, 
                                 tol=nsvr_tol_list)
        add_to_regress_queue(name,nu_svr)


# Time SeriesSupport Vector Regressor
if regress_model_box == 'time series svr':
    tssvr_c = st.sidebar.text_input("Regularization Parameter: floats only", value='1.0', key="ts_svr_c")
    tssvr_kernel = st.sidebar.text_input("Kernel: gak, linear, poly, rbf, sigmoid", value='gak', key="ts_svr_kernel")
    tssvr_degree = st.sidebar.text_input("Degree (for poly kernel): integers only", value='3', key="ts_svr_degree")
    tssvr_gamma = st.sidebar.text_input("Gamma: auto, gak, rbf, poly, sigmoid", value='auto', key="ts_svr_gamma")
    tssvr_tol = st.sidebar.text_input("Tolerance: floats only", value='0.001', key="ts_svr_tol")

if regress_model_box == 'time series svr':
    if st.sidebar.button("Add Time Series SVR to Regressor Queue"):
        tssvr_c_list = parse_text_entry(tssvr_c,'float')    
        tssvr_kernel_list = parse_text_entry(tssvr_kernel,'string')
        tssvr_degree_list = parse_text_entry(tssvr_degree,'int')
        tssvr_gamma_list = parse_text_entry(tssvr_gamma,'string')
        tssvr_tol_list = parse_text_entry(tssvr_tol,'float')

        name = "Time Series SVR"
        ts_svr = pipeBuild_TimeSeriesSVR(C=tssvr_c_list, kernel=tssvr_kernel_list, degree=tssvr_degree_list,
                                         gamma=tssvr_gamma_list, tol=tssvr_tol_list)
        add_to_regress_queue(name,ts_svr)

########################################################
# End Model Variable Entries
########################################################

if st.sidebar.button("Show Regression Model Queue"):
    print(st.session_state['regress_name_queue'])
    st.write(str(st.session_state['regress_name_queue']))

if st.sidebar.button("Clear Regression Model Queue"):
    st.session_state['regress_name_queue'] = []
    st.session_state['regress_model_queue'] = []
    st.session_state["regress log loader"] = None
    st.session_state['show_regress_log'] = False

if st.sidebar.button("Run Regression Grid Search"):
    print("Regression gridsearch started")
    st.session_state['show_regress_log'] = True
    execute_regress_gridsearch()

# if st.session_state['show_regress_log'] == True:
#     log_file = st.file_uploader("Choose a txt file", type="txt",key="regress log loader")
#     if log_file is not None:
#         root, extension = os.path.splitext(log_file.name)
#         if extension.lower() == ".txt":
#             bytes_data = log_file.getvalue()  
#             string_data = bytes_data.decode('utf-8') 
#             # Display the content
#             st.write("File Content:")
#             st.code(string_data, language="text") # Use st.code for displaying raw text            
#         else:
#             st.write("You have selected an incorrect file type")