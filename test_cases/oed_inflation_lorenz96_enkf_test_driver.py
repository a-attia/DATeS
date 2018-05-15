#!/usr/bin/env python3

"""
============================================================================================
=                                                                                          =
=   DATeS: Data Assimilation Testing Suite.                                                =
=                                                                                          =
=   Copyright (C) 2016  A. Sandu, A. Attia, P. Tranquilli, S.R. Glandon,                   =
=   M. Narayanamurthi, A. Sarshar, Computational Science Laboratory (CSL), Virginia Tech.  =
=                                                                                          =
=   Website: http://csl.cs.vt.edu/                                                         =
=   Phone: 540-231-6186                                                                    =
=                                                                                          =
=   This program is subject to the terms of the Virginia Tech Non-Commercial/Commercial    =
=   License. Using the software constitutes an implicit agreement with the terms of the    =
=   license. You should have received a copy of the Virginia Tech Non-Commercial License   =
=   with this program; if not, please contact the computational Science Laboratory to      =
=   obtain it.                                                                             =
=                                                                                          =
============================================================================================
********************************************************************************************
*   ....................................................................................   *
*    An example of a test driver.                                                          *
*    The driver file should be located in the root directory of DATeS.                     *
*                                                                                          *
*      1- Create an object of the Lorenz96 model.                                          *
*      2- Create an of the standard stochastic EnKF                                        *
*      3- Run EnKF using Lorenz96 over a defined observation, and assimilation timespan    *
*                                                                                          *
*   ....................................................................................   *
********************************************************************************************
* To Run the driver:                                                                       *
* --------------------                                                                     *
*        On the linux terminal execute the following command:                              *
*           > python oed_inflation_lorenze96_enkf_test_driver.py                           *
*                                                                                          *
********************************************************************************************
"""

import sys
import numpy as np
import scipy.optimize as optimize

# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
import dates_setup
dates_setup.initialize_dates(random_seed=0)
#
import dates_utility as utility  # import DATeS utility module(s)


# Create a callback function for the assimilation filtering class;
""" < Initial implementation is only for scalar inflation factor (both objective and gradient) > """
def a_opt_objective(inf_factor, filter_obj):
    """
    The A-Optimality objective function at the given inflation factor
    
    Inputs:
        inf_factor: inflation factor; sclar or one dimensional vector (numpy or model state)
        filter_obj: 
    Returns:
        a_opt_val: value of the A-Optimality objective
    """
    # Retrieve a pointer to the model object:
    model = filter_obj.model
    state_size = model.state_size()
    #
    if np.isscalar(inf_factor):
        local_inf_factor = np.ones(state_size) * inf_factor
    elif isinstance(inf_factor, np.ndarray):
        #
        if not(inf_factor.size==1 or inf_factor.size==state_size):
            print(" The inflation factor 'inf_factor' has to be either a scalar or \
                    a one dimensional array of size equal to state size!")
            raise ValueError
        elif(inf_factor.size==1):
            local_inf_factor = inf_factor.copy()
            for i in xrange(local_inf_factor.ndim):
                local_inf_factor = local_inf_factor[0]
            local_inf_factor = np.ones(state_size) * local_inf_factor
        elif(inf_factor.size==state_size):
            local_inf_factor = inf_factor[:]
        else:
            pass
            #
    else:
        try:
            local_inf_factor = inf_factor[:]
        except:
            print("Failed to read the inflation factor!")
            print(" The inflation factor 'inf_factor' has to be either a scalar or \
                    a one dimensional array of size equal to state size!")
            raise ValueError
    #
    analysis_ensemble = filter_obj.filter_configs['analysis_ensemble']
    a_opt_val = utility.covariance_trace(analysis_ensemble, model=filter_obj.model)
    #
    print "Objective Value:", a_opt_val
    return a_opt_val
    #
good_objective = lambda fac: a_opt_objective(fac, filter_obj)


def a_opt_gradient(inf_factor, filter_obj):
    """
    The A-Optimality objective function gradient
    
    Inputs:
        inf_factor: inflation factor; sclar or one dimensional vector (numpy or model state)
        filter_obj: 
    Returns:
        a_opt_gradient: gradient of the a_optimality function with respect to the given inflation factor
    """
    # Retrieve a pointer to the model object:
    model = filter_obj.model
    state_size = model.state_size()
    #
    if np.isscalar(inf_factor):
        local_inf_factor = np.ones(state_size) * inf_factor
    elif isinstance(inf_factor, np.ndarray):
        #
        if not(inf_factor.size==1 or inf_factor.size==state_size):
            print(" The inflation factor 'inf_factor' has to be either a scalar or \
                    a one dimensional array of size equal to state size!")
            raise ValueError
        elif(inf_factor.size==1):
            local_inf_factor = inf_factor.copy()
            for i in xrange(local_inf_factor.ndim):
                local_inf_factor = local_inf_factor[0]
            local_inf_factor = np.ones(state_size) * local_inf_factor
        elif(inf_factor.size==state_size):
            local_inf_factor = inf_factor[:]
        else:
            pass
            #
    else:
        try:
            local_inf_factor = inf_factor[:]
        except:
            print("Failed to read the inflation factor!")
            print(" The inflation factor 'inf_factor' has to be either a scalar or \
                    a one dimensional array of size equal to state size!")
            raise ValueError
    
    # Get the forecast ensemble (to use background error covariance matrix B)
    forecast_ensemble = filter_obj.filter_configs['forecast_ensemble']
    
    # construct the matrix (inefficient) I + \widetilde{B}_k H_k^T R^{-1} H_k, get it's inverse then square it
    state_size = model.state_size()
    observation_size = model.observation_vector_size()
    
    tilde_B = np.empty((state_size, state_size))
    HT_Rinv = np.empty((state_size, observation_size))
    #
    e_vec = model.observation_vector()
    for e_ind in xrange(observation_size):
        e_vec[:] = 0.0
        e_vec[e_ind] = 1.0
        R_inv_e = model.observation_error_model.invR.vector_product(e_vec)
        HT_Rinv[:, e_ind] = R_inv_e[:]
    in_state = model.state_vector()
        
    for e_ind in xrange(state_size):
        in_state[:] = HT_Rinv[e_ind, :]
        out_vec = model.observation_operator_Jacobian_T_prod_vec(None, in_state)
        out_vec = utility.ensemble_covariance_dot_state(forecast_ensemble, out_vec, model=model)
        out_vec = out_vec.get_numpy_array()
        out_vec *= local_inf_factor
        try:
            out_vec[e_ind] += 1.0
        except:
            out_vec[e_ind] = out_vec[e_ind] + 1.0
        tilde_B[e_ind, :] = out_vec[:]
    
    tilde_B = np.linalg.inv(tilde_B)
    tilde_B = tilde_B.dot(tilde_B)
    
    # TODO: Update this for vector calculations (after deriving the right formula!
    trace = 0.0
    in_state = model.state_vector()
    for state_ind in xrange(state_size):
        in_state[:] = tilde_B[:, state_ind]
        res = utility.ensemble_covariance_dot_state(forecast_ensemble, in_state, model=model)
        trace += res[state_ind]
    #
    a_opt_gradient = np.ones(state_size) * trace
    print "a_opt_gradient", a_opt_gradient
    return a_opt_gradient
    #
goeod_obj_grad = lambda fac: a_opt_gradient(fac, filter_obj)    
    
    
# This function updates the inflation parameter of the filter after each assimilation cycle:
def inflation_callback(filter_obj, inf_factor=None, criterion='a', verbose=True):
    """
    Inputs:
        inf_factor: current (initial) inflation factor
        criterion: OED-Optimality criterion (case insensitive):
            - 'A': A-Optimiality criterion; Minimize the trace of the posterior ensemble covariance matrix
            - 'D': D-Optimiality criterion; Minimize the log-det of the posterior ensemble covariance matrix
            
    Returns:
        out_inf_factor: Updated (optimal) inflation factor
        
    """
    #
    if inf_factor is None:
        inf_factor = filter_obj.filter_configs['inflation_factor']
    else:
        inf_factor = inf_factor
    #
    # Retrieve a pointer to the model object:
    model = filter_obj.model
    state_size = model.state_size()
    #
    if np.isscalar(inf_factor):
        local_inf_factor = np.ones(state_size) * inf_factor
    elif isinstance(inf_factor, np.ndarray):
        #
        if not(inf_factor.size==1 or inf_factor.size==state_size):
            print(" The inflation factor 'inf_factor' has to be either a scalar or \
                    a one dimensional array of size equal to state size!")
            raise ValueError
        elif(inf_factor.size==1):
            local_inf_factor = inf_factor.copy()
            for i in xrange(local_inf_factor.ndim):
                local_inf_factor = local_inf_factor[0]
            local_inf_factor = np.ones(state_size) * local_inf_factor
        elif(inf_factor.size==state_size):
            local_inf_factor = inf_factor[:]
        else:
            pass
            #
    else:
        try:
            local_inf_factor = inf_factor[:]
        except:
            print("Failed to read the inflation factor!")
            print "inf_factor:", type(inf_factor), inf_factor 
            print("The inflation factor 'inf_factor' has to be either a scalar or \
                    a one dimensional array of size equal to state size!")
            raise ValueError
            
            
            
    
    
    if criterion.lower() == 'a':
        # OED A-Optimality:
        # Create the optimizer, and set it's configs:
        optimizer_settings = None
        
        # Optimize for the inflation factor, and compare to the best value accross time...
        
        bnds = [1, 5]
        #
        if False:
            x0 = local_inf_factor
            bnds = [bnds] *state_size
            res = optimize.minimize(good_objective, x0,
                                                    method='L-BFGS-B',
                                                    jac=goeod_obj_grad, 
                                                    bounds=bnds)
        #
        if True:
            bnds = [bnds]*state_size
            x0 = local_inf_factor
            print x0, type(x0)
            res = optimize.fmin_l_bfgs_b(good_objective,
                                         x0,
                                         fprime=goeod_obj_grad,
                                         bounds=list(bnds),
                                         m=10,
                                         factr=10000000.0,
                                         pgtol=1e-05,
                                         epsilon=1e-08,
                                         iprint=-1,
                                         maxfun=15000,
                                         maxiter=15000, 
                                         disp=verbose)
                                     #
        # Retrieve and output the optimal design
        print res, type(res)
        out_inf_factor = res[0]
        
        
    elif criterion.lower() == 'd':
        # OED D-Optimality:
        print("D optimality yet to be implemented!")
        raise NotImplementedError
    else:
        print("Undefined optimality criterion!")
        raise ValueError
    
    if verbose:
        analysis_time = filter_obj.filter_configs['analysis_time']
        print("Inflation Parameter:")
        print("Current:")
        print(local_inf_factor)
        print("Updated:")
        print(out_inf_factor)

    # Update the filter infaltion factor and return
    filter_obj.filter_configs['inflation_factor'] = out_inf_factor[0]
    return out_inf_factor
    #



# Create a model object
# ---------------------
from lorenz_models import Lorenz96  as Lorenz
model = Lorenz(model_configs={'create_background_errors_correlations':True})
#
# create observations' and assimilation checkpoints:
obs_checkpoints = np.arange(0, 10.001, 0.01)
da_checkpoints = obs_checkpoints
#


# Create DA pieces: 
# ---------------------
# this includes:
#   i-   forecast trajectory/state
#   ii-  initial ensemble, 
#   iii- filter/smoother/hybrid object.
#
# create initial ensemble...
ensemble_size = 25
initial_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size)

# create filter object
from EnKF import DEnKF as EnKF
enkf_filter_configs = dict(model=model,
                           analysis_ensemble=initial_ensemble,
                           forecast_ensemble=None,
                           ensemble_size=ensemble_size,
                           inflation_factor=1.09,
                           obs_covariance_scaling_factor=1.0,
                           obs_adaptive_prescreening_factor=None,
                           localize_covariances=True,
                           localization_method='covariance_filtering',
                           localization_radius=4,
                           localization_function='gauss',
                           )

filter_obj = EnKF(filter_configs=enkf_filter_configs, 
                  output_configs=dict(file_output_moment_only=False)
                  )


if False:
    cov_trace = filter_obj.covariance_trace(initial_ensemble, model)
    print("trace of covariance matrix of the initial ensemble = %f" % cov_trace)

    in_state = model._reference_initial_condition.copy()
    out_state = filter_obj.ensemble_covariance_dot_state(initial_ensemble, in_state)
    print "in_state", in_state
    print "out_state", out_state


#
# IDEA: You need to add Callback function for the filtering_process class. This function is to be called after each cycle. 
# This should be useful for stuff such as tuning filter parameters (after the assimilation cycle)...
#

# OED-based Adaptive Inflation settings:


# Create sequential DA 
# processing object: 
# ---------------------
# Here this is a filtering_process object;
from filtering_process import FilteringProcess
experiment = FilteringProcess(assimilation_configs=dict(model=model,
                                                        filter=filter_obj,
                                                        obs_checkpoints=obs_checkpoints,
                                                        da_checkpoints=da_checkpoints,
                                                        forecast_first=True,
                                                        ref_initial_condition=model._reference_initial_condition.copy(),
                                                        ref_initial_time=0,  # should be obtained from the model along with the ref_IC
                                                        random_seed=0,
                                                        callback=inflation_callback,
                                                        callback_args=filter_obj
                                                        ),
                              output_configs = dict(scr_output=True,
                                                    scr_output_iter=1,
                                                    file_output=True,
                                                    file_output_iter=1)
                              )
# run the sequential filtering over the timespan created by da_checkpoints
experiment.recursive_assimilation_process()

#
# Clean executables and temporary modules
# ---------------------
utility.clean_executable_files()
#

