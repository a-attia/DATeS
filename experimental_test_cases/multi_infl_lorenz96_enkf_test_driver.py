#!/usr/bin/env python

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
*   Example Benchmarking:                                                                  *
*   Apply EnKF to Lorenz96 model with multiple inflation parameter values.                 *
*                                                                                          *
*   ....................................................................................   *
********************************************************************************************
* To Run the driver:                                                                       *
* --------------------                                                                     *
*        On the linux terminal execute the following command:                              *
*           > python multi_infl_lorenz96_enkf_test_driver.py                               *
*                                                                                          *
********************************************************************************************
"""

import sys
sys.path.insert(1, "../")

import os
import numpy as np

import re

# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
import dates_setup
dates_setup.initialize_dates()
#
import dates_utility as utility  # import DATeS utility module(s)

from lorenz_models import Lorenz96  as LorenzModel

# create filter object
from EnKF import EnKF as EnKF_Filter
from EnKF import DEnKF as DEnKF_Filter

from filtering_process import FilteringProcess


def create_model(model_name):
    """
    """
    # Create Model Object:
    if re.match(r'lorenz\Z', model_name, re.IGNORECASE):
        model_configs={'create_background_errors_correlations':True,
                             'observation_noise_level':0.05,
                             'observation_vector_size':20,
                             'background_noise_level':0.08}
        Model = LorenzModel

    elif re.match(r'\AQ(_|-)*G\Z', model_name, re.IGNORECASE):
        MREFIN = 7
        model_configs = dict(MREFIN=MREFIN,
                             model_name='QG-1.5',
                             observation_operator_type='linear',
                             observation_vector_size=300,
                             observation_error_variances=4.0,
                             observation_errors_covariance_method='diagonal',
                             background_error_variances=5.0,
                             background_errors_covariance_method='diagonal',  # 'full' if full localization is applied
                             background_errors_covariance_localization_method='Gaspari-cohn',
                             background_errors_covariance_localization_radius=8
                             )
        Model = QGModel
    else:
        raise ValueError("Model supported so far are either Lorenz96 or QG-1.5")

    # Initialize model:
    model_obj = Model(model_configs=model_configs)
    #
    return model_obj

def create_initial_ensemble(model_obj, ensemble_size):
    """
    """
    # create initial ensemble...
    try:
        initial_ensemble = model_obj.create_initial_ensemble(ensemble_size=ensemble_size, ensemble_from_repo=True)
    except(TypeError):
        initial_ensemble = model_obj.create_initial_ensemble(ensemble_size=ensemble_size)
    return initial_ensemble

def inspect_output_path(filter_name, inflation_factor, model_name, ensemble_size, localization_function):
    """
    """
    if isinstance(inflation_factor, int):
        infl = "%d" % inflation_factor
    else:
        infl = "%3.2f" % inflation_factor
        infl = infl.replace('.', '_')

    new_path = "Results/%s_Results_%s_Ensemble_%d_%s_InflationFactor_%s" %(filter_name, model_name, ensemble_size,localization_function, infl)

    return new_path

def create_filter_configs(model_name):
    """
    """
    # Observations and Assimilation Configurations
    if re.match(r'lorenz\Z', model_name, re.IGNORECASE):
        forecast_inflation_factor = 1.09  # used as initial value
        localization_radius =  4
        localization_function =  'gauss'
        obs_checkpoints = np.arange(0, 30.001, 0.1)
        #
    elif re.match(r'\AQ(_|-)*G\Z', model_name, re.IGNORECASE):
        forecast_inflation_factor = 1.09  # used as initial value
        inflation_factor =  1.0
        localization_radius =  8
        localization_function =  'Gaspari-Cohn'
        #
        # Observations' and assimilation checkpoints:
        if MREFIN == 5:
            obs_checkpoints = np.arange(0, 1250.0001, 12.5) * 4 ### 100 assymilation cycles
        elif MREFIN == 6:
            obs_checkpoints = np.arange(0, 1250.0001, 12.5) * 2
        elif MREFIN == 7:
            obs_checkpoints = np.arange(0, 12500.0001, 12.5)
        else:
            print("MREFIN has to be one of the values 5, 6, or 7")
            raise ValueError()
    else:
        raise ValueError("Model supported so far are either Lorenz96 or QG-1.5")
    da_checkpoints = obs_checkpoints
    return obs_checkpoints, da_checkpoints, localization_radius, localization_function

def create_filter(filter_name, model_obj, initial_ensemble, forecast_inflation_factor, localization_radius, localization_function,verbose=False):
    """
    """
    ensemble_size = len(initial_ensemble)
    filter_configs = dict(model=model_obj,
                          analysis_ensemble=initial_ensemble,
                          forecast_ensemble=None,
                          ensemble_size=ensemble_size,
                          forecast_inflation_factor=forecast_inflation_factor,
                          inflation_factor=1.0,  # turn-off analysis inflation
                          localize_covariances=True,
                          localization_radius=localization_radius,
                          localization_function=localization_function)

    if re.match(r'\AEnKF\Z', filter_name, re.IGNORECASE):
        Filter = EnKF_Filter
    elif re.match(r'\ADEnKF\Z', filter_name, re.IGNORECASE):
        Filter = DEnKF_Filter
    else:
        print("Unsupported Filter %s" %filter_name)
        raise ValueError

    filter_obj = Filter(filter_configs=filter_configs,
                        output_configs=dict(file_output_moment_only=False, verbose=verbose))
    return filter_obj

def create_experiment(filter_obj, obs_checkpoints, da_checkpoints, ref_IC):
    """
    """
    experiment = FilteringProcess(assimilation_configs=dict(filter=filter_obj,
                                                            obs_checkpoints=obs_checkpoints,
                                                            da_checkpoints=da_checkpoints,
                                                            forecast_first=True,
                                                            ref_initial_condition=ref_IC,
                                                            ref_initial_time=0,
                                                            random_seed=2345),
                                  output_configs = dict(scr_output=True,
                                                        scr_output_iter=1,
                                                        file_output=True,
                                                        file_output_iter=1))
    return experiment

def move_results_dir(new_path, standard_path=None):
    """
    """
    if standard_path is None:
        standard_path = "Results/Filtering_Results"
    print("Moving Results Directory:")
    print("From: %s" % standard_path)
    print("To: %s" % new_path)
    os.rename(standard_path, new_path)
    print("DONE... Proceeding ...")
    #


if __name__ == '__main__':

    filter_name = 'DEnKF'
    model_name = 'lorenz'

    localization_function_pool = ['Gauss', 'Gaspari-Cohn']
    ensemble_size_pool = np.arange(5, 61, 5)
    inflation_factors_pool = np.arange(0.95, 1.25, 0.01)
    model_obj = create_model(model_name)
    ref_IC = model_obj._reference_initial_condition.copy()

    num_experiments = len(ensemble_size_pool) * len(localization_function_pool) * len(inflation_factors_pool)
    exp_no = 0
    #
    for ensemble_size in ensemble_size_pool:
        initial_ensemble = create_initial_ensemble(model_obj, ensemble_size)

        for localization_function in localization_function_pool:

            for forecast_inflation_factor in inflation_factors_pool:

                # Print some header
                sep = "\n%s(DEnKF Experiment with various inflation factors)%s\n" % ('='*25, '='*25)
                exp_no += 1
                print("Experiment Number [%d] out of [%d]" % (exp_no, num_experiments))
                print(sep)
                print("Ensemble Size: %d" % ensemble_size)
                print("inflation Factor: %f" % forecast_inflation_factor)
                print("Localization Function: %s" % localization_function)
                print("\n%s\n" % ('='*len(sep)))

                # prepare output path:
                new_path = inspect_output_path(filter_name, forecast_inflation_factor, model_name, ensemble_size, localization_function)
                if os.path.isdir(new_path):
                    print("Fond Results Directory: [%s] Skipping..." % new_path)
                    continue

                # Ensemble and filter object:
                a_ensemble = [e.copy() for e in initial_ensemble]
                obs_checkpoints, da_checkpoints, localization_radius, _ = create_filter_configs(model_name)
                filter_obj = create_filter(filter_name, model_obj, a_ensemble, forecast_inflation_factor, localization_radius, localization_function)

                # Create and run the sequential filtering over the timespan created by da_checkpoints
                experiment = create_experiment(filter_obj, obs_checkpoints, da_checkpoints, ref_IC.copy())
                try:
                    experiment.recursive_assimilation_process()
                except:
                    print("\n\n\t >>>>>>>>>>>  Experiment with current settings has FAILED  <<<<<<<<<<<\nProceeding...\n\n")
                    continue

                # Move the output results to a marked path:
                move_results_dir(new_path)


                # Cleanup; just in-case!
                del a_ensemble, filter_obj, experiment
