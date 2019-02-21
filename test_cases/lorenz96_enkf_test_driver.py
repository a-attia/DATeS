#!/usr/bin/env python

"""
Apply Ensemble Kalman Filter to Lorenz96 model.
"""

import sys
import numpy as np


# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
import dates_setup
dates_setup.initialize_dates()
#
import dates_utility as utility  # import DATeS utility module(s)


# Create a model object
# ---------------------
from lorenz_models import Lorenz96  as Lorenz
model = Lorenz(model_configs={'create_background_errors_correlations':True,
                     'observation_noise_level':0.05,
                     'observation_vector_size':20,
                     'background_noise_level':0.08})
#
# create observations' and assimilation checkpoints:
obs_checkpoints = np.arange(0, 10.001, 0.1)
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
ensemble_size = 5
initial_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size)

# create filter object
from EnKF import DEnKF as EnKF_Filter
enkf_filter_configs = dict(model=model,
                           analysis_ensemble=initial_ensemble,
                           forecast_ensemble=None,
                           ensemble_size=ensemble_size,
                           inflation_factor=1.0,
                           forecast_inflation_factor=1.09,
                           obs_covariance_scaling_factor=1.0,
                           obs_adaptive_prescreening_factor=None,
                           localize_covariances=True,
                           localization_method='covariance_filtering',
                           localization_radius=4,
                           localization_function='gauss'
                           )

filter_obj = EnKF_Filter(filter_configs=enkf_filter_configs,
                               output_configs=dict(file_output_moment_only=False)
                               )

# Create sequential DA
# processing object:
# ---------------------
# Here this is a filtering_process object;
from filtering_process import FilteringProcess
experiment = FilteringProcess(assimilation_configs=dict(filter=filter_obj,
                                                        obs_checkpoints=obs_checkpoints,
                                                        da_checkpoints=da_checkpoints,
                                                        forecast_first=True,
                                                        ref_initial_condition=model._reference_initial_condition.copy(),
                                                        ref_initial_time=0,  # should be obtained from the model along with the ref_IC
                                                        random_seed=2345
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
# utility.clean_executable_files()
#
