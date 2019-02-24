#!/usr/bin/env python

#
# 4D-Var with Lorenz-96
#

import sys
sys.path.insert(1, "../")

import numpy as np

_random_seed = 2345

# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
import dates_setup
dates_setup.initialize_dates(random_seed=_random_seed)
#
import dates_utility as utility  # import DATeS utility module(s)


# Create a model object
# ---------------------
from lorenz_models import Lorenz96  as Lorenz
model_configs = dict(adjoint_integrator_scheme='ERK',
                     create_background_errors_correlations=True,
                     )
model = Lorenz(model_configs=model_configs)
#
# create observations' and assimilation checkpoints:
obs_checkpoints = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.75])
da_checkpoints = np.array([0.0, 0.3])
analysis_trajectory_timespan = np.arange(0, 0.75001, 0.1)
#
#

# Create DA pieces:
# ---------------------
# this includes:
#   i-   forecast trajectory/state
#   ii-  initial ensemble,
#   iii- filter/smoother/hybrid object.
#
# create initial guess or create an ensemble of size 1...
forecast_state = model.create_initial_ensemble(ensemble_size=1)[0]
ref_IC = model._reference_initial_condition.copy()

#
# create smoother object
from fourD_var import FDVAR as SMOOTHER
smoother_configs = dict(model=model,
                        reference_time=analysis_trajectory_timespan[0],
                        reference_state=ref_IC,
                        observations_list=None,
                        forecast_time=analysis_trajectory_timespan[0],
                        forecast_state=forecast_state,
                        analysis_time=analysis_trajectory_timespan[0],
                        analysis_timespan=analysis_trajectory_timespan,
                        random_seed=_random_seed
                        )
smoother_obj = SMOOTHER(smoother_configs=smoother_configs, output_configs={'verbose':False})

#
# Create sequential DA
# processing object:
# ---------------------
# Here this is a filtering_process object;
from smoothing_process import SmoothingProcess
experiment_configs = dict(smoother=smoother_obj,
                          experiment_timespan=[analysis_trajectory_timespan[0], analysis_trajectory_timespan[-1]],
                          obs_checkpoints=obs_checkpoints,
                          observations_list=None,
                          da_checkpoints=da_checkpoints,
                          initial_forecast=forecast_state,
                          ref_initial_condition=ref_IC,
                          ref_initial_time=analysis_trajectory_timespan[0],
                          analysis_timespan=analysis_trajectory_timespan,
                          # random_seed=_random_seed
                          )
experiment = SmoothingProcess(assimilation_configs=experiment_configs, output_configs={'file_output':True,'verbose':False})

#
# print("Terminated per request...")
# sys.exit()

# run the sequential filtering over the timespan created by da_checkpoints
experiment.recursive_assimilation_process()

#
# Clean executables and temporary modules
# ---------------------
utility.clean_executable_files("src")
#
