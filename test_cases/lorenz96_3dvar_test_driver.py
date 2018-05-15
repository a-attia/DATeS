#!/usr/bin/env python

#
# 3D-Var with Lorenz-96
#


import sys
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
model_configs = dict(create_background_errors_correlations=True)
model = Lorenz(model_configs=model_configs)
#
#

# create observations' and assimilation checkpoints:
obs_checkpoints = np.arange(0, 1.001, 0.1)
da_checkpoints = obs_checkpoints

# Create DA pieces: 
# ---------------------
# this includes:
#   i-   forecast trajectory/state
#   ii- filter/smoother/hybrid object.
#
# create initial guess or create an ensemble of size 1...
forecast_state = model.create_initial_ensemble(ensemble_size=1)[0]
ref_IC = model._reference_initial_condition.copy()

observation = model.evaluate_theoretical_observation(ref_IC)
observation = observation.add(model.observation_error_model.generate_noise_vec())

#
# create variational object
from threeD_var import TDVAR as FILTER
filter_configs = dict(model=model,
                      reference_state=ref_IC,
                      observation=observation,
                      forecast_state=forecast_state,
                      random_seed=_random_seed
                      )
filter_obj = FILTER(filter_configs=filter_configs, output_configs={'verbose':False})

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
                                                        random_seed=0
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


