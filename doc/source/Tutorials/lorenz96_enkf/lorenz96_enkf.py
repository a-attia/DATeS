
# 
# DATeS: Data Assimilation Testing Suite.
#

# An example of a test driver:
# ==============================
# The driver file should be located in the root directory of DATeS.
# So; if it is not there, make a copy in the DATES_ROOT_PATH!
#
#     1- Create an object of the Lorenz96 model;
#     2- Create an of the DEnKF;
#     3- Run DEnKF using Lorenz96 over a defined observation, and assimilation timespan.
# 
#     To Run the driver:
#        On the linux terminal execute the following command:
#            $ python lorenze96_enkf_test_driver.py
#


import sys
import numpy as np
import os


# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.

import dates_setup
dates_setup.initialize_dates(random_seed=0)

import dates_utility as utility  # import DATeS utility module(s)



# Create a model object
from lorenz_models import Lorenz96  as Lorenz
model = Lorenz(model_configs={'create_background_errors_correlations':True})


# Create DA pieces; this includes:
# ---------------------------------
#   i-   forecast trajectory/state
#   ii-  initial ensemble, 
#   iii- filter/smoother/hybrid object.


# create observations' and assimilation checkpoints:
obs_checkpoints = np.arange(0, 5.001, 0.1)
da_checkpoints = obs_checkpoints


# create initial ensemble:

ensemble_size = 25
initial_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size)


# import, configure, and create filter object:
from EnKF import EnKF as StochasticEnKF
enkf_filter_configs = dict(model=model,
                           analysis_ensemble=initial_ensemble,
                           forecast_ensemble=None,
                           ensemble_size=ensemble_size,
                           inflation_factor=1.05,
                           obs_covariance_scaling_factor=1.0,
                           obs_adaptive_prescreening_factor=None,
                           localize_covariances=True,
                           localization_method='covariance_filtering',
                           localization_radius=4,
                           localization_function='gauss',
                           )

filter_obj = StochasticEnKF(filter_configs=enkf_filter_configs, 
                            output_configs=dict(file_output_moment_only=False)
                            )

# Create sequential DA process:
# -----------------------------

# + processing object; here this is a filtering_process object:
from filtering_process import FilteringProcess
experiment = FilteringProcess(assimilation_configs=dict(model=model,
                                                        filter=filter_obj,
                                                        obs_checkpoints=obs_checkpoints,
                                                        da_checkpoints=da_checkpoints,
                                                        forecast_first=True,
                                                        ref_initial_condition=model._reference_initial_condition.copy(),
                                                        ref_initial_time=0,
                                                        random_seed=0
                                                        ),
                              output_configs = dict(scr_output=True,
                                                    scr_output_iter=1,
                                                    file_output=True,
                                                    file_output_iter=1)
                              )


# run the sequential filtering over the timespan created by da_checkpoints
experiment.recursive_assimilation_process()


# retrieve/read  results:
out_dir_tree_structure_file = 'Results/Filtering_Results/output_dir_structure.txt'
filtering_results = read_filter_output(out_dir_tree_structure_file)

reference_states = filtering_results[2]
forecast_ensembles = filtering_results[3]
analysis_ensembles = filtering_results[5]
forecast_times = filtering_results[8]
analysis_times = filtering_results[9]
forecast_rmse = filtering_results[11]
analysis_rmse = filtering_results[12]
filter_configs = filtering_results[13]


# 1- plot RMSE:
import matplotlib.pyplot as plt

fig_rmse = plt.figure(facecolor='white')
plt.semilogy(forecast_times, forecast_rmse, label='Forecast')
plt.semilogy(analysis_times, analysis_rmse, label=filter_configs['filter_name'])

plt.xlabel('Time')
plt.ylabel('log-RMSE')
xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), 10)]
plt.xticks(xlables, 10*np.arange(len(xlables)))

plt.legend(loc='upper right')
plt.show()


# 2- plot rank histogrmas:

_ = utility.rank_hist(forecast_ensembles, 
                      reference_states,
                      draw_hist=True, 
                      hist_type='relfreq',
                      hist_title='forecast rank histogram'
                     )
_ = utility.rank_hist(analysis_ensembles,
                      reference_states,
                      draw_hist=True,  
                      hist_type='relfreq', 
                      hist_title='analysis rank histogram'
                      )
plt.show()


# Clean executables and temporary modules:
utility.clean_executable_files()


