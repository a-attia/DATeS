
# 
# DATeS: Data Assimilation Testing Suite.
#

# An example of a test driver:
# ==============================
# The driver file should be located in the root directory of DATeS.
# So; if it is not there, make a copy in the DATES_ROOT_PATH!
#
#     1- Create an object of the Quasi-Geostrophic 'qg1p5' model;
#     2- Create an of a Deterministic DEnKF;
#     3- Run EnKF using Lorenz96 over a defined observation, and assimilation timespan.
# 
#     To Run the driver:
#        On the linux terminal execute the following command:
#            $ python lorenze96_enkf_test_driver.py
#


import sys
import numpy as np


# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
import dates_setup
dates_setup.initialize_dates(random_seed=2345)

import dates_utility as utility  # import DATeS utility module(s)


# Create a model object
from qg_1p5_model import QG1p5
model = QG1p5(model_configs = dict(MREFIN=7, model_name='QG-1.5',
                                   model_grid_type='cartesian',
                                   observation_operator_type='linear',
                                   observation_vector_size=300,
                                   observation_error_variances=4.0,
                                   observation_errors_covariance_method='diagonal',
                                   background_error_variances=5.0,
                                   background_errors_covariance_method='diagonal',
                                   )
           )  # MREFIN = 5, 6, 7 for the three models QGt, QGs, QG respectively


# Create DA pieces; this includes:
# ---------------------------------
#   i-   forecast trajectory/state
#   ii-  initial ensemble, 
#   iii- filter/smoother/hybrid object.


# create observations' and assimilation checkpoints:
obs_checkpoints = np.arange(0, 1250.5001, 50)/4.0
da_checkpoints = obs_checkpoints


# create initial ensemble:
ensemble_size = 20
initial_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size, ensemble_from_repo=True)


# import, configure, and create filter object:
from EnKF import DEnKF as DeterministicEnKF
enkf_filter_configs = dict(model=model,
                           analysis_ensemble=initial_ensemble,
                           forecast_ensemble=None,
                           ensemble_size=ensemble_size,
                           inflation_factor=1.06,
                           obs_covariance_scaling_factor=1.0,
                           obs_adaptive_prescreening_factor=None,
                           localize_covariances=True,
                           localization_method='covariance_filtering',
                           localization_radius=12,
                           localization_function='Gaspari-Cohn',
                           )

filter_obj = DeterministicEnKF(filter_configs=enkf_filter_configs, 
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
                                                        ref_initial_time=0
                                                        ),
                              output_configs = dict(scr_output=True,
                                                    scr_output_iter=1,
                                                    file_output=True,
                                                    file_output_iter=1)
                              )


# run the sequential filtering over the timespan created by da_checkpoints
experiment.recursive_assimilation_process()


# retrieve/read  results:
from filtering_results_reader import read_filter_output
out_dir_tree_structure_file = 'Results/Filtering_Results/output_dir_structure.txt'
filtering_results = read_filter_output(out_dir_tree_structure_file)

num_cycles = filtering_results[1]
reference_states = filtering_results[2]
forecast_ensembles = filtering_results[3]
forecast_means = filtering_results[4]
analysis_ensembles = filtering_results[5]
analysis_means = filtering_results[6]
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
                      first_var=0, 
                      last_var=None, 
                      var_skp=1, 
                      draw_hist=True, 
                      hist_type='relfreq', 
                      first_time_ind=20, 
                      last_time_ind=None,
                      time_ind_skp=1, 
                      hist_title= 'Frequency Rank Histogram'
                     )
_ = utility.rank_hist(analysis_ensembles,
                      reference_states,
                      first_var=0, 
                      last_var=None, 
                      var_skp=15, 
                      draw_hist=True, 
                      hist_type='relfreq', 
                      first_time_ind=2, 
                      last_time_ind=None,
                      time_ind_skp=1, 
                      hist_title= 'Analysis Rank Histogram'
                      )
plt.show()


# Show trajectories:
from matplotlib import animation, rc
from IPython.display import HTML

state_size = model.state_size()
nx = int(np.sqrt(state_size))
ny = int(state_size/nx)

# Plot reference trajectory:
fig = plt.figure(facecolor='white')
fig.suptitle("Reference Trajectory")
ims = []
for ind in xrange(num_cycles):
    state = np.reshape(np.squeeze(reference_states[:, ind]), (nx, ny), order='F')
    imgplot = plt.imshow(state, animated=True)
    if ind == 0:
        plt.colorbar()
    else:
        plt.autoscale()
    ims.append([imgplot])

anim_ref = animation.ArtistAnimation(fig, ims, interval=75, blit=True, repeat_delay=1000)
HTML(anim_ref.to_html5_video())


# Plot Forecast trajectory:
fig = plt.figure(facecolor='white')
fig.suptitle("Forecast Trajectory")
ims = []
for ind in xrange(num_cycles):
    state = np.reshape(np.squeeze(forecast_means[:, ind]), (nx, ny), order='F')
    imgplot = plt.imshow(state, animated=True)
    if ind == 0:
        plt.colorbar()
    else:
        plt.autoscale()
    ims.append([imgplot])

anim_frcst = animation.ArtistAnimation(fig, ims, interval=75, blit=True, repeat_delay=1000)
HTML(anim_frcst.to_html5_video())


# Plot Analysis trajectory:
fig = plt.figure(facecolor='white')
fig.suptitle("Analysis Trajectory")
ims = []
for ind in xrange(num_cycles):
    state = np.reshape(np.squeeze(analysis_means[:, ind]), (nx, ny), order='F')
    imgplot = plt.imshow(state, animated=True)
    if ind == 0:
        plt.colorbar()
    else:
        plt.autoscale()
    ims.append([imgplot])

anim_anls = animation.ArtistAnimation(fig, ims, interval=75, blit=True, repeat_delay=1000)
HTML(anim_anls.to_html5_video())


# Plot Forecast Error trajectory:
fig = plt.figure(facecolor='white')
fig.suptitle("Forecast Errors")
ims = []
for ind in xrange(num_cycles):
    state = np.reshape(np.squeeze(reference_states[:, ind]-forecast_means[:, ind]), (nx, ny), order='F')
    imgplot = plt.imshow(state, animated=True)
    if ind == 0:
        plt.colorbar()
    else:
        plt.autoscale()
    ims.append([imgplot])

anim_frcst_err = animation.ArtistAnimation(fig, ims, interval=75, blit=True, repeat_delay=1000)
HTML(anim_frcst_err.to_html5_video())


# Plot Analysis Error trajectory:
fig = plt.figure(facecolor='white')
fig.suptitle("Analysis Errors")
ims = []
for ind in xrange(num_cycles):
    state = np.reshape(np.squeeze(reference_states[:, ind]-analysis_means[:, ind]), (nx, ny), order='F')
    imgplot = plt.imshow(state, animated=True)
    if ind == 0:
        plt.colorbar()
    else:
        plt.autoscale()
    ims.append([imgplot])

anim_anls_err = animation.ArtistAnimation(fig, ims, interval=75, blit=True, repeat_delay=1000)
HTML(anim_anls_err.to_html5_video())



# Clean executables and temporary modules:
utility.clean_executable_files()

