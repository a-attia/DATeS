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
*      1- Create an object of the QG-1.5 (Pavel Sakov) model.                              *
*      2- Create an of the standard stochastic EnKF                                        *
*      3- Run EnKF using QG-1.5 over a defined observation, and assimilation timespan    *
*                                                                                          *
*   ....................................................................................   *
********************************************************************************************
* To Run the driver:                                                                       *
* --------------------                                                                     *
*        On the linux terminal execute the following command:                              *
*           > python lorenze96_enkf_test_driver.py                                         *
*                                                                                          *
********************************************************************************************
"""

import sys
import numpy as np  # this is just to create
from matplotlib import pyplot as plt
import matplotlib.animation as animation
# import time

# ==================================================================================================================== #
#                                           Prepare DATeS for a run:                                                   #
# -------------------------------------------------------------------------------------------------------------------- #
#  The unified start point to the package. We want to make the whole process for users to be as easy as just
# running this file...
# This is intended to include many interactive processes to chose the suitable tasks...
# -------------------------------------------------------------------------------------------------------------------- #
# Define environment variables and update Python search path
# A necessary call that must be inserted in the beginning of any driver...
#
import dates_setup
dates_setup.initialize_dates()
# -------------------------------------------------------------------------------------------------------------------- #
#
from qg_1p5_model import QG1p5
import dates_utility as utility
#


# ==================================================================================================================== #
#                               Tests for the basic (QG 1.5) model
# ==================================================================================================================== #
#
model = QG1p5()
checkpoints = np.arange(0, 1250.0001, 12.5)

# -------------------------------------------------------------------------------------------------------------------- #
# test time integration scheme, and plot the trajectory
# -------------------------------------------------------------------------------------------------------------------- #
reference_trajectory = model.integrate_state(initial_state=model._reference_initial_condition.copy(),
                                             checkpoints=checkpoints)

state_size = model._state_size
nx = int(np.sqrt(state_size))


observation = model.evaluate_theoretical_observation(model._reference_initial_condition)
model.write_observation(observation=observation, directory='/home/attia/dates', file_name='QG_observation', append=False)

ref_traj_reshaped = np.empty((len(reference_trajectory), nx, nx))
for time_ind in xrange(len(reference_trajectory)):
    # print(' t= %5.3f : State: %s ' %(checkpoints[time_ind], reference_trajectory[time_ind][:5]))
    ref_traj_reshaped[time_ind, :, :] = np.reshape(reference_trajectory[time_ind].get_numpy_array(), (nx, nx), order='F')

# # generate Observations for each point of the reference trajectory:
# # TODO: To plot observations, we need to retrieve the observation grid as well...
# observations = []
# for state in reference_trajectory:
#     observations.append(model.evaluate_theoretical_observation(state))


# generate forecast information
# Forecast:
forecast_rmse = []
forecast_state = reference_trajectory[0].copy()
forecast_state = forecast_state.add(model.background_error_model.generate_noise_vec())
forecast_trajectory = model.integrate_state(initial_state=forecast_state, checkpoints=checkpoints)

fcst_traj_reshaped = np.empty((len(reference_trajectory), nx, nx))
for time_ind in xrange(len(forecast_trajectory)):
    fcst_traj_reshaped[time_ind, :, :] = np.reshape(forecast_trajectory[time_ind].get_numpy_array(), (nx, nx), order='F')
    forecast_rmse.append(utility.calculate_rmse(forecast_trajectory[time_ind], reference_trajectory[time_ind], model._state_size))
    
    

# Plot results
# TODO: plot the reference solution, the forecast solution and the observations on the same figure.

# TODO: Ahmed start plotting here. You can add these results to the number of mixture components plot
font_size = 24
line_width = 4
marker_size = 8
font = {'weight': 'bold', 'size': font_size}
plt.rc('font', **font)
#

fig = plt.figure(facecolor='white')
fig.suptitle("Reference Trajectory",  fontsize=font_size)
ref_ims = []
for i in xrange(len(reference_trajectory)):
    imgplot = plt.imshow(np.squeeze(ref_traj_reshaped[i, :, :]), animated=True)
    if i == 0:
        plt.colorbar()
    else:
        plt.autoscale()
    ref_ims.append([imgplot])

ref_ani = animation.ArtistAnimation(fig, ref_ims, interval=50, blit=True, repeat_delay=1000)

# Set up formatting for the movie files
# Writer = animation.writers['imagemagick_file']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('qg_traject.mp4', writer=writer)
plt.draw()


# plot trajectories
fig2 = plt.figure(facecolor='white')
fig2.suptitle("Forecast Trajectory",  fontsize=font_size)
fcst_ims = []
for i in xrange(len(forecast_trajectory)):
    imgplot = plt.imshow(np.squeeze(fcst_traj_reshaped[i, :, :]), animated=True)
    if i == 0:
        plt.colorbar()
    else:
        plt.autoscale()
    fcst_ims.append([imgplot])

fcst_ani = animation.ArtistAnimation(fig2, fcst_ims, interval=50, blit=True, repeat_delay=1000)

# Set up formatting for the movie files
# Writer = animation.writers['imagemagick_file']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('qg_traject.mp4', writer=writer)
plt.draw()


# Plote Forecast RMSE
log_scale = False
#
fig3 = plt.figure(facecolor='white')
#
if log_scale:
    plt.semilogy(checkpoints, forecast_rmse, 'r--', linewidth=line_width, label='Forecast')
else:
    plt.plot(checkpoints, forecast_rmse, 'r--', linewidth=line_width, label='Forecast')
#
# Set lables and title
plt.xlabel('Time',fontsize=font_size, fontweight='bold')
if log_scale:
    plt.ylabel('log-RMSE', fontsize=font_size, fontweight='bold')
else:
    plt.ylabel('RMSE', fontsize=font_size, fontweight='bold')
plt.title('RMSE results for the model: %s' % model.model_configs['model_name'])
#
xlables = [checkpoints[i] for i in np.arange(0, len(checkpoints), 10)]
plt.xticks(xlables, 10* np.arange(len(xlables)))
# plt.ylim(0, 8)
# show the legend, show the plot
plt.legend()

plt.draw()



plt.show()





# -------------------------------------------------------------------------------------------------------------------- #
# utility.clean_executable_files()

