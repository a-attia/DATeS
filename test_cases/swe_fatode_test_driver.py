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
"""

import numpy as np
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
from cartesian_swe_model import CartesianSWE
from fatode_erk_fwd_wrapper import FatODE_ERK_FWD as ERK
# from explicit_runge_kutta import ExplicitRungeKutta as ERK
import dates_utility as utility
#
#
# ==================================================================================================================== #
#        Interactive session with user to automate the desired task:                                                   #
#        This should include all tasks ranging from a simple run with existing model, and assimilation scheme,         #
#        to adding new models and/or schemes.                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #
#



# ==================================================================================================================== #
#                               Tests for the basic (Lorenz3) model
# ==================================================================================================================== #
#
# create/construct the model
swe = CartesianSWE(64, 20)
gridsize = 64 + 2
checkpoints = np.arange(0, 30, 0.05)

# -------------------------------------------------------------------------------------------------------------------- #
# test time integration scheme, and plot the trajectory
# -------------------------------------------------------------------------------------------------------------------- #
integrator_options = {'model': swe,
                      'initial_state': swe._reference_initial_condition,
                      'checkpoints': checkpoints,
                      'step_size': 0.001}
integrator = ERK(integrator_options)
reference_trajectory = integrator.integrate()

tmp = np.empty((len(reference_trajectory), 3*gridsize*gridsize))
for time_ind in xrange(len(reference_trajectory)):
    # print(' t= %5.3f : State: %s ' %(checkpoints[time_ind], trajectory[time_ind]) )
    tmp[time_ind, :] = reference_trajectory[time_ind].get_numpy_array()

print len(reference_trajectory)

mark="o"
col="r"
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive
interactive(True)
import time
plt.ion()
fig = plt.figure()
ax = Axes3D(fig)
gridshape = (gridsize, gridsize)
x = np.reshape(np.array([(i % gridsize) for i in xrange(0,gridsize*gridsize)]), gridshape)
y = np.reshape(np.array([(i / gridsize) for i in xrange(0,gridsize*gridsize)]), gridshape)

for i in xrange(len(reference_trajectory)):
    z = np.reshape(tmp[i, 0:(gridsize*gridsize)], gridshape)
    ax.clear()
    ax.set_zlim3d([0.0, 2.0])
    frame = ax.plot_surface(x, y, z)
    plt.draw()
    time.sleep(0.01)


# -------------------------------------------------------------------------------------------------------------------- #

