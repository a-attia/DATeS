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
*           > python lorenze96_enkf_test_driver.py                                         *
*                                                                                          *
********************************************************************************************
"""

import sys, os
import numpy as np


# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
import dates_setup
dates_setup.initialize_dates(random_seed=2345)
#
import dates_utility as utility  # import DATeS utility module(s)


# Create a model object
# ---------------------
from lorenz96_model import Lorenz96  as Lorenz
model = Lorenz()
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
ensemble_size = 25
initial_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size)

# create filter object
from EnKF import EnKF as StochasticEnKF
enkf_filter_configs = dict(model=model,
                           analysis_ensemble=initial_ensemble,
                           forecast_ensemble=None,
                           ensemble_size=ensemble_size
                           )

filter_obj = StochasticEnKF(filter_configs=enkf_filter_configs, 
                            output_configs=dict(file_output_moment_only=False)
                            )

# Create sequential DA 
# processing object: 
# ---------------------
# Here this is a filtering_process object;
from filtering_process import FilteringProcess
results_dir = 'Results/Filter_Results'
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
                                                    file_output_dir=results_dir,
                                                    file_output_iter=1)
                              )
# run the sequential filtering over the timespan created by da_checkpoints
for i in xrange(3):
    print("\n%s\n%s\n%s" % ('-'*50, 'Experiment [%d]' % (i+1), '-'*50))
    if i == 0:
        backup_existing = True
    else:
        backup_existing = False
    experiment.set_filtering_output_dir(os.path.join(results_dir, str(i+1)), 
                                       rel_to_root_dir=True, 
                                       backup_existing=backup_existing
                                       )
    experiment.recursive_assimilation_process()

#
# Clean executables and temporary modules
# ---------------------
utility.clean_executable_files()
#

