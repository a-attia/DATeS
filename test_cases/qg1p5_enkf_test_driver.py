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
*      3- Run EnKF using QG-1.5 over a defined observation, and assimilation timespan      *
*                                                                                          *
*   ....................................................................................   *
********************************************************************************************
* To Run the driver:                                                                       *
* --------------------                                                                     *
*        On the linux terminal execute the following command:                              *
*           > python qg1p5_enkf_test_driver.py                                             *
*                                                                                          *
********************************************************************************************
"""

import sys
import numpy as np  # this is just to create
from matplotlib import pyplot as plt
import matplotlib.animation as animation


# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
import dates_setup
dates_setup.initialize_dates(random_seed=2345)
#
import dates_utility as utility  # import DATeS utility module(s)


# Create a model object
# ---------------------
from qg_1p5_model import QG1p5
model = QG1p5(model_configs = dict(MREFIN=5, model_name='QG-1.5',
                                   model_grid_type='cartesian',
                                   observation_operator_type='linear',
                                   observation_vector_size=50,  
                                   observation_error_variances=4.0,
                                   observation_errors_covariance_method='diagonal',
                                   background_error_variances=5.0,
                                   background_errors_covariance_method='diagonal',  # 'full' if full localization is applied
                                   background_errors_covariance_localization_method='Gaspari_Cohn',
                                   background_errors_covariance_localization_radius=8
                                   )
           )  # MREFIN = 5, 6, 7 for the Sakov's three models QGt, QGs, QG respectively
#
# create observations' and assimilation checkpoints:
obs_checkpoints = np.arange(0, 1250.0001, 12.5)
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

