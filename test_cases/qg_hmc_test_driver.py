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
*                          DATeS: Data Assimilation Testing Suite                          *
********************************************************************************************
*   ....................................................................................   *
*    An example of a test driver.                                                          *
*    The driver file should be located in the root directory of DATeS.                     *
*                                                                                          *
*      1- Create an object of the QG-1.5 (Pavel Sakov) model.                              *
*      2- Create an object of the MC-ClHMC sampling filter (Attia et. al 2016)             *
*      3- Run MC-ClHMC using QG-1.5 over a defined observation, and assimilation timespan  *
*                                                                                          *
*   ....................................................................................   *
********************************************************************************************
* To Run the driver:                                                                       *
* --------------------                                                                     *
*        On the linux terminal execute the following command:                              *
*           > python qg1p5_mcclhmc_test_driver.py                                             *
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
dates_setup.initialize_dates(random_seed=1)
#
import dates_utility as utility  # import DATeS utility module(s)


# Create a model object
# ---------------------
from qg_1p5_model import QG1p5
model = QG1p5(model_configs = dict(MREFIN=5, model_name='QG-1.5',
                                   model_grid_type='cartesian',
                                   observation_operator_type='wind-magnitude',
                                   observation_vector_size=100,
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
obs_checkpoints = np.arange(0, 5000.001, 50)/10.0
da_checkpoints = obs_checkpoints
#
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
initial_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size, ensemble_from_repo=True)

# create filter object
from hmc_filter import HMCFilter

hmc_filter_configs = dict(model=model,
                          analysis_ensemble=initial_ensemble,
                          chain_parameters=dict(Symplectic_integrator='3-stage',
                                                Hamiltonian_num_steps=15,
                                                Hamiltonian_step_size=0.015,
                                                Mass_matrix_type='prior_variances',
                                                Burn_in_num_steps=100,
                                                Mixing_steps=15,
                                                ),
                          prior_distribution='gmm',
                          gmm_prior_settings=dict(clustering_model='gmm',
                                                  cov_type='diag',
                                                  localize_covariances=False,
                                                  inf_criteria='aic',
                                                  number_of_components=None,
                                                  min_number_of_components=None,
                                                  max_number_of_components=None,
                                                  min_number_of_points_per_component=6,
                                                  invert_uncertainty_param=False,
                                                  approximate_covariances_from_comp=False,
                                                  use_oringinal_hmc_for_one_comp=True,
                                                  initialize_chain_strategy='forecast_mean'  #  highest_wight or 'forecast_mean'
                                                  ),
                          ensemble_size=ensemble_size,
                          localize_covariances=True,
                          forecast_inflation_factor=1.0,
                          analysis_inflation_factor=1.0,
                          hybrid_background_coeff=0.5  # this is multiplied by the modeled background error covariance matrix
                          )
hmc_output_configs = dict(scr_output=True,
                          file_output=False, # this will be overridden by the process anyways.
                          file_output_moment_only=False,
                          file_output_moment_name='mean',
                          file_output_separate_files=False,
                          verbose=True
                          )

filter_obj = HMCFilter(filter_configs=hmc_filter_configs, output_configs = hmc_output_configs)

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

