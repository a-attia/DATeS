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

import os
import sys
import numpy as np
import re

from coupledlorenz96_enkf_test_driver import get_model_info

try:
    Lorenz96
    Coupled_Lorenz
except(NameError):
    import dates_setup
    dates_setup.initialize_dates()
    from lorenz_models import Lorenz96, Coupled_Lorenz

import dates_utility as utility  # import DATeS utility module(s)

# Defaults:
# ---------
__BASE_RESULTS_DIR = 'CoupledLorenzResults'
__FILTER_CONFIGS_FILENAME = "oed_settings.dat"
__DEF_FILTER_CONFIGS = dict(ensemble_size=25,
                            #
                            adaptive_inflation=True,
                            inflation_factor=1.0,
                            forecast_inflation_factor=1.0,
                            inflation_design_penalty=0.05,
                            inflation_bounds=(0, 3.5),
                            #
                            adaptive_localization=False,
                            localization_function="Gaspari-Cohn",
                            localization_radius=0.5,
                            loc_direct_approach=3,  # 6 (See My Notes on OED_Localization) cases to apply localization with different radii
                            regularization_norm='l2',  # L1, L2 are supported
                            localization_space='B',  # B, R1, R2 --> B-localization, R-localization (BHT only), R-localization (both BHT, and HBHT)
                            moving_average_radius=5,
                            localization_design_penalty=0.000,
                            localization_bounds=(1e-5, 12),
                            )


def start_filtering(results_dir=None, overwrite=True, create_plots=True):
    """
    """

    # Experiment Settings:
    # ============================================================
    # Timesetup
    experiment_tspan = np.arange(0, 100.001, 0.1)

    # Model settings:
    num_Xs = 40
    num_Ys = 32
    F = 8.0
    
    observation_size = num_Xs

    # Filter settings:
    ensemble_size = 25
    #
    # ============================================================

    # Create a model object for the truth
    try:
        _, cpld_ref_IC, cpld_ref_trajectory, cpld_init_ensemble, cpld_observations = get_model_info(experiment_tspan, ensemble_size)
    except ValueError:
        coupled_model_configs = dict(num_prognostic_variables=[num_Xs, num_Ys],  # [X, Y's per each X] each of the 8 is coupled to all 32
                                     force=F,  # forcing term: F
                                     subgrid_varibles_parameters=[1.0, 10.0, 10.0],  # (h, c, b)
                                     # create_background_errors_correlations=True,
                                     observation_opertor_type='linear-coarse',
                                     observation_noise_level=0.05,
                                     background_noise_level=0.08
                                    )
        coupled_model = Coupled_Lorenz(coupled_model_configs)
        # Get Repo Info:
        _, cpld_ref_IC, cpld_ref_trajectory, cpld_init_ensemble, cpld_observations = get_model_info(experiment_tspan, ensemble_size, coupled_model)
        del coupled_model, coupled_model_configs

    #
    # Create a forecast model: i.e. use the reduced version Lorenz-96 with 40 variables
    model_configs={'create_background_errors_correlations':True,
                   'num_prognostic_variables':num_Xs,
                   'observation_error_variances':0.05,
                   # 'observation_noise_level':0.05,
                   'observation_vector_size':observation_size,  # observe everything first
                   'background_noise_level':0.08}
    model = Lorenz96(model_configs)
    model_name = model._model_name

    # return is in NumPy format
    # convert entities to model-based formats
    state_size = model.state_size()
    obs_size = model.observation_size()
    ref_IC = model.state_vector(cpld_ref_IC[: state_size].copy())
    ref_trajectory = []
    observations = []
    for i in xrange(len(experiment_tspan)):
        state = model.state_vector(cpld_ref_trajectory[: state_size, i])
        ref_trajectory.append(state)
        obs = model.state_vector(cpld_observations[: state_size, i])
        observations.append(model.evaluate_theoretical_observation(obs))

    # Create initial ensemble...
    init_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size)
    # init_ensemble = utility.inflate_ensemble(init_ensemble, 4, in_place=True)
    print("Lorenz model and corresponding observations created. Starting the Assimilation section")

    #
    # ======================================================================================== #
    #                               Inititalize the filter object                              #
    # ======================================================================================== #
    # Filter Configs
    # read settings from input file
    settings_filename = __FILTER_CONFIGS_FILENAME
    default_configs = __DEF_FILTER_CONFIGS

    if os.path.isfile(settings_filename):
        _, parser = utility.read_configs(settings_filename)
        section_name = 'filter settings'
        if not parser.has_section(section_name):
            # No configurations found: set defaults
            print("Configurations file found, with nothing in it! Setting to defaults")
            this_dir = os.path.abspath(os.path.dirname(__file__))
            utility.write_dicts_to_config_file(settings_filename, this_dir, default_configs, 'filter settings')
            fetched = False
        else:
            adaptive_inflation = parser.getboolean(section_name, 'adaptive_inflation')
            inflation_bounds = eval(parser.get(section_name, 'inflation_bounds'))
            inflation_design_penalty = parser.getfloat(section_name, 'inflation_design_penalty')
            inflation_factor = parser.getfloat(section_name, 'inflation_factor')
            forecast_inflation_factor = parser.getfloat(section_name, 'forecast_inflation_factor')
            #
            adaptive_localization = parser.getboolean(section_name, 'adaptive_localization')
            localization_function = parser.get(section_name, 'localization_function')
            localization_radius = parser.getfloat(section_name, 'localization_radius')
            localization_design_penalty = parser.getfloat(section_name, 'localization_design_penalty')
            localization_bounds = eval(parser.get(section_name, 'localization_bounds'))
            loc_direct_approach = parser.getint(section_name, 'loc_direct_approach')
            localization_space = parser.get(section_name, 'localization_space').upper().strip()
            #
            regularization_norm = parser.get(section_name, 'regularization_norm').lower().strip()
            moving_average_radius = parser.getint(section_name, 'moving_average_radius')
            ensemble_size = parser.getint(section_name, 'ensemble_size')
            #
            fetched = True
    else:
        print("Couldn't find configs file: %s" % settings_filename)
        print("Added the default values to this config file for later use...")
        this_dir = os.path.abspath(os.path.dirname(__file__))
        utility.write_dicts_to_config_file(settings_filename, this_dir, default_configs, 'filter settings')
        fetched = False

    if not fetched:
        print("Gettings things from default dict")
        for k in default_configs:
            exec("%s = default_configs['%s']" % (k, k))
        #
    #
    # Both are now implemented in Adaptive OED-EnKF ; we will test both
    if adaptive_inflation and adaptive_localization:
        forecast_inflation_factor = inflation_factor = 1.0
        if results_dir is None:
            results_dir = __BASE_RESULTS_DIR + '_ADAPTIVE_INFL_LOC'
            results_dir = os.path.join(results_dir, 'InflPenalty_%f' %(inflation_design_penalty))
            #
    elif adaptive_inflation:
        forecast_inflation_factor = inflation_factor = 1.0
        if results_dir is None:
            results_dir = __BASE_RESULTS_DIR + '_ADAPTIVE_INFL'
            results_dir = os.path.join(results_dir, 'LocRad_%f_InflPenalty_%f' %(localization_radius, inflation_design_penalty))
        #
    elif adaptive_localization:
        if results_dir is None:
            results_dir = __BASE_RESULTS_DIR + '_ADAPTIVE_LOC'
            results_dir = os.path.join(results_dir, 'InflFac_%f_LocPenalty_%f' %(forecast_inflation_factor, localization_design_penalty))
    else:
            results_dir = __BASE_RESULTS_DIR + '_NonAdaptive'
            inflation_factor = forecast_inflation_factor
            results_dir = os.path.join(results_dir, 'InflFac_%f_LocRad_%f' %(inflation_factor, localization_radius))

    #
    if os.path.isdir(results_dir):
        if overwrite:
            pass
        else:
            return None

    #
    enkf_filter_configs = dict(model=model,
                               analysis_ensemble=init_ensemble,
                               forecast_ensemble=None,
                               ensemble_size=ensemble_size,
                               #
                               adaptive_inflation=adaptive_inflation,
                               forecast_inflation_factor=forecast_inflation_factor,
                               inflation_design_penalty=inflation_design_penalty,  # penalty of the regularization parameter
                               localization_design_penalty=localization_design_penalty,  # penalty of the regularization parameter
                               inflation_factor=inflation_factor,
                               inflation_factor_bounds=inflation_bounds,
                               adaptive_localization=adaptive_localization,
                               localize_covariances=True,
                               localization_radii_bounds=localization_bounds,
                               localization_method='covariance_filtering',
                               localization_radius=localization_radius,
                               localization_function=localization_function,
                               loc_direct_approach=loc_direct_approach,
                               localization_space=localization_space,
                               regularization_norm=regularization_norm,
                               moving_average_radius=moving_average_radius,
                               )
    #
    if adaptive_inflation and adaptive_localization:
        from adaptive_EnKF import EnKF_OED_Adaptive
    elif adaptive_inflation:
        from adaptive_EnKF import EnKF_OED_Inflation as EnKF_OED_Adaptive
    elif adaptive_localization:
        from adaptive_EnKF import EnKF_OED_Localization as EnKF_OED_Adaptive
    else:
        from EnKF import DEnKF as EnKF_OED_Adaptive
        print("neither adapgtive inflation nor adaptive localization are revoked!")
        # raise ValueError
    #
    filter_obj = EnKF_OED_Adaptive(filter_configs=enkf_filter_configs,
                                   output_configs=dict(file_output_moment_only=False,verbose=False)
                                   )
    #
    # ======================================================================================== #
    #

    #
    # ======================================================================================== #
    #                        Inititalize the sequential DA process                             #
    # ======================================================================================== #
    # Create the processing object:
    # -------------------------------------
    #
    # create observations' and assimilation checkpoints:
    obs_checkpoints = experiment_tspan
    da_checkpoints = obs_checkpoints
    #

    # Here this is a filtering_process object;
    from filtering_process import FilteringProcess
    assimilation_configs=dict(filter=filter_obj,
                              obs_checkpoints=obs_checkpoints,
                              # da_checkpoints=da_checkpoints,
                              forecast_first=True,
                              ref_initial_condition=ref_IC,
                              ref_initial_time=experiment_tspan[0],
                              random_seed=2345
                             )
    assim_output_configs = dict(scr_output=True,
                                scr_output_iter=1,
                                file_output_dir=results_dir,
                                file_output=True,
                                file_output_iter=1
                               )
    experiment = FilteringProcess(assimilation_configs=assimilation_configs, output_configs=assim_output_configs)

    # Save reference Trajectory:
    results_dir = os.path.abspath(results_dir)
    np.save(os.path.join(results_dir, 'Reference_Trajectory.npy'), utility.ensemble_to_np_array(ref_trajectory, state_as_col=True))
    np.save(os.path.join(results_dir, 'Initial_Ensemble.npy'), utility.ensemble_to_np_array(init_ensemble, state_as_col=True))
    np.save(os.path.join(results_dir, 'Observations.npy'), utility.ensemble_to_np_array(observations, state_as_col=True))
    # Run the sequential filtering process:
    # -------------------------------------
    experiment.recursive_assimilation_process(observations, obs_checkpoints, da_checkpoints)
    #
    # ======================================================================================== #
    #
    if create_plots:
        print("Creating Plots")
        cmd = "python filtering_results_reader_coupledLorenz.py -f %s -r True -o True" % os.path.join(results_dir, 'output_dir_structure.txt')
        os.system(cmd)
    #
    # ======================================================================================== #
    #                        Clean executables and temporary modules                           #
    # ======================================================================================== #
    # utility.clean_executable_files(rm_extensions=['.pyc'])
    # ======================================================================================== #


if __name__ == '__main__':
    start_filtering()


