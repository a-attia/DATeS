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
import numpy as np
import pickle
import matplotlib.pyplot as plt

from coupledlorenz96_enkf_test_driver import get_model_info

try:
    Lorenz96
    Coupled_Lorenz
except(NameError):
    import dates_setup
    dates_setup.initialize_dates()
    from lorenz_models import Lorenz96, Coupled_Lorenz

import dates_utility as utility  # import DATeS utility module(s)

from test_coupled_lorenz import create_model_info, enhance_plotter


# Defaults:
# ---------
__BASE_RESULTS_DIR = 'Results/CoupledLorenzResults/VariableObsNoise'
__FILTER_CONFIGS_FILENAME = "oed_settings.dat"
__adaptive_infl_configs = dict(adaptive_inflation=True,
                               moving_average_radius=2,
                               regularization_norm='l1',
                               inflation_factor=1.0,
                               forecast_inflation_factor=1.0,
                               inflation_bounds=(1.0, 1.5),
                               inflation_design_penalty='0.003',
                               #
                               adaptive_localization=False,
                               localization_space='B',
                               loc_direct_approach=3,
                               localization_function='Gaspari-Cohn',
                               localization_radius=0.5,
                               localization_bounds=(0.0, 5.0),
                               localization_design_penalty = 0.000,
                               ensemble_size=25,
                              )
__adaptive_loc_configs = dict(adaptive_inflation=False,
                               moving_average_radius=2,
                               regularization_norm='l1',
                               inflation_factor=1.0,
                               forecast_inflation_factor=1.35,
                               inflation_bounds=(1.0, 1.5),
                               inflation_design_penalty='0.003',
                               #
                               adaptive_localization=True,
                               localization_space='B',
                               loc_direct_approach=3,
                               localization_function='Gaspari-Cohn',
                               localization_radius=1,
                               localization_bounds=(0.0, 5.0),
                               localization_design_penalty = 0.000,
                               ensemble_size=25,
                              )

def start_filtering(results_dir=None, overwrite=True, create_plots=True, background_noise_level=None, observation_noise_level=None):
    """
    """

    if background_noise_level is None:
        background_noise_level = 0.08
    if observation_noise_level is None:
        observation_noise_level = 0.05

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
    # Create model instance:
    model, checkpoints, cpld_model_info, model_info = \
            create_model_info(num_Xs=num_Xs,
                              num_Ys=num_Ys,
                              F=F,
                              observation_size=observation_size,
                              ensemble_size=ensemble_size,
                              observation_opertor_type='linear-coarse',
                              observation_noise_level=observation_noise_level,
                              background_noise_level=background_noise_level,
                              experiment_tspan=experiment_tspan
                             )
    cpld_ref_trajectory, cpld_init_ensemble, cpld_observations = cpld_model_info
    ref_trajectory, init_ensemble, observations = model_info
    cpld_ref_IC = cpld_ref_trajectory[0].copy()
    ref_IC = ref_trajectory[0].copy()

    #
    model_name = model._model_name
    state_size = model.state_size()
    obs_size = model.observation_size()
    print("Lorenz model and corresponding observations created. Starting the Assimilation section")

    #
    # ======================================================================================== #
    #                               Inititalize the filter object                              #
    # ======================================================================================== #
    # Filter Configs
    # read settings from input file
    settings_filename = __FILTER_CONFIGS_FILENAME

    if os.path.isfile(settings_filename):
        _, parser = utility.read_configs(settings_filename)
        section_name = 'filter settings'
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
    else:
        print("Couldn't find configs file: %s" % settings_filename)
        raise IOError

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
        # Try plotting results
        try:
            osf = os.path.join(results_dir, 'output_dir_structure.txt')
            cmd = "python filtering_results_reader_coupledlorenz.py -f %s -o True -r True" % osf
            print("Plotting Results:\n%s" % cmd)
            os.system(cmd)
        except:
            print("Failed to generate plots!")
            pass
    # ======================================================================================== #
    #                        Clean executables and temporary modules                           #
    # ======================================================================================== #
    # utility.clean_executable_files(rm_extensions=['.pyc'])
    # ======================================================================================== #


if __name__ == '__main__':
    """
    """
    read_results_only = True

    base_observation_noise_level = 0.05
    obs_noise_levels = np.arange(0, 0.4001, 0.5*base_observation_noise_level)
    obs_noise_levels = obs_noise_levels[1: ]
    settings_filename = __FILTER_CONFIGS_FILENAME
    this_dir = os.path.abspath(os.path.dirname(__file__))

    num_experiments = 2 * obs_noise_levels.size
    exp_no = 0
    #
    enhance_plotter()
    plots_dir = os.path.join(__BASE_RESULTS_DIR, "PLOTS")
    for adaptive_inflation in [True, False]:
        # placeholders for what need to be plotted; add more as needed
        avg_frcst_rmse = []
        avg_anl_rmse = []
        avg_free_rmse = []
        for observation_noise_level in obs_noise_levels:
            adaptive_localization = not adaptive_inflation
            postfix = "_ObsErr_%f" % observation_noise_level

            # Adjust settings file
            if adaptive_inflation:
                # get stuff from oed_adaptive_inflation
                default_configs = __adaptive_infl_configs
                exp_tag = "Adaptive_Inflation"
            else:
                # get stuff from oed_adaptive_localization
                default_configs = __adaptive_loc_configs
                exp_tag = "Adaptive_Localization"

            utility.write_dicts_to_config_file(settings_filename, this_dir, default_configs, 'filter settings')
            # Print some header
            sep = "%sEnKF Experiment%s" % ('='*25, '='*25)
            exp_no += 1
            print(sep)
            print("Experiment Number [%d] out of [%d]" % (exp_no, num_experiments))
            print(exp_tag)
            print("Observation Noise Level: %f" % observation_noise_level)
            print("%s\n" % ('='*len(sep)))

            # Prepare output path, and start assimilation
            results_dir = os.path.join(__BASE_RESULTS_DIR, "%s%s" % (exp_tag, postfix))
            if not read_results_only:
                start_filtering(results_dir, observation_noise_level=observation_noise_level)
            
            # Read RMSE results
            results_file = os.path.join(results_dir, "Collective_Results.pickle")
            cont = pickle.load(open(results_file, 'rb'))
            forecast_rmse = np.asarray(cont['forecast_rmse'])
            analysis_rmse = np.asarray(cont['analysis_rmse'])
            free_rmse = np.asarray(cont['free_run_rmse'])
            avg_frcst_rmse.append(np.mean(forecast_rmse[forecast_rmse.size*2/3: ]))
            avg_anl_rmse.append(np.mean(analysis_rmse[analysis_rmse.size*2/3: ]))
            avg_free_rmse.append(np.mean(free_rmse[free_rmse.size*2/3: ]))
        # start plotting adaptive inflation/adaptive localization
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)

        avg_frcst_rmse = np.asarray(avg_frcst_rmse)
        avg_anl_rmse = np.asarray(avg_anl_rmse)
        
        coll_dict = dict(obs_noise_levels=obs_noise_levels,
                        avg_free_rmse=avg_free_rmse,
                        avg_frcst_rmse=avg_frcst_rmse,
                        avg_anl_rmse=avg_anl_rmse)
        par_results_dir = os.path.dirname(os.path.abspath(results_dir))
        pickle.dump(coll_dict, open(os.path.join(par_results_dir, 'Collective_Results_%s.pickle'%exp_tag), 'wb'))
        
        
        #
        fig = plt.figure(figsize=(6.5,3), facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(obs_noise_levels, avg_free_rmse, '-s', label='Free')
        ax.plot(obs_noise_levels, avg_frcst_rmse, '--d', label='Forecast')
        ax.plot(obs_noise_levels, avg_anl_rmse, '-.o', label='Analysis')
        ax.set_ylim(0, np.max(free_rmse)+0.015)
        ax.set_xlabel("Observation noise level")
        ax.set_ylabel("Average RMSE")
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)

        file_name = os.path.join(plots_dir, "%s.pdf" % exp_tag)
        print("Saving: %s" %file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True,     bbox_inches='tight')
        
        # resaving with grid
        plt.minorticks_on()
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle='-.')
        file_name = os.path.join(plots_dir, "%s_grid.pdf" % exp_tag)
        print("Saving: %s" %file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True,     bbox_inches='tight')
