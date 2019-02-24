#!/usr/bin/env python

"""
Apply Ensemble Kalman Filter to Coupled Lorenz96 model.
"""

import sys
sys.path.insert(1, "../")

import os
import numpy as np
import pickle
import shutil
import matplotlib.pyplot as plt
from test_coupled_lorenz import create_model_info

try:
    Lorenz96
    Coupled_Lorenz
except(NameError):
    import dates_setup
    dates_setup.initialize_dates()
    from lorenz_models import Lorenz96, Coupled_Lorenz
from EnKF import DEnKF as EnKF_Filter


# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
#
import dates_utility as utility  # import DATeS utility module(s)

def get_model_info(timespan, ensemble_size, num_Xs, num_Ys, F, observation_size, observation_noise_level, background_noise_level):
    """
    """
    if background_noise_level is None:
        background_noise_level = 0.08
    if observation_noise_level is None:
        observation_noise_level = 0.05

    model, timespan, _, model_info = \
            create_model_info(num_Xs=num_Xs,
                              num_Ys=num_Ys,
                              F=F,
                              observation_size=observation_size,
                              ensemble_size=ensemble_size,
                              observation_opertor_type='linear-coarse',
                              observation_noise_level=observation_noise_level,
                              background_noise_level=background_noise_level,
                              experiment_tspan=timespan
                             )
    ref_trajectory, init_ensemble, observations = model_info
    ref_IC = ref_trajectory[0].copy()

    return model, timespan, ref_IC, ref_trajectory, init_ensemble, observations


if __name__ == '__main__':

    overwrite_results = True
    read_only = True
    individual_plots = False

    base_file_output_dir="Results/CoupledLorenz_Benchmark/VarObsError"
    inflation_factor = 1.35
    localization_radius = 0.5


    # ============================================================
    # Experiment Settings:
    # ============================================================
    # Timesetup
    experiment_tspan = np.arange(0, 100.001, 0.1)

    # Model settings:
    num_Xs = 40
    num_Ys = 32
    F = 8.0
    observation_size = num_Xs

    avg_frcst_rmse = []
    avg_anl_rmse = []
    avg_free_rmse = []

    # Filter settings:
    ensemble_size = 25

    base_observation_noise_level = 0.05
    obs_noise_levels = np.arange(0, 0.4001, 0.5*base_observation_noise_level)
    obs_noise_levels = obs_noise_levels[1: ]
    background_noise_level = 0.08

    plots_dir = os.path.join(base_file_output_dir, "PLOTS")

    for observation_noise_level in obs_noise_levels:
        print("*"*30)
        print("Observation Noise Level: %f" % observation_noise_level)
        print("*"*30)

        file_output_dir = os.path.join(base_file_output_dir, "ObsErr_%f" % observation_noise_level)
        print(file_output_dir)

        if not read_only:

            #
            # ============================================================
            if os.path.isdir(file_output_dir):
                if overwrite_results:
                    print("To overwrite, I'll remvoe stuff")
                    shutil.rmtree(file_output_dir)
                else:
                    if overwrite_results:
                        print("Cleaning up existing directory")
                        shutil.rmtree(file_output_dir)
                    else:
                        continue
            else:
                pass


            # Create a model object for the truth
            model, timespan, ref_IC, ref_trajectory, init_ensemble, observations = \
                    get_model_info(experiment_tspan, ensemble_size, num_Xs, num_Ys, F, observation_size, observation_noise_level, background_noise_level)
            # return is in NumPy format
            # convert entities to model-based formats
            state_size = model.state_size()
            obs_size = model.observation_size()

            #
            print("Lorenz model and corresponding observations created. Starting the Assimilation section")

            # Create DA pieces:
            # ---------------------
            # this includes:
            #   i-   forecast trajectory/state
            #   ii-  initial ensemble,
            #   iii- filter/smoother/hybrid object.
            #
            # create initial ensemble...
            #
            print("Creating filter object...")
            # create filter object
            # read infaltion factor and localization radius
            enkf_filter_configs = dict(model=model,
                                       analysis_ensemble=init_ensemble,
                                       forecast_ensemble=None,
                                       ensemble_size=ensemble_size,
                                       inflation_factor=1.0,
                                       forecast_inflation_factor=inflation_factor,
                                       obs_covariance_scaling_factor=1.0,
                                       localize_covariances=True,
                                       localization_method='covariance_filtering',
                                       localization_radius=localization_radius,
                                       localization_function='gaspari-cohn'
                                       )

            filter_obj = EnKF_Filter(filter_configs=enkf_filter_configs,
                                     output_configs=dict(verbose=False,file_output_moment_only=False)
                                    )

            # Create sequential DA
            # processing object:
            # ---------------------
            # Here this is a filtering_process object;
            print("Creating filtering-process object...")
            obs_checkpoints = experiment_tspan
            da_checkpoints = obs_checkpoints
            #
            from filtering_process import FilteringProcess
            assimilation_configs=dict(filter=filter_obj,
                                      obs_checkpoints=obs_checkpoints,
                                      # da_checkpoints=da_checkpoints,
                                      forecast_first=True,
                                      ref_initial_condition=ref_IC,
                                      ref_initial_time=experiment_tspan[0],  # should be obtained from the model along with the ref_IC
                                      random_seed=2345
                                      )
            assim_output_configs = dict(scr_output=True,
                                        scr_output_iter=1,
                                        file_output=True,
                                        file_output_dir=file_output_dir,
                                        file_output_iter=1
                                       )
            experiment = FilteringProcess(assimilation_configs, output_configs=assim_output_configs)

            # Save reference Trajectory:
            trgt_dir = file_output_dir
            np.save(os.path.join(trgt_dir, 'Reference_Trajectory.npy'), utility.ensemble_to_np_array(ref_trajectory, state_as_col=True))
            np.save(os.path.join(trgt_dir, 'Initial_Ensemble.npy'), utility.ensemble_to_np_array(init_ensemble, state_as_col=True))
            np.save(os.path.join(trgt_dir, 'Observations.npy'), utility.ensemble_to_np_array(observations, state_as_col=True))

            # run the sequential filtering over the timespan created by da_checkpoints
            experiment.recursive_assimilation_process(observations, obs_checkpoints, da_checkpoints)

        if individual_plots:
            cmd = "python filtering_results_reader_coupledLorenz.py -f %s" % os.path.join(file_output_dir, 'output_dir_structure.txt')
            os.system(cmd)

        # Collect RMSE plots
        results_file = os.path.join(file_output_dir, "Collective_Results.pickle")
        if not os.path.isfile(results_file):
            cmd = "python filtering_results_reader_coupledLorenz.py -f %s" % os.path.join(file_output_dir, 'output_dir_structure.txt')
            os.system(cmd)

        # Update RMSE results
        cont = pickle.load(open(results_file, 'rb'))
        forecast_rmse = np.asarray(cont['forecast_rmse'])
        analysis_rmse = np.asarray(cont['analysis_rmse'])
        free_rmse = np.asarray(cont['free_run_rmse'])
        avg_frcst_rmse.append(np.mean(forecast_rmse[forecast_rmse.size*2/3: ]))
        avg_anl_rmse.append(np.mean(analysis_rmse[analysis_rmse.size*2/3: ]))
        avg_free_rmse.append(np.mean(free_rmse[free_rmse.size*2/3: ]))

    #
    # start plotting adaptive inflation/adaptive localization
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    avg_frcst_rmse = np.asarray(avg_frcst_rmse)
    avg_anl_rmse = np.asarray(avg_anl_rmse)

    coll_dict = dict(obs_noise_levels=obs_noise_levels,
                     avg_free_rmse=avg_free_rmse,
                     avg_frcst_rmse=avg_frcst_rmse,
                     avg_anl_rmse=avg_anl_rmse)
    pickle.dump(coll_dict, open(os.path.join(base_file_output_dir, 'Collective_Results.pickle'), 'wb'))


    #
    fig = plt.figure(facecolor='white')
    ax = fig.gca()
    ax.plot(obs_noise_levels, avg_free_rmse, '-s', label='Free')
    ax.plot(obs_noise_levels, avg_frcst_rmse, '--d', label='Forecast')
    ax.plot(obs_noise_levels, avg_anl_rmse, '-.o', label='Analysis')
    ax.set_ylim(0, np.max(free_rmse)+0.015)
    ax.set_xlabel("Observation noise level")
    ax.set_ylabel("Average RMSE")
    ax.legend(loc='best', framealpha=0.65)

    file_name = os.path.join(plots_dir, "ObsError_vs_RMSE.pdf")
    print("Saving: %s" %file_name)
    plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True,     bbox_inches='tight')

    # Plot ensemble sizes vs. avg_rmses
    #
    # Clean executables and temporary modules
    # ---------------------
    # utility.clean_executable_files()
    #
