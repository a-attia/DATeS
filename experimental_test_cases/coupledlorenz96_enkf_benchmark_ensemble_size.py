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

try:
    Lorenz96
    Coupled_Lorenz
except(NameError):
    import dates_setup
    dates_setup.initialize_dates()
    from lorenz_models import Lorenz96, Coupled_Lorenz


# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
#
import dates_utility as utility  # import DATeS utility module(s)

def get_model_info(timespan, ensemble_size=25, model=None, load_ensemble=False, ignore_existing_repo=False, repo_file=None):
    """
    """
    timespan = np.array(timespan).flatten()
    # print("Requesting timespan: ", timespan)
    # Check the experiment repository
    if repo_file is None:
        file_name = 'Coupled_Lorenz_Repository.pickle'
        file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))
    else:
        file_name = os.path.abspath(repo_file)

    # Check repo if needed, and found:
    if ignore_existing_repo:
        ref_IC_np = ref_trajectory_np = init_ensemble_np = observations_np = None
        #
    elif os.path.isfile(file_name):
        print("Loading model repo from : %s" % file_name)
        cont = pickle.load(open(file_name, 'rb'))
        tspan = cont['tspan']

        # Get the initial condition anyways
        ref_IC_np = cont['ref_trajectory'][:, 0].copy()

        # Verify timespan, and get reference trajectory
        if tspan.size < timespan.size:
            ref_trajectory_np = None
            t_indexes = None
            save_info = True
            #
        elif tspan.size > timespan.size:
            # Look into the reference trajectory to find matches, if existing
            t_indexes = []
            for i, t in enumerate(timespan):
                loc = np.where(np.isclose(tspan, t, rtol=1e-12))[0]
                if loc.size>0:
                    loc = loc[0]
                    t_indexes.append(loc)
                else:
                    t_indexes = None
                    break
            #
            if t_indexes is not None:
                t_indexes.sort()
                t_indexes = np.asarray(t_indexes)
                ref_trajectory_np = cont['ref_trajectory'][:, t_indexes]
            else:
                ref_trajectory_np = None
                #
        else:
            if np.isclose(tspan, timespan).all():
                ref_trajectory_np = cont['ref_trajectory']
                t_indexes = np.arange(timespan.size)
                #

        # Observations:
        if t_indexes is None or ref_trajectory_np is None:  # TODO: this is overkill!
            observations_np = None
        else:
            observations_np = cont['observations'][:, t_indexes]

        # Read the ensemble
        if load_ensemble:
            init_ensemble_np = cont['init_ensemble']
            if np.size(init_ensemble_np, 1) <= ensemble_size:
                init_ensemble_np = init_ensemble_np[:, : ensemble_size]
            else:
                init_ensemble_np = None
        else:
            init_ensemble_np = None

    else:
        print("The coupled-lorenz model repo is not found. This should be the first time to run this function/script.")
        ref_IC_np = ref_trajectory_np = init_ensemble_np = observations_np = None

    # Check the integrity of the loaded data:
    if ref_trajectory_np is None or (load_ensemble and init_ensemble_np is None) or observations_np is None:
        #
        # Generate stuff, and save them
        if model is None:
            print("YOU MUST pass a MODEL object because the repo is not found in %s, or some information is missing!" % file_name)
            raise ValueError
        else:
            print("Got the model >> Creating experiemnt info...")

        # Reference IC
        if ref_IC_np is None:
            ref_IC = model._reference_initial_condition.copy()
            ref_IC_np = ref_IC.get_numpy_array()

        # Reference Trajectory (Truth)
        if ref_trajectory_np is None:
            if ref_IC_np is not None:
                ref_IC = model.state_vector(ref_IC_np.copy())
            else:
                ref_IC = model._reference_initial_condition.copy()
            # generate observations from coupled lorenz:
            ref_trajectory = model.integrate_state(ref_IC, timespan)
            ref_trajectory_np = utility.ensemble_to_np_array(ref_trajectory, state_as_col=True)

        # Initial (forecast) Ensmble:
        if load_ensemble and init_ensemble_np is None:
            prior_noise_model = model.background_error_model
            prior_noise_sample = [prior_noise_model.generate_noise_vec() for i in xrange(ensemble_size)]
            noise_mean = utility.ensemble_mean(prior_noise_sample)
            ic = prior_noise_model.generate_noise_vec().add(ref_IC)
            for i in xrange(ensemble_size):
                prior_noise_sample[i].axpy(-1.0, noise_mean)
                prior_noise_sample[i].add(ic, in_place=True)
            init_ensemble = prior_noise_sample
            init_ensemble_np = utility.ensemble_to_np_array(init_ensemble, state_as_col=True)

        if observations_np is None:
            # Create observations' and assimilation checkpoints:
            if ref_trajectory is None:
                if ref_trajectory_np is None:
                    ref_trajectory = model.integrate_state(ref_IC, timespan)
                else:
                    ref_trajectory = [model.state_vectory(ref_trajectory_np[i, :].copy()) for i in xrange(np.size(ref_trajectory, 1))]
            observations = [model.evaluate_theoretical_observation(x) for x in ref_trajectory]
            # Perturb observations:
            obs_noise_model = model.observation_error_model
            for obs_ind in xrange(len(timespan)):
                observations[obs_ind].add(obs_noise_model.generate_noise_vec(), in_place=True)
            observations_np = utility.ensemble_to_np_array(observations, state_as_col=True)

        print("Experiment repo created; saving for later use...")
        # save results for later use
        out_dict = dict(tspan=timespan,
                        ref_trajectory=ref_trajectory_np,
                        init_ensemble=init_ensemble_np,
                        observations=observations_np)
        pickle.dump(out_dict, open(file_name, 'wb'))
        print("...done...")
        #
    else:
        print("All information properly loaded from file")
        pass

    return timespan, ref_IC_np, ref_trajectory_np, init_ensemble_np, observations_np


if __name__ == '__main__':

    overwrite_results = True
    read_only = True
    individual_plots = False

    base_file_output_dir="Results/CoupledLorenz_Benchmark/VarEnsembleSize"
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

    avg_frcst_rmse = []
    avg_anl_rmse = []
    avg_free_rmse = []

    # Filter settings:
    ensemble_size_pool = np.arange(5, 26, 5)

    plots_dir = os.path.join(base_file_output_dir, "PLOTS")

    for ensemble_size in ensemble_size_pool:
        print("*"*30)
        print("Ensemble Size: %d " % ensemble_size)
        print("*"*30)

        file_output_dir = os.path.join(base_file_output_dir, "EnsembleSize_%d" % ensemble_size)
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
                           'observation_vector_size':num_Xs,  # observe everything first
                           'background_noise_level':0.08}
            model = Lorenz96(model_configs)

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

            if True:
                init_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size)
            else:
                init_ensemble = []
                for i in xrange(ensemble_size):
                    state = model.state_vector(cpld_init_ensemble[: state_size, i])
                    init_ensemble.append(state)
                #
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
            from EnKF import DEnKF as EnKF_Filter
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
        try:
            results_file = os.path.join(file_output_dir, "Collective_Results.pickle")
        except(IOError):
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
    #
    coll_dict = dict(ensemble_size_pool=ensemble_size_pool,
                     avg_free_rmse=avg_free_rmse,
                     avg_frcst_rmse=avg_frcst_rmse,
                     avg_anl_rmse=avg_anl_rmse)
    pickle.dump(coll_dict, open(os.path.join(base_file_output_dir, 'Collective_Results.pickle'), 'wb'))


    fig = plt.figure(facecolor='white')
    ax = fig.gca()
    ax.plot(ensemble_size_pool, avg_free_rmse, '-s', label='Free')
    ax.plot(ensemble_size_pool, avg_frcst_rmse, '--d', label='Forecast')
    ax.plot(ensemble_size_pool, avg_anl_rmse, '-.o', label='Analysis')
    ax.set_ylim(0, np.max(free_rmse)+0.015)
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Average RMSE")
    ax.legend(loc='best', framealpha=0.65)

    file_name = os.path.join(file_output_dir, "EnsembleSize_vs_RMSE.pdf")
    print("Saving: %s" %file_name)
    plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True,     bbox_inches='tight')

    # Plot ensemble sizes vs. avg_rmses
    #
    # Clean executables and temporary modules
    # ---------------------
    # utility.clean_executable_files()
    #
