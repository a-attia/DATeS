#!/usr/bin/env python

"""
"""
import sys
sys.path.insert(1, "../")

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
import dates_setup
dates_setup.initialize_dates(random_seed=0)
#
from lorenz_models import Lorenz96, Coupled_Lorenz
from coupledlorenz96_enkf_test_driver import get_model_info
import dates_utility as utility

def enhance_plotter():
    """
    """
    font_size = 18
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : font_size}
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=True)
    cmap = matplotlib.cm.jet
    cmap.set_bad('white',1.)

def create_model_info(num_Xs=40,
                      num_Ys=32,
                      F=8.0,
                      observation_size=None,
                      ensemble_size=25,
                      observation_opertor_type='linear-coarse',
                      observation_noise_level=0.05,
                      background_noise_level=0.08,
                      experiment_tspan=None):
    """
    """
    if observation_size is None:
        observation_size = num_Xs
    if experiment_tspan is None:
        experiment_tspan=np.arange(0, 100.001, 0.1)

    #
    repo_dir = "CoupledLorenzRepository"
    if not os.path.isdir(repo_dir):
        os.create_dirs(repo_dir)
    repo_file_name = "Coupled_Lorenz_Repository_ObsErr_%f_PrErr_%f.pickle" % (observation_noise_level, background_noise_level)
    repo_file_name = os.path.join(repo_dir, repo_file_name)
    print("Looking-for/Creating Repo: %s " % repo_file_name)

    # ============================================================
    # Create a model object for the truth
    # ============================================================
    try:
        checkpoints, coupled_ref_IC, coupled_ref_trajectory, coupled_init_ensemble, coupled_observations = get_model_info(experiment_tspan, ignore_existing_repo=False, repo_file=repo_file_name)
    except(ValueError):
        coupled_model_configs = dict(num_prognostic_variables=[num_Xs, num_Ys],  # [X, Y's per each X] each of the 8 is coupled to all 32
                                     force=F,  # forcing term: F
                                     subgrid_varibles_parameters=[1.0, 10.0, 10.0],  # (h, c, b)
                                     observation_opertor_type=observation_opertor_type,
                                     observation_noise_level=observation_noise_level,
                                     background_noise_level=background_noise_level
                                    )
        coupled_model = Coupled_Lorenz(coupled_model_configs)
        # Get Repo Info:
        checkpoints, coupled_ref_IC, coupled_ref_trajectory, coupled_init_ensemble, coupled_observations = get_model_info(experiment_tspan, model=coupled_model, ignore_existing_repo=True, repo_file=repo_file_name)

    #
    # Create a forecast model: i.e. use the reduced version Lorenz-96 with 40 variables
    model_configs={'create_background_errors_correlations':True,
                   'num_prognostic_variables':num_Xs,
                   'observation_error_variances':observation_noise_level,
                   'observation_vector_size':observation_size,  # observe everything first
                   'background_noise_level':background_noise_level}
    model = Lorenz96(model_configs)

    # convert entities to model-based formats
    state_size = model.state_size()
    obs_size = model.observation_size()
    ref_IC = model.state_vector(coupled_ref_IC[: state_size].copy())
    ref_trajectory = []
    observations = []
    for i in xrange(len(experiment_tspan)):
        state = model.state_vector(coupled_ref_trajectory[: state_size, i])
        ref_trajectory.append(state)
        obs = model.state_vector(coupled_observations[: state_size, i])
        observations.append(model.evaluate_theoretical_observation(obs))

    # Create initial ensemble...
    init_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size)

    #
    return model, checkpoints,\
            [coupled_ref_trajectory, coupled_init_ensemble, coupled_observations], \
            [ref_trajectory, init_ensemble, observations]



if __name__ == '__main__':

    plots_dir = "CoupledLorenz_PLOTS"
    show_plots = False
    num_Xs = 40
    num_Ys = 32
    F = 8.0
    observation_size = None
    ensemble_size = 25
    observation_opertor_type = 'linear-coarse'
    background_noise_level = 0.08
    experiment_tspan = None
    base_observation_noise_level = 0.05
    observation_noise_levels = np.arange(21)*0.5*base_observation_noise_level
    # observation_noise_levels = [base_observation_noise_level]
    #
    for observation_noise_level in observation_noise_levels:
        postfix = "_ObsErr_%f_PrErr_%f" % (observation_noise_level, background_noise_level)

        # Create model instance:
        model, checkpoints, coupled_model_info, model_info = \
                create_model_info(num_Xs=num_Xs,
                                  num_Ys=num_Ys,
                                  F=F,
                                  observation_size=observation_size,
                                  ensemble_size=ensemble_size,
                                  observation_opertor_type=observation_opertor_type,
                                  observation_noise_level=observation_noise_level,
                                  background_noise_level=background_noise_level,
                                  experiment_tspan=experiment_tspan
                                 )
        coupled_ref_trajectory, coupled_init_ensemble, coupled_observations = coupled_model_info
        ref_trajectory, init_ensemble, observations = model_info
        IC = ref_trajectory[0]


        # Forecast/Free-Run Trajectory:
        forecast_trajectory = model.integrate_state(utility.ensemble_mean(init_ensemble), checkpoints)
        forecast_trajectory_fine = model.integrate_state(utility.ensemble_mean(init_ensemble), np.arange(checkpoints[0], checkpoints[-1], 0.005))

        #
        print("Model Info Generated; Started Plotting...")
        enhance_plotter()
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)
        #

        # Coupled Lorenz:
        coupled_state_size = len(coupled_ref_trajectory[1])
        trajectory = np.asarray(coupled_ref_trajectory)
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        extent = [checkpoints[0], checkpoints[-1], 1, coupled_state_size]
        cax = ax.imshow(trajectory, aspect='auto', origin='upper', extent=extent, interpolation='nearest')
        # ax.set_title("Coupled Lorenz Model")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$x_{i}\, \forall i=1,\ldots,%d$" % coupled_state_size)
        ax.set_yticklabels(coupled_state_size - ax.get_yticks().astype(int) + 1)  # invert ticklables
        fig.colorbar(cax)
        file_name = "Coupled_Lorenz_Reference_trajectory%s.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

        # 2- Low-dimensional Lorenz (coarse
        # ----------------------------------
        state_size = model.state_size()
        trajectory = np.asarray(ref_trajectory)
        fig = plt.figure(figsize=(8.10, 3.15), facecolor='white')
        ax = fig.add_subplot(111)
        extent = [checkpoints[0], checkpoints[-1], 1, num_Xs]
        cax = ax.imshow(trajectory.T, aspect='auto', origin='upper', extent=extent, interpolation='nearest')
        # ax.set_title("Lorenz Model")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$x_{k}\, \forall k=1,2,\ldots,%d$" % num_Xs)
        ax.set_yticklabels(num_Xs - ax.get_yticks().astype(int) + 1)  # invert ticklables
        fig.colorbar(cax)
        file_name = "Lorenz_Reference_trajectory%s.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

        # 3- Low-dimensional Lorenz (coarse
        # ----------------------------------
        trajectory = np.asarray(forecast_trajectory)
        fig = plt.figure(figsize=(8.10, 3.15), facecolor='white')
        ax = fig.add_subplot(111)
        extent = [checkpoints[0], checkpoints[-1], 1, num_Xs]
        cax = ax.imshow(trajectory.T, aspect='auto', origin='upper', extent=extent, interpolation='nearest')
        # ax.set_title("Lorenz Model")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$x_{k}\, \forall k=1,2,\ldots,%d$" % num_Xs)
        ax.set_yticklabels(num_Xs - ax.get_yticks().astype(int) + 1)  # invert ticklables
        fig.colorbar(cax)
        file_name = "Lorenz_Forecast_trajectory%s.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

        # 4- Initial True state vs. initial forecast state
        # ----------------------------------
        fig = plt.figure(figsize=(8.10, 3.15), facecolor='white')
        ax = fig.add_subplot(111)
        xvals = np.arange(1, num_Xs+1)
        ax.plot(xvals, ref_trajectory[0][:], '-d', label='True IC')
        ax.plot(xvals, forecast_trajectory[0][:], '-.o', label='Prior IC')
        # ax.set_title("Lorenz Model")
        # ax.set_xlabel(r"$x_{k}\, \forall k=1,2,\ldots,%d$" % num_Xs)
        plt.minorticks_on()
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle='-.')
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
        file_name = "Lorenz_IC%s.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

        # 5- Noise variances
        fig = plt.figure(figsize=(8.10, 3.15), facecolor='white')
        ax = fig.add_subplot(111)
        xvals = np.arange(1, num_Xs+1)
        obs_noise = np.sqrt(model.observation_error_model.R.diagonal())
        prior_noise = np.sqrt(model.background_error_model.B.diagonal())
        label = r'$\sigma_k^b(t_0);\, k=1,\ldots,%d$'%num_Xs
        ax.plot(xvals, prior_noise, '-d', label=label)
        label = r'$\sigma_k^o;\, k=1,\ldots,%d$'%num_Xs
        ax.plot(xvals, obs_noise, '-.o', label=label)
        # ax.set_title("Lorenz Model")
        plt.minorticks_on()
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle='-.')
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
        file_name = "Lorenz_Noise%s.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


        # 6- 3D Plot of 3 entries of the reference solutions of both coupled Lorenz, and lorenz96
        # ----------------------------------
        trajectory = np.asarray(coupled_ref_trajectory).T
        print(trajectory.shape)
        #
        xs = [0, 0, num_Xs/2]
        ys = [1, num_Xs/4, num_Xs/2+1]
        zs = [2, num_Xs/2, num_Xs-1]
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        for ind, [x_ind, y_ind, z_ind] in enumerate(zip(xs, ys, zs)):
            label = r"$x_{%d}, x_{%d}, x_{%d}$" % (x_ind+1, y_ind+1, z_ind+1)
            ax.plot(trajectory[:, x_ind], trajectory[:, y_ind], trajectory[:, z_ind], label=label, alpha=0.75)
        ax.legend(loc='upper left')
        file_name = "Coupled_Lorenz_Reference_trajectory_3D_1%s.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

        # Repeat for free-run/forecast trajectory
        f_trajectory = np.asarray(forecast_trajectory_fine)
        sinds = np.arange(0, np.size(f_trajectory, 0), 10)
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        for ind, [x_ind, y_ind, z_ind] in enumerate(zip(xs, ys, zs)):
            label = r"$x_{%d}, x_{%d}, x_{%d}$" % (x_ind+1, y_ind+1, z_ind+1)
            ax.plot(f_trajectory[sinds, x_ind], f_trajectory[sinds, y_ind], f_trajectory[sinds, z_ind], label=label, alpha=0.75)
        ax.legend(loc='upper left')
        file_name = "Coupled_Lorenz_Forecast_trajectory_3D_1%s.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

        #
        xs = [num_Xs+i for i in xs]
        ys = [num_Xs+i for i in ys]
        zs = [num_Xs+i for i in zs]
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        for ind, [x_ind, y_ind, z_ind] in enumerate(zip(xs, ys, zs)):
            label = r"$x_{%d}, x_{%d}, x_{%d}$" % (x_ind+1, y_ind+1, z_ind+1)
            ax.plot(trajectory[:, x_ind], trajectory[:, y_ind], trajectory[:, z_ind], label=label, alpha=0.75)
        ax.legend(loc='upper left')
        file_name = "Coupled_Lorenz_Reference_trajectory_3D_2%s.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

        # Only the True nitial condition
        fig = plt.figure(figsize=(8.10, 3.15), facecolor='white')
        ax = fig.add_subplot(111)
        xvals = np.arange(num_Xs) * num_Ys + 1
        yvals = coupled_ref_trajectory[: num_Xs, 0]
        label = r"$x_{k};\, k=1,\ldots,%d$" % num_Xs
        ax.plot(xvals, yvals, '-.o', label=label)
        #
        yvals = coupled_ref_trajectory[num_Xs: , 0]
        xvals = np.arange(1, yvals.size+1)
        label = r"$z_{j};\, j=1,\ldots,%d$" % (num_Xs*num_Ys)
        ax.plot(xvals, yvals, '-', label=label)
        #
        plt.minorticks_on()
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle='-.')
        #
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
        #
        file_name = "Coupled_Lorenz_IC%s.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

        ax.set_xticklabels([])
        file_name = "Coupled_Lorenz_IC%s_tickless.pdf" % postfix
        file_name = os.path.join(plots_dir, file_name)
        print("Saving Plot: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
        # Show Plot:
        if show_plots:
            plt.show()
        plt.close('all')
