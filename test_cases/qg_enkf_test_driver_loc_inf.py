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
*   Specific description:                                                                  *
*       This driver is to test EnKF with QG-1.5 for different values of the                *
*       1- covariance localization radius, and                                             *
*       2- covariance inflation coefficient.                                               *
*                                                                                          *
*   Note: What you may want to play with are:                                              *
*       a) read_results_only: this is IMPORTANT to avoid rerunning experiments             *
*       b) localization_radii_pool: a pool of values of the localization radius you want   *
*          to test                                                                         *
*       c) localization_function: a pool of localization functions to test                 *
*       d) inflation_factors_pool: a pool of values of the inflation factor to test        *
*                                                                                          *
*   ....................................................................................   *
********************************************************************************************
* To Run the driver:                                                                       *
* --------------------                                                                     *
*        On the linux terminal execute the following command:                              *
*           > python qg_enkf_test_driver_loc_inf.py                                        *
*                                                                                          *
********************************************************************************************
"""

import sys
import os
import numpy as np
import copy
from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


try:
    import cpickle
except:
    import cPickle as pickle

from filtering_results_reader import read_filter_output, rank_hist

#
# ================================================================================================ #
def get_output_configs_file_name():
    """
    A simple function to decide upon the file name to which experiments settings are pickled/written
    """
    dates_root_dir = os.environ.get('DATES_ROOT_PATH')
    output_configs_file = os.path.join(file_output_dir_rel_root, 'output_configs.p')
    output_configs_file = os.path.join(dates_root_dir, output_configs_file)
    
    return output_configs_file

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.set_adjustable('box-forced')
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)
            
def make_titles_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)
# ================================================================================================ #
#


# Define environment variables and update Python search path;
# this is a necessary call that must be inserted in the beginning of any driver.
import dates_setup
dates_setup.initialize_dates(random_seed=1)
#
import dates_utility as utility  # import DATeS utility module(s)

# If the results directory already exist, don't rerun experiments
# Set it to False only if you want to run all the experiments, and save the results
read_results_only = False  

# Plot relative frequency in rank histograms
plot_relative_freq = True

# Output directory
file_output_dir_rel_root = 'Results/'

if not read_results_only:
    # ============================================================================================ #
    #                                  Create a model object                                       #
    # ============================================================================================ #
    #
    from qg_1p5_model import QG1p5
    MREFIN = 5  # MREFIN = 5, 6, 7 for the Sakov's three models QGt, QGs, QG respectively
    obs_dim = 300
    model = QG1p5(model_configs = dict(MREFIN=MREFIN, 
                                       observation_operator_type='linear',
                                       observation_vector_size=obs_dim,
                                       observation_error_variances=4.0,
                                       # observation_errors_covariance_method='diagonal',
                                       background_error_variances=5.0,
                                       # background_errors_covariance_method='diagonal',
                                       # background_errors_covariance_localization_method='Gaspari-cohn',
                                       # background_errors_covariance_localization_radius=12
                                       )
               )

    #
    # ============================================================================================ #
    #                            Create Data Assimilation pieces                                   #
    # ============================================================================================ #
    #
    # i) create initial ensemble...
    ensemble_size = 25
    initial_ensemble = model.create_initial_ensemble(ensemble_size=ensemble_size, ensemble_from_repo=True)

    #
    # ii) create observations' and assimilation checkpoints:
    base_checkpoints = np.arange(0, 1250.0001, 12.5)
    if MREFIN == 5:
        obs_checkpoints = base_checkpoints * 4
    elif MREFIN == 6:
        obs_checkpoints = base_checkpoints * 2
    elif MREFIN == 7:
        obs_checkpoints = base_checkpoints
    else:
        print("MREFIN has to be one of the values 5, 6, or 7")
        raise ValueError()
    da_checkpoints = obs_checkpoints  # Synchronous DA
    #

    # iii) Localization radii, and inflation factors to test:
    inflation_factors_pool = np.arange(1.01, 1.1001, 0.0025)

    localization_functions_pool = ["Gaspari-Cohn", "Gauss"]
    if MREFIN == 5:
        localization_radii_pool = np.arange(3,12)
    elif MREFIN == 6:
        localization_radii_pool = np.arange(3,15)
    elif MREFIN == 7:
        localization_radii_pool = np.arange(3,20)
    else:
        print("MREFIN has to be one of the values 5, 6, or 7")
        raise ValueError()


    #
    # Initialize an EnKF filter:
    from EnKF import DEnKF as StochasticEnKF
    enkf_filter_configs = dict(model=model,
                               analysis_ensemble=initial_ensemble,
                               forecast_ensemble=None,
                               ensemble_size=ensemble_size,
                               localize_covariances=True,
                               localization_method='covariance_filtering')

    filter_obj = StochasticEnKF(filter_configs=enkf_filter_configs, 
                                output_configs=dict(file_output_moment_only=False))


    #
    # Sequential DA 
    # ---------------------
    # Initialize a filtering_process object;
    from filtering_process import FilteringProcess
    experiment = FilteringProcess(assimilation_configs=dict(model=model,
                                                            filter=filter_obj,
                                                            obs_checkpoints=obs_checkpoints,
                                                            da_checkpoints=da_checkpoints,
                                                            forecast_first=True,
                                                            ref_initial_condition=model._reference_initial_condition.copy(),
                                                            ref_initial_time=0,  # should be obtained from the model along with the ref_IC
                                                            random_seed=None
                                                            ),
                                  output_configs = dict(scr_output=True,
                                                        scr_output_iter=1,
                                                        file_output=True,
                                                        file_output_iter=1,
                                                        file_output_dir=file_output_dir_rel_root))

    #
    # Start testing all combinations of localization functions/radii and inflation factors:
    results_settings_configs = OrderedDict()  # an ordered dictionary holding experiments filenames and settings
    for localization_function in localization_functions_pool:
        for inflation_factor in inflation_factors_pool:
            for localization_radius in localization_radii_pool:
                #
                print("Running DEnKF with QG 1.5 with:  \
                       \n\r\tInflation Factor: %f  \
                       \n\r\tLocalization Function: %s  \
                       \n\r\tLocalization radius: %f" % (inflation_factor,
                                                       localization_function, 
                                                       localization_radius))
                
                #
                # Update the localization and inflation settings for the current experiment
                results_dir = 'Infl_%f_Loc_%s_rad_%s' % (inflation_factor, 
                                                         localization_function, 
                                                         localization_radius
                                                         )
                file_output_dir = os.path.join(file_output_dir_rel_root, results_dir)
                
                results_settings_configs.update({file_output_dir:{'localization_function':localization_function,
                                                         'localization_radius':localization_radius,
                                                         'inflation_factor':inflation_factor}})

                # Update the filtering object configurations:
                filter_obj.filter_configs['localization_function'] = localization_function
                filter_obj.filter_configs['localization_radius'] = localization_radius
                filter_obj.filter_configs['inflation_factor'] = inflation_factor
                filter_obj.output_configs['file_output_dir'] = file_output_dir
                
                # Update and run the sequential filtering over the timespan created by da_checkpoints:
                experiment.output_configs['file_output_dir'] = file_output_dir  # Not necessary
                experiment.recursive_assimilation_process()
                #

    # Write/Dump the output file/settings dictionary:
    #
    output_configs_file = get_output_configs_file_name()
    pickle.dump(results_settings_configs, open(output_configs_file, 'wb'))


# ============================================================================================ #
#                                  Read and plot the results                                   #
# ============================================================================================ #
#

# Read filtering results, and extract some features to plot:
if read_results_only:
    output_configs_file = get_output_configs_file_name()
    results_settings_configs = pickle.load(open(output_configs_file, 'rb'))
else:
    # results_settings_configs is supposed to be already in memory
    try:
        results_settings_configs
    except NameError:
        print("Was not able to retrieve 'NameError'\n This shouldn't happen!")
        raise
    
#
# Start collecting results:
dates_root_dir = os.environ.get('DATES_ROOT_PATH')

# new holders; just in case something is missing
spinup_cycles = 5 
_localization_functions = []
_localization_radii = []
_inflation_factors = []

use_log_scale = False
add_lables = False  # add lales to the RMSE plots or not
#

# Set plots configurations:
font_size = 22
line_width = 5
font = {'weight': 'bold', 'size': font_size}
plt.rc('font', **font)

# Quick pass to get correct sizes:
for file_output_dir in results_settings_configs:
    #
    exp_settings = results_settings_configs[file_output_dir]
    localization_function = exp_settings['localization_function']
    localization_radius = exp_settings['localization_radius']
    inflation_factor = exp_settings['inflation_factor']
    #
    _localization_functions.append(localization_function)
    _localization_radii.append(localization_radius)
    _inflation_factors.append(inflation_factor)

_localization_functions = list(set(_localization_functions))
_localization_radii = list(set(_localization_radii))
_localization_radii.sort()
_inflation_factors = list(set(_inflation_factors))
_inflation_factors.sort()

num_of_loc_funcs = len(_localization_functions)
num_of_loc_radii = len(_localization_radii)
num_of_infl_facs = len(_inflation_factors)


# TODO: Can do better than that!
localization_functions = []
localization_radii = []
inflation_factors = []
rmse_averages = []
rmse_plots = []
rhist_plots = []

#
for loc_function in _localization_functions:
    #
    rmse_scatter_fig, rmse_scatter_axes = plt.subplots(nrows=num_of_infl_facs,
                                                       ncols=num_of_loc_radii, 
                                                       facecolor='white')
    rmse_scatter_fig.suptitle("Localization Function: %s" % localization_function)
    #
    rhist_scatter_fig, rhist_scatter_axes = plt.subplots(nrows=num_of_infl_facs,
                                                         ncols=num_of_loc_radii, 
                                                         facecolor='white')
    rhist_scatter_fig.suptitle("Localization Function: %s" % localization_function)
    
    #
    for file_output_dir in results_settings_configs:
        #
        exp_settings = results_settings_configs[file_output_dir]
        localization_function = exp_settings['localization_function']
        if loc_function.lower() != localization_function.lower():
            break
        localization_radius = exp_settings['localization_radius']
        inflation_factor = exp_settings['inflation_factor']
        #
        inflation_factors.append(inflation_factor)
        localization_radii.append(localization_radius)
        #
        
        xind = np.where(_inflation_factors==inflation_factor)[0][0]
        yind = np.where(_localization_radii==localization_radius)[0][0]
        
        #
        # Start reading experiment results:
        results_path = os.path.join(dates_root_dir, file_output_dir)
        results_structure_file_path = os.path.join(results_path, 'output_dir_structure.txt')
        retrieved_results = read_filter_output(results_structure_file_path)
         
        #
        # extract what we need from returned results: TODO: maintain only what's needed!
        analysis_times = retrieved_results[9]
        analysis_rmse = retrieved_results[12]
        analysis_ensembles = retrieved_results[5]
        reference_states = retrieved_results[2]
        state_size = reference_states.shape[0]
        
        #
        # RMSE plot, and average (after a spinup period):
        rmse_average = np.asarray(analysis_rmse[spinup_cycles:]).mean()
        rmse_averages.append(rmse_average)
        
        # calculate rank statistics:
        Ranks, Bins_bounds , _, Bins_relative_counts = rank_hist(ensembles_repo=analysis_ensembles,
                                           reference_states=reference_states, 
                                           first_var=1, 
                                           last_var=state_size-1, 
                                           skp=1,
                                           plot_hist=False, 
                                           type_hist=0,
                                           initial_time_ind=0
                                           )

        # Add RMSE plot:
        if use_log_scale:
            rmse_scatter_axes[xind, yind].semilogy(analysis_times, analysis_rmse, 'bd-', linewidth=line_width)
        else:
            rmse_scatter_axes[xind, yind].plot(analysis_times, analysis_rmse, 'bd-', linewidth=line_width)
        
        # Add Rank-histogram plot:
        tmp = rhist_scatter_axes[xind, yind].bar(Bins_bounds , Bins_relative_counts, width=1, color='blue')
        
    #
    # Fine-tune figures, then draw it; 
    rmse_scatter_fig.subplots_adjust(hspace=0.05, wspace=0.05)
    tmp = plt.setp([a.get_xticklabels() for a in rmse_scatter_fig.axes], visible=False)
    tmp = plt.setp([a.get_yticklabels() for a in rmse_scatter_fig.axes], visible=False)
    make_ticklabels_invisible(rmse_scatter_fig)
    rmse_scatter_fig.suptitle("RMSE ScatterPlot: Inflation (X-axis) vs. Localization radius (Y-axis)")
    
    rhist_scatter_fig.subplots_adjust(hspace=0.05, wspace=0.05)
    tmp = plt.setp([a.get_xticklabels() for a in rhist_scatter_fig.axes], visible=False)
    tmp = plt.setp([a.get_yticklabels() for a in rhist_scatter_fig.axes], visible=False)
    make_ticklabels_invisible(rhist_scatter_fig)
    rhist_scatter_fig.suptitle("Talagrand ScatterPlot: Inflation (X-axis) vs. Localization radius (Y-axis)")
    
    # x, y lables:
    for i in xrange(num_of_infl_facs):
        inf_fac = _inflation_factors[i]
        rmse_scatter_axes[i, 0].set_ylabel(str(inf_fac))
        rhist_scatter_axes[i, 0].set_ylabel(str(inf_fac))
    print "num_of_loc_radii", num_of_loc_radii
    print "rmse_scatter_axes.shape", rmse_scatter_axes.shape
    for i in xrange(num_of_loc_radii):
        loc_rad = _localization_radii[i]
        rmse_scatter_axes[-1, i].set_xlabel(str(loc_rad))
        rhist_scatter_axes[-1, i].set_xlabel(str(loc_rad))

    #
    # Create an RMSE surface
    X = _inflation_factors
    Y = _localization_radii
    X, Y = np.meshgrid(X, Y)
    Z = np.empty_like(X)
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            inf_fac, loc_rad = X[i, j], Y[i, j]
            cand1 = np.where(inf_fac==inflation_factors)[0]
            cand2 = np.where(loc_rad==localization_radii)[0]
            
            Z[i, j] = rmse_averages[list(set.intersection(set(cand1), set(cand2)))[0]]
            
    # Plot RMSE-Average surface:
    rmse_surf_fig = plt.figure()
    ax = rmse_surf_fig.gca(projection='3d')    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    rmse_surf_fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('\nInflation Factor', fontsize=font_size)
    ax.set_ylabel('\nLocalization_radius', fontsize=font_size)
    ax.set_zlabel('Average RMSE', fontsize=font_size)
    rmse_surf_fig.suptitle("Localization Function: %s" % localization_function)

plt.show()




# ============================================================================================ #
#                           Clean executables and temporary modules                            #
# ============================================================================================ #
#
utility.clean_executable_files()
#
# ============================================================================================ #


