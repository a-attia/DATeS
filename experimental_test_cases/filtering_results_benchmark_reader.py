#!/usr/bin/env python

# This is a script to read the ouptput of EnKF, and HMC filter.

import os
import sys, getopt
import ConfigParser
import numpy as np
import scipy
import scipy.special as sp

import scipy.io as sio
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import re
import shutil
import pickle

import dates_setup
dates_setup.initialize_dates()
import dates_utility as utility
from filtering_results_reader_coupledLorenz import read_filter_output
from test_coupled_lorenz import enhance_plotter


# Example on how to change things recursively in output_dir_structure.txt file:
# find . -type f -name 'output_dir_structure.txt' -exec sed -i '' s/nfs2/Users/ {} +


def min_index_for_rows(X):
    """
    get the index (for each row) for where the values of X is minimum across this row; first index is returned if multiple occurances are found
    The uniqueness of this function is that it discards np.NaN

    Returns:
        indexes: of size X.shape[0]; with ith entry is the index that gives the minimum entry of the ith row

    """
    nx, ny = X.shape
    min_vals = np.nanmin(X, 1)
    indexes = []
    for ind in xrange(nx):
        locs = np.where(X[ind, :] == min_vals[ind])[0]
        if locs.size == 0:
            locs = np.array([0])  # just give it a zero index (either all is Nan or Inf
            # pass
        elif locs.size == 1:
            pass
        else:
            locs = np.array([locs[0]])
        indexes.append(locs)
    return np.asarray(indexes)


def min_index_for_columns(X):
    """
    get the index (for each column) for where the values of X is minimum accoross this column; first index is returned if multiple occurances are found
    The uniqueness of this function is that it discards np.NaN

    Returns:
        indexes: of size X.shape[1]; with ith entry is the index that gives the minimum entry of the ith column

    """
    nx, ny = X.shape
    min_vals = np.nanmin(X, 0)
    indexes = []
    for ind in xrange(ny):
        locs = np.where(X[:, ind] == min_vals[ind])[0]
        if locs.size == 0:
            locs = np.array([0])  # just give it a zero index (either all is Nan or Inf
            # pass
        elif locs.size == 1:
            pass
        else:
            locs = np.array([locs[0]])
        indexes.append(locs)
    return np.asarray(indexes)


def plot_results(inflation_factors,
                 localization_radii,
                 average_forecast_rmse_repo,
                 average_analysis_rmse_repo,
                 forecast_KL_dist_repo,
                 analysis_KL_dist_repo,
                 forecast_rank_hist_tracker,
                 analysis_rank_hist_tracker,
                 res_dir,
                 rmse_filter_threshold=np.inf,
                 kl_filter_threshold=np.inf,
                 cmap=None):
    """
    """
    plots_dir = os.path.join(res_dir, "PLOTS")
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # X and Y ticks, ticklabels:
    num_ticks = 10
    xticks = np.arange(0, inflation_factors.size, int(max(np.ceil(inflation_factors.size/float(num_ticks)), 1)) )
    xticklabels = [inflation_factors[i] for i in xticks]
    yticks = np.arange(0, localization_radii.size, int(max(np.ceil(localization_radii.size/float(num_ticks)), 1)) )
    if yticks[-1] < localization_radii.size-1:
        yticks = np.append(yticks, localization_radii.size-1)
    yticklabels = [localization_radii[i] for i in yticks]

    extent = [inflation_factors.min()-0.03, inflation_factors.max()+0.03,
                localization_radii.min()-0.03, localization_radii.max()+0.03]
    extent = None

    _average_forecast_rmse_repo = average_forecast_rmse_repo.copy()
    _average_forecast_rmse_repo[_average_forecast_rmse_repo>rmse_filter_threshold] = np.NaN
    _average_analysis_rmse_repo = average_analysis_rmse_repo.copy()
    _average_analysis_rmse_repo[_average_analysis_rmse_repo>rmse_filter_threshold] = np.NaN

    _forecast_KL_dist_repo = forecast_KL_dist_repo.copy()
    _forecast_KL_dist_repo[_forecast_KL_dist_repo>kl_filter_threshold] = np.NaN
    _analysis_KL_dist_repo = analysis_KL_dist_repo.copy()
    _analysis_KL_dist_repo[_analysis_KL_dist_repo>kl_filter_threshold] = np.NaN

    # Plot RMSEs
    fig = plt.figure(figsize=(6.5, 3), facecolor='white')
    ax = fig.add_subplot(111)
    cax = ax.imshow(_average_forecast_rmse_repo.T, aspect='auto', origin='lower', extent=extent, interpolation='nearest', cmap=cmap)
    ax.set_xlabel("Inflation factor")
    ax.set_ylabel("Localization radius")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.xaxis.set_ticks_position('bottom')
    fig.colorbar(cax)
    file_name = os.path.join(plots_dir, 'Average_Forecast_RMSE.pdf')
    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

    fig = plt.figure(figsize=(6.5, 3), facecolor='white')
    ax = fig.add_subplot(111)
    cax = ax.imshow(_average_analysis_rmse_repo.T, aspect='auto', origin='lower', extent=extent, interpolation='nearest', cmap=cmap)
    ax.set_xlabel("Inflation factor")
    ax.set_ylabel("Localization radius")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.xaxis.set_ticks_position('bottom')
    fig.colorbar(cax)
    file_name = os.path.join(plots_dir, 'Average_Analysis_RMSE.pdf')
    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

    fig = plt.figure(figsize=(9.5, 6), facecolor='white')
    ax = fig.add_subplot(111)
    cax = ax.imshow(_forecast_KL_dist_repo.T, aspect='auto', origin='lower', extent=extent, interpolation='nearest', cmap=cmap)
    ax.set_xlabel("Inflation Factor")
    ax.set_ylabel("Localization Radius")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.xaxis.set_ticks_position('bottom')
    fig.colorbar(cax)
    file_name = os.path.join(plots_dir, 'Forecast_KL_Distance.pdf')
    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

    fig = plt.figure(figsize=(9.5, 6), facecolor='white')
    ax = fig.add_subplot(111)
    cax = ax.imshow(_analysis_KL_dist_repo.T, aspect='auto', origin='lower', extent=extent, interpolation='nearest', cmap=cmap)
    ax.set_xlabel("Inflation Factor")
    ax.set_ylabel("Localization Radius")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.xaxis.set_ticks_position('bottom')
    fig.colorbar(cax)
    file_name = os.path.join(plots_dir, 'Analysis_KL_Distance.pdf')
    print("Saving: %s " % file_name)
    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

    # Create a 2D-Plot with lines
    # Closer rank histogram to unifirmty
    # REMARK: all repositories are of size Inflation-Facors-Size x Localization-Radii-Size
    num_infl_facs, num_loc_radii = _average_analysis_rmse_repo.shape
    marker_size = ms = 65
    # Analysis rank histogram Uniformity (according minimum KL distance to uniform distribution)
    if False:
        # best inflation_factors:
        kl_sorter = np.argsort(_analysis_KL_dist_repo, 0)
        # x_cord, y_cord = np.where(kl_sorter==0)  # INCORRECT! SEE UPDATE BELOW
        x_cord = rmse_sorter[0, :]
        y_cord = np.arange(np.size(_analysis_KL_dist_repo, 1))
        opt_infl = []
        for ind in xrange(num_loc_radii):
            infl = inflation_factors[x_cord[np.where(y_cord==ind)][0]]
            opt_infl.append(infl)
        opt_infl = np.asarray(opt_infl)
        # best localization radii
        kl_sorter = np.argsort(_analysis_KL_dist_repo, 1)
        # x_cord, y_cord = np.where(kl_sorter==0)  # INCORRECT! SEE UPDATE BELOW
        x_cord = np.arange(np.size(_analysis_KL_dist_repo, 0))
        y_cord = rmse_sorter[:, 0]
        opt_loc = []
        for ind in xrange(num_infl_facs):
            rad = localization_radii[y_cord[np.where(x_cord==ind)][0]]
            opt_loc.append(rad)
        opt_loc = np.asarray(opt_loc)
        #
    else:
        opt_infl_inds = min_index_for_columns(_analysis_KL_dist_repo)
        opt_infl = inflation_factors[opt_infl_inds]  # optimal inflation factor for each localization radius

        opt_loc_inds = min_index_for_rows(_analysis_KL_dist_repo)
        opt_loc = localization_radii[opt_loc_inds]  # optimal localization radius for each inflation factor

    # start plotting
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlabel("Inflation Factor")
    ax.set_ylabel("Localization Radius")
    ax.set_xticks(xticklabels)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    kl_label = r'KL(Rank hist | U)'
    ax.scatter(opt_infl, np.arange(localization_radii.size), ms, c='olivedrab', marker='>', label=kl_label, alpha=0.5)
    ax.scatter(inflation_factors, [np.where(opt_loc==i)[0][0] for i in opt_loc], ms, c='maroon', marker='^', label=' ', alpha=0.5)
    fig.legend(loc='upper center', fancybox=True, ncol=1, shadow=False, framealpha=0.9, bbox_to_anchor=(0.48, 0.98))
    #
    plt.minorticks_on()
    ax.grid(True, which='major', linestyle='-')
    ax.grid(True, which='minor', linestyle='-.')
    file_name = 'Analysis_Uniformity_Scatter.pdf'
    file_name = os.path.join(plots_dir, file_name)
    print("Saving: %s " % file_name)
    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

    #
    # RMSE minima
    #
    if False:  # TODO: Remove after Debugging...
        # best inflation_factors:
        rmse_sorter = np.argsort(_average_analysis_rmse_repo, 0)  # sorter of each column
        # x_cord, y_cord = np.where(rmse_sorter==0)  # INCORRECT! SEE UPDATE BELOW
        x_cord = rmse_sorter[0, :]
        y_cord = np.arange(np.size(_average_analysis_rmse_repo, 1))
        opt_infl = []
        for ind in xrange(num_loc_radii):
            infl = inflation_factors[x_cord[np.where(y_cord==ind)][0]]
            opt_infl.append(infl)
        opt_infl = np.asarray(opt_infl)
    
        # best localization radii
        rmse_sorter = np.argsort(_average_analysis_rmse_repo, 1)
        # x_cord, y_cord = np.where(rmse_sorter==0)  # INCORRECT! SEE UPDATE BELOW
        x_cord = np.arange(np.size(_average_analysis_rmse_repo, 0))
        y_cord = rmse_sorter[:, 0]
        opt_loc = []
        for ind in xrange(num_infl_facs):
            rad = localization_radii[y_cord[np.where(x_cord==ind)][0]]
            opt_loc.append(rad)
        opt_loc = np.asarray(opt_loc)
    
    else:
        opt_infl_inds = min_index_for_columns(_average_analysis_rmse_repo)
        opt_infl = inflation_factors[opt_infl_inds]  # optimal inflation factor for each localization radius

        opt_loc_inds = min_index_for_rows(_average_analysis_rmse_repo)
        opt_loc = localization_radii[opt_loc_inds]  # optimal localization radius for each inflation factor

    # start plotting
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlabel("Inflation Factor")
    ax.set_xticks(xticklabels)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Localization Radius")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    loc_radii_fl = opt_loc.copy()
    flags = np.isinf(loc_radii_fl)
    loc_radii_fl[flags] = np.nanmax(loc_radii_fl[np.invert(flags)]) + 1
    ax.scatter(opt_infl, np.arange(num_loc_radii), ms, c='olivedrab', marker='>', label='RMSE', alpha=0.5)
    ax.scatter(inflation_factors, [np.where(opt_loc==i)[0][0] for i in opt_loc], ms, c='maroon', marker='^', label=' ', alpha=0.5)
    plt.minorticks_on()
    ax.grid(True, which='major', linestyle='-')
    ax.grid(True, which='minor', linestyle='-.')
    fig.legend(loc='upper center', fancybox=True, ncol=1, shadow=False, framealpha=0.9, bbox_to_anchor=(0.48, 0.98))
    #
    file_name = 'Analysis_RMSE_Scatter.pdf'
    file_name = os.path.join(plots_dir, file_name)
    print("Saving: %s " % file_name)
    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
    

    # Plots the inflation factor that resulted in minimum average RMSE for a given localization radiu, highlighted on all inflation factors
    selected_radius_ind = min(5, localization_radii.size-1)
    selected_radius = localization_radii[selected_radius_ind]
    rmse_sorter = np.argsort(_average_analysis_rmse_repo, 0)
    fig = plt.figure(facecolor='white', figsize=(8.10, 3.1))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Inflation factor")
    ax.set_ylabel("Average RMSE")
    #
    ax.plot(inflation_factors, _average_analysis_rmse_repo[:, selected_radius_ind], '-d', label=r'$L=%3.2f$'%selected_radius)
    ax.grid(True, which='major', axis='both')
    #
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Mark the best inflation factor
    infl_loc = opt_infl_inds[selected_radius_ind][0]
    ax.plot(inflation_factors[infl_loc], _average_analysis_rmse_repo[infl_loc, selected_radius_ind], 'ro', ms = 12)
    lbl = r"$\lambda^{\rm opt}=%3.2f;\, RMSE=%4.3f $"%(inflation_factors[infl_loc], average_analysis_rmse_repo[infl_loc, selected_radius_ind])
    ax.plot(inflation_factors[infl_loc], _average_analysis_rmse_repo[infl_loc, selected_radius_ind], 'g*', ms = 12, label=lbl)
    ax.arrow(x=inflation_factors[infl_loc], y=ylim[0], dx=0, dy=_average_analysis_rmse_repo[infl_loc, selected_radius_ind]-ylim[0], color='maroon',
             fc='maroon', ec='maroon', alpha=0.75)
    ax.arrow(x=inflation_factors[infl_loc], y=_average_analysis_rmse_repo[infl_loc, selected_radius_ind], dx=xlim[0]-inflation_factors[infl_loc], dy=0, color='maroon',
             fc='maroon', ec='maroon', alpha=0.75)

    plt.minorticks_on()
    ax.grid(True, which='major', linestyle='-')
    ax.grid(True, which='minor', linestyle='-.')
    plt.legend(loc='best', framealpha=0.65)
    #
    file_name = 'AnalysisRMSE_Selective_Loc_BestRMSE.pdf'
    file_name = os.path.join(plots_dir, file_name)
    print("Saving: %s " % file_name)
    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
    #

    # Plots the average RMSE corresponding to multiple localization radii for the best inflation factor
    selected_fac_ind = inflation_factors.size-1
    selected_infl_fac = inflation_factors[selected_fac_ind]
    rmse_sorter = np.argsort(_average_analysis_rmse_repo, 1)
    rad_loc = opt_loc_inds[selected_fac_ind][0]
    fig = plt.figure(facecolor='white', figsize=(6.5, 3))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Localization radius")
    ax.set_ylabel("Average RMSE")
    ax.plot(localization_radii[rad_loc], _average_analysis_rmse_repo[selected_fac_ind, rad_loc], 'ro', ms = 12)
    plt.minorticks_on()
    ax.grid(True, which='major', linestyle='-')
    ax.grid(True, which='minor', linestyle='-.')
    file_name = "AnalysisRMSE_Selective_inflfac_BestRMSE.pdf"
    file_name = os.path.join(plots_dir, file_name)
    print("Saving Plot: %s" % file_name)
    plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


    #
    fig = plt.figure(facecolor='white', figsize=(12, 6))
    ax = fig.add_subplot(111)
    #
    ax.set_xlabel("Inflation Factor")
    ax.set_ylabel("Average RMSE")
    #
    line_colors = utility.unique_colors(localization_radii.size)
    for selected_radius_ind in xrange(localization_radii.size):
        selected_radius = localization_radii[selected_radius_ind]
        infl_loc = opt_infl_inds[selected_radius_ind][0]
        #
        color = line_colors[selected_radius_ind]
        lbl = r"$L=%3.2f; \, \lambda^{\rm opt}=%3.2f$"%(selected_radius, inflation_factors[infl_loc])
        ax.plot(inflation_factors, _average_analysis_rmse_repo[:, selected_radius_ind], '-o', color=color, label=lbl)
        ax.plot(inflation_factors[infl_loc], _average_analysis_rmse_repo[infl_loc, selected_radius_ind], '*', color=color, ms = 12)
    plt.legend(loc='best', ncol=3, framealpha=0.65, fontsize=8)
    plt.minorticks_on()
    ax.grid(True, which='major', linestyle='-')
    ax.grid(True, which='minor', linestyle='-.')
    #
    file_name = 'AnalysisRMSE_BestInflation_foreach_LocRadius.pdf'
    file_name = os.path.join(plots_dir, file_name)
    print("Saving: %s " % file_name)
    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

    # Create a figure with subplots containing Rank histograms of inflation vs localization
    # Each entry of the following variables is a tuple (ranks_freq, ranks_rel_freq, bins_bounds)
    print("\n****\nPlotting an array of Forecast rank histogram \n...This will take a few minutes...\n ***\n")
    subsize = 2.25  # subplot size in inches
    num_infl_facs = len(inflation_factors)
    num_loc_radii = len(localization_radii)
    fig_size = (min(30, subsize*num_loc_radii), min(23, subsize*num_infl_facs))
    fig, axes = plt.subplots(num_infl_facs, num_loc_radii, sharex=True, sharey=True, facecolor='white', figsize=fig_size)
    for i, inflation_factor in enumerate(inflation_factors):
        for j, localization_radius in enumerate(localization_radii):
            # Array of forecast histograms:
            _, ranks_rel_freq, bins_bounds = forecast_rank_hist_tracker[i, j]
            # print(i, j, inflation_factor, localization_radius, ranks_rel_freq)
            ax = axes[i, j]
            ax.bar(bins_bounds, ranks_rel_freq, width=1, color='green', edgecolor='black')
            ax.set_xlim(bins_bounds[0]-0.5, bins_bounds[-1]+0.5)
            ax.set_ylim(0, np.max(ranks_rel_freq))
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            # Adjust ticks and labels
            if j == 0:
                ax.set_ylabel(r"$\lambda=%3.2f$"%inflation_factor, fontsize=5)
            if i == num_infl_facs-1:
                ax.set_xlabel(r"$L=%4.3f$" % localization_radius, fontsize=7)
                #
    # Adjust subplots
    fig.subplots_adjust(wspace=0.01, hspace=0.02)

    file_name = 'Forecast_Rank_histogram_Array.pdf'
    file_name = os.path.join(plots_dir, file_name)
    print("Saving: %s " % file_name)
    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

    # Close all open figures
    plt.close('all')


def get_args(input_args, recollect_results=False, plot_individual_experiment_results=False, output_repository_dir='./CoupledLorenzResults'):
    """
    Get command line arguments
    """

    try:
        opts, args = getopt.getopt(input_args,"hd:r:i:",["doutputdir=","rrecollect-results=","iindiv-plots="])
    except getopt.GetoptError:
        print 'filtering_results_benchmark_reader.py -d <output_repository_dir> -r <recollect_results> -i <plot_individual_experiment_results>'
        sys.exit(2)

    # Default Values

    for opt, arg in opts:
        if opt == '-h':
            print 'filtering_results_benchmark_reader.py -d <output_repository_dir> -r <recollect_results> -i <plot_individual_experiment_results>'
            sys.exit()
        elif opt in ("-r", "--rrecollect-results"):
            recollect_results = utility.str2bool(arg)
        elif opt in ("-d", "--doutputdir"):
            output_repository_dir = arg
        elif opt in ("-i", "--iindiv-plots"):
            plot_individual_experiment_results = utility.str2bool(arg)

    return output_repository_dir, recollect_results, plot_individual_experiment_results

#
if __name__ == '__main__':

    read_from_pickle = True
    plot_individual_experiment_results = False
    overwrite_individual_results_plots = False
    target_results_repository = './CoupledLorenzResults'

    init_time_ind = None
    rmse_filter_threshold = 10.0
    kl_filter_threshold = 30

    output_repository_dir, recollect_results, plot_individual_experiment_results = get_args(sys.argv[1: ],
                                                                                            recollect_results=not read_from_pickle,
                                                                                            plot_individual_experiment_results=plot_individual_experiment_results,
                                                                                            output_repository_dir=target_results_repository,
                                                                                           )
    joint_results_file_name = os.path.join(output_repository_dir, 'JointResults.pickle')
    #
    # =====================================================================
    # General Plotting settings
    # =====================================================================
    log_scale = True
    enhance_plotter()

    # set colormap:
    colormap = None  # vs 'jet'
    # plt.set_cmap(colormap)
    interpolation = None  # 'bilinear'
    # =====================================================================

    if not read_from_pickle:
        list_of_dirs = utility.get_list_of_subdirectories(output_repository_dir, ignore_root=True, return_abs=True, ignore_special=True, recursive=False)

        # get all inflation factors and localization radii
        inflation_factors  = []
        localization_radii = []
        for res_dir in list_of_dirs:
            # Get inflation factor and localization radius from folder name
            _, dirname = os.path.split(res_dir)
            l = dirname.split('_')
            try:
                inflation_factor = float(l[-3])
                localization_radius = float(l[-1])
            except(IndexError):
                continue
            inflation_factors.append(inflation_factor)
            localization_radii.append(localization_radius)
            #
        inflation_factors = list(set(inflation_factors))
        inflation_factors.sort()
        inflation_factors = np.array(inflation_factors)
        localization_radii = list(set(localization_radii))
        localization_radii.sort()
        localization_radii = np.array(localization_radii)

        # Create repositories for results:
        average_forecast_rmse_repo = np.empty((inflation_factors.size, localization_radii.size))
        average_forecast_rmse_repo[...] = np.nan
        average_analysis_rmse_repo = average_forecast_rmse_repo.copy()
        forecast_KL_dist_repo = average_forecast_rmse_repo.copy()
        analysis_KL_dist_repo = average_forecast_rmse_repo.copy()
        forecast_rank_hist_tracker = [[None]*(localization_radii.size)]*(inflation_factors.size)
        analysis_rank_hist_tracker = [[None]*(localization_radii.size)]*(inflation_factors.size)

        #
        sep = "*" * 80
        for res_dir in list_of_dirs:

            print("\n%s\nCollecting Results and Plotting: '%s'\n%s\n" % (sep, res_dir, sep))
            out_dir_tree_structure_file = os.path.join(res_dir, 'output_dir_structure.txt')

            _, dirname = os.path.split(res_dir)
            l = dirname.split('_')
            try:
                inflation_factor = float(l[-3])
                localization_radius = float(l[-1])
            except(IndexError):
                continue
            infl_ind = np.where(inflation_factors == inflation_factor)[0][0]
            loc_ind  = np.where(localization_radii == localization_radius)[0][0]

            #
            # =====================================================================
            # Start reading the output of the assimilation process
            # =====================================================================
            if plot_individual_experiment_results:
                print("Plotting experiment results...")
                cmd = "python filtering_results_reader_coupledLorenz.py -f %s -o %s " % (out_dir_tree_structure_file, overwrite_individual_results_plots)
                print("Executing: %s " % cmd)
                os.system(cmd)

            try:  # for incomplete results
                cycle_prefix, num_cycles, reference_states, forecast_ensembles, forecast_means, analysis_ensembles, \
                analysis_means, observations, forecast_times, analysis_times, observations_times, \
                forecast_rmse, analysis_rmse, filter_configs, gmm_results, model_configs, mardiaTest_results,  \
                inflation_opt_results, localization_opt_results = read_filter_output(out_dir_tree_structure_file)

                if init_time_ind is None:
                    init_time_ind = len(forecast_times) / 3 * 2
            except:
                continue
            #
            try:
                filter_name
                model_name
            except NameError:
                filter_name = filter_configs['filter_name']
                filter_name = filter_name.replace('_', ' ')
                model_name = model_configs['model_name']
            try:
                state_size = model_configs['state_size']
            except KeyError:
                state_size = np.size(forecast_ensembles, 0)

            # Get reference states and initial ensemble for free run:
            # Load reference states:
            try:
                reference_states = np.load(os.path.join(res_dir, 'Reference_Trajectory.npy'))  # generated and saved offline for this experiment
            except(IOError):
                continue
            initial_ensemble = np.load(os.path.join(res_dir, 'Initial_Ensemble.npy'))  # generated and saved offline for this experiment
            initial_mean = np.mean(initial_ensemble, 1)
            forecast_rmse[0] = analysis_rmse[0] = np.sqrt(np.linalg.norm((initial_mean-reference_states[:state_size, 0]), 2)/ state_size)

            reference_states = reference_states[ :state_size, np.size(reference_states, 1)-num_cycles:]
            try:  # for still-running experiments with incomplete results
                for i in xrange(num_cycles):
                    forecast_rmse[i+1] = np.sqrt(np.linalg.norm((forecast_means[:, i]-reference_states[:, i]), 2)/ state_size)
                    analysis_rmse[i+1] = np.sqrt(np.linalg.norm((analysis_means[:, i]-reference_states[:, i]), 2)/ state_size)
            except(IndexError):
                continue
            # average RMSEs
            average_forecast_rmse_repo[infl_ind, loc_ind] = forecast_rmse[init_time_ind: ].mean()
            average_analysis_rmse_repo[infl_ind, loc_ind] = analysis_rmse[init_time_ind: ].mean()
            #
            # =====================================================================


            #
            # =====================================================================
            # Plot Rank Histograms for forecast and analysis ensemble
            # =====================================================================
            #
            ignore_indexes = None
            if forecast_ensembles is not None:
                f_out = utility.rank_hist(forecast_ensembles,
                                  reference_states, first_var=0,
                                  last_var=None,
                                  var_skp=5,
                                  draw_hist=False,
                                  hist_type='relfreq',
                                  first_time_ind=init_time_ind,
                                  last_time_ind=None,
                                  time_ind_skp=1,
                                #   hist_title='forecast rank histogram',
                                  hist_title=None,
                                  hist_max_height=None,
                                  ignore_indexes=ignore_indexes
                          )
            ranks_freq, ranks_rel_freq, bins_bounds , fig_hist = f_out[:]
            forecast_rank_hist_tracker[infl_ind][loc_ind] = (ranks_freq, ranks_rel_freq, bins_bounds)
            forecast_rank_hist_kl_dist, _ = utility.calc_kl_dist(ranks_freq, bins_bounds)
            forecast_KL_dist_repo[infl_ind, loc_ind] = forecast_rank_hist_kl_dist
            #
            if analysis_ensembles is not None:
                a_out = utility.rank_hist(analysis_ensembles,
                                  reference_states, first_var=0,
                                  last_var=None,
                                  var_skp=5,
                                  draw_hist=False,
                                  hist_type='relfreq',
                                  first_time_ind=init_time_ind,
                                  last_time_ind=None,
                                  time_ind_skp=1,
                                #   hist_title='analysis rank histogram',
                                  hist_title=None,
                                  hist_max_height=None,
                                  ignore_indexes=ignore_indexes
                          )
            ranks_freq, ranks_rel_freq, bins_bounds , fig_hist = a_out[:]
            analysis_rank_hist_tracker[infl_ind][loc_ind] = (ranks_freq, ranks_rel_freq, bins_bounds)
            analysis_rank_hist_kl_dist, _ = utility.calc_kl_dist(ranks_freq, bins_bounds)
            analysis_KL_dist_repo[infl_ind, loc_ind] = analysis_rank_hist_kl_dist

            # =====================================================================

            # back to original folder
            del cycle_prefix, num_cycles, reference_states, forecast_ensembles, forecast_means, analysis_ensembles, \
            analysis_means, observations, forecast_times, analysis_times, observations_times, \
            forecast_rmse, analysis_rmse, filter_configs, gmm_results, model_configs, mardiaTest_results,  \
            inflation_opt_results, localization_opt_results

            # =====================================================================

        #
        # =====================================================================
        # Collective RMSE Plot, RMSE average & boxplot (after spinup),  and collective
        # =====================================================================
        #
        # Save collective RMSE and rank histogram results
        file_name = joint_results_file_name
        results_dict = dict(inflation_factors=inflation_factors,
                            localization_radii=localization_radii,
                            average_forecast_rmse_repo=average_forecast_rmse_repo,
                            average_analysis_rmse_repo=average_analysis_rmse_repo,
                            forecast_KL_dist_repo=forecast_KL_dist_repo,
                            analysis_KL_dist_repo=analysis_KL_dist_repo,
                            forecast_rank_hist_tracker=np.asarray(forecast_rank_hist_tracker),
                            analysis_rank_hist_tracker=np.asarray(analysis_rank_hist_tracker),
                            )
        pickle.dump(results_dict, open(file_name, 'wb'))

    else:
        try:
            cont = pickle.load(open(joint_results_file_name, 'rb'))
        except(IOError):
            print("Failed to load results from file %s " % joint_results_file_name)
            raise
        inflation_factors = cont['inflation_factors']
        localization_radii = cont['localization_radii']
        average_forecast_rmse_repo = cont['average_forecast_rmse_repo']
        average_analysis_rmse_repo = cont['average_analysis_rmse_repo']
        forecast_KL_dist_repo = cont['forecast_KL_dist_repo']
        analysis_KL_dist_repo = cont['analysis_KL_dist_repo']
        forecast_rank_hist_tracker = cont['forecast_rank_hist_tracker']
        analysis_rank_hist_tracker = cont['analysis_rank_hist_tracker']

    # =====================================================================
    plot_results(inflation_factors,
                 localization_radii,
                 average_forecast_rmse_repo,
                 average_analysis_rmse_repo,
                 forecast_KL_dist_repo,
                 analysis_KL_dist_repo,
                 forecast_rank_hist_tracker,
                 analysis_rank_hist_tracker,
                 output_repository_dir,
                 rmse_filter_threshold,
                 kl_filter_threshold,
                 cmap=colormap)




