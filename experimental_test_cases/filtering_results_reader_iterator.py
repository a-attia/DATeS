#!/usr/bin/env python

# Loop by recalling filtering results reader for all results directory in a given path, and save results

import os
import sys
import ConfigParser
import numpy as np
import scipy.io as sio
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import re
import shutil
try:
    import cpickle
except:
    import cPickle as pickle

import dates_setup
dates_setup.initialize_dates()

import dates_utility as utility
#

from filtering_results_reader import read_filter_output

# default values fro filtering_results_reader
out_dir_tree_structure_file = 'Results/Filtering_Results/output_dir_structure.txt'
standard_results_dir = os.path.abspath("Results/Filtering_Results")

# Example on how to change things recursively in output_dir_structure.txt file:
# find . -type f -name 'output_dir_structure.txt' -exec sed -i '' s/nfs2/Users/ {} +


#
def read_and_plot(ref_min_rmse_infl_fac,
                  ref_min_kl_infl_fac,
                  adaptive_inflation,
                  adaptive_localization,
                  compare_to_EnKF,
                  _filter_name_compare,
                  l_norm=1,
                  add_fitted_Beta_toRhist=False,
                  overwrite_plots=False):
    #
    # =====================================================================
    # Adjust RESULTS PATHs  (Adjust to point to path of results' dirs)
    # =====================================================================

    # Root path to objective results directory; (no recursion)
    if adaptive_inflation is adaptive_localization is True:
        raise ValueError("Both localization and inflation set to Active is a case yet to be implemented!")
    elif adaptive_inflation:
        target_results_dir = "Results/OED_EnKF_Results/OED_Adaptive_Inflation/L%d_Norm" % l_norm
    elif adaptive_localization:
        target_results_dir = "Results/OED_EnKF_Results/OED_Adaptive_Localization/L%d_Norm" % l_norm
    else:
        target_results_dir = "Results/EnKF_Results/%s" % _filter_name_compare

    # Plots directory settings
    plots_dir_name = "Plots" # name of directory (under results_dir) to save plots in

    # Ignore assimilation cycles (e.g. replace results with Nans), where the optimizer fails
    remove_optimizer_failures = True

    #
    # =====================================================================
    # General Plotting settings
    # =====================================================================
    # Font, and Texts:
    font_size = 22
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : font_size}
    #
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=True)

    # Drawing Lines:
    line_width = 2
    marker_size = 4

    # set colormap:
    colormap = None  # vs 'jet'
    # plt.set_cmap(colormap)
    interpolation = None  # 'bilinear'

    # format of saved figures
    plots_format = 'pdf'

    # =====================================================================

    #
    # =====================================================================
    # OED-EnKF additional settings
    # =====================================================================
    # inflation lower and upper bounds for plotting:
    infl_lb = 0.0
    infl_ub = 3.0

    # localization radii lower and upper bounds for plotting:
    loc_lb = None
    loc_ub = None

    # =====================================================================

    #
    # =====================================================================
    # Additional settings
    # =====================================================================
    plots_init_time_index = 100
    use_smoothed_opt_results = False
    #
    # bound_axes (use for producing final results with unified axes and limits:
    bound_axes = False

    # Show means on boxplots:
    means_on_boxplots = False
    # =====================================================================


    # =====================================================================
    # Read EnKF empirical inflation results:
    # =====================================================================
    if ref_min_rmse_infl_fac is not None and ref_min_kl_infl_fac is not None:
        enkf_results_dir = os.path.join("Results/EnKF_Results", _filter_name_compare)
        filter_optimal_inflation_values = None  # this is not needed in this case
    else:
        if compare_to_EnKF:
            enkf_results_dir = os.path.join("Results/EnKF_Results", _filter_name_compare)
            empirical_inflation_results_file = os.path.join(enkf_results_dir, "%s_optimal_inflation_values.pickle" % _filter_name_compare)
            if not os.path.isfile(empirical_inflation_results_file):
                compare_to_EnKF = False
        else:
            empirical_inflation_results_file = None

        if compare_to_EnKF:
            filter_optimal_inflation_values = pickle.load(open(empirical_inflation_results_file, 'rb'))
        else:
            filter_optimal_inflation_values = None
    # =====================================================================


    # =====================================================================
    # Get list of directories, and start reading results:
    # =====================================================================
    list_of_dirs = utility.get_list_of_subdirectories(target_results_dir, ignore_root=True, return_abs=False, ignore_special=True, recursive=False)
    #
    sep = "*" * 80
    for res_dir in list_of_dirs:
        print("\n%s\nCollecting Results and Plotting: '%s'\n%s\n" % (sep, res_dir, sep))
        res_dir = os.path.abspath(res_dir)

        if not os.path.isdir(res_dir):
            print("This doesn't make any sense; couldn't find the directory '%s'" % res_dir)

        plots_dir = os.path.join(res_dir, plots_dir_name)
        if os.path.isdir(plots_dir):
            if overwrite_plots:
                # Clenup; not necessary, but just in-case
                if False:
                    shutil.rmtree(plots_dir, ignore_errors=True)
                    os.makedirs(plots_dir)
                else:
                    # Just overwrite each file
                    pass
            else:
                print("found plots dir [%s] SKIPPING ..." % plots_dir)
                continue
        else:
            os.makedirs(plots_dir)
        #
        # =====================================================================
        # Start reading the output of the assimilation process
        # =====================================================================
        os.rename(res_dir, standard_results_dir)
        #
        cycle_prefix, num_cycles, reference_states, forecast_ensembles, forecast_means, analysis_ensembles, \
        analysis_means, observations, forecast_times, analysis_times, observations_times, \
        forecast_rmse, analysis_rmse, filter_configs, gmm_results, model_configs, mardiaTest_results,  \
        inflation_opt_results, localization_opt_results = read_filter_output(out_dir_tree_structure_file)
        #
        os.rename(standard_results_dir, res_dir)

        # Reading best results:
        ensemble_size = filter_configs['ensemble_size']
        try:
            localization_function = filter_configs['localization_function']
        except:
            if 'Gauss' in res_dir:
                localization_function = 'Gauss'
            else:
                localization_function = 'Gaspari-Cohn'

        if compare_to_EnKF:
            if ref_min_rmse_infl_fac is not None and ref_min_kl_infl_fac is not None:
                min_rmse_infl_fac = ref_min_rmse_infl_fac
                min_kl_infl_fac = ref_min_kl_infl_fac
            else:
                try:
                    ens_size_ind = np.where(filter_optimal_inflation_values[localization_function]['ensemble_size_pool'] == ensemble_size)[0][0]
                except(IndexError):
                    print("Ensemble size %d is not tested in the empirical experiments." % ensemble_size)
                    print("Only available ensemble sizes are: ")
                    print(filter_optimal_inflation_values[localization_function]['ensemble_size_pool'])
                    ens_size_ind = None
                finally:
                    if ensemble_size is None:
                        min_rmse_infl_fac = None
                        min_kl_infl_fac = None
                    else:
                        min_rmse_infl_fac = filter_optimal_inflation_values[localization_function]['optimal_inflations'][0][ens_size_ind]
                        min_kl_infl_fac = filter_optimal_inflation_values[localization_function]['optimal_inflations'][1][ens_size_ind]
            #
            postfix = "%s_Results_lorenz_Ensemble_%s_%s_InflationFactor_%s" % (_filter_name_compare, ensemble_size, localization_function, str.replace("%3.2f" % min_rmse_infl_fac, '.', '_'))
            min_rmse_dir = os.path.join(enkf_results_dir, postfix)
            rmse_file_path = os.path.join(min_rmse_dir, "Filter_Statistics/rmse.dat")
            rmse_file_contents = np.loadtxt(rmse_file_path, skiprows=2)
            ref_min_rmse = rmse_file_contents[:, 4]

            if min_kl_infl_fac == min_rmse_infl_fac:
                ref_min_kl = ref_min_rmse
            else:
                postfix = "%s_Results_lorenz_Ensemble_%s_%s_InflationFactor_%s" % (_filter_name_compare, ensemble_size, localization_function, str.replace("%3.2f" % min_kl_infl_fac, '.', '_'))
                min_kl_dir = os.path.join(enkf_results_dir, postfix)
                rmse_file_path = os.path.join(min_kl_dir, "Filter_Statistics/rmse.dat")
                rmse_file_contents = np.loadtxt(rmse_file_path, skiprows=2)
                ref_min_kl = rmse_file_contents[:, 4]
        else:
            # Not a comparative study; just load filter results and plot
            pass

        #
        filter_name = filter_configs['filter_name'].replace('_', ' ')
        model_name = model_configs['model_name']
        try:
            state_size = model_configs['state_size']
        except KeyError:
            state_size = np.size(forecast_ensembles, 0)
        #
        # print(reference_states, forecast_ensembles, forecast_means, analysis_ensembles, analysis_means, observations)

        # =====================================================================
        if plots_init_time_index is None:
            plots_init_time_index = 0
        if plots_init_time_index >= len(analysis_times):
            print("plots_init_time_index %d is invalid; resetting to zero")
            plots_init_time_index = 0

        #
        # =====================================================================
        # Plot RMSE
        # =====================================================================
        # Full RMSE:
        #
        fig1 = plt.figure(facecolor='white')
        if compare_to_EnKF:
                if min_kl_infl_fac == min_rmse_infl_fac:
                    plt.plot(analysis_times[plots_init_time_index: ], ref_min_rmse[plots_init_time_index: ], 'r--o', linewidth=line_width, label='Optimal DEnKF')
                else:
                    plt.plot(analysis_times[plots_init_time_index: ], ref_min_rmse[plots_init_time_index: ], 'r--o', linewidth=line_width, label='%s: min average RMSE' % _filter_name_compare)
                    plt.plot(analysis_times[plots_init_time_index: ], ref_min_kl[plots_init_time_index: ], 'g-.*', linewidth=line_width, label=r'%s: min KL-D to $\mathcal{U}$ ' % _filter_name_compare)
        else:
            plt.plot(analysis_times[plots_init_time_index: ], forecast_rmse[plots_init_time_index: ], 'g-.*', linewidth=line_width, label='Forecast' )
        plt.plot(analysis_times[plots_init_time_index: ], analysis_rmse[plots_init_time_index: ], 'bd-', linewidth=line_width, label='OED-%s' % _filter_name_compare)
        #
        # Set lables and title
        plt.xlabel("Time (assimilation cycles)")
        plt.ylabel('RMSE', fontsize=font_size, fontweight='bold')
        # plt.title('RMSE results for the model: %s' % model_name)
        #
        times = forecast_times[plots_init_time_index: ]
        skips = max(1, len(forecast_times) / 10)   # - (len(forecast_times) % 10)
        xlabels = [times[i] for i in xrange(0, len(times), skips)]
        ticklabels = skips*np.arange(len(xlabels)) + plots_init_time_index
        plt.xticks(xlabels, ticklabels)
        # show the legend, show the plot
        plt.legend(loc='best')
        plt.draw()
        #
        file_name = os.path.join(plots_dir, "RMSE.%s" % plots_format)
        print("Plotting: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')


        # Log-scale RMSE plot
        #
        fig1 = plt.figure(facecolor='white')
        if compare_to_EnKF:
            if ref_min_rmse is not None:
                if min_kl_infl_fac == min_rmse_infl_fac:
                    plt.semilogy(analysis_times[plots_init_time_index: ], ref_min_rmse[plots_init_time_index: ], 'r--o', linewidth=line_width, label='Optimal %s'%_filter_name_compare)
                else:
                    plt.semilogy(analysis_times[plots_init_time_index: ], ref_min_rmse[plots_init_time_index: ], 'r--o', linewidth=line_width, label='%s: min average RMSE' % _filter_name_compare)
                    plt.semilogy(analysis_times[plots_init_time_index: ], ref_min_kl[plots_init_time_index: ], 'g-.*', linewidth=line_width, label=r'%s: min KL-D to $\mathcal{U}$ ' % _filter_name_compare)
            else:
                pass
        else:
            plt.semilogy(analysis_times[plots_init_time_index: ], forecast_rmse[plots_init_time_index: ], 'g-.*', linewidth=line_width, label='Forecast' )
        plt.semilogy(analysis_times[plots_init_time_index: ], analysis_rmse[plots_init_time_index: ], 'bd-', linewidth=line_width, label='OED-%s' % _filter_name_compare)  # label=filter_name
        #
        # Set lables and title
        plt.xlabel("Time (assimilation cycles)")
        plt.ylabel('log-RMSE', fontsize=font_size, fontweight='bold')
        # plt.title('RMSE results for the model: %s' % model_name)
        #
        times = forecast_times[plots_init_time_index: ]
        skips = max(1, len(forecast_times) / 10)   # - (len(forecast_times) % 10)
        xlabels = [times[i] for i in xrange(0, len(times), skips)]
        ticklabels = skips*np.arange(len(xlabels)) + plots_init_time_index
        plt.xticks(xlabels, ticklabels)
        # show the legend, show the plot
        plt.legend(loc='best')
        plt.draw()
        #
        log_postfix = '_logscale'
        file_name = os.path.join(plots_dir, "RMSE%s.%s" % (log_postfix, plots_format))
        print("Plotting: %s" % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

        # =====================================================================


        #
        # =====================================================================
        # Adaptive Inflation and/or localization results
        # =====================================================================
        #

        if len(inflation_opt_results) > 0:
            if len(analysis_times)-1 != len(inflation_opt_results):
                print("Check len(inflation_opt_results), and len(forecast_times)")
                print("len(inflation_opt_results)", len(inflation_opt_results))
                print("len(forecast_times)", len(analysis_times)-1)
                raise ValueError

            inflation_stats = np.zeros((5, len(inflation_opt_results)))
            # first row:  optimal objective (without regularization)
            # second row: optimal objective (with regularization)
            # third row:  L2 norm of optimal solution
            # fourth row: average inflation factor
            # fifth row:  standard deviation inflation factor
            #
            optimal_sols = []  # smoothed and rounded solutions
            original_optimal_sols = []
            #
            for i in xrange(len(inflation_opt_results)):
                opt_x = inflation_opt_results[i][1]['opt_x']
                orig_opt_x = inflation_opt_results[i][1]['orig_opt_x']
                try:
                    success = inflation_opt_results[i][1]['success']
                except(KeyError):
                    try:
                        success = not inflation_opt_results[i][1]['opt_info_d']['warnflag']
                    except:
                        try:
                            success = inflation_opt_results[i][1]['full_opt_results']['success']
                        except:
                            success = True  # Failed to retrieve the convergence flag; just take the value
                #
                if remove_optimizer_failures and not success:
                    opt_x[:] = np.nan
                    orig_opt_x[:] = np.nan
                    post_trace = np.nan
                    min_f = np.nan
                    l2_norm = np.nan
                    avrg = np.nan
                    stdev = np.nan
                else:
                    post_trace = inflation_opt_results[i][1]['post_trace']
                    min_f = inflation_opt_results[i][1]['min_f']
                    if not use_smoothed_opt_results:
                        l2_norm = np.linalg.norm(orig_opt_x)
                        avrg = np.mean(orig_opt_x)
                        stdev = np.std(orig_opt_x)
                    else:
                        l2_norm = np.linalg.norm(opt_x)
                        avrg = np.mean(opt_x)
                        stdev = np.std(opt_x)
                #
                optimal_sols.append(opt_x)
                original_optimal_sols.append(orig_opt_x)
                inflation_stats[:, i] = post_trace, min_f, l2_norm, avrg, stdev
                #

            _, ax_adap_inf = plt.subplots(facecolor='white')
            #
            ax_adap_inf.plot(analysis_times[plots_init_time_index+1:], inflation_stats[0, plots_init_time_index:], 'bd-', linewidth=line_width, label=r"$Trace(\widetilde{\mathbf{A}})$")
            # ax_adap_inf.plot(analysis_times[plots_init_time_index+1:], inflation_stats[1, plots_init_time_index:], 'gd-', linewidth=line_width, label="optimal objective")
            # ax_adap_inf.plot(analysis_times[1:], inflation_stats[2, :], 'r-.', linewidth=line_width, label=r"$\|\mathbf{L}\|$")
            ax_adap_inf.plot(analysis_times[plots_init_time_index+1:], inflation_stats[3, plots_init_time_index:], 'c--', linewidth=line_width, label=r"$\overline{\mathbf{\alpha}}$")
            ax_adap_inf.plot(analysis_times[plots_init_time_index+1:], inflation_stats[4, plots_init_time_index:], 'm--', linewidth=line_width, label=r"$\sigma_{\mathbf{\alpha}}$")
            #
            # Set lables and title
            ax_adap_inf.set_xlabel("Time (assimilation cycles)")
            # ax_adap_inf.set_title('OED-Adaptive Inflation results for the model: %s' % model_name)
            ax_adap_inf.set_xlim(analysis_times[plots_init_time_index], analysis_times[-1])
            #
            times = forecast_times[plots_init_time_index: ]
            skips = max(1, len(times) / 10)   # - (len(forecast_times) % 10)
            xticks = [times[i] for i in xrange(0, len(times), skips)]
            xticklabels = skips*np.arange(len(xticks)) + plots_init_time_index
            plt.xticks(xticks, xticklabels)
            # show the legend, show the plot
            plt.legend(loc='best')
            #
            plt.draw()
            #
            file_name = os.path.join(plots_dir, "InflationOED_Objective.%s" % plots_format)
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            #
            _, ax_adap_inf = plt.subplots(facecolor='white')
            #
            ax_adap_inf.semilogy(analysis_times[plots_init_time_index+1:], inflation_stats[0, plots_init_time_index:], 'bd-', linewidth=line_width, label=r"$Trace(\widetilde{\mathbf{A}})$")
            # ax_adap_inf.semilogy(analysis_times[plots_init_time_index+1:], inflation_stats[1, plots_init_time_index:], 'gd-', linewidth=line_width, label="optimal objective")
            # ax_adap_inf.semilogy(analysis_times[1:], inflation_stats[2, :], 'r-.', linewidth=line_width, label=r"$\|\mathbf{\alpha}\|$")
            ax_adap_inf.semilogy(analysis_times[plots_init_time_index+1:], inflation_stats[3, plots_init_time_index:], 'c--', linewidth=line_width, label=r"$\overline{\mathbf{\alpha}}$")
            ax_adap_inf.semilogy(analysis_times[plots_init_time_index+1:], inflation_stats[4, plots_init_time_index:], 'm--', linewidth=line_width, label=r"$\sigma_{\mathbf{\alpha}}$")
            #
            # Set lables and title
            ax_adap_inf.set_xlabel("Time (assimilation cycles)")
            # ax_adap_inf.set_title('OED-Adaptive Inflation results for the model: %s' % model_name)
            ax_adap_inf.set_xlim(analysis_times[plots_init_time_index], analysis_times[-1])
            #
            times = forecast_times[plots_init_time_index: ]
            skips = max(1, len(times) / 10)   # - (len(forecast_times) % 10)
            xticks = [times[i] for i in xrange(0, len(times), skips)]
            xticklabels = skips*np.arange(len(xticks)) + plots_init_time_index
            plt.xticks(xticks, xticklabels)
            # show the legend, show the plot
            plt.legend(loc='best')
            #
            plt.draw()
            #
            log_postfix= '_logscale'
            file_name = os.path.join(plots_dir, "InflationOED_Objective%s.%s" % (log_postfix, plots_format))
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')


            # Bounds:
            if bound_axes:
                infl_vmin = min(0.0, infl_lb)
                infl_vmax = max(2.0, infl_lb)
                infl_vmax = max(infl_vmax, np.asarray([optimal_sols]).max())
            else:
                infl_vmin = infl_vmax = None

            if not use_smoothed_opt_results:
                target_optimal_sols = original_optimal_sols
            else:
                target_optimal_sols = optimal_sols

            # histogram of inflation factor
            _, ax_adap_inf_hist = plt.subplots(facecolor='white')
            data = np.asarray(target_optimal_sols[plots_init_time_index]).flatten()
            data = data[~np.isnan(data)]  # removing optimizer failures, i.e. np.nan; if any
            if data.size == 0:
                pass
            else:
                weights = np.zeros_like(data) + 1.0 / data.size
                ax_adap_inf_hist.hist(data, weights=weights, bins=50)
                ax_adap_inf_hist.set_xlabel(r"Inflation factors $\lambda_i$")
                ax_adap_inf_hist.set_ylabel("Relative frequency")
                plt.draw()
                #
                file_name = os.path.join(plots_dir, "InflationSpaceTimeHistogram.%s" % plots_format)
                print("Plotting: %s" % file_name)
                plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            # boxplots of inflation factors over time
            _, ax_adap_inf_bplot = plt.subplots(facecolor='white')
            ax_adap_inf_bplot.boxplot(target_optimal_sols[plots_init_time_index: ], showmeans=means_on_boxplots, notch=True, patch_artist=True, sym='+', vert=1, whis=1.5)
            if bound_axes:
                ax_adap_inf_bplot.set_ylim(infl_vmin, infl_vmax)
            ax_adap_inf_bplot.set_xlabel("Time (assimilation cycles)")
            ax_adap_inf_bplot.set_ylabel(r"Inflation factors $\lambda_i$")
            #
            times = forecast_times[plots_init_time_index: ]
            skips = max(1, len(times) / 5)   # - (len(forecast_times) % 10)
            xticks = [i for i in xrange(0, len(times), skips)]
            xticklabels = skips*np.arange(len(xticks)) + plots_init_time_index
            # print(xticks, xticklabels)
            ax_adap_inf_bplot.set_xticks(xticks)
            ax_adap_inf_bplot.set_xticklabels(xticklabels)
            plt.draw()
            #
            file_name = os.path.join(plots_dir, "InflationSpaceTimeBoxPlot.%s" % plots_format)
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            # colorplot/imagesec of inflation factors over space and time
            fig_adap_inf_imsec, ax_adap_inf_imsec = plt.subplots(facecolor='white')
            cax = ax_adap_inf_imsec.imshow(np.array(target_optimal_sols[plots_init_time_index: ]).squeeze().T, aspect='auto', interpolation=interpolation, cmap=colormap)
            # cax = ax_adap_inf_imsec.imshow(np.array([optimal_sols]).squeeze().T, vmin=infl_vmin, vmax=infl_vmax, aspect='auto', interpolation=interpolation, cmap=colormap)
            if bound_axes:
                skip = (infl_vmax-infl_vmin)/7
                cbar = fig_adap_inf_imsec.colorbar(cax, ticks=np.arange(infl_vmin,infl_vmax, skip), orientation='vertical')
            else:
                cbar = fig_adap_inf_imsec.colorbar(cax, orientation='vertical')
            ax_adap_inf_imsec.set_xlabel("Time (assimilation cycles)")
            ax_adap_inf_imsec.set_ylabel("State variables")
            if bound_axes:
                ax_adap_inf_imsec.set_yticks(np.arange(0, state_size, state_size/10))
            # ax_adap_inf_imsec.set_title("Inflation factors over space and time")

            times = forecast_times[plots_init_time_index: ]
            skips = max(1, len(times) / 5)   # - (len(forecast_times) % 10)
            xticks = [i for i in xrange(0, len(times), skips)]
            xticklabels = [x+plots_init_time_index for x in xticks]
            # xticklabels = skips*np.arange(len(xticks)) + plots_init_time_index
            ax_adap_inf_imsec.set_xticks(xticks)
            ax_adap_inf_imsec.set_xticklabels(xticklabels)
            plt.draw()
            #
            file_name = os.path.join(plots_dir, "InflationSpaceTimeImSec.%s" % plots_format)
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, bbox_inches='tight')

        #
        if len(localization_opt_results)>0:
            if len(analysis_times)-1 != len(localization_opt_results):
                print("Check len(localization_opt_results), and len(forecast_times)")
                print("len(localization_opt_results)", len(localization_opt_results))
                print("len(forecast_times)", len(analysis_times)-1)
                raise ValueError
            localization_stats = np.zeros((5, len(localization_opt_results)))
            # first row:  optimal objective (without regularization)
            # second row: optimal objective (with regularization)
            # third row:  L2 norm of optimal solution
            # fourth row: average inflation factor
            # fifth row:  standard deviation inflation factor
            #
            optimal_sols = []  # smoothed and rounded solutions
            original_optimal_sols = []
            #
            for i in xrange(len(localization_opt_results)):
                opt_x = localization_opt_results[i][1]['opt_x']
                orig_opt_x = localization_opt_results[i][1]['orig_opt_x']
                try:
                    success = localization_opt_results[i][1]['success']
                except(KeyError):
                    try:
                        success = not localization_opt_results[i][1]['opt_info_d']['warnflag']
                    except:
                        try:
                            success = localization_opt_results[i][1]['full_opt_results']['success']
                        except:
                            success = True  # Failed to retrieve the convergence flag; just take the value
                #
                if remove_optimizer_failures and not success:
                    opt_x[:] = np.nan
                    orig_opt_x[:] = np.nan
                    post_trace = np.nan
                    min_f = np.nan
                    l2_norm = np.nan
                    avrg = np.nan
                    stdev = np.nan
                else:
                    post_trace = localization_opt_results[i][1]['post_trace']
                    min_f = localization_opt_results[i][1]['min_f']
                    if not use_smoothed_opt_results:
                        l2_norm = np.linalg.norm(orig_opt_x)
                        avrg = np.mean(orig_opt_x)
                        stdev = np.std(orig_opt_x)
                    else:
                        l2_norm = np.linalg.norm(opt_x)
                        avrg = np.mean(opt_x)
                        stdev = np.std(opt_x)
                #
                optimal_sols.append(opt_x)
                original_optimal_sols.append(orig_opt_x)
                localization_stats[:, i] = post_trace, min_f, l2_norm, avrg, stdev
                #

            #
            _, ax_adap_loc = plt.subplots(facecolor='white')
            #
            ax_adap_loc.plot(analysis_times[plots_init_time_index+1:], localization_stats[0, plots_init_time_index:], 'bd-', linewidth=line_width, label=r"$Trace(\widehat{\mathbf{A}})$")
            # ax_adap_loc.plot(analysis_times[plots_init_time_index+1:], localization_stats[1, plots_init_time_index:], 'gd-', linewidth=line_width, label="optimal objective")
            # ax_adap_loc.plot(analysis_times[1:], localization_stats[2, :], 'r-.', linewidth=line_width, label=r"$\|\mathbf{L}\|$")
            ax_adap_loc.plot(analysis_times[plots_init_time_index+1:], localization_stats[3, plots_init_time_index:], 'c--', linewidth=line_width, label=r"$\overline{\mathbf{\ell_i}}$")
            ax_adap_loc.plot(analysis_times[plots_init_time_index+1:], localization_stats[4, plots_init_time_index:], 'm--', linewidth=line_width, label=r"$\sigma_{\mathbf{L}}$")
            #
            # Set lables and title
            ax_adap_loc.set_xlabel("Time (assimilation cycles)")
            # ax_adap_loc.set_title('OED-Adaptive Localization results for the model: %s' % model_name)
            ax_adap_loc.set_xlim(analysis_times[plots_init_time_index], analysis_times[-1])
            #
            times = forecast_times[plots_init_time_index: ]
            skips = max(1, len(times) / 5)   # - (len(forecast_times) % 10)
            xticks = [times[i] for i in xrange(0, len(times), skips)]
            xticklabels = skips*np.arange(len(xticks))
            plt.xticks(xticks, xticks)
            # show the legend, show the plot
            plt.legend(loc='best')
            #
            file_name = os.path.join(plots_dir, "LocalizationOED_Objective.%s" % plots_format)
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
            _, ax_adap_loc = plt.subplots(facecolor='white')

            #
            ax_adap_loc.semilogy(analysis_times[plots_init_time_index+1:], localization_stats[0, plots_init_time_index:], 'bd-', linewidth=line_width, label=r"$Trace(\widehat{ \mathbf{A}})$")
            # ax_adap_loc.semilogy(analysis_times[plots_init_time_index+1:], localization_stats[1, plots_init_time_index:], 'gd-', linewidth=line_width, label="optimal objective")
            # ax_adap_loc.semilogy(analysis_times[1:], localization_stats[2, :], 'r-.', linewidth=line_width, label=r"$\|\mathbf{\alpha}\|$")
            ax_adap_loc.semilogy(analysis_times[plots_init_time_index+1:], localization_stats[3, plots_init_time_index:], 'c--', linewidth=line_width, label=r"$\overline{\mathbf{\gamma}}$")
            ax_adap_loc.semilogy(analysis_times[plots_init_time_index+1:], localization_stats[4, plots_init_time_index:], 'm--', linewidth=line_width, label=r"$\sigma_{\mathbf{\gamma}}$")
            #
            # Set lables and title
            ax_adap_loc.set_xlabel("Time (assimilation cycles)")
            # ax_adap_loc.set_title('OED-Adaptive Localization results for the model: %s' % model_name)
            ax_adap_loc.set_xlim(analysis_times[plots_init_time_index], analysis_times[-1])
            #
            times = forecast_times[plots_init_time_index: ]
            skips = max(1, len(times) / 10)   # - (len(forecast_times) % 10)
            xticks = [times[i] for i in xrange(0, len(times), skips)]
            xticklabels = skips*np.arange(len(xticks))
            plt.xticks(xticks, xticks)
            # show the legend, show the plot
            plt.legend(loc='best')
            #
            log_postfix = '_logscale'
            file_name = os.path.join(plots_dir, "LocalizationOED_Objective%s.%s" % (log_postfix, plots_format))
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            if use_smoothed_opt_results:
                target_optimal_sols = optimal_sols
            else:
                target_optimal_sols = original_optimal_sols

            # histogram of localization coefficients
            _, ax_adap_loc_hist = plt.subplots(facecolor='white')
            data = np.asarray(target_optimal_sols[plots_init_time_index: ]).flatten()
            data = data[~np.isnan(data)]  # removing optimizer failures, i.e. np.nan; if any
            if data.size == 0:
                pass
            else:
                weights = np.zeros_like(data) + 1.0 / data.size
                ax_adap_loc_hist.hist(data, weights=weights, bins=50)
                if bound_axes:
                    ax_adap_loc_hist.set_xlim(0, state_size)
                ax_adap_loc_hist.set_xlabel(r"Localization radii $\ell_i$")
                ax_adap_loc_hist.set_ylabel("Relative frequency")
                plt.draw()

                file_name = os.path.join(plots_dir, "LocalizationSpaceTimeHistogram.%s" % plots_format)
                print("Plotting: %s" % file_name)
                plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            # boxplots of localization coefficients over time
            _, ax_adap_loc_bplot = plt.subplots(facecolor='white')
            ax_adap_loc_bplot.boxplot(target_optimal_sols[plots_init_time_index: ], showmeans=means_on_boxplots, notch=True, patch_artist=True, sym='+', vert=1, whis=1.5)
            # ax_adap_loc_bplot.set_ylim(0, state_size)
            ax_adap_loc_bplot.set_xlabel("Time (assimilation cycles)")
            ax_adap_loc_bplot.set_ylabel(r"Localization radii $\ell_i$")
            #
            times = forecast_times[plots_init_time_index: ]
            skips = max(1, len(times) / 5)   # - (len(forecast_times) % 10)
            xticks = [i for i in xrange(0, len(times), skips)]
            ax_adap_loc_bplot.set_xticks(xticks)
            ax_adap_loc_bplot.set_xticklabels([x+plots_init_time_index for x in xticks])
            plt.draw()
            #
            file_name = os.path.join(plots_dir, "LocalizationSpaceTimeBoxPlot.%s" % plots_format)
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')


            # colorplot/imagesec of localization parameters over space and time
            vmin, vmax = 0, state_size-1

            fig_adap_loc_imsec, ax_adap_loc_imsec = plt.subplots(facecolor='white')
            if not bound_axes:
                cax = ax_adap_loc_imsec.imshow(np.asarray(target_optimal_sols[plots_init_time_index: ]).squeeze().T, aspect='auto', interpolation=interpolation, cmap=colormap)
            else:
                cax = ax_adap_loc_imsec.imshow(np.asarray(target_optimal_sols[plots_init_time_index: ]).squeeze().T, vmin=vmin, vmax=vmax, aspect='auto', interpolation=interpolation, cmap=colormap)
            cbar = fig_adap_loc_imsec.colorbar(cax, ticks=np.arange(1,state_size, state_size/10), orientation='vertical')
            ax_adap_loc_imsec.set_xlabel("Time (assimilation cycles)")
            ax_adap_loc_imsec.set_ylabel("State variables")
            if bound_axes:
                ax_adap_loc_imsec.set_yticks(np.arange(0, state_size, state_size/10))
                ax_adap_loc_imsec.set_title("Localization parameters over space and time")

            times = forecast_times[plots_init_time_index: ]
            skips = max(1, len(times) / 5)   # - (len(forecast_times) % 10)
            xticks = [i for i in xrange(0, len(times), skips)]
            ax_adap_loc_imsec.set_xticks(xticks)
            ax_adap_loc_imsec.set_xticklabels([x+plots_init_time_index for x in xticks])
            plt.draw()
            #
            file_name = os.path.join(plots_dir, "LocalizationSpaceTimeImSec.%s" % plots_format)
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')


        # =====================================================================


        #
        # =====================================================================
        # Plot Rank Histograms for forecast and analysis ensemble
        # =====================================================================
        model_name= model_name.lower()
        if model_name == 'lorenz96':
            ignore_indexes = None
        elif model_name == 'qg-1.5':
            nx = int(np.sqrt(state_size))
            #
            top_bounds = np.arange(nx)
            right_bounds = np.arange(2*nx-1, nx**2-nx+1, nx)
            left_bounds = np.arange(nx, nx**2-nx, nx )
            down_bounds = np.arange(nx**2-nx, nx**2)
            side_bounds = np.reshape(np.vstack((left_bounds, right_bounds)), (left_bounds.size+right_bounds.size), order='F')
            ignore_indexes = np.concatenate((top_bounds, side_bounds, down_bounds))
        else:
            raise ValueError("Model is not supported here yet...")

        #
        add_fitted_Beta = add_fitted_Beta_toRhist
        for add_optimal_rhist in [True, False]:
            if add_optimal_rhist:
                fit_postfix = '_FittedUBeta'
            else:
                fit_postfix = ''

            #
            if True:
                title = None
            else:
                title = 'Forecast rank histogram'
            if forecast_ensembles is not None:
                f_out = utility.rank_hist(forecast_ensembles,
                                reference_states, first_var=0,
                                last_var=None,
                                var_skp=5,
                                draw_hist=True,
                                hist_type='relfreq',
                                first_time_ind=plots_init_time_index,
                                last_time_ind=None,
                                time_ind_skp=1,
                                hist_title=title,
                                hist_max_height=None,
                                ignore_indexes=ignore_indexes,
                                add_fitted_beta=add_fitted_Beta,
                                add_uniform=add_optimal_rhist
                        )
            fig_hist = f_out[-1]
            file_name = os.path.join(plots_dir, "ForecastRankHistogram%s.%s" % (fit_postfix, plots_format))
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            #
            if True:
                title = None
            else:
                title = "Analysis rank histogram"
            if analysis_ensembles is not None:
                a_out = utility.rank_hist(analysis_ensembles,
                                reference_states, first_var=0,
                                last_var=None,
                                var_skp=5,
                                draw_hist=True,
                                hist_type='relfreq',
                                first_time_ind=plots_init_time_index,
                                last_time_ind=None,
                                time_ind_skp=1,
                                hist_title=title,
                                hist_max_height=None,
                                ignore_indexes=ignore_indexes,
                                add_fitted_beta=add_fitted_Beta,
                                add_uniform=add_optimal_rhist
                        )
            fig_hist = a_out[-1]
            file_name = os.path.join(plots_dir, "AnalysisRankHistogram%s.%s" % (fit_postfix, plots_format))
            print("Plotting: %s" % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
            #
        # =====================================================================

        plt.close('all')
        print("Cool, check: %s" % res_dir)
        # plt.show()


#
if __name__ == '__main__':

    ref_min_rmse_infl_fac = 1.05  # set it manully instead of inspecting it;
    ref_min_kl_infl_fac = ref_min_rmse_infl_fac

    # Adaptive inflation vs. Adaptive localization, to get the right results directory
    adaptive_inflation = True
    adaptive_localization = False

    # Regularization Norm: either L1 or L2 norm results; use 1 or 2
    if adaptive_inflation:
        l_norm = 1
    else:
        l_norm = 2
    # If plotting something to compare to optimal EnKF results; turn this ON; of course, pickled file must exist
    compare_to_EnKF = True
    _filter_name_compare = 'DEnKF'  # Filter to compare results to

    overwrite_plots = True

    # Start reading and plotting (Lots of redundancy :)
    try:
        read_and_plot(ref_min_rmse_infl_fac=ref_min_rmse_infl_fac,
                      ref_min_kl_infl_fac=ref_min_kl_infl_fac,
                      adaptive_inflation=adaptive_inflation,
                      adaptive_localization=adaptive_localization,
                      compare_to_EnKF=compare_to_EnKF,
                      _filter_name_compare=_filter_name_compare,
                      l_norm=l_norm,
                      overwrite_plots=overwrite_plots
                     )
    except:
        print("Caught an Error here... Continuing")
        raise
        pass


