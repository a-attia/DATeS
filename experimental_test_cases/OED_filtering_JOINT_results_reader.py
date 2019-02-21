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
try:
    import pickle
except:
    import cPickle as pickle

import dates_setup
dates_setup.initialize_dates()

from filtering_results_reader_coupledLorenz import str2bool, read_assimilation_results, get_args, enhance_plotter

import dates_utility as utility
#


def get_args(input_args,
             output_repository_dir=None,
            norm=None,
            moving_average=None,
            recollect_results=False):
    """
    Get command line arguments
    """

    try:
        opts, args = getopt.getopt(input_args,"hn:d:a:r:",["nnorm=","amoving-average=", "doutputdir=", "rrecollect-results="])
    except getopt.GetoptError:
        print 'OED_filtering_JOINT_results_reader.py -n <norm_l1_l2> -d <output_repository_dir> -a <moving_average> -r <recollect_results_flag>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'filtering_results_reader_coupledLorenz.py -n <norm_l1_l2> -d <output_repository_dir>  -r <recollect_results_flag>'
            sys.exit()
        elif opt in ("-a", "--amoving-average"):
            moving_average = arg
        elif opt in ("-n", "--nnorm"):
            norm = arg
        elif opt in ("-d", "--doutputdir"):
            output_repository_dir = arg
        elif opt in ("-r", "--recollect-results"):
            recollect_results = str2bool(arg)

    return output_repository_dir, norm, moving_average, recollect_results



def start_reading(repo_dir, norm=None, moving_average_radius=None, recollect_results=False, plot_individual_results=False, collective_res_filename="Collective_Results.pickle",
                  rmse_threshold=1, kl_threshold=10):
    """
    """
    list_of_dirs = utility.get_list_of_subdirectories(repo_dir, ignore_root=True, return_abs=False, ignore_special=True, recursive=False)

    # check if directories have different moving averages
    if moving_average_radius is None:
        moving_average_radii = []
        for d in list_of_dirs:
            l = d.split('_MovingAvgRad_')
            if len(l) > 1:
                moving_average_radii.append(l[-1])
        moving_average_radii = list(set(moving_average_radii))
        if len(moving_average_radii) > 0:
            moving_average_radii.sort()
        else:
            moving_average_radii = [None]
    else:
        moving_average_radii = [moving_average_radius]

    #
    plots_dir = os.path.join(repo_dir, 'PLOTS')
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    if norm is None:
        norms = ['l1', 'l2']
    else:
        if re.match(r'\Al(-|_| )*1\Z', norm, re.IGNORECASE):
            norms = ['l1']
        elif re.match(r'\Al(-|_| )*2\Z', norm, re.IGNORECASE):
            norms = ['l2']
        else:
            print("The passed norm after -n flag, must contain 'l1' or 'l2'")
            print("The flag [%s] is not recognized" % norm)
            sys.exit()

    # Create individual plots if needed
    if plot_individual_results:
        print("Creating Individual plots")
        for res_dir in list_of_dirs:
            cmd = "python filtering_results_reader_coupledLorenz.py -f %s -r True -o True" % os.path.join(res_dir, 'output_dir_structure.txt')
            try:
                os.system(cmd)
            except:
                print("Failed to plot results for %s" % res_dir)
                continue
            #

    #
    # =====================================================================
    # General Plotting settings
    # =====================================================================
    # Font, and Texts:
    enhance_plotter()

    # Drawing Lines:
    line_width = 2
    marker_size = 4

    # set colormap:
    colormap = None  # vs 'jet'
    # plt.set_cmap(colormap)
    interpolation = None  # 'bilinear'

    # =====================================================================
    for moving_average_radius in moving_average_radii:
        for norm in norms:
            if re.match(r'\Al(-|_| )*1\Z', norm, re.IGNORECASE):
                norm_val = 1
            elif re.match(r'\Al(-|_| )*2\Z', norm, re.IGNORECASE):
                norm_val = 2
            else:
                raise ValueError("Only l1, l2 norms are supported.")
            
            print("\n\n READING RESULTS FOR :")
            print("%s\n" % ('.'*100))
            print("> NORM: %s " % norm)
            print("> Moving Average Radius: %s " % repr(moving_average_radius))
            print("%s\n" % ('.'*100))
            fname, ext = os.path.splitext(collective_res_filename)
            prefix = fname
            postfix = '_Norm_%s' % norm
            if moving_average_radius is not None:
                postfix += '_MovingAvgRad_%s' % moving_average_radius
            prefix += postfix
            _collective_res_filename = prefix + ext
            # collective_res_file = os.path.join(repo_dir, _collective_res_filename)
            collective_res_file = os.path.join(plots_dir, _collective_res_filename)

            if os.path.isfile(collective_res_file):
                if recollect_results:
                    pass
                else:
                    try:
                        results_dict = pickle.load(open(collective_res_file, 'rb'))
                        adaptive_inflation = results_dict['adaptive_inflation']
                        adaptive_localization = results_dict['adaptive_localization']
                        lorenz_design_penalties = results_dict['lorenz_design_penalties']
                        lorenz_forecast_times = results_dict['lorenz_forecast_times']
                        lorenz_forecast_rmses = results_dict['lorenz_forecast_rmses']
                        lorenz_analysis_times = results_dict['lorenz_analysis_times']
                        lorenz_analysis_rmses = results_dict['lorenz_analysis_rmses']
                        lorenz_rank_hist_kl_dist = results_dict['lorenz_rank_hist_kl_dist']
                        lorenz_opt_sol_norm = results_dict['lorenz_opt_sol_norm']
                        lorenz_opt_iterations = results_dict['lorenz_opt_iterations']
                        lorenz_opt_post_trace = results_dict['lorenz_opt_post_trace']
                        qg_design_penalties = results_dict['qg_design_penalties']
                        qg_forecast_times = results_dict['qg_forecast_times']
                        qg_forecast_rmses = results_dict['qg_forecast_rmses']
                        qg_analysis_times = results_dict['qg_analysis_times']
                        qg_analysis_rmses = results_dict['qg_analysis_rmses']
                        qg_rank_hist_kl_dist = results_dict['qg_rank_hist_kl_dist']
                        qg_opt_sol_norm = results_dict['qg_opt_sol_norm']
                        qg_opt_iterations = results_dict['qg_opt_iterations']
                        qg_opt_post_trace = results_dict['qg_opt_post_trace']
                        recollect_results = False
                        print("Data Collected SUCCESSFULLY from Pickled file...")
                    except:
                        recollect_results = True
            else:
                print("Couldn't find file: %s " % collective_res_file)
                recollect_results = True
            # print("Target collective file: %s " % collective_res_file)

            if recollect_results:
                print("Recollecting data from results' source files... this will take a moment ...")

                lorenz_design_penalties  = []
                lorenz_forecast_times    = []
                lorenz_forecast_rmses    = []
                lorenz_analysis_times    = []
                lorenz_analysis_rmses    = []
                lorenz_rank_hist_kl_dist = []
                lorenz_opt_sol_norm      = []
                lorenz_opt_iterations    = []
                lorenz_opt_post_trace    = []

                qg_design_penalties  = []
                qg_forecast_times    = []
                qg_forecast_rmses    = []
                qg_analysis_times    = []
                qg_analysis_rmses    = []
                qg_rank_hist_kl_dist = []
                qg_opt_sol_norm      = []
                qg_opt_iterations    = []
                qg_opt_post_trace    = []

                #
                sep = "*" * 80
                for res_dir in list_of_dirs:

                    norm_pattern = "Norm_%s_" % norm
                    if re.search(r"%s"%norm_pattern, res_dir, re.IGNORECASE):
                        # print("dir HIT with norm", res_dir, norm)
                        pass
                    else:
                        # print("dir mismatch with norm", res_dir, norm_pattern, norm)
                        continue

                    avg_pattern = r"MovingAvgRad_%s" % moving_average_radius
                    if re.search(avg_pattern, res_dir, re.IGNORECASE):
                        # print("dir HIT with average radius", res_dir, avg_pattern, moving_average_radius)
                        pass
                    else:
                        # print("dir mismatch with average radius", res_dir, avg_pattern, moving_average_radius)
                        continue

                    if not os.path.isdir(res_dir):
                        print("This doesn't make any sense; couldn't find the directory '%s'" % res_dir)

                    print("*** reading results from %s " % os.path.relpath(res_dir))
                    out_dir_tree_structure_file = os.path.join(res_dir, 'output_dir_structure.txt')

                    #
                    # =====================================================================
                    # Start reading the output of the assimilation process
                    # =====================================================================
                    cycle_prefix, num_cycles, reference_states, forecast_ensembles, forecast_means, analysis_ensembles, \
                    analysis_means, observations, forecast_times, analysis_times, observations_times, \
                    forecast_rmse, analysis_rmse, filter_configs, gmm_results, model_configs, mardiaTest_results,  \
                    inflation_opt_results, localization_opt_results = read_assimilation_results(out_dir_tree_structure_file, rebuild_truth=True, read_results_only=True)

                    init_time_ind = forecast_times.size * 2 / 3

                    #
                    filter_name = filter_configs['filter_name']
                    model_name = model_configs['model_name']
                    try:
                        state_size = model_configs['state_size']
                    except KeyError:
                        state_size = np.size(forecast_ensembles, 0)
                    #
                    # print(reference_states, forecast_ensembles, forecast_means, analysis_ensembles, analysis_means, observations)
                    #
                    if forecast_ensembles is None and analysis_ensembles is None:
                        moments_only = True
                    else:
                        moments_only = False
                    # =====================================================================

                    if len(inflation_opt_results) > 0:
                        adaptive_inflation = True
                    else:
                        adaptive_inflation = False

                    if len(localization_opt_results) > 0:
                        adaptive_localization = True
                    else:
                        adaptive_localization = False

                    #
                    # =====================================================================
                    # Adaptive Inflation and/or localization results
                    # =====================================================================
                    #
                    if len(inflation_opt_results) > 0:
                        design_penalty = inflation_opt_results[0][1]['alpha']
                        opt_sol_norm = []
                        opt_post_trace = []
                        opt_iterations = []

                        optimal_sols = []
                        for i in xrange(len(inflation_opt_results)):
                            opt_post_trace.append(inflation_opt_results[i][1]['post_trace'])
                            opt_sol_norm.append(np.linalg.norm(inflation_opt_results[i][1]['opt_x'], norm_val))
                            niter = inflation_opt_results[i][1]['full_opt_results']['nit']
                            try:
                                funcalls = inflation_opt_results[i][1]['full_opt_results']['nfev']
                            except(KeyError):
                                funcalls = inflation_opt_results[i][1]['full_opt_results']['funcalls']
                            opt_iterations.append((niter, funcalls))
                        opt_iterations = np.asarray(opt_iterations)

                    if len(localization_opt_results)>0:
                        design_penalty = localization_opt_results[0][1]['alpha']

                        opt_sol_norm = []
                        opt_post_trace = []
                        opt_iterations = []

                        optimal_sols = []
                        for i in xrange(len(localization_opt_results)):
                            opt_post_trace.append(localization_opt_results[i][1]['post_trace'])
                            opt_sol_norm.append(np.linalg.norm(localization_opt_results[i][1]['opt_x'], norm_val))
                            niter = localization_opt_results[i][1]['full_opt_results']['nit']
                            try:
                                funcalls = localization_opt_results[i][1]['full_opt_results']['nfev']
                            except(KeyError):
                                funcalls = localization_opt_results[i][1]['full_opt_results']['funcalls']
                            opt_iterations.append((niter, funcalls))
                        opt_iterations = np.asarray(opt_iterations)

                    # =====================================================================


                    #
                    # =====================================================================
                    # Plot Rank Histograms for forecast and analysis ensemble
                    # =====================================================================
                    model_name= model_name.lower()
                    if model_name.startswith('lorenz'):
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
                    ranks_freq, _, bins_bounds , fig_hist = f_out[:]
                    forecast_rank_hist_kl_dist, _ = utility.calc_kl_dist(ranks_freq, bins_bounds)

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
                    ranks_freq, _, bins_bounds , fig_hist = a_out[:]
                    analysis_rank_hist_kl_dist, _ = utility.calc_kl_dist(ranks_freq, bins_bounds)

                    # =====================================================================

                    #
                    if model_name.startswith('lorenz'):
                        lorenz_design_penalties.append(design_penalty)
                        lorenz_forecast_times.append(forecast_times)
                        lorenz_forecast_rmses.append(forecast_rmse)
                        lorenz_analysis_times.append(analysis_times)
                        lorenz_analysis_rmses.append(analysis_rmse)
                        lorenz_rank_hist_kl_dist.append(analysis_rank_hist_kl_dist)
                        lorenz_opt_sol_norm.append(opt_sol_norm)
                        lorenz_opt_iterations.append(opt_iterations)
                        lorenz_opt_post_trace.append(opt_post_trace)

                    elif model_name == 'qg-1.5':
                        qg_design_penalties.append(design_penalty)
                        qg_forecast_times.append(forecast_times)
                        qg_forecast_rmses.append(forecast_rmse)
                        qg_analysis_times.append(analysis_times)
                        qg_analysis_rmses.append(analysis_rmse)
                        qg_rank_hist_kl_dist.append(analysis_rank_hist_kl_dist)
                        qg_opt_sol_norm.append(opt_sol_norm)
                        qg_opt_iterations.append(opt_iterations)
                        qg_opt_post_trace.append(opt_post_trace)

                    else:
                        raise ValueError("Model is not supported here yet...")

                    # =====================================================================
                # Pickle results:
                print("Joint results collected; saving for later use to:\n\t%s" % collective_res_file)
                results_dict = dict(adaptive_inflation=adaptive_inflation,
                                    adaptive_localization=adaptive_localization,
                                    lorenz_design_penalties=lorenz_design_penalties,
                                    lorenz_forecast_times=lorenz_forecast_times,
                                    lorenz_forecast_rmses=lorenz_forecast_rmses,
                                    lorenz_analysis_times=lorenz_analysis_times,
                                    lorenz_analysis_rmses=lorenz_analysis_rmses,
                                    lorenz_rank_hist_kl_dist=lorenz_rank_hist_kl_dist,
                                    lorenz_opt_sol_norm=lorenz_opt_sol_norm,
                                    lorenz_opt_iterations=lorenz_opt_iterations,
                                    lorenz_opt_post_trace=lorenz_opt_post_trace,
                                    qg_design_penalties=qg_design_penalties,
                                    qg_forecast_times=qg_forecast_times,
                                    qg_forecast_rmses=qg_forecast_rmses,
                                    qg_analysis_times=qg_analysis_times,
                                    qg_analysis_rmses=qg_analysis_rmses,
                                    qg_rank_hist_kl_dist=qg_rank_hist_kl_dist,
                                    qg_opt_sol_norm=qg_opt_sol_norm,
                                    qg_opt_iterations=qg_opt_iterations,
                                    qg_opt_post_trace=qg_opt_post_trace
                                   )
                pickle.dump(results_dict, open(collective_res_file, 'wb'))

            #
            if adaptive_inflation:
                file_prefix = 'Adaptive_Inflation'
            elif adaptive_localization:
                file_prefix = 'Adaptive_Localization'
            else:
                print("This driver is designed for Adaptive inflation or localization...!")
                raise ValueError
            #
            # =====================================================================
            # Collective RMSE Plot, RMSE average & boxplot (after spinup),  and collective
            # =====================================================================
            try:
                init_time_ind
            except(NameError):
                init_time_ind = lorenz_analysis_times[0].size * 2 / 3

            # Regularization parameter against average RMSE
            sortargs = np.argsort(np.asarray(lorenz_design_penalties))
            fig = plt.figure(figsize=(8.10, 3.15), facecolor='white')
            ax = fig.add_subplot(111)
            avg_trace = np.empty(len(lorenz_design_penalties))
            for i in xrange(len(lorenz_design_penalties)):
                avg_trace[i] = np.mean(lorenz_opt_post_trace[i][init_time_ind:])
            ax.plot([lorenz_design_penalties[i] for i in sortargs], [avg_trace[i] for i in sortargs], '-d', linewidth=line_width)
            ax.set_xlabel("Penalty Parameter")
            ax.set_ylabel("Average posterior Trace")
            #
            plt.minorticks_on()
            ax.grid(True, which='major', linestyle='-')
            ax.grid(True, which='minor', linestyle='-.')
            #
            file_name = os.path.join(plots_dir, "%s_Regularization_vs_Average_PostTrace%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

            # Regularization parameter against average RMSE
            sortargs = np.argsort(np.asarray(lorenz_design_penalties))
            fig = plt.figure(figsize=(8.10, 3.15), facecolor='white')
            ax = fig.add_subplot(111)
            avg_rmse = np.empty(len(lorenz_design_penalties))
            for i in xrange(len(lorenz_design_penalties)):
                avg_rmse[i] = np.mean(lorenz_analysis_rmses[i][init_time_ind:])
            avg_rmse[avg_rmse>rmse_threshold] = np.nan
            ax.plot([lorenz_design_penalties[i] for i in sortargs], [avg_rmse[i] for i in sortargs], '-d', linewidth=line_width)
            ax.set_xlabel("Penalty Parameter")
            ax.set_ylabel("Average RMSE")
            #
            plt.minorticks_on()
            ax.grid(True, which='major', linestyle='-')
            ax.grid(True, which='minor', linestyle='-.')
            #
            file_name = os.path.join(plots_dir, "%s_Regularization_vs_Average_RMSE%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

            # Regularization parameter against Number of iterations, function evaluations
            sortargs = np.argsort(np.asarray(lorenz_design_penalties))
            fig = plt.figure(figsize=(8.10, 3.15), facecolor='white')
            ax = fig.add_subplot(111)
            avg_niter = np.empty(len(lorenz_design_penalties))
            avg_nfeval = np.empty(len(lorenz_design_penalties))
            lorenz_opt_iterations = np.asarray(lorenz_opt_iterations)
            for i in xrange(len(lorenz_design_penalties)):
                avg_niter[i]  = np.mean(lorenz_opt_iterations[i][init_time_ind:, 0])
                avg_nfeval[i] = np.mean(lorenz_opt_iterations[i][init_time_ind:, 1])
            ax.plot([lorenz_design_penalties[i] for i in sortargs], [avg_niter[i] for i in sortargs], '-d', label='Avg niter', markersize=marker_size*2)
            ax.plot([lorenz_design_penalties[i] for i in sortargs], [avg_nfeval[i] for i in sortargs], '-.s', label='Avg nfeval', markersize=marker_size*1.25)
            ax.set_xlabel("Penalty Parameter")
            plt.minorticks_on()
            ax.grid(True, which='major', linestyle='-')
            ax.grid(True, which='minor', linestyle='-.')
            ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
            #
            file_name = os.path.join(plots_dir, "%s_Regularization_vs_Opt_Iterations%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

            # Time Vs. RMSE for multiple regularization parameters
            log_scale = True
            skp = 1
            leg_ncol = 6
            #
            fig1 = plt.figure(facecolor='white', figsize=(12, 7))
            ax = fig1.gca()
            sortargs = np.argsort(np.asarray(lorenz_design_penalties))
            lines, lables = [], []
            line_colors = utility.unique_colors(len(lorenz_design_penalties))
            for i in xrange(0, len(lorenz_design_penalties), skp):
                ind = sortargs[i]
                d = lorenz_design_penalties[ind]
                err = lorenz_analysis_rmses[ind]
                tm = lorenz_analysis_times[ind]
                if adaptive_inflation:
                    lbl = r"$\alpha$ = %+6.5g" % d
                else:
                    lbl = r"$\gamma$ = %+6.5g" % d

                if log_scale:
                    lne, = ax.semilogy(tm, err, color=line_colors[i], linewidth=line_width, label=lbl)
                else:
                    lne, = ax.plot(tm, err, color=line_colors[i], linewidth=line_width, label=lbl)
                lines.append(lne)
                lables.append(lbl)
            #
            # Set lables and title
            ax.set_xlabel("Time")
            if log_scale:
                ax.set_ylabel(r'RMSE ($log-scale$)')
            else:
                ax.set_ylabel(r'RMSE')
            # plt.title('RMSE results for the model: %s' % model_name)
            #
            ftimes = np.asarray(lorenz_analysis_times[0])
            skps = max(ftimes.size/20, 1)
            xlables = [ftimes[i] for i in xrange(0, len(ftimes), skps)]
            # ax.set_xlim(ftimes[0], ftimes[-1])
            ax.set_xticks(xlables)
            ax.set_xticklabels(xlables)
            # show the legend, show the plot
            # plt.legend(loc='upper right', ncol=leg_ncol)
            plt.legend(lines, lables, ncol=4, loc='best', fontsize=12, framealpha=0.45)
            #
            plt.minorticks_on()
            ax.grid(True, which='major', linestyle='-')
            ax.grid(True, which='minor', linestyle='-.')
            #
            file_name = os.path.join(plots_dir, "%s_Lorenz_Joint_RMSE%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
            ax.set_xlim(tm[init_time_ind], tm[-1])
            file_name = os.path.join(plots_dir, "%s_Lorenz_Joint_RMSE_trimmed%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


            # Plot KL Distances
            fig2 = plt.figure(facecolor='white')
            pen = np.asarray(lorenz_design_penalties)
            kl_dists = np.asarray(lorenz_rank_hist_kl_dist)
            kl_dists[kl_dists>kl_threshold] = np.nan
            plt.plot(pen[np.argsort(pen)], kl_dists[np.argsort(pen)], '-*')
            #
            # Set lables and title
            plt.xlabel("Penalty Parameter")
            plt.ylabel(r'$D_{KL}(\mathbf{P}^{a}|\mathcal{U}\,)$')
            #
            file_name = os.path.join(plots_dir, "%s_Regularization_vs_KL_Dist%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

            fig4 = plt.figure(facecolor='white', figsize=(12, 7))
            ax = fig4.gca()
            sortargs = np.argsort(np.asarray(lorenz_design_penalties))
            post_traces = np.asarray(lorenz_opt_post_trace)
            lines, lables = [], []
            line_colors = utility.unique_colors(len(lorenz_design_penalties))
            _mx = 0
            for i in xrange(0, len(lorenz_design_penalties), skp):
                ind = sortargs[i]
                d = lorenz_design_penalties[ind]
                pt = np.asarray(post_traces[ind])
                tm = lorenz_analysis_times[ind][1: ]
                if adaptive_inflation:
                    lbl = r"$\alpha$ = %+6.5g" % d
                else:
                    lbl = r"$\gamma$ = %+6.5g" % d
                lne, = ax.plot(tm, pt, color=line_colors[i], linewidth=line_width, label=lbl)
                lines.append(lne)
                lables.append(lbl)
                try:
                    _pt_mx = np.nanmax(pt[init_time_ind: ])
                    if _pt_mx > _mx:
                        _mx = _pt_mx
                except:
                    print(pt.size, init_time_ind)
                    raise
            #
            # Set lables and title
            ax.set_xlabel("Time")
            ax.set_ylabel("Posterior Trace")
            plt.legend(lines, lables, ncol=4, loc='best', fontsize=12, framealpha=0.45)
            #
            plt.minorticks_on()
            ax.grid(True, which='major', linestyle='-')
            ax.grid(True, which='minor', linestyle='-.')
            #
            file_name = os.path.join(plots_dir, "%s_Lorenz_Joint_PostTrace%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
            #
            ax.set_xlim(tm[init_time_ind], tm[-1])
            ax.set_ylim(ax.get_ylim()[0], _mx*1.5)
            file_name = os.path.join(plots_dir, "%s_Lorenz_Joint_PostTrace_trimmed%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


            # L-Curves
            post_traces = np.asarray(lorenz_opt_post_trace)  # num-of-penalties x num time points; for every penalty value, full-length experiment is given
            post_LNorms = np.asarray(lorenz_opt_sol_norm)  # num-of-penalties x num time points
            post_rmses = np.asarray(lorenz_analysis_rmses)

            # get best penalty for each time point
            forecast_times = lorenz_forecast_times[0]
            min_rmse_penalties = np.empty(forecast_times.size-1)
            min_rmse_penalties[...] = np.nan
            min_rmses = min_rmse_penalties.copy()
            min_rmse_norms = min_rmse_penalties.copy()
            min_rmse_posttrace = min_rmse_penalties.copy()
            #
            for t_ind in xrange(forecast_times.size-1):
                min_rmse = np.inf
                min_rmse_ind = None
                opt_penalty = np.nan
                opt_norm = np.nan
                opt_trace = np.nan
                for p_ind in xrange(0, len(lorenz_design_penalties), skp):
                    rmse = lorenz_analysis_rmses[p_ind][t_ind]
                    if rmse < min_rmse:
                        min_rmse = rmse
                        opt_penalty = lorenz_design_penalties[p_ind]
                        opt_norm = lorenz_opt_sol_norm[p_ind][t_ind]
                        opt_trace = lorenz_opt_post_trace[p_ind][t_ind]

                min_rmse_penalties[t_ind] = opt_penalty
                min_rmses[t_ind] = min_rmse
                min_rmse_norms[t_ind] = opt_norm
                min_rmse_posttrace[t_ind] = opt_trace

            # Selective curves
            selective_time_indexes = [init_time_ind + i for i in xrange(10)]
            fig = plt.figure(facecolor='white', figsize=(11, 6))
            ax = fig.gca()
            lines, lables = [], []
            line_colors = utility.unique_colors(len(selective_time_indexes))
            for ind, t_ind in enumerate(selective_time_indexes):
                # at every selected time index, plot norm of optimal solution, vs. posterior trace
                xvals = post_LNorms[sortargs, t_ind]
                yvals = post_traces[sortargs, t_ind]
                if adaptive_inflation:
                    lbl = r"$t=%5.3f;\, \alpha=%6.5f$" % (forecast_times[t_ind], min_rmse_penalties[t_ind])
                else:
                    lbl = r"$t=%5.3f;\, \gamma=%6.5f$" % (forecast_times[t_ind], min_rmse_penalties[t_ind])
                lne, = ax.plot(xvals, yvals, '-', color=line_colors[ind], label=lbl)
                _ = ax.scatter([min_rmse_norms[t_ind]], [min_rmse_posttrace[t_ind]])  # circle the one with slowest RMSE
                lines.append(lne)
                lables.append(lbl)

            if re.match(r'\Al(-|_| )*1\Z', norm, re.IGNORECASE):
                xlabel = r"$|| \lambda^{\rm opt} ||_1 $"
            elif re.match(r'\Al(-|_| )*2\Z', norm, re.IGNORECASE):
                xlabel = r"$|| \lambda^{\rm opt} ||_2 $"
            else:
                # Just use Generic norm
                print("***WARNING*** I am expecting L1 or L2 norm; received %s \nProceed using a generic norm..." % norm)
                xlabel = r"$|| \lambda^{\rm opt} || $"
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Posterior Trace")
            plt.legend(lines, lables, ncol=max(len(selective_time_indexes)/5, 1), loc='best', fontsize=12, framealpha=0.45)
            #
            plt.minorticks_on()
            ax.grid(True, which='major', linestyle='-')
            ax.grid(True, which='minor', linestyle='-.')
            #
            file_name = os.path.join(plots_dir, "%s_Lorenz_Selective_LCurves_PostTrace%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
            
            # Selective curves
            selective_time_indexes = [init_time_ind + i for i in xrange(10)]
            fig = plt.figure(facecolor='white', figsize=(11, 6))
            ax = fig.gca()
            lines, lables = [], []
            line_colors = utility.unique_colors(len(selective_time_indexes))
            for ind, t_ind in enumerate(selective_time_indexes):
                # at every selected time index, plot norm of optimal solution, vs. posterior trace
                xvals = post_LNorms[sortargs, t_ind]
                yvals = post_rmses[sortargs, t_ind]
                if adaptive_inflation:
                    lbl = r"$t=%5.3f;\, \alpha=%6.5f$" % (forecast_times[t_ind], min_rmse_penalties[t_ind])
                else:
                    lbl = r"$t=%5.3f;\, \gamma=%6.5f$" % (forecast_times[t_ind], min_rmse_penalties[t_ind])
                lne, = ax.plot(xvals, yvals, '-', color=line_colors[ind], label=lbl)
                _ = ax.scatter([min_rmse_norms[t_ind]], [min_rmses[t_ind]])  # circle the one with slowest RMSE
                lines.append(lne)
                lables.append(lbl)

            if re.match(r'\Al(-|_| )*1\Z', norm, re.IGNORECASE):
                xlabel = r"$|| \lambda^{\rm opt} ||_1 $"
            elif re.match(r'\Al(-|_| )*2\Z', norm, re.IGNORECASE):
                xlabel = r"$|| \lambda^{\rm opt} ||_2 $"
            else:
                # Just use Generic norm
                print("***WARNING*** I am expecting L1 or L2 norm; received %s \nProceed using a generic norm..." % norm)
                xlabel = r"$|| \lambda^{\rm opt} || $"
            ax.set_xlabel(xlabel)
            ax.set_ylabel("RMSE")
            plt.legend(lines, lables, ncol=max(len(selective_time_indexes)/5, 1), loc='best', fontsize=12, framealpha=0.45)
            #
            plt.minorticks_on()
            ax.grid(True, which='major', linestyle='-')
            ax.grid(True, which='minor', linestyle='-.')
            #
            file_name = os.path.join(plots_dir, "%s_Lorenz_Selective_LCurves_RMSE%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
            

            # One L-Curve, with all penalties on it
            prop_cycle = plt.rcParams['axes.prop_cycle'][: 5]
            def_colors = prop_cycle.by_key()['color']
            def_markers = ['o', 'd', 's', '*', 'v', '^', '>', '<']
            for t_ind in selective_time_indexes:
                lorenz_design_penalties = np.asarray(lorenz_design_penalties)
                rmses = np.empty_like(lorenz_design_penalties)
                post_traces = np.empty_like(lorenz_design_penalties)
                opt_norms = np.empty_like(lorenz_design_penalties)
                #   
                rmses[...] = np.nan
                post_traces[...] = np.nan
                opt_norms[...] = np.nan
                #   
                for p_ind in xrange(lorenz_design_penalties.size):
                    rmses[p_ind] = lorenz_analysis_rmses[p_ind][t_ind]
                    post_traces[p_ind] = lorenz_opt_post_trace[p_ind][t_ind]
                    opt_norms[p_ind] = lorenz_opt_sol_norm[p_ind][t_ind]
                sorter = np.argsort(lorenz_design_penalties)
                #   
                marker_colors = utility.unique_colors(len(lorenz_design_penalties))
                c1 = 'k' 
                c2 = 'b' 
                #   
                fig, axs = plt.subplots(2, 1, figsize=(12, 5), facecolor='white', sharex=True, sharey=True)
                ax = axs[0]
                # ax.set_ylim(ylim[0], ylim[1])
                ax.plot(opt_norms[sorter], post_traces[sorter])
                for p_ind in xrange(lorenz_design_penalties.size):
                    ind = sorter[p_ind]
                    if adaptive_inflation:
                        lbl = r"$\alpha=%5.4f$" % lorenz_design_penalties[ind]
                    else:
                        lbl = r"$\gamma=%5.4f$" % lorenz_design_penalties[ind]
                    
                    # ax.plot(opt_norms[ind], post_traces[ind], 'o', ms = 9, color=marker_colors[p_ind], markeredgecolor=c1, markeredgewidth=0.5, label=lbl)
                    color = def_colors[p_ind%len(def_colors)]
                    marker = def_markers[(p_ind//len(def_colors))%len(def_markers)]
                    ax.plot(opt_norms[ind], post_traces[ind], marker, ms = 9, color=color, label=lbl)
                
                ax.set_ylabel('Posterior trace')
                ax.minorticks_on()
                ax.grid(True, which='major', linestyle='-')
                ax.grid(True, which='minor', linestyle='-.')
                ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
                #
                ax = axs[1]
                ax.plot(opt_norms[sorter], rmses[sorter])
                ax.set_ylabel('RMSE')
                for p_ind in xrange(lorenz_design_penalties.size):
                    ind = sorter[p_ind]
                    if adaptive_inflation:
                        lbl = r"$\alpha=%5.4f$" % lorenz_design_penalties[ind]
                    else:
                        lbl = r"$\gamma=%5.4f$" % lorenz_design_penalties[ind]
                    # ax.plot(opt_norms[ind], rmses[ind], 'o', ms = 9, color=marker_colors[p_ind], markeredgewidth=0.5, markeredgecolor=c2, label=lbl)
                    color = def_colors[p_ind%len(def_colors)]
                    marker = def_markers[(p_ind//len(def_colors))%len(def_markers)]
                    ax.plot(opt_norms[ind], rmses[ind], marker, ms = 9, color=color, label=lbl)

                if re.match(r'\Al(-|_| )*1\Z', norm, re.IGNORECASE):
                    xlabel = r"$|| \lambda^{\rm opt} ||_1 $"
                elif re.match(r'\Al(-|_| )*2\Z', norm, re.IGNORECASE):
                    xlabel = r"$|| \lambda^{\rm opt} ||_2 $"
                else:
                    # Just use Generic norm
                    print("***WARNING*** I am expecting L1 or L2 norm; received %s \nProceed using a generic norm..." % norm)
                    xlabel = r"$|| \lambda^{\rm opt} || $"
                ax.set_xlabel(xlabel)
                ax.minorticks_on()
                ax.grid(True, which='major', linestyle='-')
                ax.grid(True, which='minor', linestyle='-.')

                fig.subplots_adjust(hspace=0.01)
                file_name = os.path.join(plots_dir, "%s_Lorenz_Selective_LCurves_Tind_%d%s.pdf" % (file_prefix, t_ind, postfix))
                print("Saving: %s" %file_name)
                plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


            # optimal penalty parameter over time
            fig = plt.figure(facecolor='white', figsize=(12, 7))
            ax = fig.gca()
            ax.plot(forecast_times[1:], min_rmse_penalties, '-o', color='darkblue', linewidth=line_width)
            ax.set_xlabel("Time")
            if adaptive_inflation:
                ax.set_ylabel(r"$\alpha^{\rm opt}$")
            else:
                ax.set_ylabel(r"$\gamma^{\rm opt}$")
            #
            plt.minorticks_on()
            ax.grid(True, which='major', linestyle='-')
            ax.grid(True, which='minor', linestyle='-.')
            #
            file_name = os.path.join(plots_dir, "%s_Lorenz_Time_vs_Optimal_Penalties%s.pdf" % (file_prefix, postfix))
            print("Saving: %s" %file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
            
            # 3D-Surfaces



            plt.close('all')

            # =====================================================================



#
if __name__ == '__main__':
    # =====================================================================
    collective_res_filename = "Collective_Results.pickle"
    plot_individual_results = False

    repo_dir, norm, moving_average, recollect_results = get_args(sys.argv[1:])
    if recollect_results is None:
        recollect_results = False

    if repo_dir is None:
        print 'You need to pass the result directory by running :\n OED_filtering_JOINT_results_reader.py -d <output_repository_dir>'
        sys.exit()
        #
    else:
        start_reading(repo_dir, norm, moving_average,
                      recollect_results=recollect_results,
                      plot_individual_results=plot_individual_results)  # Set to False after full sync of all running machines

