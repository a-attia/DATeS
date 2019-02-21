#!/usr/bin/env python


from filtering_results_reader import *

import heapq

from mpl_toolkits.mplot3d import Axes3D

from OED_filtering_JOINT_results_reader import calc_kl_dist

import beautify_plots

# global variables
standard_results_dir = os.path.abspath("Results/Filtering_Results")
out_dir_tree_structure_file = 'Results/Filtering_Results/output_dir_structure.txt'


def inspect_output_path(results_dir, design_penalty, model_name, ensemble_size, localization_function, adaptive_inflation=True):
    """
    """
    if isinstance(design_penalty, int):
        penalty = "%d" % design_penalty
    else:
        penalty = "%6.5f" % design_penalty
        penalty = penalty.replace('.', '_')

    if adaptive_inflation:
        new_path = "OED_Adaptive_Inflation_%s_Ensemble_%d_%s_Penalty_%s" %(model_name, ensemble_size,localization_function, penalty)
    else:
        new_path = "OED_Adaptive_Localization_%s_Ensemble_%d_%s_Penalty_%s" %(model_name, ensemble_size,localization_function, penalty)
    new_path = os.path.join(results_dir, new_path)
    return new_path


def get_pools(list_of_dirs, filter_name, model_name):
    """
    """
    # matching results directories:
    matching_dirs = []
    for d in list_of_dirs:
        _d = d.lower()
        if filter_name.lower() in _d and model_name.lower() in _d:
            matching_dirs.append(d)

    # check all penalty parameters, localization functions, and sample sizes
    design_penalties = []
    sample_sizes = []
    loc_functions = []
    for d in matching_dirs:
        _d = d.split('_Ensemble_')[-1].split('_')
        sample_sizes.append(int(_d[0]))
        func = ' '.join(_d[1:-3])
        if func not in loc_functions:
            loc_functions.append(func)
        design_penalties.append(float('.'.join(_d[-2:])))
    design_penalties = list(set(design_penalties))
    design_penalties.sort()
    sample_sizes = list(set(sample_sizes))
    sample_sizes.sort()

    return design_penalties, sample_sizes, loc_functions


def read_from_results(base_results_dir, adaptive_inflation=True):
    """
    """
    analysis_rmse_vectors = None
    rmse_mat = np.empty((2, len(ensemble_size_pool), len(design_penalty_pool)))
    rhist_uniformity_mat = np.empty((2, len(ensemble_size_pool), len(design_penalty_pool), 3))

    inflation_stats = None
    localization_stats = None

    num_experiments = len(ensemble_size_pool) * len(localization_function_pool) * len(design_penalty_pool)
    exp_no = 0

    for e_ind, ensemble_size in enumerate(ensemble_size_pool):
        for f_ind, design_penalty in enumerate(design_penalty_pool):

            # Print some header
            results_dir = inspect_output_path(base_results_dir, design_penalty, model_name, ensemble_size, localization_function, adaptive_inflation)

            sep = "\n%s(Reading %s Experiment Results [ %03d / %03d ]:)%s\n" % ('='*25, filter_name, exp_no, num_experiments, '='*25)
            print("Results from: %s\n" % (results_dir)),
            exp_no += 1

            if not os.path.isdir(results_dir):
                print("\nWARNING: Didn't find this results directory: %s !\n ...SKIPPING...\n" % results_dir)
                rmse_mat[:, e_ind, f_ind] = np.nan
                rhist_uniformity_mat[:, e_ind, f_ind, :] = np.nan
                continue
            else:
                os.rename(results_dir, standard_results_dir)

            # =====================================================================
            # Start reading the output of the assimilation process
            # =====================================================================
            _, _, reference_states, forecast_ensembles, _, analysis_ensembles, _, _, _, analysis_times, _, \
                forecast_rmse, analysis_rmse, _, _, _, _, \
                inflation_opt_results, localization_opt_results = read_filter_output(out_dir_tree_structure_file)

            # update analysis rmse collection
            if analysis_rmse_vectors is None:
                analysis_rmse_vectors = np.empty((len(ensemble_size_pool), len(design_penalty_pool), len(analysis_times)))
                analysis_rmse_vectors[...] = np.nan
            analysis_rmse_vectors[e_ind, f_ind, :len(analysis_rmse)] = analysis_rmse

            # Collect results:
            # 1- ensemble size vs. penalty parameter -> RMSE, and distance to uniformity
            # Collect results:
            rmse_mat[0, e_ind, f_ind] = np.nanmean(forecast_rmse[init_time_ind:])
            rmse_mat[1, e_ind, f_ind] = np.nanmean(analysis_rmse[init_time_ind:])
            #

            f_out = utility.rank_hist(forecast_ensembles, reference_states, first_var=0,
                                                                            last_var=None,
                                                                            var_skp=5,
                                                                            draw_hist=False,
                                                                            hist_type='relfreq',
                                                                            first_time_ind=init_time_ind,
                                                                            last_time_ind=None,
                                                                            time_ind_skp=2)
            ranks_freq, _, bins_bounds , fig_hist = f_out[:]
            out = calc_kl_dist(ranks_freq, bins_bounds)
            forecast_rank_hist_kl_dist = [out[0], out[1][0], out[1][1]]

            for i in xrange(3):
                rhist_uniformity_mat[0, e_ind, f_ind, i] = forecast_rank_hist_kl_dist[i]

            #
            a_out = utility.rank_hist(analysis_ensembles, reference_states, first_var=0,
                                                                            last_var=None,
                                                                            var_skp=5,
                                                                            draw_hist=False,
                                                                            hist_type='relfreq',
                                                                            first_time_ind=init_time_ind,
                                                                            last_time_ind=None,
                                                                            time_ind_skp=2)
            ranks_freq, _, bins_bounds , fig_hist = a_out[:]
            out = calc_kl_dist(ranks_freq, bins_bounds)
            analysis_rank_hist_kl_dist = [out[0], out[1][0], out[1][1]]
            #
            for i in xrange(3):
                rhist_uniformity_mat[1, e_ind, f_ind, i] = analysis_rank_hist_kl_dist[i]

            # =====================================================================
            # Rename Results Directory back to original folder, and Cleanup
            # =====================================================================
            os.rename(standard_results_dir, results_dir)
            print("...DONE...\n%s\n" % ('='*len(sep)))
            # Cleanup:
            # del reference_states, forecast_ensembles, analysis_ensembles, forecast_rmse, analysis_rmse
            # =====================================================================

            # Extract Inflation/Localization Stats
            if len(inflation_opt_results) > 0:
                if inflation_stats is None:
                    inflation_stats = np.empty((len(ensemble_size_pool), len(design_penalty_pool), 5, len(inflation_opt_results)))
                    inflation_stats[...] = np.nan
                # first row:  optimal objective (without regularization)
                # second row: optimal objective (with regularization)
                # third row:  L2 norm of optimal solution
                # fourth row: average inflation factor
                # fifth row:  standard deviation inflation factor
                #
                for i in xrange(len(inflation_opt_results)):
                    post_trace = inflation_opt_results[i][1]['post_trace']
                    min_f = inflation_opt_results[i][1]['min_f']
                    # opt_x = inflation_opt_results[i][1]['opt_x']
                    orig_opt_x = inflation_opt_results[i][1]['orig_opt_x']
                    l2_norm = np.linalg.norm(orig_opt_x)
                    avrg = np.mean(orig_opt_x)
                    stdev = np.std(orig_opt_x)
                    inflation_stats[e_ind, f_ind, :, i] = post_trace, min_f, l2_norm, avrg, stdev
            else:
                inflation_stats = None

            if len(localization_opt_results) > 0:
                if localization_stats is None:
                    localization_stats = np.empty((len(ensemble_size_pool), len(design_penalty_pool), 5, len(localization_opt_results)))
                    localization_stats[...] = np.nan
                for i in xrange(len(localization_opt_results)):
                    post_trace = localization_opt_results[i][1]['post_trace']
                    min_f = localization_opt_results[i][1]['min_f']
                    # opt_x = localization_opt_results[i][1]['opt_x']
                    orig_opt_x = localization_opt_results[i][1]['orig_opt_x']
                    l2_norm = np.linalg.norm(orig_opt_x)
                    avrg = np.mean(orig_opt_x)
                    stdev = np.std(orig_opt_x)
                    localization_stats[e_ind, f_ind, :, i] = post_trace, min_f, l2_norm, avrg, stdev
            else:
                localization_stats = None
                #
    return analysis_times, rmse_mat, rhist_uniformity_mat, inflation_stats, localization_stats, analysis_rmse_vectors

#
def load_from_pickled(file_name):
    """
    """
    cont = pickle.load(open(file_name, 'rb'))
    analysis_times = cont['analysis_times']
    rmse_mat = cont['rmse_mat']
    analysis_rmse_vectors=cont['analysis_rmse_vectors']
    rhist_uniformity_mat = cont['rhist_uniformity_mat']
    # design_penalty_pool = cont['design_penalty_pool']
    # ensemble_size_pool = cont['ensemble_size_pool']
    # localization_function_pool = cont['localization_function_pool']
    inflation_stats = cont['inflation_stats']
    localization_stats = cont['localization_stats']
    return analysis_times, rmse_mat, rhist_uniformity_mat, inflation_stats, localization_stats, analysis_rmse_vectors


def find_min_ranks(mat, count=1):
    """
    Given a 2D-matrix, find the location of the (count) minimum across rows, and columns,
    I am not relying on argsort because np.nan confuses it...

    Args:
        mat

    Returns:
        rows_min_ranks: rank of minimum value for each row
        cols_min_ranks: rank of minimum value for each column

    """
    assert np.ndim(mat) == 2, "mat must be 2-D np.ndarray"

    if count > np.min(mat.shape):
        print("Can't find %d minimum numbers with either of mat shapes less than that count. Shape: %s " % (count, repr(mat.shape)))
        raise ValueError

    # indices with Nan
    nans = np.isnan(mat)
    mat[nans] = np.inf

    # cols_mins = np.nanmin(mat, 0)  # minimum across columns
    # rows_mins = np.nanmin(mat, 1)  # minimum across rows

    # find rank of minimum accross rows
    num_rows, num_cols = mat.shape

    aggr_rows_min_ranks = []
    aggr_cols_min_ranks = []

    for min_val_ind in xrange(count):
        rows_min_ranks, cols_min_ranks = [], []
        for row_ind in xrange(num_rows):
            row_min_n = heapq.nsmallest(min_val_ind+1, mat[row_ind, :])[-1]
            vals = mat[row_ind, :]
            if np.isnan(vals).all():
                rnk = np.nan
            else:
                rnk = np.where(vals==row_min_n)[0][0]
            rows_min_ranks.append(rnk)

        for col_ind in xrange(num_cols):
            col_min_n = heapq.nsmallest(min_val_ind+1, mat[:, col_ind])[-1]
            vals = mat[:, col_ind]
            if np.isnan(vals).all():
                rnk = np.nan
            else:
                rnk = np.where(vals==col_min_n)[0][0]
            cols_min_ranks.append(rnk)

        #
        if count == 1:
            aggr_rows_min_ranks = rows_min_ranks
            aggr_cols_min_ranks = cols_min_ranks
        else:
            aggr_rows_min_ranks.append(rows_min_ranks)
            aggr_cols_min_ranks.append(cols_min_ranks)
    #
    mat[nans] = np.nan
    #
    return aggr_rows_min_ranks, aggr_cols_min_ranks


# if __name__ == '__main__':


show_grids = False
show_plots = False
alpha = 0.75

plot_Lcurve_only = True
trimmed_only = False

plot_individual_lcurves = False

adaptive_inflation = True
adaptive_loclization = not adaptive_inflation
#
if adaptive_inflation:
    L_Norm_num = 1  # L_1 vs L2 number (value is 1 or 2).
else:
    L_Norm_num = 2

results_dir = "./Results/OED_EnKF_Results"
if adaptive_inflation:
    filter_name = 'OED_Adaptive_Inflation'
else:
    filter_name = 'OED_Adaptive_Localization'
# I have added subdirs:
results_dir = os.path.join(results_dir, filter_name, "L%d_Norm" % L_Norm_num)

model_name = 'lorenz'

plots_format = 'pdf'

init_time_ind = 100

rmse_thresh = 0.7  # 0.7
rhist_thresh = 2.5  # 2.5

post_trace_thresh_l = 0.0
post_trace_thresh_u = 10  # 10

LNormthresh_l = 0
LNormthresh_u = np.inf

num_min_penalties = 5  # number of minimum penalties to plot

load_from_pickle = True

# Font, and Texts:
beautify_plots.enhance_plots(font_size=18, font_weight='bold', use_tex=True)
cmap = matplotlib.cm.jet
cmap.set_bad('white',1.)

rmse_vmin, rmse_vmax = 0, rmse_thresh
rhist_vmin, rhist_vmax = 0, rhist_thresh

# get list of results directories:
list_of_dirs = utility.get_list_of_subdirectories(results_dir, ignore_root=True, return_abs=False, ignore_special=True, recursive=False)
design_penalty_pool, ensemble_size_pool, localization_function_pool = get_pools(list_of_dirs, filter_name, model_name)

#
for localization_function in ['Gaspari-Cohn']:  # localization_function_pool:
    file_prefix = '%s_%s_Results_Localization_%s' % (model_name, filter_name, localization_function)
    file_name = '%s.pickle' % (file_prefix)
    file_name = os.path.join(results_dir, file_name)
    if load_from_pickle:
        analysis_times, rmse_mat, rhist_uniformity_mat, inflation_stats, localization_stats, analysis_rmse_vectors = load_from_pickled(file_name)
    else:
        analysis_times, rmse_mat, rhist_uniformity_mat, inflation_stats, localization_stats,analysis_rmse_vectors = read_from_results(results_dir, adaptive_inflation)
        # pickle Results:
        results_dict = {'analysis_times':analysis_times,
                        'rmse_mat':rmse_mat,
                        'analysis_rmse_vectors':analysis_rmse_vectors,
                        'rhist_uniformity_mat':rhist_uniformity_mat,
                        'inflation_stats':inflation_stats,
                        'localization_stats':localization_stats,
                        'design_penalty_pool':design_penalty_pool,
                        'ensemble_size_pool':ensemble_size_pool,
                        'localization_function_pool':localization_function_pool,
                        'init_time_ind':init_time_ind}
        pickle.dump(results_dict, open(file_name, 'wb'))

    # threshold rmse:
    try:
        rmse_mat[np.where(rmse_mat>rmse_thresh)] = np.nan
    except:
        # print("np.where(rmse_mat>rmse_thresh): ", np.where(rmse_mat>rmse_thresh))
        # raise
        pass
    try:
        rhist_uniformity_mat[np.where(rhist_uniformity_mat>rhist_thresh)] = np.nan
    except:
        # print("np.where(rhist_uniformity_mat>rhist_thresh): ", np.where(rhist_uniformity_mat>rhist_thresh))
        # raise
        pass


    # RMSE min Ranks:
    design_penalty_pool = np.asarray(design_penalty_pool).flatten()
    ensemble_size_pool = np.asarray(ensemble_size_pool).flatten()
    rows_min_ranks, cols_min_ranks = find_min_ranks(rmse_mat[1, :, :])  # from analysis
    rmse_design_penalty_opt = []
    for r in rows_min_ranks:
        if np.isnan(r):
            rmse_design_penalty_opt.append(np.nan)
        else:
            rmse_design_penalty_opt.append(design_penalty_pool[r])
    rmse_design_penalty_opt = np.array(rmse_design_penalty_opt)

    rmse_ens_size_opt = []
    for r in cols_min_ranks:
        if np.isnan(r):
            rmse_ens_size_opt.append(np.nan)
        else:
            rmse_ens_size_opt.append(ensemble_size_pool[r])
    rmse_ens_size_opt = np.array(rmse_ens_size_opt)

    # Rank Hist Ranks:
    rhist_design_penalty_opt_multi_crit = []
    rhist_ens_size_opt_multi_crit = []
    for i in xrange(3):
        rows_min_ranks, cols_min_ranks = find_min_ranks(rhist_uniformity_mat[1, :, :, i])  # from analysis
        rhist_design_penalty_opt = []
        for r in rows_min_ranks:
            if np.isnan(r):
                rhist_design_penalty_opt.append(np.nan)
            else:
                rhist_design_penalty_opt.append(design_penalty_pool[r])
        rhist_design_penalty_opt = np.array(rhist_design_penalty_opt)
        rhist_design_penalty_opt_multi_crit.append(rhist_design_penalty_opt)

        rhist_ens_size_opt = []
        for r in cols_min_ranks:
            if np.isnan(r):
                rhist_ens_size_opt.append(np.nan)
            else:
                rhist_ens_size_opt.append(ensemble_size_pool[r])
        rhist_ens_size_opt = np.array(rhist_ens_size_opt)
        rhist_ens_size_opt_multi_crit.append(rhist_ens_size_opt)

    # sys.exit("Terminating")  # break
    # Start Plotting
    if not plot_Lcurve_only:
        #
        legend_labels = [r"KL-Distance to $\mathcal{U}$",  r"Distance to $\mathcal{U}$", r"Distance of fitted $\beta$ to $\mathcal{U}$"]
        for i, rhist_ens_size_opt in enumerate(rhist_ens_size_opt_multi_crit):
            # Create a 2D-Plot with lines
            fig0 = plt.figure(facecolor='white')
            ax = fig0.add_subplot(111)
            #
            ax.set_xlabel(r'Ensemble size ($N_{\rm ens}$)')
            ax.set_ylabel(r'Penalty ($\alpha$)')
            ax.set_xticks(ensemble_size_pool)
            ax.set_xticklabels(ensemble_size_pool)

            ms = 65
            ax.scatter(ensemble_size_pool, rmse_design_penalty_opt, ms, c='maroon', marker='^', label='RMSE')
            # ax.scatter(rmse_ens_size_opt, design_penalty_pool, ms, c='olivedrab', marker='>', label=' ')

            kl_label = legend_labels[i]
            ax.scatter(ensemble_size_pool, rhist_design_penalty_opt_multi_crit[i], ms+10, c='darkblue', marker='2', label=kl_label)
            # ax.scatter(rhist_ens_size_opt_multi_crit[i], design_penalty_pool, ms+10, c='red', marker='4', label=' ')

            # ax.legend(loc='best', fancybox=True, framealpha=alpha)
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            #
            file_name = '%s_analysis_rmse_scatter_%d.%s' % (file_prefix, i, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
            #

        #
        grid_x = []
        for i in xrange(len(ensemble_size_pool)-1):
            grid_x.append((ensemble_size_pool[i]+ensemble_size_pool[i+1])/2.0)

        grid_y = []
        for i in xrange(len(design_penalty_pool)-1):
            grid_y.append((design_penalty_pool[i]+design_penalty_pool[i+1])/2.0)

        if False:
            extent = [ensemble_size_pool.min()-5, ensemble_size_pool.max()+5,
                    design_penalty_pool.min()-0.03, design_penalty_pool.max()+0.03]
        else:
            extent = None

        # X and Y ticks, ticklabels:
        xticks = np.arange(0, len(ensemble_size_pool), max(len(ensemble_size_pool)/12, 1))
        xticklabels = [ensemble_size_pool[i] for i in xticks]
        yticks = np.arange(0, len(design_penalty_pool), max(len(design_penalty_pool)/10, 1))
        yticklabels = [design_penalty_pool[i] for i in yticks]

        for log_scale in [False, True]:
            if log_scale:
                postfix = '_logscale'
            else:
                postfix = ''

            #
            # Plot RMSE checkboards (forecast and analysis):
            if log_scale:
                data = np.log(rmse_mat[0,:,:].squeeze()).T
            else:
                data = rmse_mat[0,:,:].squeeze().T
            #
            fig1 = plt.figure(facecolor='white')
            ax = fig1.add_subplot(111)
            cax = ax.matshow(data, aspect='auto', origin='lower', extent=extent, interpolation='nearest', cmap=cmap)
            ax.set_xlabel(r'Ensemble size ($N_{\rm ens}$)')
            ax.set_ylabel(r'Penalty ($\alpha$)')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.xaxis.set_ticks_position('bottom')

            # ax.set_xlim(ensemble_size_pool.min()-5, ensemble_size_pool.max()+5)
            # ax.set_ylim(design_penalty_pool.min()-0.03, design_penalty_pool.max()+0.03)
            fig1.colorbar(cax)
            #
            file_name = '%s_forecast_rmse%s.%s' % (file_prefix, postfix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            # Plot RMSE checkboards (forecast and analysis):
            if log_scale:
                data = np.log(rmse_mat[1,:,:].squeeze()).T
            else:
                data = rmse_mat[1,:,:].squeeze().T
            #
            fig2 = plt.figure(facecolor='white')
            ax = fig2.add_subplot(111)
            cax = ax.matshow(data, aspect='auto', origin='lower', extent=extent, interpolation='nearest', cmap=cmap)
            ax.set_xlabel(r'Ensemble size ($N_{\rm ens}$)')
            ax.set_ylabel(r'Penalty ($\alpha$)')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.xaxis.set_ticks_position('bottom')
            # ax.set_xlim(ensemble_size_pool.min()-5, ensemble_size_pool.max()+5)
            # ax.set_ylim(design_penalty_pool.min()-0.03, design_penalty_pool.max()+0.03)
            fig2.colorbar(cax)
            #
            file_name = file_name = '%s_analysis_rmse%s.%s' % (file_prefix, postfix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            plt.close('all')

        # Plot Rankhistogram uniformity checkboard:
        for i in xrange(3):
            data = rhist_uniformity_mat[0,:,:, i].squeeze().T
            #
            fig3 = plt.figure(facecolor='white')
            ax = fig3.add_subplot(111)
            cax = ax.matshow(data, aspect='auto', origin='lower', extent=extent, interpolation='nearest', cmap=cmap)
            ax.set_xlabel(r'Ensemble size ($N_{\rm ens}$)')
            ax.set_ylabel(r'Penalty ($\alpha$)')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.xaxis.set_ticks_position('bottom')
            fig3.colorbar(cax)
            #
            file_name = file_name = '%s_forecast_rhist_uniformity_%d.%s' % (file_prefix, i, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            data = rhist_uniformity_mat[1,:,:, i].squeeze().T
            #
            fig4 = plt.figure(facecolor='white')
            ax = fig4.add_subplot(111)
            cax = ax.matshow(data, aspect='auto', origin='lower', extent=extent, interpolation='nearest', cmap=cmap)
            ax.set_xlabel(r'Ensemble size ($N_{\rm ens}$)')
            ax.set_ylabel(r'Penalty ($\alpha$)')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.xaxis.set_ticks_position('bottom')
            fig4.colorbar(cax)
            #
            file_name = file_name = '%s_analysis_rhist_uniformity_%d.%s' % (file_prefix, i, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
        #
        plt.close('all')

    # Move on, to the L-Curve plots
    #
    if inflation_stats is not None:
        target_stats = inflation_stats
    elif localization_stats is not None:
        target_stats = localization_stats
    else:
        target_stats = None


    # Now, generate L-Curve plots
    # 1- 3D plot of L-curves over assimilation times in the testing timespan
    #
    if target_stats is not None:
        target_ensemble_size = 25  # ensemble size at which plot L-curves
        ens_ind = np.where(ensemble_size_pool==target_ensemble_size)[0][0]
        LNormtarget_vals = target_stats[ens_ind, :, 2, init_time_ind:].squeeze()
        PostTrace_target_vals = target_stats[ens_ind, :, 0, init_time_ind:].squeeze()
        target_times = analysis_times[init_time_ind: ] 

        # Threshold Posterior trace values
        l_inds = np.where(PostTrace_target_vals<post_trace_thresh_l)
        u_inds = np.where(PostTrace_target_vals>post_trace_thresh_u)
        if l_inds[0].size>0 or u_inds[0].size>0:
            print("Some indexes of PostTrace_target_vals out of limit")
            print("l_inds", l_inds)
            print("u_inds", u_inds)
        PostTrace_target_vals[l_inds] = np.nan
        PostTrace_target_vals[u_inds] = np.nan
        LNormtarget_vals[l_inds] = np.nan
        LNormtarget_vals[u_inds] = np.nan
        #
        l_inds = np.where(LNormtarget_vals<LNormthresh_l)
        u_inds = np.where(LNormtarget_vals>LNormthresh_u)
        if l_inds[0].size>0 or u_inds[0].size>0:
            print("Some indexes of LNormtarget_vals out of limit")
            print("l_inds", l_inds)
            print("u_inds", u_inds)
        LNormtarget_vals[l_inds] = np.nan
        LNormtarget_vals[u_inds] = np.nan
        PostTrace_target_vals[l_inds] = np.nan
        PostTrace_target_vals[u_inds] = np.nan


        min_x = target_times[0]  # minimum on Time axis
        max_x = target_times[-1]+(target_times[-1]-target_times[-2])
        min_y = np.nanmin(LNormtarget_vals) - 0.1
        max_y = np.nanmax(LNormtarget_vals) + 0.1
        min_z = np.nanmin(PostTrace_target_vals) - 0.01
        max_z = np.nanmax(PostTrace_target_vals) + 0.01


        # get optimal choice of penalty parameter for each timepoint:
        target_analysis_rmse = analysis_rmse_vectors[ens_ind,:,:].squeeze()
        min_rmse_vals = np.nanmin(target_analysis_rmse, 0)
        min_rmse_inds = []
        for j in xrange(min_rmse_vals.size):
            min_ind = np.where(target_analysis_rmse[:, j]==min_rmse_vals[j])[0][0]
            min_rmse_inds.append(min_ind)
        min_rmse_inds = np.asarray(min_rmse_inds)
        min_rmse_inds = min_rmse_inds[init_time_ind+1: ]
        #
        near_optimal_penalties = np.empty_like(min_rmse_inds, dtype=np.float)
        near_optimal_penalties[:] = np.nan
        near_optimal_objective = near_optimal_penalties.copy()
        near_optimal_L2 = near_optimal_objective.copy()
        #
        for i in xrange(target_times[1:].size):
            try:
                ind = min_rmse_inds[i]
                near_optimal_penalties[i] = design_penalty_pool[ind]
                near_optimal_objective[i] = PostTrace_target_vals[ind, i]
                near_optimal_L2[i] = LNormtarget_vals[ind, i]
            except:
                print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                pass

        X_vals = np.empty_like(LNormtarget_vals)
        for i in xrange(X_vals.shape[0]):
            X_vals[i, :] = analysis_times[init_time_ind+1:]

        #
        num_min_vals = num_min_penalties
        if not trimmed_only:
            #
            avg_target_analysis_rmse = rmse_mat[1,:,:].squeeze()
            avg_rows_min_ranks, avg_cols_min_ranks = find_min_ranks(avg_target_analysis_rmse, num_min_vals)  # Get smallest 5 RMSE entries
            avg_target_analysis_rmse = avg_target_analysis_rmse[ens_ind, :]
            _rows_min_ranks = []
            for r in avg_rows_min_ranks:
                _rows_min_ranks.append(r[ens_ind])  # minimum average RMSE for each row
            avg_rows_min_ranks = _rows_min_ranks
            #/
            avg_near_optimal_penalties = np.empty_like(min_rmse_inds, dtype=np.float)
            avg_near_optimal_penalties[:] = np.nan
            avg_near_optimal_objective = avg_near_optimal_penalties.copy()
            avg_near_optimal_L2 = avg_near_optimal_objective.copy()


            # scatter Design Penalty Pool Vs. Average RMSE
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xlims = [np.min(design_penalty_pool), np.max(design_penalty_pool)]
            xlims[0] -= design_penalty_pool[1] - design_penalty_pool[0]
            xlims[1] += design_penalty_pool[-1] - design_penalty_pool[-2]
            ax.set_xlim(xlims[0], xlims[1])
            ax.plot(design_penalty_pool, avg_target_analysis_rmse, '-d', linewidth=3)
            ylims = ax.get_ylim()
            nan_locs = np.isnan(avg_target_analysis_rmse)
            ax.scatter(design_penalty_pool[nan_locs], [ylims[0]]*np.count_nonzero(nan_locs), facecolors='red', edgecolors='red', marker='X', s=110, linewidth=2, zorder=1)
            ax.set_ylim(ylims[0], ylims[1])
            # ax = beautify_plots.show_grids(ax)
            ylabel = 'Average RMSE'
            if adaptive_inflation:
                xlabel = r'$\alpha$'
            else:
                xlabel = r'$\gamma$'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            file_name = '%s_Penalty_VS_AverageRMSE.%s' % (file_prefix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            # scatter Design Penalty Pool Vs. KL_Dist to Uniformity
            target_Rhist_kl = rhist_uniformity_mat[1,ens_ind,:, 0]
            target_Rhist_kl[nan_locs]=np.nan  # unify with RMSE
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlim(xlims[0], xlims[1])
            ax.plot(design_penalty_pool, target_Rhist_kl, '-d', linewidth=3)
            ylims = ax.get_ylim()
            rhist_nan_locs = np.isnan(target_Rhist_kl)
            nan_locs += rhist_nan_locs
            ax.scatter(design_penalty_pool[nan_locs], [ylims[0]]*np.count_nonzero(nan_locs), facecolors='red', edgecolors='red', marker='X', s=110, linewidth=2, zorder=1)
            ax.set_ylim(ylims[0], ylims[1])
            # ax = beautify_plots.show_grids(ax)
            ylabel = r'Rank histogram $D_{\rm KL}\, to\, \mathcal{U}$'
            if adaptive_inflation:
                xlabel = r'$\alpha$'
            else:
                xlabel = r'$\gamma$'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            file_name = '%s_Penalty_VS_RhistUniformity.%s' % (file_prefix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            # Combine last two plots
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlim(xlims[0], xlims[1])
            ax.plot(design_penalty_pool, avg_target_analysis_rmse, '-o', linewidth=3, label='Average RMSE')
            ax.plot(design_penalty_pool, target_Rhist_kl, '-d', linewidth=3, label=r'Rank histogram $D_{\rm KL}\, to\,\, \mathcal{U}$')
            ylims = ax.get_ylim()
            ax.scatter(design_penalty_pool[nan_locs], [ylims[0]]*np.count_nonzero(nan_locs), facecolors='red', edgecolors='red', marker='X', s=110, linewidth=2, zorder=1)
            ax.set_ylim(ylims[0], max(ylims[1], 0.12))
            # ax = beautify_plots.show_grids(ax)
            if adaptive_inflation:
                xlabel = r'$\alpha$'
            else:
                xlabel = r'$\gamma$'
            ax.set_xlabel(xlabel)
            ax.legend(loc='upper right')
            file_name = '%s_Penalty_VS_RMSERhistUniformity.%s' % (file_prefix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')


            # Surface plot of L-Curves
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_ylim(min_x, max_x)
            ax.set_xlim(min_y, max_y)
            ax.set_zlim(min_z, max_z)
            ax.plot_surface(LNormtarget_vals, X_vals, PostTrace_target_vals, alpha=alpha, zorder=0)

            markers = ['o', 'd', 'p', '*', 's']
            marker_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for v in xrange(num_min_vals)]

            for v in xrange(num_min_vals):
                avg_near_optimal_penalties[:] = np.nan
                avg_near_optimal_objective[:] = np.nan
                avg_near_optimal_L2[:] = np.nan
                for i in xrange(target_times[1:].size):
                    try:
                        ind = avg_rows_min_ranks[v]
                        avg_near_optimal_penalties[i] = design_penalty_pool[ind]
                        avg_near_optimal_objective[i] = PostTrace_target_vals[ind, i]
                        avg_near_optimal_L2[i] = LNormtarget_vals[ind, i]
                    except:
                        print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                        raise
                        # pass
                color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
                marker = markers[v%len(markers)]
                ind = avg_rows_min_ranks[v]
                if adaptive_inflation:
                    label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                else:
                    label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                ax.scatter(LNormtarget_vals[ind, :], target_times[1: ], PostTrace_target_vals[ind, :], facecolors='none', edgecolors=color, marker=marker, s=70, linewidth=3, zorder=num_min_vals-v, label=label)
            # ax.scatter(near_optimal_L2, analysis_times[init_time_ind+1: ], near_optimal_objective, facecolors='none', edgecolors='red', marker='o', s=120, linewidth=3, zorder=1)
            ax.set_ylabel("Time")
            if adaptive_inflation:
                ax.set_xlabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
                zlabel = r"$\Psi^{\rm %s}(\mathbf{\lambda})$" % "Infl"
                ax.set_zlabel(zlabel)
            else:
                ax.set_xlabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
                zlabel = r"$\Psi^{\rm %s}(\mathbf{L})$" % "Loc"
                ax.set_zlabel(zlabel)

            # Update limits and lables of time index:
            time_lables = np.asarray(ax.get_yticks(), dtype=int)
            ax.set_yticks(time_lables)
            ax.set_yticklabels(time_lables*10)
            
            ax.set_ylim(min_x, max_x)
            ax.set_xlim(min_y, max_y)

            ax.legend(loc='upper center', bbox_to_anchor=(-0.20, 0.9), ncol=1, fancybox=True, shadow=True, fontsize=12)
            #
            file_name = '%s_LCurve_surface.%s' % (file_prefix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            # Countourf representation of the Surface plot of L-Curves
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_ylim(min_x, max_x)
            ax.set_xlim(min_y, max_y)
            cs = ax.pcolor(LNormtarget_vals, X_vals, PostTrace_target_vals, cmap='terrain', zorder=0)
            fig.colorbar(cs, ax=ax, shrink=0.9)

            markers = ['o', 'd', 'p', '*', 's']
            # marker_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for v in xrange(num_min_vals)]

            for v in xrange(num_min_vals):
                avg_near_optimal_penalties[:] = np.nan
                avg_near_optimal_objective[:] = np.nan
                avg_near_optimal_L2[:] = np.nan
                for i in xrange(target_times[1:].size):
                    try:
                        ind = avg_rows_min_ranks[v]
                        avg_near_optimal_penalties[i] = design_penalty_pool[ind]
                        avg_near_optimal_objective[i] = PostTrace_target_vals[ind, i]
                        avg_near_optimal_L2[i] = LNormtarget_vals[ind, i]
                    except:
                        print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                        raise
                        # pass
                color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
                marker = markers[v%len(markers)]
                ind = avg_rows_min_ranks[v]
                if adaptive_inflation:
                    label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                else:
                    label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                # ax.scatter(LNormtarget_vals[ind, :], target_times[1: ], PostTrace_target_vals[ind, :], facecolors='none', edgecolors=color, marker=marker, s=70, linewidth=3, zorder=num_min_vals-v, label=label)
                ax.scatter(LNormtarget_vals[ind, :], target_times[1: ], facecolors='none', edgecolors='black', marker=marker, s=50, zorder=num_min_vals-v, label=label)
            # ax.scatter(near_optimal_L2, analysis_times[init_time_ind+1: ], near_optimal_objective, facecolors='none', edgecolors='red', marker='o', s=120, linewidth=3, zorder=1)
            ax.set_ylabel("Time")
            if adaptive_inflation:
                ax.set_xlabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
            else:
                ax.set_xlabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)

            # Update limits and lables of time index:
            time_lables = np.asarray(ax.get_yticks(), dtype=int)
            ax.set_yticks(time_lables)
            ax.set_yticklabels(time_lables*10)
            
            ax.set_ylim(min_x, max_x)
            ax.set_xlim(min_y, max_y)
            # ax.legend(loc='upper center', bbox_to_anchor=(-0.05, 0.0), ncol=2, fancybox=True, shadow=True, fontsize=12)
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode='expand', borderaxespad=0.)
            #
            file_name = '%s_LCurve_contourf.%s' % (file_prefix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
            plt.show()  # Adjust plot manually!
            # sys.exit("Terminating as requested")

            # wire plot of L-Curves
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_ylim(min_x, max_x)
            ax.set_xlim(min_y, max_y)
            ax.set_zlim(min_z, max_z)
            ax.plot_wireframe(LNormtarget_vals, X_vals, PostTrace_target_vals, alpha=alpha, zorder=0)

            for v in xrange(num_min_vals):
                avg_near_optimal_penalties[:] = np.nan
                avg_near_optimal_objective[:] = np.nan
                avg_near_optimal_L2[:] = np.nan
                for i in xrange(target_times[1:].size):
                    try:
                        ind = avg_rows_min_ranks[v]
                        avg_near_optimal_penalties[i] = design_penalty_pool[ind]
                        avg_near_optimal_objective[i] = PostTrace_target_vals[ind, i]
                        avg_near_optimal_L2[i] = LNormtarget_vals[ind, i]
                    except:
                        print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                        pass
                color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
                marker = markers[v%len(markers)]
                ind = avg_rows_min_ranks[v]
                if adaptive_inflation:
                    label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                else:
                    label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                ax.scatter(LNormtarget_vals[ind, :], target_times[1: ], PostTrace_target_vals[ind, :], facecolors='none', edgecolors=color, marker=marker, s=70, linewidth=3, zorder=num_min_vals-v, label=label)
            # ax.scatter(near_optimal_L2, analysis_times[init_time_ind+1: ], near_optimal_objective, facecolors='none', edgecolors='red', marker='o', s=120, linewidth=3, zorder=1)
            ax.set_ylabel("Time")
            if adaptive_inflation:
                ax.set_xlabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
                zlabel = r"$\Psi^{\rm %s}(\mathbf{\lambda})$" % "Infl"
                ax.set_zlabel(zlabel)
            else:
                ax.set_xlabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
                zlabel = r"$\Psi^{\rm %s}(\mathbf{L})$" % "Loc"
                ax.set_zlabel(zlabel)

            # Update limits and lables of time index:
            time_lables = np.asarray(ax.get_yticks(), dtype=int)
            ax.set_yticks(time_lables)
            ax.set_yticklabels(time_lables*10)
            
            ax.set_ylim(min_x, max_x)
            ax.set_xlim(min_y, max_y)

            ax.legend(loc='upper center', bbox_to_anchor=(-0.20, 0.9), ncol=1, fancybox=True, shadow=True, fontsize=12)
            #
            file_name = '%s_LCurve_wireframe.%s' % (file_prefix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

            # scatter plot of L-Curves
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_ylim(min_x, max_x)
            ax.set_xlim(min_y, max_y)
            ax.set_zlim(min_z, max_z)
            ax.scatter(LNormtarget_vals, X_vals, PostTrace_target_vals)
            ax.set_ylabel("Time")
            if adaptive_inflation:
                ax.set_xlabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
                zlabel = r"$\Psi^{\rm %s}(\mathbf{\lambda})$" % "Infl"
                ax.set_zlabel(zlabel)
            else:
                ax.set_xlabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
                zlabel = r"$\Psi^{\rm %s}(\mathbf{L})$" % "Loc"
                ax.set_zlabel(zlabel)

            ax.legend(loc='upper center', bbox_to_anchor=(-0.20, 0.9), ncol=1, fancybox=True, shadow=True, fontsize=12)
            #
            file_name = '%s_LCurve_scatter.%s' % (file_prefix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
            plt.close('all')
            
            if plot_individual_lcurves:
                # Single line plots (each at an assimilation time):
                for t_ind, t in enumerate(target_times[1: ]):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    #
                    ax.set_xlim(min_y, max_y)
                    ax.set_ylim(min_z, max_z)
                    if show_grids:
                        ax = beautify_plots.show_grids(ax)
                    #
                    ax.plot(LNormtarget_vals[:, t_ind], PostTrace_target_vals[:, t_ind], '-o', linewidth=3, markersize=4, alpha=alpha, zorder=0)
                    for v in xrange(num_min_vals):
                        try:
                            ind = avg_rows_min_ranks[v]
                            avg_near_optimal_penalty = design_penalty_pool[ind]
                            avg_near_optimal_objective = PostTrace_target_vals[ind, t_ind]
                            avg_near_optimal_L2 = LNormtarget_vals[ind, t_ind]
                        except:
                            print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                            pass
                        color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
                        marker = markers[v%len(markers)]
                        ind = avg_rows_min_ranks[v]
                        if adaptive_inflation:
                            label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                        else:
                            label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                        ax.scatter(avg_near_optimal_L2, avg_near_optimal_objective, facecolors='none', edgecolors='red', marker=marker, s=100, linewidth=3, zorder=num_min_vals-v, label=label)
                    # ax.scatter([near_optimal_L2[t_ind]], near_optimal_objective[t_ind], facecolors='none', edgecolors='red', marker='o', s=140, label=lbl, linewidth=3, zorder=1)

                    if adaptive_inflation:
                        ax.set_xlabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
                        ax.set_ylabel(r"$\Psi^{\rm Infl}(\mathbf{\lambda})$")
                    else:
                        ax.set_xlabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
                        ax.set_ylabel(r"$\Psi^{\rm Loc}(\mathbf{L})$")
                    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
                    if adaptive_inflation:
                        loc = 'upper left'
                    else:
                        loc = 'upper right'
                    ax.legend(loc=loc, ncol=1, fancybox=True, shadow=True, fontsize=12)
                    #
                    file_name = '%s_LCurve_timeind_%d.%s' % (file_prefix, t_ind, plots_format)
                    file_name = os.path.join(results_dir, file_name)
                    print("Plotting: %s " % file_name)
                    plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
                    plt.close(fig)

            # Time-trimmed Contourf representation of prvious plot
            init_t_ind = target_times.size * 3/4
            fin_t_ind = target_times.size - 1
            _min_x = target_times[init_t_ind]  # minimum on Time axis
            _max_x = target_times[fin_t_ind]+(target_times[-1]-target_times[-2])
            _min_y = np.nanmin(LNormtarget_vals[:, init_t_ind:fin_t_ind+1]) - 0.1
            _max_y = np.nanmax(LNormtarget_vals[:, init_t_ind:fin_t_ind+1]) + 0.1
            _min_z = np.nanmin(PostTrace_target_vals[:, init_t_ind:fin_t_ind+1]) - 0.01
            _max_z = np.nanmax(PostTrace_target_vals[:, init_t_ind:fin_t_ind+1]) + 0.01
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cs = ax.pcolor(LNormtarget_vals[:, init_t_ind:fin_t_ind+1], X_vals[:, init_t_ind:fin_t_ind+1], PostTrace_target_vals[:, init_t_ind:fin_t_ind+1], cmap='terrain', zorder=0)
            fig.colorbar(cs, ax=ax, shrink=0.9)
            for v in xrange(num_min_vals):
                avg_near_optimal_penalties[:] = np.nan
                avg_near_optimal_objective[:] = np.nan
                avg_near_optimal_L2[:] = np.nan
                for i in xrange(target_times[1:].size):
                    try:
                        ind = avg_rows_min_ranks[v]
                        avg_near_optimal_penalties[i] = design_penalty_pool[ind]
                        avg_near_optimal_objective[i] = PostTrace_target_vals[ind, i]
                        avg_near_optimal_L2[i] = LNormtarget_vals[ind, i]
                    except:
                        print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                        raise
                        # pass
                color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
                marker = markers[v%len(markers)]
                ind = avg_rows_min_ranks[v]
                if adaptive_inflation:
                    label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                else:
                    label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                # ax.scatter(LNormtarget_vals[ind, :], target_times[1: ], PostTrace_target_vals[ind, :], facecolors='none', edgecolors=color, marker=marker, s=70, linewidth=3, zorder=num_min_vals-v, label=label)
                ax.scatter(LNormtarget_vals[ind, init_t_ind:fin_t_ind+1], target_times[1+init_t_ind:fin_t_ind+1 ], facecolors='none', edgecolors='black', marker=marker, s=60, zorder=num_min_vals-v, label=label)
            # ax.scatter(near_optimal_L2, analysis_times[init_time_ind+1: ], near_optimal_objective, facecolors='none', edgecolors='red', marker='o', s=120, linewidth=3, zorder=1)
            ax.set_ylabel("Time")
            if adaptive_inflation:
                ax.set_xlabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
            else:
                ax.set_xlabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
            
            time_lables = np.asarray(ax.get_yticks(), dtype=int)
            ax.set_yticks(time_lables)
            ax.set_yticklabels(time_lables*10)
            ax.set_ylim(_min_x, _max_x)
            ax.set_xlim(_min_y, _max_y)
            

            # ax.legend(loc='upper center', bbox_to_anchor=(-0.05, 0.0), ncol=2, fancybox=True, shadow=True, fontsize=12)
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=16)
            #
            file_name = '%s_LCurve_Contourf_TimeTrimmed.%s' % (file_prefix, plots_format)
            file_name = os.path.join(results_dir, file_name)
            print("Plotting: %s " % file_name)
            plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
            plt.show()  # Adjust plot manually!


        # Plot everythin trimmed after third index; need to show that weird behaviour for tiny penalties
        trim_ind = 3
        #
        X_vals = X_vals[trim_ind:, :]
        LNormtarget_vals = LNormtarget_vals[trim_ind:, :]
        PostTrace_target_vals = PostTrace_target_vals[trim_ind:, :]
        # get optimal choice of penalty parameter for each timepoint:
        target_analysis_rmse = analysis_rmse_vectors[ens_ind, trim_ind:, :].squeeze()
        min_rmse_vals = np.nanmin(target_analysis_rmse, 0)
        min_rmse_inds = []
        for j in xrange(min_rmse_vals.size):
            min_ind = np.where(target_analysis_rmse[:, j]==min_rmse_vals[j])[0][0]
            min_rmse_inds.append(min_ind)
        min_rmse_inds = np.asarray(min_rmse_inds)
        min_rmse_inds = min_rmse_inds[init_time_ind+1: ]
        #
        near_optimal_penalties = np.empty_like(min_rmse_inds, dtype=np.float)
        near_optimal_penalties[:] = np.nan
        near_optimal_objective = near_optimal_penalties.copy()
        near_optimal_L2 = near_optimal_objective.copy()
        #
        for i in xrange(target_times[1:].size):
            try:
                ind = min_rmse_inds[i]
                near_optimal_penalties[i] = design_penalty_pool[ind]
                near_optimal_objective[i] = PostTrace_target_vals[ind, i]
                near_optimal_L2[i] = LNormtarget_vals[ind, i]
            except:
                print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                pass

        # update plots limits
        min_x = target_times[0]  # minimum on Time axis
        max_x = target_times[-1]+(target_times[-1]-target_times[-2])
        min_y = np.nanmin(LNormtarget_vals) - 0.1
        max_y = np.nanmax(LNormtarget_vals) + 0.1
        min_z = np.nanmin(PostTrace_target_vals) - 0.01
        max_z = np.nanmax(PostTrace_target_vals) + 0.01

        # Surface plot of L-Curves
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        X_vals = np.empty_like(LNormtarget_vals)
        for i in xrange(X_vals.shape[0]):
            X_vals[i, :] = analysis_times[init_time_ind+1:]
        ax.plot_surface(X_vals, LNormtarget_vals, PostTrace_target_vals, alpha=alpha, zorder=0)
        ax.scatter(target_times[1: ], near_optimal_L2, near_optimal_objective, facecolors='none', edgecolors='black', marker='o', s=120, linewidth=3, zorder=1)
        ax.set_xlabel("Time")
        if adaptive_inflation:
            ax.set_ylabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
            zlabel = r"$\Psi^{\rm %s}(\mathbf{\lambda})$" % "Infl"
            ax.set_zlabel(zlabel)
        else:
            ax.set_ylabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
            zlabel = r"$\Psi^{\rm %s}(\mathbf{L})$" % "Loc"
            ax.set_zlabel(zlabel)

        # Update limits and lables of time index:
        time_lables = np.asarray(ax.get_yticks(), dtype=int)
        ax.set_xticks(time_lables)
        ax.set_xticklabels(time_lables*10)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        #
        file_name = '%s_LCurve_surface_trimmed_%d.%s' % (file_prefix, trim_ind, plots_format)
        file_name = os.path.join(results_dir, file_name)
        print("Plotting: %s " % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
        

        # Plot things using average
        avg_target_analysis_rmse = rmse_mat[1,:,:].squeeze()
        avg_rows_min_ranks, avg_cols_min_ranks = find_min_ranks(avg_target_analysis_rmse, num_min_vals)  # Get smallest 5 RMSE entries
        avg_target_analysis_rmse = avg_target_analysis_rmse[ens_ind, :]
        _rows_min_ranks = []
        for r in avg_rows_min_ranks:
            _rows_min_ranks.append(r[ens_ind])  # minimum average RMSE for each row
        avg_rows_min_ranks = _rows_min_ranks
        avg_rmse_best_penalties = design_penalty_pool[avg_rows_min_ranks]

        avg_near_optimal_penalties = np.empty_like(min_rmse_inds, dtype=np.float)
        avg_near_optimal_penalties[:] = np.nan
        avg_near_optimal_objective = avg_near_optimal_penalties.copy()
        avg_near_optimal_L2 = avg_near_optimal_objective.copy()

        # Contourf representation of prvious plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(min_x, max_x)
        ax.set_xlim(min_y, max_y)
        cs = ax.pcolor(LNormtarget_vals, X_vals, PostTrace_target_vals, cmap='terrain', zorder=0)
        fig.colorbar(cs, ax=ax, shrink=0.9)
        for v in xrange(num_min_vals):
            avg_near_optimal_penalties[:] = np.nan
            avg_near_optimal_objective[:] = np.nan
            avg_near_optimal_L2[:] = np.nan
            for i in xrange(target_times[1:].size):
                try:
                    ind = avg_rows_min_ranks[v]
                    avg_near_optimal_penalties[i] = design_penalty_pool[ind]
                    avg_near_optimal_objective[i] = PostTrace_target_vals[ind-trim_ind, i]
                    avg_near_optimal_L2[i] = LNormtarget_vals[ind-trim_ind, i]
                except:
                    print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                    raise
                    # pass
            color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
            marker = markers[v%len(markers)]
            ind = avg_rows_min_ranks[v]
            if adaptive_inflation:
                label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            else:
                label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            # ax.scatter(LNormtarget_vals[ind, :], target_times[1: ], PostTrace_target_vals[ind, :], facecolors='none', edgecolors=color, marker=marker, s=70, linewidth=3, zorder=num_min_vals-v, label=label)
            ax.scatter(LNormtarget_vals[ind-trim_ind, :], target_times[1: ], facecolors='none', edgecolors='black', marker=marker, s=60, zorder=num_min_vals-v, label=label)
        # ax.scatter(near_optimal_L2, analysis_times[init_time_ind+1: ], near_optimal_objective, facecolors='none', edgecolors='red', marker='o', s=120, linewidth=3, zorder=1)
        ax.set_ylabel("Time")
        if adaptive_inflation:
            ax.set_xlabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
        else:
            ax.set_xlabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)

        # Update limits and lables of time index:
        time_lables = np.asarray(ax.get_yticks(), dtype=int)
        ax.set_yticks(time_lables)
        ax.set_yticklabels(time_lables*10)
        ax.set_ylim(min_x, max_x)
        ax.set_xlim(min_y, max_y)
        # ax.legend(loc='upper center', bbox_to_anchor=(-0.05, 0.0), ncol=2, fancybox=True, shadow=True, fontsize=12)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=16)
        #
        file_name = '%s_LCurve_Contourf_trimmed_%d.%s' % (file_prefix, trim_ind, plots_format)
        file_name = os.path.join(results_dir, file_name)
        print("Plotting: %s " % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
         
        # Time-trimmed Contourf representation of prvious plot
        init_t_ind = target_times.size * 3/4
        fin_t_ind = target_times.size - 1
        _min_x = target_times[init_t_ind]  # minimum on Time axis
        _max_x = target_times[fin_t_ind]+(target_times[-1]-target_times[-2])
        _min_y = np.nanmin(LNormtarget_vals[:, init_t_ind:fin_t_ind+1]) - 0.1
        _max_y = np.nanmax(LNormtarget_vals[:, init_t_ind:fin_t_ind+1]) + 0.1
        _min_z = np.nanmin(PostTrace_target_vals[:, init_t_ind:fin_t_ind+1]) - 0.01
        _max_z = np.nanmax(PostTrace_target_vals[:, init_t_ind:fin_t_ind+1]) + 0.01
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.pcolor(LNormtarget_vals[:, init_t_ind:fin_t_ind+1], X_vals[:, init_t_ind:fin_t_ind+1], PostTrace_target_vals[:, init_t_ind:fin_t_ind+1], cmap='terrain', zorder=0)
        fig.colorbar(cs, ax=ax, shrink=0.9)
        for v in xrange(num_min_vals):
            avg_near_optimal_penalties[:] = np.nan
            avg_near_optimal_objective[:] = np.nan
            avg_near_optimal_L2[:] = np.nan
            for i in xrange(target_times[1:].size):
                try:
                    ind = avg_rows_min_ranks[v]
                    avg_near_optimal_penalties[i] = design_penalty_pool[ind]
                    avg_near_optimal_objective[i] = PostTrace_target_vals[ind-trim_ind, i]
                    avg_near_optimal_L2[i] = LNormtarget_vals[ind-trim_ind, i]
                except:
                    print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                    raise
                    # pass
            color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
            marker = markers[v%len(markers)]
            ind = avg_rows_min_ranks[v]
            if adaptive_inflation:
                label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            else:
                label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            # ax.scatter(LNormtarget_vals[ind, :], target_times[1: ], PostTrace_target_vals[ind, :], facecolors='none', edgecolors=color, marker=marker, s=70, linewidth=3, zorder=num_min_vals-v, label=label)
            ax.scatter(LNormtarget_vals[ind-trim_ind, init_t_ind:fin_t_ind+1], target_times[1+init_t_ind:fin_t_ind+1 ], facecolors='none', edgecolors='black', marker=marker, s=60, zorder=num_min_vals-v, label=label)
        # ax.scatter(near_optimal_L2, analysis_times[init_time_ind+1: ], near_optimal_objective, facecolors='none', edgecolors='red', marker='o', s=120, linewidth=3, zorder=1)
        ax.set_ylabel("Time")
        if adaptive_inflation:
            ax.set_xlabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
        else:
            ax.set_xlabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
        
        time_lables = np.asarray(ax.get_yticks(), dtype=int)
        ax.set_yticks(time_lables)
        ax.set_yticklabels(time_lables*10)
        ax.set_ylim(_min_x, _max_x)
        ax.set_xlim(_min_y, _max_y)
        

        # ax.legend(loc='upper center', bbox_to_anchor=(-0.05, 0.0), ncol=2, fancybox=True, shadow=True, fontsize=12)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=16)
        #
        file_name = '%s_LCurve_Contourf_trimmed_%d_TimeTrimmed.%s' % (file_prefix, trim_ind, plots_format)
        file_name = os.path.join(results_dir, file_name)
        print("Plotting: %s " % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
        plt.show()  # Adjust plot manually!
        # sys.exit("Terminating...")

        #
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        X_vals = np.empty_like(LNormtarget_vals)
        for i in xrange(X_vals.shape[0]):
            X_vals[i, :] = analysis_times[init_time_ind+1:]
        ax.plot_surface(X_vals, LNormtarget_vals, PostTrace_target_vals, alpha=alpha, zorder=0)

        markers = ['o', 'd', 'p', '*', 's']
        marker_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for v in xrange(num_min_vals)]

        for v in xrange(num_min_vals):
            avg_near_optimal_penalties[:] = np.nan
            avg_near_optimal_objective[:] = np.nan
            avg_near_optimal_L2[:] = np.nan
            for i in xrange(target_times[1:].size):
                try:
                    ind = avg_rows_min_ranks[v]
                    avg_near_optimal_penalties[i] = design_penalty_pool[ind]
                    avg_near_optimal_objective[i] = PostTrace_target_vals[ind-trim_ind, i]
                    avg_near_optimal_L2[i] = LNormtarget_vals[ind-trim_ind, i]
                except:
                    print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                    pass
            color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
            marker = markers[v%len(markers)]
            ind = avg_rows_min_ranks[v]
            if adaptive_inflation:
                label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            else:
                label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            ax.scatter(target_times[1: ], avg_near_optimal_L2, avg_near_optimal_objective, facecolors='none', edgecolors=color, marker=marker, s=120, linewidth=3, zorder=num_min_vals-v, label=label)
        ax.set_xlabel("Time")
        if adaptive_inflation:
            ax.set_ylabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
            zlabel = r"$\Psi^{\rm %s}(\mathbf{\lambda})$" % "Infl"
            ax.set_zlabel(zlabel)
        else:
            ax.set_ylabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
            zlabel = r"$\Psi^{\rm %s}(\mathbf{L})$" % "Loc"
            ax.set_zlabel(zlabel)

        ax.legend(loc='upper center', bbox_to_anchor=(-0.20, 0.9), ncol=1, fancybox=True, shadow=True, fontsize=12)
        #
        file_name = '%s_LCurve_surface_trimmed_Min_Average_%d.%s' % (file_prefix, trim_ind, plots_format)
        file_name = os.path.join(results_dir, file_name)
        print("Plotting: %s " % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')


        # wire plot of L-Curves
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        ax.plot_wireframe(X_vals, LNormtarget_vals, PostTrace_target_vals, alpha=alpha, zorder=0)
        # ax.scatter(target_times[1: ], near_optimal_L2, near_optimal_objective, facecolors='none', edgecolors='red', marker='o', s=120, linewidth=3, zorder=1)

        for v in xrange(num_min_vals):
            avg_near_optimal_penalties[:] = np.nan
            avg_near_optimal_objective[:] = np.nan
            avg_near_optimal_L2[:] = np.nan
            for i in xrange(target_times[1:].size):
                try:
                    ind = avg_rows_min_ranks[v]
                    avg_near_optimal_penalties[i] = design_penalty_pool[ind]
                    avg_near_optimal_objective[i] = PostTrace_target_vals[ind-trim_ind, i]
                    avg_near_optimal_L2[i] = LNormtarget_vals[ind-trim_ind, i]
                except:
                    print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                    pass
            color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
            marker = markers[v%len(markers)]
            ind = avg_rows_min_ranks[v]
            if adaptive_inflation:
                label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            else:
                label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            ax.scatter(target_times[1: ], avg_near_optimal_L2, avg_near_optimal_objective, facecolors='none', edgecolors=color, marker=marker, s=120, linewidth=3, zorder=num_min_vals-v, label=label)

        ax.set_xlabel("Time")
        if adaptive_inflation:
            ax.set_ylabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
            zlabel = r"$\Psi^{\rm %s}(\mathbf{\lambda})$" % "Infl"
            ax.set_zlabel(zlabel)
        else:
            ax.set_ylabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
            zlabel = r"$\Psi^{\rm %s}(\mathbf{L})$" % "Loc"
            ax.set_zlabel(zlabel)
        #
        ax.legend(loc='upper center', bbox_to_anchor=(-0.20, 0.9), ncol=1, fancybox=True, shadow=True, fontsize=12)
        file_name = '%s_LCurve_wireframe_trimmed_%d.%s' % (file_prefix, trim_ind, plots_format)
        file_name = os.path.join(results_dir, file_name)
        print("Plotting: %s " % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')

        # scatter plot of L-Curves
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        ax.scatter(X_vals, LNormtarget_vals, PostTrace_target_vals)

        for v in xrange(num_min_vals):
            avg_near_optimal_penalties[:] = np.nan
            avg_near_optimal_objective[:] = np.nan
            avg_near_optimal_L2[:] = np.nan
            for i in xrange(target_times[1:].size):
                try:
                    ind = avg_rows_min_ranks[v]
                    avg_near_optimal_penalties[i] = design_penalty_pool[ind]
                    avg_near_optimal_objective[i] = PostTrace_target_vals[ind-trim_ind, i]
                    avg_near_optimal_L2[i] = LNormtarget_vals[ind-trim_ind, i]
                except:
                    print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                    pass
            color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
            marker = markers[v%len(markers)]
            ind = avg_rows_min_ranks[v]
            if adaptive_inflation:
                label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            else:
                label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
            ax.scatter(target_times[1: ], avg_near_optimal_L2, avg_near_optimal_objective, facecolors='none', edgecolors=color, marker=marker, s=120, linewidth=3, zorder=num_min_vals-v, label=label)

        ax.set_xlabel("Time")
        if adaptive_inflation:
            ax.set_ylabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
            zlabel = r"$\Psi^{\rm %s}(\mathbf{\lambda})$" % "Infl"
            ax.set_zlabel(zlabel)
        else:
            ax.set_ylabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
            zlabel = r"$\Psi^{\rm %s}(\mathbf{L})$" % "Loc"
            ax.set_zlabel(zlabel)
        #
        ax.legend(loc='upper center', bbox_to_anchor=(-0.20, 0.9), ncol=1, fancybox=True, shadow=True, fontsize=12)
        file_name = '%s_LCurve_scatter_trimmed_%d.%s' % (file_prefix, trim_ind, plots_format)
        file_name = os.path.join(results_dir, file_name)
        print("Plotting: %s " % file_name)
        plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')


        plt.close('all')



        if plot_individual_lcurves:
            # Single line plots (each at an assimilation time):
            for t_ind, t in enumerate(target_times[1: ]):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                #
                ax.set_xlim(min_y, max_y)
                ax.set_ylim(min_z, max_z)
                if show_grids:
                    ax = beautify_plots.show_grids(ax)
                #
                ax.plot(LNormtarget_vals[:, t_ind], PostTrace_target_vals[:, t_ind], '-o', linewidth=3, markersize=4, alpha=alpha, zorder=0)

                for v in xrange(num_min_vals):
                    try:
                        ind = avg_rows_min_ranks[v]
                        avg_near_optimal_penalty = design_penalty_pool[ind]
                        avg_near_optimal_objective = PostTrace_target_vals[ind-trim_ind, t_ind]
                        avg_near_optimal_L2 = LNormtarget_vals[ind-trim_ind, t_ind]
                    except:
                        print("Failed to get stats at time %s. Passing ." % str(target_times[i]))
                        pass
                    color = marker_colors[v]  # '#%06X' % np.random.randint(0, 0xFFFFFF)
                    marker = markers[v%len(markers)]
                    ind = avg_rows_min_ranks[v]
                    if adaptive_inflation:
                        label = r'$\alpha=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                    else:
                        label = r'$\gamma=%5.4f,\, RMSE=%5.4f$' %(design_penalty_pool[ind], avg_target_analysis_rmse[ind])
                    ax.scatter(avg_near_optimal_L2, avg_near_optimal_objective, facecolors='none', edgecolors='red', marker=marker, s=120, linewidth=3, zorder=num_min_vals-v, label=label)
                if False:
                    if adaptive_inflation:
                        lbl = r'$\alpha=%5.4f\,;\, \|\mathbf{\lambda}\|_%d=%4.3f\,;\, \Psi^{\rm Infl}$=%3.2e' % (near_optimal_penalty, L_Norm_num, near_optimal_L2[t_ind], near_optimal_objective[t_ind])
                    else:
                        lbl = r'$\alpha=%5.4f\,;\, \|\mathbf{L}\|_%d=%4.3f\,;\, \Psi^{\rm Loc}$=%3.2e' % (near_optimal_penalty, L_Norm_num, near_optimal_L2[t_ind], near_optimal_objective[t_ind])
                    ax.scatter([near_optimal_L2[t_ind]], near_optimal_objective[t_ind], facecolors='none', edgecolors='red', marker='o', s=140, label=lbl, linewidth=3, zorder=1)

                if adaptive_inflation:
                    ax.set_xlabel(r"$\|\mathbf{\lambda}\|_%d$" % L_Norm_num)
                    ylabel = r"$\Psi^{\rm %s}(\mathbf{\lambda})$" % "Infl"
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_xlabel(r"$\|\mathbf{L}\|_%d$" % L_Norm_num)
                    ylabel = r"$\Psi^{\rm %s}(\mathbf{L})$" % "Loc"
                    ax.set_ylabel(ylabel)

                # ax.legend(loc='upper center', bbox_to_anchor=(1.25, 0.9), ncol=1, fancybox=True, shadow=True, fontsize=12)
                ax.legend(loc='upper left', ncol=1, fancybox=True, shadow=True, fontsize=12)
                #
                file_name = '%s_LCurve_timeind_%d_trimmed_%d.%s' % (file_prefix, t_ind, trim_ind, plots_format)
                file_name = os.path.join(results_dir, file_name)
                print("Plotting: %s " % file_name)
                plt.savefig(file_name, dpi=300, facecolor='w', format=plots_format, transparent=True, bbox_inches='tight')
                plt.close(fig)

if show_plots:
    plt.show()
plt.close('all')
