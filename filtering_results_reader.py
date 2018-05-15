#!/usr/bin/env python

# This is a script to read the ouptput of EnKF, and HMC filter.

import os
import sys
import ConfigParser
import numpy as np
import scipy.io as sio
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import dates_setup
dates_setup.initialize_dates()

import dates_utility as utility
#

def_r_script_str = """
cwd <- getwd()
MVN_PATH <- "/home/attia/Downloads/MVN/R"
setwd(MVN_PATH)
for (f in list.files(pattern="*.R")) { source(f) }
for (f in list.files(pattern="*.rda")) { load(f) }

setwd(cwd)
class('hz')
library(R.matlab)

# ------ forecast ensemble reader ------
contents <- readMat('forecast_ensemble.mat')

setEPS()
postscript(file = paste('qqplot_forecast_', file_name, '.eps', sep=''), width = 7, height = 7, family = "Helvetica")
forecast_mardiaTest_results <- mardiaTest(contents$S, qqplot=TRUE)
forecast_mardiaTest_results
dev.off()
rm('forecast_mardiaTest_results')

# forecast_hzTest_results <- hzTest(contents$S, qqplot=False)
# forecast_hzTest_results
# rm('forecast_hzTest_results', 'contents')
rm('contents')
# ------ analysis ensemble reader ------
contents <- readMat('analysis_ensemble.mat')

setEPS()
postscript(file = paste("qqplot_analysis_", file_name, ".eps", sep=''), width = 7, height = 7, family = "Helvetica")
analysis_mardiaTest_results <- mardiaTest(contents$S, qqplot=TRUE)
analysis_mardiaTest_results
dev.off()
rm('analysis_mardiaTest_results')

# analysis_hzTest_results <- hzTest(contents$S, qqplot=False)
# analysis_hzTest_results
# rm('analysis_hzTest_results', 'contents')
rm('contents')

"""

def read_filter_output(out_dir_tree_structure_file, apply_statisticsl_tests=False):
    """
    Read the output of a filter (EnKF, HMC so far) from one or more cycles.

    :param out_dir_tree_structure_file: the file in which output directory structure is saved as a config-file.
            out_dir_tree_structure = dict(file_output_separate_files=file_output_separate_files,
                                          file_output_directory=file_output_directory,
                                          model_states_dir=model_states_dir,
                                          observations_dir=observations_dir,
                                          filter_statistics_dir=filter_statistics_dir,
                                          cycle_prefix=cycle_prefix
                                         )
    :return:
            reference_states,
            forecast_ensembles,
            forecast_means,
            analysis_ensembles,
            analysis_means,
            observations
    """
    # TODO: Currently I am making the reader aware that each cycle the prior can change
    
    if not os.path.isfile(out_dir_tree_structure_file):
        raise IOError("File Not Found!")

    #
    output_configs = ConfigParser.ConfigParser()
    output_configs.read(out_dir_tree_structure_file)

    section_header = 'out_dir_tree_structure'
    if not output_configs.has_section(section_header):
        raise KeyError("Couldn't find the proper section header [%s]" % section_header)
    else:
        out_dir_tree_structure = dict()
        options = output_configs.options(section_header)

        for option in options:
            if option == 'file_output_separate_files':
                option_val = output_configs.getboolean(section_header, option)
            else:
                option_val = output_configs.get(section_header, option)
            out_dir_tree_structure.update({option: option_val})

    try:
        file_output_separate_files = out_dir_tree_structure['file_output_separate_files']
        file_output_directory = out_dir_tree_structure['file_output_directory']
        model_states_dir = out_dir_tree_structure['model_states_dir']
        observations_dir = out_dir_tree_structure['observations_dir']
        filter_statistics_dir = out_dir_tree_structure['filter_statistics_dir']
        cycle_prefix = out_dir_tree_structure['cycle_prefix']
    except(KeyError, ValueError, AttributeError):
        raise KeyError("Couldn't find some the required variables in the config file")

    # Now start reading based on the structure read from configuration file
    # Will assume we have mat files containing results as instructed by QG1.5 with full ensembles saved


    model_configs_parser = ConfigParser.ConfigParser()
    model_configs_parser.read(os.path.join(file_output_directory, 'setup.dat'))
    section_header = 'Model Configs'
    if not model_configs_parser.has_section(section_header):
        raise KeyError("Couldn't find the proper section header [%s]" % section_header)
    else:
        options = model_configs_parser.options(section_header)
        model_configs = dict()
        for option in options:
            if option in ['dx', 'dy', 'model_error_variances', 'background_error_variances', 'observation_error_variances']:
                try:
                    option_val = model_configs_parser.getfloat(section_header, option)
                except:
                    option_val = model_configs_parser.get(section_header, option)
                    if option_val == 'None':
                        pass
                    else:
                        print("ValueError exception raised while reading %s\n retrieved '%s' as a string!" %(option, option_val))
                
            elif option in ['state_size', 'background_errors_covariance_localization_radius',
                            'model_errors_steps_per_model_steps', 'observation_vector_size', 'nx', 'ny', 'mrefin']:
                try:
                    option_val = model_configs_parser.getint(section_header, option)
                except:
                    option_val = model_configs_parser.get(section_header, option)
                    if option_val == 'None':
                        pass
                    else:
                        print("ValueError exception raised while reading %s\n retrieved '%s' as a string!" %(option, option_val))
            else:
                option_val = model_configs_parser.get(section_header, option)
            model_configs.update({option: option_val})

    # get a proper name for the folder (cycle_*) under the model_states_dir path

    # First sweep to find number of cycles:
    # get a proper name for the folder (cycle_*) under the model_states_dir path
    # read states and observations
    num_cycles = 0
    while True:
        cycle_dir = cycle_prefix + str(num_cycles)
        cycle_states_out_dir = os.path.join(model_states_dir, cycle_dir)  # full path where states will be saved for the current cycle
        if os.path.isdir(cycle_states_out_dir):
            # check for next cycle
            num_cycles += 1
        else:
            break

    if num_cycles == 0:
        return None
    else:
        cycle_dir = cycle_prefix + str(0)
        cycle_states_out_dir = os.path.join( model_states_dir, cycle_dir)  # full path where states will be saved for the current cycle
        filter_config_parser = ConfigParser.ConfigParser()
        try:
            filter_config_parser.read(os.path.join(model_states_dir, cycle_dir,'setup.dat'))
        except IOError:
            raise IOError("The file setup.dat is not found. You need the filter configurations to be saved in it to continue!")
        section_header = 'Filter Configs'
        filter_config_options = filter_config_parser.options(section_header)
        filter_configs = dict()
        for option in filter_config_options:
            if option in ['observation_time', 'forecast_time', 'analysis_time']:
                option_val = filter_config_parser.getfloat(section_header, option)
            elif option in ['forecast_first', 'apply_preprocessing', 'apply_postprocessing']:
                option_val = filter_config_parser.getboolean(section_header, option)
            elif option == 'ensemble_size':
                option_val = filter_config_parser.getint(section_header, option)
            elif option in ['prior_distribution', 'filter_name']:
                option_val = filter_config_parser.get(section_header, option)
            # elif option == 'filter_statistics':
            #     option_val = eval(filter_config_parser.get(section_header, option))
            else:
                pass
            filter_configs.update({option: option_val})
        ensemble_size = filter_configs['ensemble_size']
        filter_name = filter_configs['filter_name']
        #
        # Check the prior distribution:
        prior_distribution = filter_configs['prior_distribution'].lower()
        if prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            # filter_statistics = filter_configs['filter_statistics']
            section_header = 'GMM-Prior Configs'
            if not filter_config_parser.has_section(section_header):
                # this means the prior is a converted to a Gaussian as the number of mixture components detected is 1
                # raise KeyError("How is the GMM section header not found?")
                gmm_results = dict(gmm_num_components=dict(),
                                   gmm_weights=dict(),
                                   gmm_lables=dict(),
                                   gmm_inf_criteria=dict()
                                   )
                pass
            else:
                gmm_results = dict(gmm_num_components=dict(),
                                   gmm_weights=dict(),
                                   gmm_lables=dict(),
                                   gmm_inf_criteria=dict()
                                   )
        else:
            gmm_results = None

        #
        section_header = 'Output Configs'
        filter_config_options = filter_config_parser.options(section_header)
        output_configs = dict()
        for option in filter_config_options:
            if option in ['file_output_separate_files', 'file_output_moment_only', '']:
                option_val = filter_config_parser.getboolean(section_header, option)
            elif option in ['model_states_dir', 'observations_dir', 'filter_statistics_dir',
                            'filter_name', 'file_output_dir', 'file_output_moment_name']:
                option_val = filter_config_parser.get(section_header, option)
            else:
                pass
            output_configs.update({option:option_val})
        file_output_moment_only = output_configs['file_output_moment_only']

        #
        if os.path.isdir(cycle_states_out_dir):
            cycle_observations_out_dir = os.path.join(observations_dir, cycle_dir)
            if not os.path.isdir(cycle_observations_out_dir):
                read_observations = False
            else:
                read_observations = True
        else:
            raise IOError("How is this even possible!")

        # read filter configurations:

        #
        if not file_output_moment_only:
            contents = sio.loadmat(os.path.join(cycle_states_out_dir,'forecast_ensemble.mat'))
            state_size = contents['n']
            # ensemble_size = contents['n_sample']
        else:
            contents = sio.loadmat(os.path.join(cycle_states_out_dir,'forecast_mean.mat'))
            state_size = contents['n']

        if np.isscalar(state_size):
            pass
        elif isinstance(state_size, np.ndarray):
            for i in xrange(state_size.ndim):
                state_size = state_size[0]
        else:
            print("state_size is of unrecognized type %s" %Type(state_size))
            raise TypeError
            
        # create place-holders for state ensembles
        reference_states = np.empty((state_size, num_cycles))
        forecast_means = np.empty((state_size, num_cycles))
        analysis_means = np.empty((state_size, num_cycles))
        #
        if not file_output_moment_only:
            # create place-holders for state ensembles
            forecast_ensembles = np.empty((state_size, ensemble_size, num_cycles))
            analysis_ensembles = np.empty((state_size, ensemble_size, num_cycles))
        else:
            forecast_ensembles = None
            analysis_ensembles = None

        if read_observations:
            cycle_observations_out_dir = os.path.join( observations_dir, cycle_dir)
            contents = sio.loadmat(os.path.join(cycle_observations_out_dir, 'observation.mat'))
            observation_size = contents['m']
            if np.isscalar(observation_size):
                pass
            elif isinstance(observation_size, np.ndarray):
                for i in xrange(observation_size.ndim):
                    observation_size = observation_size[0]
            else:
                print("observation_size is of unrecognized type %s" %Type(observation_size))
                raise TypeError
                
            observations = np.empty((observation_size, num_cycles))
        else:
            observations = None

    # # place_holders for forecast and analysis times
    # forecast_times = np.empty((num_cycles))
    # analysis_times = np.empty((num_cycles))
    # observations_times = np.empty((num_cycles))

    # read times and RMSE results:
    rmse_file_name = 'rmse.dat'
    rmse_file_path = os.path.join(filter_statistics_dir, rmse_file_name)
    rmse_file_contents = np.loadtxt(rmse_file_path, skiprows=2)
    observations_times = rmse_file_contents[:, 0]
    forecast_times = rmse_file_contents[:, 1]
    analysis_times = rmse_file_contents[:, 2]
    forecast_rmse = rmse_file_contents[:, 3]
    analysis_rmse = rmse_file_contents[:, 4]
    if apply_statisticsl_tests:
        mardiaTest_results = {}

    section_header = 'Filter Configs'
    # read states and observations
    for suffix in xrange(num_cycles):
        cycle_dir = cycle_prefix + str(suffix)

        # # read times (forecast and analysis)
        # try:
        #     filter_config_parser.read(os.path.join(model_states_dir, cycle_dir, 'setup.dat'))
        # except IOError:
        #     raise IOError("The file setup.dat is not found. You need the filter configurations to be saved in it to continue!")
        # filter_config_options = filter_config_parser.options(section_header)
        # for option in filter_config_options:
        #     if option == 'observation_time':
        #         observations_times[suffix] = filter_config_parser.getfloat(section_header, option)
        #     elif option == 'forecast_time':
        #         forecast_times[suffix] = filter_config_parser.getfloat(section_header, option)
        #     elif option == 'analysis_time':
        #         analysis_times[suffix] = filter_config_parser.getfloat(section_header, option)
        #     else:
        #         pass

        # read states output:
        # reference_state
        contents = sio.loadmat(os.path.join(model_states_dir, cycle_dir, 'reference_state.mat'))
        reference_states[:, suffix]= contents['S'][:, 0].copy()  # each state is written as a column in the mat file...
        # print('reference_states', reference_states[:, suffix])

        if not file_output_moment_only:
            # forecast_ensemble
            contents = sio.loadmat(os.path.join(model_states_dir, cycle_dir, 'forecast_ensemble.mat'))
            forecast_ensembles[:, :, suffix]= contents['S'][:, :].copy()
            forecast_means[:, suffix] = np.mean(np.squeeze(forecast_ensembles[:, :, suffix]), 1)

            if apply_statisticsl_tests:
                # Run R script and analyze ensembles
                cwd = os.getcwd()
                os.chdir(os.path.join(model_states_dir, cycle_dir))
                r_script_text = "file_name <- '%s' \n %s" % (cycle_dir, def_r_script_str)
                r_script_name = 'r_analyzer.R'
                with open(r_script_name, 'w') as file:
                    file.write(r_script_text)
                os.system("R CMD BATCH %s" % r_script_name)
                
                # Now read normality test results from output file
                r_out_file_name = r_script_name + 'out'
                with open(r_out_file_name) as r_output_f:
                    r_out_text = r_output_f.readlines()
                # forecast ensemble results:
                mardia_header_indices = []
                for r_out_ln_ind in xrange(len(r_out_text)):
                    if str.find(r_out_text[r_out_ln_ind], "Mardia's Multivariate Normality Test") != -1:
                        mardia_header_indices.append(r_out_ln_ind)

                if len(mardia_header_indices) <= 1:
                    mardia_test_results = None
                else:
                    forecast_mardiaTest_results = dict()
                    for r_out_ln_ind in xrange(mardia_header_indices[0]+1, mardia_header_indices[1]):
                        line_r_out_text = r_out_text[r_out_ln_ind].strip('\n').split(':')
                        if line_r_out_text[0].strip(' ') == 'g1p':
                            forecast_mardiaTest_results.update({'g1p': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'chi.skew':
                            forecast_mardiaTest_results.update({'chi.skew': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'p.value.skew':
                            forecast_mardiaTest_results.update({'p.value.skew': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'g2p':
                            forecast_mardiaTest_results.update({'g2p': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'z.kurtosis':
                            forecast_mardiaTest_results.update({'z.kurtosis': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'p.value.kurt':
                            forecast_mardiaTest_results.update({'p.value.kurt': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'chi.small.skew':
                            forecast_mardiaTest_results.update({'chi.small.skew': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'p.value.small':
                            forecast_mardiaTest_results.update({'p.value.small': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'Result':
                            forecast_mardiaTest_results.update({'Result': line_r_out_text[1].strip(' ')})
                        else:
                            pass
                    
                    analysis_mardiaTest_results = dict()
                    for r_out_ln_ind in xrange(mardia_header_indices[0]+1, mardia_header_indices[1]):
                        line_r_out_text = r_out_text[r_out_ln_ind].strip('\n').split(':')
                        if line_r_out_text[0].strip(' ') == 'g1p':
                            analysis_mardiaTest_results.update({'g1p': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'chi.skew':
                            analysis_mardiaTest_results.update({'chi.skew': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'p.value.skew':
                            analysis_mardiaTest_results.update({'p.value.skew': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'g2p':
                            analysis_mardiaTest_results.update({'g2p': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'z.kurtosis':
                            analysis_mardiaTest_results.update({'z.kurtosis': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'p.value.kurt':
                            analysis_mardiaTest_results.update({'p.value.kurt': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'chi.small.skew':
                            analysis_mardiaTest_results.update({'chi.small.skew': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'p.value.small':
                            analysis_mardiaTest_results.update({'p.value.small': eval(line_r_out_text[1].strip(' '))})
                        elif line_r_out_text[0].strip(' ') == 'Result':
                            analysis_mardiaTest_results.update({'Result': line_r_out_text[1].strip(' ')})
                        else:
                            pass

                    mardiaTest_results[cycle_dir] = dict(forecast_mardiaTest_results=forecast_mardiaTest_results, 
                                                         analysis_mardiaTest_results=analysis_mardiaTest_results
                                                         )
                    # remove the R script file
                    os.remove(r_script_name)
                    os.chdir(cwd)
            else:
                mardiaTest_results = None                

        else:
            mardiaTest_results = None
            # forecast_mean
            contents = sio.loadmat(os.path.join(model_states_dir, cycle_dir, 'forecast_mean.mat'))
            forecast_means[:, suffix] = contents['S'][:, 0].copy()

        if not file_output_moment_only:
            # analysis_ensemble
            contents = sio.loadmat(os.path.join(model_states_dir, cycle_dir, 'analysis_ensemble.mat'))
            analysis_ensembles[:, :, suffix]= contents['S'][:, :].copy()
            analysis_means[:, suffix] = np.mean(np.squeeze(analysis_ensembles[:, :, suffix]), 1)
        else:
            # analysis_mean
            contents = sio.loadmat(os.path.join(model_states_dir, cycle_dir, 'analysis_mean.mat'))
            analysis_means[:, suffix] = contents['S'][:, 0].copy()

        # read observations
        if read_observations:
            contents = sio.loadmat(os.path.join(observations_dir, cycle_dir, 'observation.mat'))
            observations[:, suffix]= contents['Obs'][:, 0].copy()


        if gmm_results is not None:
            # read gmm_results in the case of a GMM prior
            gmm_configs = ConfigParser.ConfigParser()
            gmm_results_file = os.path.join(model_states_dir, cycle_dir, 'setup.dat')
            gmm_configs.read(gmm_results_file)

            section_header = 'GMM-Prior Configs'
            if not gmm_configs.has_section(section_header):
                # the prior is gmm but converted temporarily to Gaussian (1 mixture component is detected).
                gmm_results['gmm_num_components'].update({cycle_dir: 1})
                gmm_results['gmm_weights'].update({cycle_dir: 1})
                gmm_results['gmm_lables'].update({cycle_dir: 0})
                gmm_results['gmm_inf_criteria'].update({cycle_dir: 'None'})
                # raise KeyError("Couldn't find the proper section header [%s]" % section_header)
            #
            else:
                options = gmm_configs.options(section_header)
                for option in options:
                    if option == 'gmm_num_components':
                        option_val = gmm_configs.getint(section_header, option)
                        gmm_results['gmm_num_components'].update({cycle_dir: option_val})
                    elif option == 'gmm_weights':
                        option_str = str.strip(gmm_configs.get(section_header, option), '[]')
                        option_str = ','.join(option_str.split())
                        option_val = np.asarray(eval("[%s]" % option_str))
                        gmm_results['gmm_weights'].update({cycle_dir: option_val})
                    elif option == 'gmm_lables':
                        option_str = str.strip(gmm_configs.get(section_header, option), '[]')
                        option_str = ','.join(option_str.split())
                        option_val = np.asarray(eval("[%s]" % option_str))
                        gmm_results['gmm_lables'].update({cycle_dir: option_val})
                    elif option == 'gmm_inf_criteria':
                        option_val = gmm_configs.get(section_header, option)
                        gmm_results['gmm_inf_criteria'].update({cycle_dir: option_val})
                    else:
                        pass

    #
    return cycle_prefix, num_cycles, reference_states, forecast_ensembles, forecast_means, analysis_ensembles, \
           analysis_means, observations, forecast_times, analysis_times, observations_times, \
           forecast_rmse, analysis_rmse, filter_configs, gmm_results, model_configs, mardiaTest_results




#
if __name__ == '__main__':
    # =====================================================================
    input_args = sys.argv
    if len(input_args) > 2:
        raise IOError("Incorrect input")
    elif len(input_args) == 2:
        out_dir_tree_structure_file = input_args[1]
    elif len(input_args) == 1:
        out_dir_tree_structure_file = 'Results/Filtering_Results/output_dir_structure.txt'
    else:
        raise ValueError("This means the number of arguments is negative Hahhh!")
    # =====================================================================

    #
    # =====================================================================
    # Start reading the output of the assimilation process
    # =====================================================================
    cycle_prefix, num_cycles, reference_states, forecast_ensembles, forecast_means, analysis_ensembles, \
    analysis_means, observations, forecast_times, analysis_times, observations_times, \
    forecast_rmse, analysis_rmse, filter_configs, gmm_results, model_configs, mardiaTest_results = read_filter_output(out_dir_tree_structure_file)
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

    #
    # =====================================================================
    # Plot RMSE
    # =====================================================================
    log_scale = False
    font_size = 8
    line_width = 2
    marker_size = 4
    font = {'weight': 'bold', 'size': font_size}
    #
    fig1 = plt.figure(facecolor='white')
    plt.rc('font', **font)
    #
    if log_scale:
        plt.semilogy(forecast_times, forecast_rmse, 'r--o', linewidth=line_width, label='Forecast')
        plt.semilogy(analysis_times, analysis_rmse, 'bd-', linewidth=line_width, label=filter_name)
    else:
        plt.plot(forecast_times, forecast_rmse, 'r--o', linewidth=line_width, label='Forecast')
        plt.plot(analysis_times, analysis_rmse, 'bd-', linewidth=line_width, label=filter_name)
    #
    # Set lables and title
    plt.xlabel('Time',fontsize=font_size, fontweight='bold')
    if log_scale:
        plt.ylabel('log-RMSE', fontsize=font_size, fontweight='bold')
    else:
        plt.ylabel('RMSE', fontsize=font_size, fontweight='bold')
    plt.title('RMSE results for the model: %s' % model_name)
    #
    xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), 10)]
    plt.xticks(xlables, 10*np.arange(len(xlables)))
    # show the legend, show the plot
    plt.legend(loc='upper left')
    plt.draw()
    # =====================================================================


    # =====================================================================
    # Plot the results obtained from mardiaTest for forecast and analysis ensembles:
    # =====================================================================
       
        
    if mardiaTest_results is not None:
        g1p = np.empty((num_cycles, 2))  # first column for forecast, second for analysis ensemble
        chi_skew = np.empty((num_cycles, 2))  # first column for forecast, second for analysis ensemble
        p_value_small = np.empty((num_cycles, 2))  # first column for forecast, second for analysis ensemble
        g2p = np.empty((num_cycles, 2))  # first column for forecast, second for analysis ensemble
        z_kurtosis = np.empty((num_cycles, 2))  # first column for forecast, second for analysis ensemble
        p_value_kurt = np.empty((num_cycles, 2))  # first column for forecast, second for analysis ensemble
        p_value_skew = np.empty((num_cycles, 2))  # first column for forecast, second for analysis ensemble
        chi_small_skew = np.empty((num_cycles, 2))  # first column for forecast, second for analysis ensemble
        ensemble_multivariate_normal = np.empty((num_cycles, 2))  # first column for forecast, second for analysis ensemble
        
        for cycle in xrange(num_cycles):
            cycle_name = cycle_prefix + str(cycle)
            forecast_mardiaTest_results = mardiaTest_results[cycle_name]['forecast_mardiaTest_results']
            analysis_mardiaTest_results = mardiaTest_results[cycle_name]['analysis_mardiaTest_results']
            
            g1p[cycle, 0], g1p[cycle, 1] = forecast_mardiaTest_results['g1p'], analysis_mardiaTest_results['g1p']
            chi_skew[cycle, 0], chi_skew[cycle, 1] = forecast_mardiaTest_results['chi.skew'], analysis_mardiaTest_results['chi.skew']
            p_value_small[cycle, 0], p_value_small[cycle, 1] = forecast_mardiaTest_results['p.value.small'], analysis_mardiaTest_results['p.value.small']
            g2p[cycle, 0], g2p[cycle, 1] = forecast_mardiaTest_results['g2p'], analysis_mardiaTest_results['g2p']
            z_kurtosis[cycle, 0], z_kurtosis[cycle, 1] = forecast_mardiaTest_results['z.kurtosis'], analysis_mardiaTest_results['z.kurtosis']
            p_value_kurt[cycle, 0], p_value_kurt[cycle, 1] = forecast_mardiaTest_results['p.value.kurt'], analysis_mardiaTest_results['p.value.kurt']
            p_value_skew[cycle, 0], p_value_skew[cycle, 1] = forecast_mardiaTest_results['p.value.skew'], analysis_mardiaTest_results['p.value.skew']
            chi_small_skew[cycle, 0], chi_small_skew[cycle, 1] = forecast_mardiaTest_results['chi.small.skew'], analysis_mardiaTest_results['chi.small.skew']
            
            if str.find(forecast_mardiaTest_results['Result'], 'not') != -1:
                ensemble_multivariate_normal[cycle, 0] = 0
            else:
                ensemble_multivariate_normal[cycle, 0] = 1
                
            if str.find(analysis_mardiaTest_results['Result'], 'not') != -1:
                ensemble_multivariate_normal[cycle, 1] = 0
            else:
                ensemble_multivariate_normal[cycle, 1] = 1
                
        # TODO: Ahmed start plotting here. You can add these results to the number of mixture components plot
        font_size = 16
        line_width = 4
        marker_size = 8
        font = {'weight': 'bold', 'size': font_size}
        #
        fig = plt.figure(facecolor='white')
        plt.rc('font', **font)
        #
        plt.semilogy(np.arange(1, num_cycles+1), g1p[:,0], 'bd-', linewidth=line_width, markersize=marker_size, label='g1p')
        plt.semilogy(np.arange(1, num_cycles+1), chi_skew[:,0], 'gd-', linewidth=line_width, markersize=marker_size, label='chi.skew')
        plt.semilogy(np.arange(1, num_cycles+1), p_value_small[:,0], 'rd-', linewidth=line_width, markersize=marker_size, label='p-value.small')
        plt.semilogy(np.arange(1, num_cycles+1), g2p[:,0], 'cd-', linewidth=line_width, markersize=marker_size, label='g2p')
        plt.semilogy(np.arange(1, num_cycles+1), z_kurtosis[:,0], 'md-', linewidth=line_width, markersize=marker_size, label='z.kurtosis')
        plt.semilogy(np.arange(1, num_cycles+1), p_value_kurt[:,0], 'yd-', linewidth=line_width, markersize=marker_size, label='p-value.kurt')
        plt.semilogy(np.arange(1, num_cycles+1), p_value_skew[:,0], 'kd-', linewidth=line_width, markersize=4+marker_size, label='p-value.skew')
        # plt.plot(np.arange(1, num_cycles+1), ensemble_multivariate_normal[:,0], color='#64B5CD' , marker='>', linewidth=line_width, markersize=4+marker_size, label='Gaussian')
        
        # show the legend, show the plot
        plt.legend()
        #
        # Set lables and title
        plt.xlabel('Time',fontsize=font_size, fontweight='bold')
        plt.ylabel('Mardia-test statistics', fontsize=font_size, fontweight='bold')
        plt.xticks(np.arange(0, num_cycles+2))
        plt.yticks(np.arange(-4, 8, 2))
        plt.title('Forecast Gaussianity test')
        #
        plt.draw()
        #
        fig = plt.figure(facecolor='white')
        plt.rc('font', **font)
        #
        plt.semilogy(np.arange(1, num_cycles+1), g1p[:, 1], 'bd-', linewidth=line_width, markersize=marker_size, label='g1p')
        plt.semilogy(np.arange(1, num_cycles+1), chi_skew[:, 1], 'gd-', linewidth=line_width, markersize=marker_size, label='chi.skew')
        plt.semilogy(np.arange(1, num_cycles+1), p_value_small[:, 1], 'rd-', linewidth=line_width, markersize=marker_size, label='p-value.small')
        plt.semilogy(np.arange(1, num_cycles+1), g2p[:, 1], 'cd-', linewidth=line_width, markersize=marker_size, label='g2p')
        plt.semilogy(np.arange(1, num_cycles+1), z_kurtosis[:, 1], 'md-', linewidth=line_width, markersize=marker_size, label='z.kurtosis')
        plt.semilogy(np.arange(1, num_cycles+1), p_value_kurt[:, 1], 'yd-', linewidth=line_width, markersize=marker_size, label='p-value.kurt')
        plt.semilogy(np.arange(1, num_cycles+1), p_value_skew[:,1], 'kd-', linewidth=line_width, markersize=4+marker_size, label='p-value.skew')
        # plt.plot(np.arange(1, num_cycles+1), ensemble_multivariate_normal[:,1], color='#64B5CD' , marker='>', linewidth=line_width, markersize=4+marker_size, label='Gaussian')
        
        # show the legend, show the plot
        plt.legend()
        #
        # Set lables and title
        plt.xlabel('Time',fontsize=font_size, fontweight='bold')
        plt.ylabel('Mardia-test statistics', fontsize=font_size, fontweight='bold')
        plt.xticks(np.arange(0, num_cycles+2))
        plt.yticks(np.arange(-4, 8, 2))
        plt.title('Analysis Gaussianity test')
        #
        plt.draw()


    #
    # =====================================================================

    #
    # =====================================================================
    # Plot number of mixture components for the case of GMM prior
    # =====================================================================
    if gmm_results is not None:
        # max_num_components = 6
        gmm_num_components = np.empty(num_cycles)
        try:
            gmm_inf_criteria = gmm_results['gmm_inf_criteria']['cycle_0']
        except:
            gmm_inf_criteria = None

        gmm_inf_criteria_al_similar = True
        for cycle in xrange(num_cycles):
            cycle_name = cycle_prefix + str(cycle)
            gmm_num_components[cycle] = gmm_results['gmm_num_components'][cycle_name]
            if gmm_inf_criteria is not None:
                try:
                    if gmm_inf_criteria != gmm_results['gmm_inf_criteria'][cycle_name]:
                        gmm_inf_criteria_al_similar = False
                except(KeyError, NameError, ValueError):
                    pass

        # Start plotting and use configurations set above for RMSE plots
        fig2 = plt.figure(facecolor='white')
        plt.rc('font', **font)
        #
        if gmm_inf_criteria_al_similar and gmm_inf_criteria is not None:
            plt.plot(forecast_times[1: ], gmm_num_components, 'bd-', linewidth=line_width, markersize=marker_size, label=gmm_inf_criteria.upper())
            # show the legend, show the plot
            plt.legend()
        else:
            plt.plot(forecast_times[1: ], gmm_num_components, 'bd-', linewidth=line_width, markersize=marker_size)
        #
        # Set lables and title
        plt.xlabel('Time',fontsize=font_size, fontweight='bold')
        plt.ylabel('GMM number of components', fontsize=font_size, fontweight='bold')
        #
        xlables = [forecast_times[i] for i in xrange(len(forecast_times))]
        plt.xticks(forecast_times[1: ])
        # plt.xticks(np.arange(0, num_cycles+2))
        # plt.yticks(np.arange(max_num_components+1))
        plt.title('GMM Number of Mixture Components')
        #
        plt.draw()
        
        if mardiaTest_results is not None:
            fig = plt.figure(facecolor='white')
            plt.rc('font', **font)
            #
            plt.plot(np.arange(1, num_cycles+1), ensemble_multivariate_normal[:,0], 'b*-', linewidth=line_width, markersize=marker_size, label='Gauss-Forecast')
            plt.plot(np.arange(1, num_cycles+1), ensemble_multivariate_normal[:,1], 'ko-.', linewidth=line_width, markersize=2+marker_size, label='Gauss-Analysis')
            if gmm_inf_criteria_al_similar and gmm_inf_criteria is not None:
                plt.plot(forecast_times[1: ], gmm_num_components, 'bd-', linewidth=line_width, markersize=marker_size, label=gmm_inf_criteria.upper())
                # show the legend, show the plot
                plt.legend()
            else:
                plt.plot(forecast_times[1: ], gmm_num_components, 'bd-', linewidth=line_width, markersize=marker_size)
            # show the legend, show the plot
            plt.legend()
            #
            # Set lables and title
            plt.xlabel('Time',fontsize=font_size, fontweight='bold')
            plt.ylabel('Mardia-test, & num. GMM comp.', fontsize=font_size, fontweight='bold')
            plt.xticks(np.arange(0, num_cycles+2))
            plt.yticks(np.arange(max_num_components+1))
            plt.title('Mardia-test, & number GMM components')
            #
            plt.draw() 

    #
    # =====================================================================
    # Plot Rank Histograms for forecast and analysis ensemble
    # =====================================================================
    if forecast_ensembles is not None:
        f_out = utility.rank_hist(forecast_ensembles,
                          reference_states, first_var=0, 
                          last_var=None, 
                          var_skp=5, 
                          draw_hist=True, 
                          hist_type='freq', 
                          first_time_ind=2, 
                          last_time_ind=None,
                          time_ind_skp=1, 
                          hist_title='forecast rank histogram',
                          hist_max_height=None
                  )
                  
    #
    if analysis_ensembles is not None:
        a_out = utility.rank_hist(analysis_ensembles,
                          reference_states, first_var=0, 
                          last_var=None, 
                          var_skp=5, 
                          draw_hist=True, 
                          hist_type='relfreq', 
                          first_time_ind=2, 
                          last_time_ind=None,
                          time_ind_skp=1, 
                          hist_title='analysis rank histogram',
                          hist_max_height=None
                  )
    # =====================================================================


    #
    # =====================================================================
    # Plot the states based on the model name.
    # This supports only lorenz96 an QG-1.5 model results
    # =====================================================================
    model_name= model_name.lower()
    if model_name == 'lorenz96':
        pass
    elif model_name == 'qg-1.5':
        # from matplotlib import pyplot as plt
        # import matplotlib.animation as animation

        nx = int(np.sqrt(state_size))
        ny = int(state_size/nx)

        fig = plt.figure(facecolor='white')
        fig.suptitle("Reference Trajectory",  fontsize=font_size)
        ims = []
        for ind in xrange(num_cycles):
            state = np.reshape(np.squeeze(reference_states[:, ind]), (nx, ny), order='F')
            imgplot = plt.imshow(state, animated=True)
            if ind == 0:
                plt.colorbar()
            else:
                plt.autoscale()
            ims.append([imgplot])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        plt.draw()

        fig = plt.figure(facecolor='white')
        fig.suptitle("Forecast Trajectory",  fontsize=font_size)
        ims = []
        for ind in xrange(num_cycles):
            state = np.reshape(np.squeeze(forecast_means[:, ind]), (nx, ny), order='F')
            imgplot = plt.imshow(state, animated=True)
            if ind == 0:
                plt.colorbar()
            else:
                plt.autoscale()
            ims.append([imgplot])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        plt.draw()
        #
        fig = plt.figure(facecolor='white')
        fig.suptitle("Forecast Errors",  fontsize=font_size)
        ims = []
        for ind in xrange(num_cycles):
            state = np.reshape(np.squeeze(forecast_means[:, ind]-reference_states[:, ind]), (nx, ny), order='F')
            imgplot = plt.imshow(state, animated=True)
            if ind == 0:
                plt.colorbar()
            else:
                plt.autoscale()
            ims.append([imgplot])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        plt.draw()

        fig = plt.figure(facecolor='white')
        fig.suptitle("Analysis Trajectory",  fontsize=font_size)
        ims = []
        for ind in xrange(num_cycles):
            state = np.reshape(np.squeeze(analysis_means[:, ind]), (nx, ny), order='F')
            imgplot = plt.imshow(state, animated=True)
            if ind == 0:
                plt.colorbar()
            else:
                plt.autoscale()
            ims.append([imgplot])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        plt.draw()
        #
        fig = plt.figure(facecolor='white')
        fig.suptitle("Analysis Errors",  fontsize=font_size)
        ims = []
        for ind in xrange(num_cycles):
            state = np.reshape(np.squeeze(analysis_means[:, ind]-reference_states[:, ind]), (nx, ny), order='F')
            imgplot = plt.imshow(state, animated=True)
            if ind == 0:
                plt.colorbar()
            else:
                plt.autoscale()
            ims.append([imgplot])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        plt.draw()

    else:
        raise ValueError("Model is not supported here yet...")
    #
    plt.show()
    # =====================================================================
    #

    # =====================================================================
    # Conduct and plot results of multivariate normality tests.from
    # Probably we need to use the MVN package in R.
    # Note that MVN has problems being installed on UBUNTU.
    # Check with Fedora...
    # =====================================================================
