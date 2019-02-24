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

# This is a script to read and plot the ouptput of EnKF flavors, and HMC filter.

import sys
sys.path.insert(1, "../")

import os
import getopt
import numpy as np
import scipy.io as sio
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import shutil
import re

try:
    import ConfigParser
except ImportError:
    import configparser

try:
    import cpickle
except:
    import cPickle as pickle

import dates_setup
dates_setup.initialize_dates()
#
import dates_utility as utility
#
from lorenz_models import Lorenz96



__def_out_dir_tree_structure_file = "Results/Filtering_Results/output_dir_structure.txt"

__NUM_X_TICKS = 10


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
# Example on how to change things recursively in output_dir_structure.txt file:
# find . -type f -name 'output_dir_structure.txt' -exec sed -i '' s/nfs2/Users/ {} +

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

def read_filter_output(out_dir_tree_structure_file, apply_statisticsl_tests=False):
    """
    Read the output of a filter (EnKF, HMC so far) from one or more cycles.

    Args:
        out_dir_tree_structure_file: the file in which output directory structure is saved as a config-file.
        out_dir_tree_structure = dict(file_output_separate_files=file_output_separate_files,
                                      file_output_directory=file_output_directory,
                                      model_states_dir=model_states_dir,
                                      observations_dir=observations_dir,
                                      filter_statistics_dir=filter_statistics_dir,
                                      cycle_prefix=cycle_prefix)
        apply_statisticsl_tests: Apply, e.g. Mardia, tests of Gaussianity

    Returns:
        cycle_prefix:
        num_cycles:
        reference_states:
        forecast_ensembles:
        forecast_means:
        analysis_ensembles:
        analysis_means:
        observations:
        forecast_times:
        analysis_times:
        observations_times:
        forecast_rmse:
        analysis_rmse:
        filter_configs:
        gmm_results:
        model_configs:
        mardiaTest_results:
        inflation_opt_results:
        localization_opt_results:

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
        print("Check File: ", os.path.join(file_output_directory, 'setup.dat'))
        print("Secitons found", model_configs_parser.sections())
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
                        option_val = None
                    else:
                        print("ValueError exception raised while reading %s\n retrieved '%s' as a string!" %(option, option_val))
            elif option in ['model_name']:
                option_val = model_configs_parser.get(section_header, option)

            elif option in ['state_size', 'background_errors_covariance_localization_radius',
                            'model_errors_steps_per_model_steps', 'observation_vector_size', 'nx', 'ny', 'mrefin']:
                try:
                    option_val = model_configs_parser.getint(section_header, option)
                except:
                    option_val = model_configs_parser.get(section_header, option)
                    if option_val == 'None':
                        option_val = None
                    else:
                        print("ValueError exception raised while reading %s\n retrieved '%s' as a string!" %(option, option_val))
            else:
                try:
                    option_val = eval(model_configs_parser.get(section_header, option))
                except:
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
            elif option in ['prior_distribution', 'filter_name', 'localization_function']:
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
    observations_times, forecast_times, analysis_times, forecast_rmse, analysis_rmse = read_rmse_results(rmse_file_path)

    if apply_statisticsl_tests:
        mardiaTest_results = {}

    # Optimization results for OED-Aaptive EnKF
    inflation_opt_results = []
    localization_opt_results = []
    inf_opt_results_file = 'inflation_opt_results.p'
    loc_opt_results_file = 'localization_opt_results.p'


    # Filter configs
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
        try:
            contents = sio.loadmat(os.path.join(model_states_dir, cycle_dir, 'reference_state.mat'))
            reference_states[:, suffix]= contents['S'][:, 0].copy()  # each state is written as a column in the mat file...
            # print('reference_states', reference_states[:, suffix])
        except IOError:
            if reference_states is not None:
                reference_states = None

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

        #
        # Read adaptive inflation and localization results:
        inf_opt_path = os.path.join(model_states_dir, cycle_dir, inf_opt_results_file)
        if os.path.isfile(inf_opt_path):
            inf_opt_dict = pickle.load(open(inf_opt_path, 'rb'))
            inflation_opt_results.append((suffix, inf_opt_dict))
        #
        loc_opt_path = os.path.join(model_states_dir, cycle_dir, loc_opt_results_file)
        if os.path.isfile(loc_opt_path):
            loc_opt_dict = pickle.load(open(loc_opt_path, 'rb'))
            localization_opt_results.append((suffix, loc_opt_dict))
        #

        #
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
           forecast_rmse, analysis_rmse, filter_configs, gmm_results, model_configs, mardiaTest_results, inflation_opt_results, localization_opt_results


def read_rmse_results(rmse_file_path):
    # read times and RMSE results:
    rmse_file_contents = np.loadtxt(rmse_file_path, skiprows=2)
    observations_times = rmse_file_contents[:, 0]
    forecast_times = rmse_file_contents[:, 1]
    analysis_times = rmse_file_contents[:, 2]
    forecast_rmse = rmse_file_contents[:, 3]
    analysis_rmse = rmse_file_contents[:, 4]
    #
    return observations_times, forecast_times, analysis_times, forecast_rmse, analysis_rmse


def plot_rank_histogram(ensembles, truth,
                        first_var=0,
                        last_var=None,
                        var_skp=1,
                        draw_hist=True,
                        hist_type='relfreq',
                        first_time_ind=0,
                        last_time_ind=None,
                        time_ind_skp=0,
                        font_size=8,
                        color='skyblue',
                        edgecolor='skyblue',
                        hist_title=None,
                        hist_max_height=None,
                        ignore_indexes=None,
                        add_fitted_beta=False,
                        file_name=None,
                        add_uniform=False
                       ):
    """
    """
    f_out = utility.rank_hist(ensembles,
                              truth,
                              first_var=first_var,
                              last_var=last_var,
                              var_skp=var_skp,
                              draw_hist=draw_hist,
                              hist_type=hist_type,
                              first_time_ind=first_time_ind,
                              last_time_ind=last_time_ind,
                              time_ind_skp=time_ind_skp,
                              font_size=font_size,
                              color=color,
                              edgecolor=edgecolor,
                              hist_title=hist_title,
                              hist_max_height=hist_max_height,
                              ignore_indexes=ignore_indexes,
                              add_fitted_beta=add_fitted_beta,
                              file_name=file_name,
                              add_uniform=add_uniform
                             )

    return f_out


def read_assimilation_results(out_dir_tree_structure_file, show_plots=False, use_logscale=True, overwrite=True, rebuild_truth=False, recollect_from_source=False, read_results_only=False, ignore_err=False):
    """
    """
    out_dir_tree_structure_file = os.path.abspath(out_dir_tree_structure_file)
    if not os.path.isfile(out_dir_tree_structure_file):
        sep = "\n" + ("*"*100) + "\n"
        print("%sThe out_dir_tree_structure_file: \n\t%s\n is not a valid file!%s" % (sep, out_dir_tree_structure_file, sep))
        raise IOError

    # =====================================================================
    results_dir = os.path.dirname(out_dir_tree_structure_file)
    cwd = os.getcwd()
    os.chdir(results_dir)
    os.system("find . -type f -name 'output_dir_structure.txt' -exec sed -i '' s/nfs2/Users/ {} +")
    os.chdir(cwd)

    print("Reading Results from: %s " % results_dir)
    #
    # =====================================================================
    # Start reading the output of the assimilation process
    # =====================================================================
    print("- Collecting results")
    collective_res_file = os.path.join(results_dir, 'Collective_Results.pickle')
    if not recollect_from_source and os.path.isfile(collective_res_file):
        results_dict = pickle.load(open(collective_res_file, 'rb'))
        #
        try:
            cycle_prefix = results_dict['cycle_prefix']
            num_cycles = results_dict['num_cycles']
            reference_states = results_dict['reference_states']
            forecast_ensembles = results_dict['forecast_ensembles']
            forecast_means = results_dict['forecast_means']
            analysis_ensembles = results_dict['analysis_ensembles']
            analysis_means = results_dict['analysis_means']
            observations = results_dict['observations']
            forecast_times = results_dict['forecast_times']
            analysis_times = results_dict['analysis_times']
            observations_times = results_dict['observations_times']
            forecast_rmse = results_dict['forecast_rmse']
            analysis_rmse = results_dict['analysis_rmse']
            free_run_rmse = results_dict['free_run_rmse']
            filter_configs = results_dict['filter_configs']
            gmm_results = results_dict['gmm_results']
            model_configs = results_dict['model_configs']
            mardiaTest_results = results_dict['mardiaTest_results']
            inflation_opt_results = results_dict['inflation_opt_results']
            localization_opt_results = results_dict['localization_opt_results']
            #
            if free_run_rmse is None:
                results_loaded = False
                rebuild_truth = True
            else:
                results_loaded = True
        except:
            results_loaded = False
        #
    else:
        results_loaded = False
        rebuild_truth = True

    if not results_loaded:
        print("\treading from original files...")
        cycle_prefix, num_cycles, reference_states, forecast_ensembles, forecast_means, analysis_ensembles, \
        analysis_means, observations, forecast_times, analysis_times, observations_times, \
        forecast_rmse, analysis_rmse, filter_configs, gmm_results, model_configs, mardiaTest_results,  \
        inflation_opt_results, localization_opt_results = read_filter_output(out_dir_tree_structure_file)
        #
        rebuild_truth = True
        #
        free_run_rmse = None
        initial_ensemble = None
        initial_mean = None
        #
    else:
        print("\tdata collected from pickled collective file...")  # TODO: remove after debugging
        rebuild_truth = False

    #
    model = Lorenz96(model_configs)
    state_size = model.state_size()
    #
    if rebuild_truth:
        print("Rebuilding Truth")
        #
        # Load reference states:
        reference_states = np.load(os.path.join(os.path.dirname(out_dir_tree_structure_file), 'Reference_Trajectory.npy'))  # generated and saved offline for this experiment
        initial_ensemble = np.load(os.path.join(os.path.dirname(out_dir_tree_structure_file), 'Initial_Ensemble.npy'))  # generated and saved offline for this experiment
        initial_mean = np.mean(initial_ensemble, 1)
        forecast_rmse[0] = analysis_rmse[0] = np.sqrt(np.linalg.norm((initial_mean-reference_states[:state_size, 0]), 2)/ state_size)

        print("- Generating Free run results")
        free_run_trajectory = model.integrate_state(model.state_vector(initial_mean.copy()), checkpoints=analysis_times)
        free_run_rmse = forecast_rmse.copy()

        reference_states = reference_states[ :state_size, 1:1+num_cycles]
        for i in xrange(num_cycles):
            forecast_rmse[i+1] = np.sqrt(np.linalg.norm((forecast_means[:, i]-reference_states[:, i]), 2)/ state_size)
            analysis_rmse[i+1] = np.sqrt(np.linalg.norm((analysis_means[:, i]-reference_states[:, i]), 2)/ state_size)
            free_run_rmse[i+1] = np.sqrt(np.linalg.norm((free_run_trajectory[i+1][:]-reference_states[:, i]), 2)/ state_size)
            #

        if False:
            xrmse = []
            for i in xrange(num_cycles):
                tt = model.integrate_state(model.state_vector(analysis_means[:, i]), checkpoints=analysis_times[i: i+2])
                rmse = np.sqrt(np.linalg.norm(tt[-1][:]-reference_states[:, i], 2) /state_size)
                xrmse.append(rmse)
            print("Rec RMSES: ", xrmse)

    if not results_loaded:
        results_dict = dict(cycle_prefix=cycle_prefix,
                            num_cycles=num_cycles,
                            reference_states=reference_states,
                            forecast_ensembles=forecast_ensembles,
                            forecast_means=forecast_means,
                            analysis_ensembles=analysis_ensembles,
                            analysis_means=analysis_means,
                            observations=observations,
                            forecast_times=forecast_times,
                            analysis_times=analysis_times,
                            observations_times=observations_times,
                            forecast_rmse=forecast_rmse,
                            analysis_rmse=analysis_rmse,
                            free_run_rmse=free_run_rmse,
                            filter_configs=filter_configs,
                            gmm_results=gmm_results,
                            model_configs=model_configs,
                            mardiaTest_results=mardiaTest_results,
                            inflation_opt_results=inflation_opt_results,
                            localization_opt_results=localization_opt_results,
                            initial_ensemble=initial_ensemble,
                            initial_mean=initial_mean,
                           )
        pickle.dump(results_dict, open(collective_res_file, 'wb'))


    if not read_results_only:
        #
        plots_dir = os.path.abspath(os.path.join(results_dir, "PLOTS"))
        premat_term = False
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)
        else:
            if overwrite:
                shutil.rmtree(plots_dir)
                os.makedirs(plots_dir)
            else:
                out = utility.get_list_of_files(plots_dir, recursive=False)
                if len(out) == 0:
                    pass
                else:
                    print("The plots directory is not empty; set overwrite to True if you want to update results;")
                    print("List of files found in %s: " % plots_dir)
                    for i, f in enumerate(out):
                        print("%03d: % s " % (i, f))
                    premat_term = True

        #
        if not premat_term:
            filter_name = filter_configs['filter_name']
            filter_name = filter_name.replace('_', ' ')
            model_name = model_configs['model_name']
            try:
                state_size = model_configs['state_size']
            except KeyError:
                state_size = np.size(forecast_ensembles, 0)

            # print(reference_states, forecast_ensembles, forecast_means, analysis_ensembles, analysis_means, observations)
            #
            if forecast_ensembles is None and analysis_ensembles is None:
                moments_only = True
            else:
                moments_only = False
            # =====================================================================

            # Initial time for plotting (secondary)
            init_time_ind = int(forecast_times.size *2/3)

            #
            # =====================================================================
            # General Plotting settings
            # =====================================================================
            print("- Generating Plots...")
            # Font, and Texts:
            enhance_plotter()

            # Drawing Lines:
            font_size = 18
            line_width = 2
            marker_size = 3

            # set colormap:
            colormap = None  # 'jet'  # vs 'jet'
            # plt.set_cmap(colormap)

            interpolation = None  # 'bilinear'

            # =====================================================================


            #
            print("\nPlots are saved in %s "% plots_dir)
            #
            # =====================================================================
            # Plot RMSE
            # =====================================================================
            log_scale = use_logscale
            fig1 = plt.figure(figsize=(15, 6), facecolor='white')
            #
            if log_scale:
                plt.semilogy(analysis_times, free_run_rmse, '-.', color='red', linewidth=line_width, markersize=marker_size, label='Free Run')
                plt.semilogy(forecast_times, forecast_rmse, '-d', color='maroon', linewidth=line_width, markersize=marker_size, label='Forecast')
                plt.semilogy(analysis_times, analysis_rmse, '-o', color='darkgreen', linewidth=line_width, markersize=marker_size, label=filter_name)
                #
            else:
                plt.plot(analysis_times, free_run_rmse, '-', color='red', linewidth=line_width, markersize=marker_size, label='Free Run')
                plt.plot(forecast_times, forecast_rmse, '-d', color='maroon', linewidth=line_width, markersize=marker_size, label='Forecast')
                plt.plot(analysis_times, analysis_rmse, '-o', color='darkgreen', linewidth=line_width, markersize=marker_size, label=filter_name)

            #
            # Set lables and title
            ax = fig1.gca()
            ax.set_xlabel(r'Time')
            # plt.xlabel("Time (Assimilation cycles)")
            if log_scale:
                plt.ylabel(r'RMSE ($\textit{log-scale}$)', fontsize=font_size, fontweight='bold')
                # Adjust yticklables
                ylim = ax.get_ylim()
                yticks = ax.get_yticks()
                yticks_minor = ax.get_yticks(minor=True)
                lb = np.where(yticks_minor>=yticks[0])[0][0]
                ub = np.where(yticks_minor>yticks[-1])[0][0]
                yticks = [yticks_minor[2*i] for i in xrange(len(yticks_minor)/2)]
                yticklabels = np.round(np.log(yticks), 2)
                # ax.set_yticks([])
                # ax.set_yticks(yticks)
                # ax.set_yticklabels([])
                # ax.set_yticklabels(yticklabels)
                # ax.set_ylim(ylim)
            else:
                plt.ylabel('RMSE', fontsize=font_size, fontweight='bold')
            # plt.title('RMSE results for the model: %s' % model_name)
            #
            skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
            xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
            ax.set_xticks(xlables)
            ax.set_xticklabels(forecast_times[skips*np.arange(len(xlables))])
            ax.yaxis.grid(True, which='both', alpha=0.45)
            # show the legend, show the plot
            plt.legend(loc='best')
            file_name = os.path.join(plots_dir, 'RMSE.pdf')
            plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
            #
            # Adjust and save
            ax.set_xlim(forecast_times[init_time_ind], forecast_times[-1])
            file_name = os.path.join(plots_dir, 'RMSE_trimmed.pdf')
            plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
            #
            plt.grid(False)
            # =====================================================================

            # Plot trajectories:
            xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
            # selected_var_inds = np.arange(7, 40, 12)
            selected_var_inds = np.array([7, 31])
            ylims = [np.min(reference_states[selected_var_inds, init_time_ind:])-0.05, np.max(reference_states[selected_var_inds, init_time_ind:])+0.05]
            margin = (ylims[-1] - ylims[0]) / 5.0
            ylims = [ylims[0]-margin, ylims[-1]+margin]
            # selected_var_inds = np.array([0, 1, 2, 38, 39])  # TODO: Decide what to plot...
            num_selected_vars = selected_var_inds.size
            fig2, axs2 = plt.subplots(num_selected_vars, 1, figsize=(15,5.5), facecolor='white', sharex=True, sharey=True)
            plt.minorticks_on()
            for i, var_ind in enumerate(selected_var_inds):
                ax = axs2[i]
                ref_sol = reference_states[selected_var_inds[i], :]
                anl_sol = analysis_means[selected_var_inds[i], :]
                frcst_sol = forecast_means[selected_var_inds[i], :]
                if np.size(observations, 0) == np.size(reference_states, 0):
                    obs_sol = observations[selected_var_inds[i], :]
                else:
                    obs_sol = None

                line_width = 1.5
                l1, = ax.plot(forecast_times[1: ], ref_sol, '-', color='black', linewidth=3.5, label='Truth')
                # l2, = ax.plot(forecast_times[1: ], frcst_sol, 'o', color='red',  markersize=4, label='Forecast')
                l2, = ax.plot(forecast_times[1: ], frcst_sol, 'o', color='red',  linewidth=line_width, label='Forecast', alpha=0.85)
                if obs_sol is not None:
                    lo, = ax.plot(forecast_times[1: ], obs_sol, 's', color='c', markersize=4.5, label='Observation', alpha=0.85)
                l3, = ax.plot(forecast_times[1: ], anl_sol, '--', color='green', linewidth=2.5, markersize=3, label='Analysis')
                #
                ax.set_ylabel(r'$x_{%d}$' % (1+selected_var_inds[i]))
                ax.grid(True, which='major', linestyle='-')
                ax.grid(True, which='minor', linestyle='-.')
                ax.xaxis.grid(True, which='major', color='k', alpha=0.4)
                ax.set_xticks(xlables)
                if i < num_selected_vars - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xticklabels(forecast_times[skips*np.arange(len(xlables))])
                    ax.set_xlabel(r'$Time$')
                    #
                ax.set_xlim(forecast_times[init_time_ind], forecast_times[-1])
                ax.set_ylim(ylims[0], ylims[-1])
                #

            if obs_sol is None:
                axs2[0].legend((l1, l2, l3), ('Truth', 'Forecast', 'Analysis'), ncol=3, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0)
            else:
                axs2[0].legend((l2, l1, lo, l3), ('Forecast', 'Truth', 'Observations', 'Analysis'), ncol=4, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0)

            fig2.subplots_adjust(hspace=0.01)
            file_name = os.path.join(plots_dir, 'Selected_Trajects_1.pdf')
            plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

            # Plot trajectories at the boundaries:
            selected_var_inds = np.array([0, 1, 38, 39])  # TODO: Decide what to plot...
            ylims = [np.min(reference_states[selected_var_inds, init_time_ind:])-0.05, np.max(reference_states[selected_var_inds, init_time_ind:])+0.05]
            margin = (ylims[-1] - ylims[0]) / 5.0
            ylims = [ylims[0]-margin, ylims[-1]+margin]
            #
            num_selected_vars = selected_var_inds.size
            fig2, axs2 = plt.subplots(num_selected_vars, 1, figsize=(15,9.5), facecolor='white', sharex=True, sharey=True)
            for i, var_ind in enumerate(selected_var_inds):
                ax = axs2[i]
                ref_sol = reference_states[selected_var_inds[i], :]
                anl_sol = analysis_means[selected_var_inds[i], :]
                frcst_sol = forecast_means[selected_var_inds[i], :]
                if np.size(observations, 0) == np.size(reference_states, 0):
                    obs_sol = observations[selected_var_inds[i], :]
                else:
                    obs_sol = None

                line_width = 1.5
                l1, = ax.plot(forecast_times[1: ], ref_sol, '-', color='black', linewidth=line_width, label='Truth')
                # l2, = ax.plot(forecast_times[1: ], frcst_sol, 'o', color='red',  markersize=4, label='Forecast')
                l2, = ax.plot(forecast_times[1: ], frcst_sol, 'o', color='red',  linewidth=line_width, label='Forecast', alpha=0.65)
                if obs_sol is not None:
                    lo, = ax.plot(forecast_times[1: ], obs_sol, 'd', color='c', markersize=4, label='Observation', alpha=0.65)
                l3, = ax.plot(forecast_times[1: ], anl_sol, '--', color='green', linewidth=line_width, markersize=3, label='Analysis')
                #
                ax.set_ylabel(r'$x_{%d}$' % selected_var_inds[i])
                ax.xaxis.grid(True, which='major', color='k', alpha=0.4)
                ax.set_xticks(xlables)
                if i < num_selected_vars - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xticklabels(forecast_times[skips*np.arange(len(xlables))])
                    ax.set_xlabel(r'$Time$')
                    #
                ax.set_xlim(forecast_times[init_time_ind], forecast_times[-1])
                ax.set_ylim(ylims[0], ylims[-1])
                #

            if obs_sol is None:
                fig2.legend((l1, l2, l3), ('Truth', 'Forecast', 'Analysis'), 'upper center', ncol=3, fancybox=True, shadow=False, framealpha=0.9, bbox_to_anchor=(0.42, 0.88))
            else:
                fig2.legend((l2, l1, lo, l3), ('Forecast', 'Truth', 'Observations', 'Analysis'), 'upper center', ncol=4, fancybox=True, shadow=False, framealpha=0.9, bbox_to_anchor=(0.42, 0.88))

            fig2.subplots_adjust(hspace=0.01)
            file_name = os.path.join(plots_dir, 'Selected_Trajects_2.pdf')
            plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

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
                optimal_sols = []  # rounded, and moving average might be applied
                original_optimal_sols = []
                num_func_evaluations = []  # Number of function evaluations per each cycle
                num_iterations = []  # # Number of function evaluations per each cycle
                for i in xrange(len(inflation_opt_results)):
                    post_trace = inflation_opt_results[i][1]['post_trace']
                    min_f = inflation_opt_results[i][1]['min_f']
                    opt_x = inflation_opt_results[i][1]['opt_x']
                    orig_opt_x = inflation_opt_results[i][1]['orig_opt_x']
                    optimal_sols.append(opt_x)
                    original_optimal_sols.append(orig_opt_x)
                    l2_norm = np.linalg.norm(opt_x)
                    avrg = np.mean(opt_x)
                    stdev = np.std(opt_x)
                    inflation_stats[:, i] = post_trace, min_f, l2_norm, avrg, stdev
                    #
                    num_iterations.append(inflation_opt_results[i][1]['full_opt_results']['nit'])
                    try:
                        num_func_evaluations.append(inflation_opt_results[i][1]['full_opt_results']['nfev'])
                    except(KeyError):
                        num_func_evaluations.append(inflation_opt_results[i][1]['full_opt_results']['funcalls'])

                #
                log_scale = False
                #
                _, ax_adap_inf = plt.subplots(figsize=(15, 6), facecolor='white')
                #
                if log_scale:
                    ax_adap_inf.semilogy(analysis_times[1:], inflation_stats[0, :], 'bd-', linewidth=line_width, label=r"$Trace(\widetilde{\mathbf{A}})$")
                    ax_adap_inf.semilogy(analysis_times[1:], inflation_stats[1, :], 'gd-', linewidth=line_width, label="optimal objective")
                    # ax_adap_inf.semilogy(analysis_times[1:], inflation_stats[2, :], 'r-.', linewidth=line_width, label=r"$\|\mathbf{\lambda}\|$")
                    ax_adap_inf.semilogy(analysis_times[1:], inflation_stats[3, :], 'c--', linewidth=line_width, label=r"$\overline{\mathbf{\lambda}}$")
                    ax_adap_inf.semilogy(analysis_times[1:], inflation_stats[4, :], 'm--', linewidth=line_width, label=r"$\sigma_{\mathbf{\lambda}}$")
                else:
                    ax_adap_inf.plot(analysis_times[1:], inflation_stats[0, :], 'bd-', linewidth=line_width, label=r"$Trace(\widetilde{\mathbf{A}})$")
                    ax_adap_inf.plot(analysis_times[1:], inflation_stats[1, :], 'gd-', linewidth=line_width, label="optimal objective")
                    # ax_adap_inf.plot(analysis_times[1:], inflation_stats[2, :], 'r-.', linewidth=line_width, label=r"$\|\mathbf{L}\|$")
                    ax_adap_inf.plot(analysis_times[1:], inflation_stats[3, :], 'c--', linewidth=line_width, label=r"$\overline{\mathbf{\lambda}}$")
                    ax_adap_inf.plot(analysis_times[1:], inflation_stats[4, :], 'm--', linewidth=line_width, label=r"$\sigma_{\mathbf{\lambda}}$")
                #
                # Set lables and title
                ax_adap_inf.set_xlabel("Time")
                # ax_adap_inf.set_title('OED-Adaptive Inflation results for the model: %s' % model_name)
                ax_adap_inf.set_xlim(analysis_times[0], analysis_times[-1])
                #
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                ax_adap_inf.set_xticks(xlables)
                ax_adap_inf.set_xticklabels(forecast_times[skips*np.arange(len(xlables))])
                # show the legend, show the plot
                plt.legend(loc='best')
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Inflation_Stats.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


                # Number of iterations, function evaluations, and Gradient evaluations
                _, ax = plt.subplots(figsize=(15, 6), facecolor='white')
                ax.plot(analysis_times[1:], num_iterations, 'b-', label='Iterations')
                ax.plot(analysis_times[1:], num_func_evaluations, 'g--', label='Function calls')
                # Set lables and title
                ax.set_xlabel("Time")
                ax.set_xlim(analysis_times[0], analysis_times[-1])
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                ax.set_xticks(xlables)
                ax.set_xticklabels(forecast_times[skips*np.arange(len(xlables))])
                # Add subplot with zoomed results
                subax = utility.add_subplot_axes(ax)
                subax.plot(analysis_times[1+init_time_ind:], num_iterations[init_time_ind: ], 'b-')
                subax.plot(analysis_times[1+init_time_ind:], num_func_evaluations[init_time_ind: ], 'g--')
                ax.legend(loc='best')
                file_name = os.path.join(plots_dir, 'Optimization_Iterations.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


                # histogram of inflation factor
                _, ax_adap_inf_hist = plt.subplots(facecolor='white')
                data = np.asarray(optimal_sols).flatten()
                weights = np.zeros_like(data) + 1.0 / data.size
                ax_adap_inf_hist.hist(data, weights=weights, bins=50)
                # ax_adap_inf_hist.set_xlabel(r"Inflation factors $\lambda_i$")
                ax_adap_inf_hist.set_xlabel(r"$\lambda_i$")
                ax_adap_inf_hist.set_ylabel("Relative frequency")
                # ax_adap_inf_hist.set_title("Smoothed; Rounded Solution")
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Inflation_Rhist_Smoothed.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

                # Do the same for the original solution:
                _, ax_adap_inf_hist = plt.subplots(facecolor='white')
                data = np.asarray(original_optimal_sols).flatten()
                weights = np.zeros_like(data) + 1.0 / data.size
                ax_adap_inf_hist.hist(data, weights=weights, bins=50)
                # ax_adap_inf_hist.set_xlabel(r"Inflation factors $\lambda_i$")
                ax_adap_inf_hist.set_xlabel(r"$\lambda_i$")
                ax_adap_inf_hist.set_ylabel("Relative frequency")
                # ax_adap_inf_hist.set_title("Original solution")
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Inflation_Rhist_Original.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

                # boxplots of inflation factors over time
                _, ax_adap_inf_bplot = plt.subplots(figsize=(15, 6), facecolor='white')
                ax_adap_inf_bplot.boxplot(original_optimal_sols, notch=True, patch_artist=True, sym='+', vert=1, whis=1.5)
                ax_adap_inf_bplot.set_xlabel("Time")
                # ax_adap_inf_bplot.set_ylabel(r"Inflation factors $\lambda_i$")
                ax_adap_inf_bplot.set_ylabel(r"$\lambda_i$")
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xticks = [i for i in xrange(0, len(forecast_times), skips)]
                ax_adap_inf_bplot.set_xticks(xticks)
                ax_adap_inf_bplot.set_xticklabels([forecast_times[i] for i in xticks])
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Inflation_BoxPlot_Original.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

                # boxplots of inflation factors over time
                _, ax_adap_inf_bplot = plt.subplots(figsize=(15, 6), facecolor='white')
                ax_adap_inf_bplot.boxplot(optimal_sols, notch=True, patch_artist=True, sym='+', vert=1, whis=1.5)
                ax_adap_inf_bplot.set_xlabel("Time")
                # ax_adap_inf_bplot.set_ylabel(r"Inflation factors $\lambda_i$")
                ax_adap_inf_bplot.set_ylabel(r"$\lambda_i$")
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xticks = [i for i in xrange(0, len(forecast_times), skips)]
                ax_adap_inf_bplot.set_xticks(xticks)
                ax_adap_inf_bplot.set_xticklabels([forecast_times[i] for i in xticks])
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Inflation_BoxPlot_Smoothed.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

                # colorplot/imagesec of inflation factors over space and time
                fig_adap_inf_imsec, ax_adap_inf_imsec = plt.subplots(figsize=(15, 6), facecolor='white')
                cax = ax_adap_inf_imsec.imshow(np.array(original_optimal_sols).squeeze().T, aspect='auto', interpolation=interpolation, cmap=colormap, origin='lower')
                vmin, vmax = 0, state_size-1
                # cbar = fig_adap_inf_imsec.colorbar(cax, ticks=np.arange(1,1.26, 0.05), orientation='vertical')
                cbar = fig_adap_inf_imsec.colorbar(cax, orientation='vertical')
                ax_adap_inf_imsec.set_xlabel("Time")
                ax_adap_inf_imsec.set_ylabel(r"$\lambda_i$")
                # ax_adap_inf_imsec.set_yticks(np.arange(0, state_size, state_size/10))
                # ax_adap_inf_imsec.set_title("Inflation factors over space and time")
                #
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xticks = [i for i in xrange(0, len(forecast_times), skips)]
                xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                ax_adap_inf_imsec.set_xticks(xticks)
                ax_adap_inf_imsec.set_xticklabels(xlables)
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Inflation_ImSec.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

                # Same as the previous plot, but a bit enhanced
                # colorplot/imagesec of inflation factors over space and time
                fig_adap_inf_imsec, ax_adap_inf_imsec = plt.subplots(figsize=(7.5,3), facecolor='white')
                cax = ax_adap_inf_imsec.imshow(np.array(original_optimal_sols).squeeze().T, aspect='auto', interpolation=interpolation, cmap=colormap, origin='lower')
                vmin, vmax = 0, state_size-1
                # cbar = fig_adap_inf_imsec.colorbar(cax, ticks=np.arange(1,1.26, 0.05), orientation='vertical')
                cbar = fig_adap_inf_imsec.colorbar(cax, orientation='vertical')
                ax_adap_inf_imsec.set_xlabel("Time")
                ax_adap_inf_imsec.set_ylabel(r"$\lambda_i^{\rm opt}$")
                # ax_adap_inf_imsec.set_yticks(np.arange(0, state_size, state_size/10))
                # ax_adap_inf_imsec.set_title("Inflation factors over space and time")
                #
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xticks = [i for i in xrange(0, len(forecast_times), skips)]
                xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                ax_adap_inf_imsec.set_xticks(xticks)
                ax_adap_inf_imsec.set_xticklabels(xlables)
                #
                ylim = ax_adap_inf_imsec.get_ylim()
                yticks = ax_adap_inf_imsec.get_yticks() + 4
                yticklabels = np.array(yticks, dtype=np.int) + 1
                ax_adap_inf_imsec.set_yticks(yticks)
                ax_adap_inf_imsec.set_yticklabels(yticklabels)
                ax_adap_inf_imsec.set_ylim(ylim[0], ylim[-1])
                file_name = os.path.join(plots_dir, 'Adaptive_Inflation_ImSec_2.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')
                xlim = ax_adap_inf_imsec.get_xlim()
                ax_adap_inf_imsec.set_xlim(init_time_ind-0.5, xlim[-1])
                file_name = os.path.join(plots_dir, 'Adaptive_Inflation_ImSec_2_zoomed.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')



                # Plot the evolution of inflation factors over time, for selected state indexes:
                for pool_ind, selected_var_inds in enumerate([ np.arange(7, 40, 12), np.array([0, 1, 38, 39]) ]):
                    optimal_sols_np = np.asarray(optimal_sols).T
                    xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                    ylims = [np.nanmin(optimal_sols_np[selected_var_inds, init_time_ind:])-0.05, np.nanmax(optimal_sols_np[selected_var_inds, init_time_ind:])+0.05]
                    margin = (ylims[-1] - ylims[0]) / 5.0
                    ylims = [ylims[0]-margin, ylims[-1]+margin]
                    #
                    num_selected_vars = selected_var_inds.size
                    fig2, axs2 = plt.subplots(num_selected_vars, 1, figsize=(15,9.5), facecolor='white', sharex=True, sharey=True)
                    for i, var_ind in enumerate(selected_var_inds):
                        ax = axs2[i]
                        sol = optimal_sols_np[var_ind, :]

                        line_width = 1.5
                        l1, = ax.plot(forecast_times[1: ], sol, '-o', color='darkblue', linewidth=line_width)
                        #
                        ax.set_ylabel(r'$\lambda_{%d}$' % var_ind)
                        ax.xaxis.grid(True, which='major', color='k', alpha=0.4)
                        ax.set_xticks(xlables)
                        if i < num_selected_vars - 1:
                            ax.set_xticklabels([])
                        else:
                            ax.set_xticklabels(forecast_times[skips*np.arange(len(xlables))])
                            ax.set_xlabel(r'$Time$')
                            #
                        ax.set_xlim(forecast_times[init_time_ind], forecast_times[-1])
                        ax.set_ylim(ylims[0], ylims[-1])
                        #

                    fig2.subplots_adjust(hspace=0.01)
                    file_name = os.path.join(plots_dir, 'Selected_Inflation_Trajects_%d.pdf' %(pool_ind+1) )
                    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


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
                # fourth row: average localization radii
                # fifth row:  standard deviation localization radii
                #
                optimal_sols = []
                original_optimal_sols = []
                num_func_evaluations = []  # Number of function evaluations per each cycle
                num_iterations = []  # # Number of function evaluations per each cycle
                for i in xrange(len(localization_opt_results)):
                    post_trace = localization_opt_results[i][1]['post_trace']
                    min_f = localization_opt_results[i][1]['min_f']
                    opt_x = localization_opt_results[i][1]['opt_x']
                    orig_opt_x = localization_opt_results[i][1]['orig_opt_x']
                    # print("orig_opt_x: ", orig_opt_x)
                    optimal_sols.append(opt_x)
                    original_optimal_sols.append(orig_opt_x)
                    l2_norm = np.linalg.norm(opt_x)
                    avrg = np.mean(opt_x)
                    stdev = np.std(opt_x)
                    localization_stats[:, i] = post_trace, min_f, l2_norm, avrg, stdev
                    #
                    num_iterations.append(localization_opt_results[i][1]['full_opt_results']['nit'])
                    try:
                        num_func_evaluations.append(localization_opt_results[i][1]['full_opt_results']['nfev'])
                    except(KeyError):
                        num_func_evaluations.append(localization_opt_results[i][1]['full_opt_results']['funcalls'])

                #
                _, ax_adap_loc = plt.subplots(figsize=(15, 6), facecolor='white')
                #
                if log_scale:
                    ax_adap_loc.semilogy(analysis_times[1:], localization_stats[0, :], 'bd-', linewidth=line_width, label=r"$Trace(\widehat{ \mathbf{A}})$")
                    ax_adap_loc.semilogy(analysis_times[1:], localization_stats[1, :], 'gd-', linewidth=line_width, label="optimal objective")
                    # ax_adap_loc.semilogy(analysis_times[1:], localization_stats[2, :], 'r-.', linewidth=line_width, label=r"$\|\mathbf{L}\|$")
                    ax_adap_loc.semilogy(analysis_times[1:], localization_stats[3, :], 'c--', linewidth=line_width, label=r"$\overline{\mathbf{L}}$")
                    ax_adap_loc.semilogy(analysis_times[1:], localization_stats[4, :], 'm--', linewidth=line_width, label=r"$\sigma_{\mathbf{L}}$")
                else:
                    ax_adap_loc.plot(analysis_times[1:], localization_stats[0, :], 'bd-', linewidth=line_width, label=r"$Trace(\widehat{\mathbf{A}})$")
                    ax_adap_loc.plot(analysis_times[1:], localization_stats[1, :], 'gd-', linewidth=line_width, label="optimal objective")
                    # ax_adap_loc.plot(analysis_times[1:], localization_stats[2, :], 'r-.', linewidth=line_width, label=r"$\|\mathbf{L}\|$")
                    ax_adap_loc.plot(analysis_times[1:], localization_stats[3, :], 'c--', linewidth=line_width, label=r"$\overline{\mathbf{\ell_i}}$")
                    ax_adap_loc.plot(analysis_times[1:], localization_stats[4, :], 'm--', linewidth=line_width, label=r"$\sigma_{\mathbf{L}}$")
                #
                # Set lables and title
                ax_adap_loc.set_xlabel("Time")
                # ax_adap_loc.set_title('OED-Adaptive Localization results for the model: %s' % model_name)
                ax_adap_loc.set_xlim(analysis_times[0], analysis_times[-1])
                #
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                ax_adap_loc.set_xticks(xlables)
                ax_adap_loc.set_xticklabels(forecast_times[skips*np.arange(len(xlables))])
                # show the legend, show the plot
                plt.legend(loc='best')
                #
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Localization_Stats.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

                # Number of iterations, function evaluations, and Gradient evaluations
                _, ax = plt.subplots(figsize=(15, 6), facecolor='white')
                ax.plot(analysis_times[1:], num_iterations, 'b-', label='Iterations')
                ax.plot(analysis_times[1:], num_func_evaluations, 'g--', label='Function calls')
                # Set lables and title
                ax.set_xlabel("Time")
                ax.set_xlim(analysis_times[0], analysis_times[-1])
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                ax.set_xticks(xlables)
                ax.set_xticklabels(forecast_times[skips*np.arange(len(xlables))])
                # Add subplot with zoomed results
                subax = utility.add_subplot_axes(ax)
                subax.plot(analysis_times[1+init_time_ind:], num_iterations[init_time_ind: ], 'b-')
                subax.plot(analysis_times[1+init_time_ind:], num_func_evaluations[init_time_ind: ], 'g--')
                #
                ax.legend(loc='best')
                file_name = os.path.join(plots_dir, 'Optimization_Iterations.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


                # TODO: Add plots for the original solution; not just rounded, and smoothed solution
                # histogram of localization coefficients
                _, ax_adap_loc_hist = plt.subplots(facecolor='white')
                data = np.asarray(optimal_sols).flatten()
                weights = np.zeros_like(data) + 1.0 / data.size
                try:
                    ax_adap_loc_hist.hist(data, weights=weights, bins=50)
                except ValueError:
                    valid_locs = np.bitwise_not(np.isnan(data))
                    valid_data = data[valid_locs]
                    valid_weights = weights[valid_locs]
                    ax_adap_loc_hist.hist(valid_data, weights=valid_weights, bins=50)
                # ax_adap_loc_hist.set_xlabel(r"Localization parameters $\ell_i$")
                ax_adap_loc_hist.set_xlabel(r"$\ell_i$")
                ax_adap_loc_hist.set_ylabel("Relative frequency")
                # ax_adap_loc_hist.set_title("Smoothed; Rounded Solution")
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Localization_Rhist_Smoothed.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

                _, ax_adap_loc_hist = plt.subplots(facecolor='white')
                data = np.asarray(original_optimal_sols).flatten()
                weights = np.zeros_like(data) + 1.0 / data.size
                try:
                    ax_adap_loc_hist.hist(data, weights=weights, bins=50)
                except ValueError:
                    valid_locs = np.bitwise_not(np.isnan(data))
                    valid_data = data[valid_locs]
                    valid_weights = weights[valid_locs]
                    ax_adap_loc_hist.hist(valid_data, weights=valid_weights, bins=50)
                # ax_adap_loc_hist.set_xlabel(r"Localization parameters $\ell_i$")
                ax_adap_loc_hist.set_xlabel(r"$\ell_i$")
                ax_adap_loc_hist.set_ylabel("Relative frequency")
                # ax_adap_loc_hist.set_title("Original Solution")
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Localization_Rhist_Original.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

                # boxplots of localization coefficients over time
                _, ax_adap_loc_bplot = plt.subplots(figsize=(15, 6), facecolor='white')
                ax_adap_loc_bplot.boxplot(optimal_sols, notch=True, patch_artist=True, sym='+', vert=1, whis=1.5)
                ax_adap_loc_bplot.set_xlabel("Time")
                # ax_adap_loc_bplot.set_ylabel(r"Localization parameters $\ell_i$")
                ax_adap_loc_bplot.set_ylabel(r"$\ell_i$")
                #
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xticks = [i for i in xrange(0, len(forecast_times), skips)]
                xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                ax_adap_loc_bplot.set_xticks(xticks)
                ax_adap_loc_bplot.set_xticklabels(xlables)
                # show the legend, show the plot
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Localization_BoxPlot_Smoothed.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

                _, ax_adap_loc_bplot = plt.subplots(figsize=(15, 6), facecolor='white')
                ax_adap_loc_bplot.boxplot(original_optimal_sols, notch=True, patch_artist=True, sym='+', vert=1, whis=1.5)
                ax_adap_loc_bplot.set_xlabel("Time")
                # ax_adap_loc_bplot.set_ylabel(r"Localization parameters $\ell_i$")
                ax_adap_loc_bplot.set_ylabel(r"$\ell_i$")
                #
                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xticks = [i for i in xrange(0, len(forecast_times), skips)]
                xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                ax_adap_loc_bplot.set_xticks(xticks)
                ax_adap_loc_bplot.set_xticklabels(xlables)
                # show the legend, show the plot
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Localization_BoxPlot_Original.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')



                # colorplot/imagesec of localization parameters over space and time
                fig_adap_loc_imsec, ax_adap_loc_imsec = plt.subplots(figsize=(15, 6), facecolor='white')
                cax = ax_adap_loc_imsec.imshow(np.array(original_optimal_sols).squeeze().T, aspect='auto', interpolation=interpolation, cmap=colormap, origin='lower')
                vmin, vmax = 0, state_size-1
                # cbar = fig_adap_loc_imsec.colorbar(cax, ticks=np.arange(1,1.26, 0.05), orientation='vertical')
                cbar = fig_adap_loc_imsec.colorbar(cax, orientation='vertical')
                ax_adap_loc_imsec.set_xlabel("Time")
                ax_adap_loc_imsec.set_ylabel(r"$\ell_i$")
                # ax_adap_loc_imsec.set_yticks(np.arange(0, state_size, state_size/10))
                # ax_adap_loc_imsec.set_title("Localization parameters over space and time")

                skips = max(1, len(forecast_times) / __NUM_X_TICKS)   # - (len(forecast_times) % 10)
                xticks = [i for i in xrange(0, len(forecast_times), skips)]
                xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                ax_adap_loc_imsec.set_xticks(xticks)
                ax_adap_loc_imsec.set_xticklabels(xlables)
                #
                file_name = os.path.join(plots_dir, 'Adaptive_Localization_ImSec.pdf')
                plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')


                # Plot the evolution of inflation factors over time, for selected state indexes:
                for pool_ind, selected_var_inds in enumerate([ np.asarray([7, 31]), np.asarray([0, 1, 38, 39]) ]):
                    optimal_sols_np = np.asarray(optimal_sols).T
                    xlables = [forecast_times[i] for i in xrange(0, len(forecast_times), skips)]
                    ylims = [np.nanmin(optimal_sols_np[selected_var_inds, init_time_ind:])-0.05, np.nanmax(optimal_sols_np[selected_var_inds, init_time_ind:])+0.05]
                    if np.isnan(ylims).any() or np.isinf(ylims).any():
                        print ylims
                        continue
                    margin = (ylims[-1] - ylims[0]) / 5.0
                    ylims = [ylims[0]-margin, ylims[-1]+margin]
                    #
                    num_selected_vars = selected_var_inds.size
                    fig2, axs2 = plt.subplots(num_selected_vars, 1, figsize=(7, 3), facecolor='white', sharex=True, sharey=True)
                    for i, var_ind in enumerate(selected_var_inds):
                        ax = axs2[i]
                        sol = optimal_sols_np[selected_var_inds[i], :]
                        line_width = 1.5
                        l1, = ax.plot(forecast_times[1: ], sol, '-', color='darkblue')
                        #
                        ax.set_ylabel(r'$\ell_{%d}$' % (selected_var_inds[i]+1))
                        ax.xaxis.grid(True, which='major', color='k', alpha=0.4)
                        ax.set_xticks(xlables)
                        if i < num_selected_vars - 1:
                            ax.set_xticklabels([])
                        else:
                            ax.set_xticklabels(forecast_times[skips*np.arange(len(xlables))])
                            ax.set_xlabel(r'$Time$')
                            #
                        ax.set_xlim(forecast_times[init_time_ind], forecast_times[-1])
                        ax.set_ylim(ylims[0], ylims[-1])
                        plt.minorticks_on()
                        ax.grid(True, which='major', linestyle='-')
                        ax.grid(True, which='minor', linestyle='-.')
                        #

                    fig2.subplots_adjust(hspace=0.01)
                    file_name = os.path.join(plots_dir, 'Selected_Localization_Trajects_%d.pdf' %(pool_ind+1) )
                    plt.savefig(file_name, dpi=500, facecolor='w', format='pdf', transparent=True, bbox_inches='tight')

            # =====================================================================


            #
            # =====================================================================
            # Plot Rank Histograms for forecast and analysis ensemble
            # =====================================================================
            model_name = model_name.lower()
            if model_name == 'lorenz96':
                ignore_indexes = None
            elif model_name == 'qg-1.5':
                if False:
                    nx = int(np.sqrt(state_size))
                    #
                    top_bounds = np.arange(nx)
                    right_bounds = np.arange(2*nx-1, nx**2-nx+1, nx)
                    left_bounds = np.arange(nx, nx**2-nx, nx )
                    down_bounds = np.arange(nx**2-nx, nx**2)
                    side_bounds = np.reshape(np.vstack((left_bounds, right_bounds)), (left_bounds.size+right_bounds.size), order='F')
                    ignore_indexes = np.concatenate((top_bounds, side_bounds, down_bounds))
                else:
                    ignore_indexes = None
            else:
                raise ValueError("Model is not supported here yet...")

            #
            for state_histogram in [True, False]:
                for iter_ind, ensembles in enumerate([forecast_ensembles, analysis_ensembles]):
                    if state_histogram:
                        pref = "State"
                    else:
                        pref = "Observation"

                    if iter_ind == 0:
                        file_name = os.path.join(plots_dir, '%s_Forecast_Rhist.pdf' % pref)
                    elif iter_ind == 1:
                        file_name = os.path.join(plots_dir, '%s_Analysis_Rhist.pdf' % pref)
                    else:
                        raise ValueError

                    # Histogram of states vs. observations
                    if state_histogram:
                        # State-based histograms
                        if ensembles is None or reference_states is None:
                            truth = None
                            continue
                        else:
                            truth = reference_states
                        #
                    else:
                        # create rank histogram of obsevations instead of states
                        if ensembles is None or observations is None:
                            continue
                        else:
                            truth = observations

                        # Get dimensions:
                        observation_size = np.size(truth, 0)
                        num_times = np.size(ensembles, 2)
                        ensemble_size = np.size(ensembles, 1)

                        # Place holders for observations/truth
                        obs_ensembles = np.empty((observation_size, ensemble_size, num_times))
                        obs_truth = np.empty((observation_size, num_times))

                        for time_ind in xrange(num_times):
                            for obs_ind in xrange(ensemble_size):
                                obs = model.evaluate_theoretical_observation(model.state_vector(ensembles[:, obs_ind, time_ind]))
                                obs_ensembles[:, obs_ind, time_ind] = obs.get_numpy_array()
                                obs = model.evaluate_theoretical_observation(model.state_vector(truth[:, obs_ind]))
                                obs_truth[:, time_ind] = obs.get_numpy_array()

                        ensembles = obs_ensembles
                        truth = obs_truth

                    # Plot Histograms:
                    f_out = plot_rank_histogram(ensembles,
                                                truth,
                                                first_var=0,
                                                last_var=None,
                                                var_skp=1,
                                                draw_hist=True,
                                                hist_type='relfreq',
                                                first_time_ind=init_time_ind,
                                                last_time_ind=None,
                                                time_ind_skp=0,
                                                font_size=font_size,
                                                hist_title=None,
                                                hist_max_height=None,
                                                ignore_indexes=ignore_indexes,
                                                add_fitted_beta=True,
                                                file_name=file_name,
                                                add_uniform=True
                                                )

            #
            # =====================================================================


            #
            # =====================================================================
            # Plot the states based on the model name.
            # This supports only lorenz96 an QG-1.5 model results
            # =====================================================================
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
                    imgplot = plt.imshow(state, animated=True, interpolation=interpolation, cmap=colormap)
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
                    imgplot = plt.imshow(state, animated=True, interpolation=interpolation, cmap=colormap)
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
                    imgplot = plt.imshow(state, animated=True, interpolation=interpolation, cmap=colormap)
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
                    imgplot = plt.imshow(state, animated=True, interpolation=interpolation, cmap=colormap)
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
                    imgplot = plt.imshow(state, animated=True, interpolation=interpolation, cmap=colormap)
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
            if show_plots:
                plt.show()

            plt.close('all')
            # =====================================================================
            #

    return cycle_prefix, num_cycles, reference_states, forecast_ensembles, forecast_means, analysis_ensembles, \
        analysis_means, observations, forecast_times, analysis_times, observations_times, \
        forecast_rmse, analysis_rmse, filter_configs, gmm_results, model_configs, mardiaTest_results,  \
        inflation_opt_results, localization_opt_results

def str2bool(v):
    if v.lower().strip() in ('yes', 'true', 't', 'y', '1'):
        val = True
    elif v.lower().strip() in ('no', 'false', 'f', 'n', '0'):
        val = False
    else:
        print("Boolean Value is Expected Yes/No, Y/N, True/False, T/F, 1/0")
        raise ValueError
    return val

def get_args(input_args, output_repository_dir=None, out_dir_tree_structure_file=None, overwrite=None, recollect=False):
    """
    Get command line arguments
    """

    try:
        opts, args = getopt.getopt(input_args,"hf:d:o:r:",["fstructurefile=","doutputdir=","ooverwrite=", "recollect="])
    except getopt.GetoptError:
        print 'filtering_results_reader_coupledLorenz.py -f <out_dir_tree_structure_file> -d <output_repository_dir> -o <overwrite> -r <recollect-from-source>'
        sys.exit(2)

    # Default Values
    for opt, arg in opts:
        if opt == '-h':
            print 'filtering_results_reader_coupledLorenz.py -f <out_dir_tree_structure_file> -d <output_repository_dir> -o <overwrite-flag> -r <recollect-from-source>'
            sys.exit()
        elif opt in ("-o", "--ooverwrite"):
            overwrite = str2bool(arg)
        elif opt in ("-f", "--fstructurefile"):
            out_dir_tree_structure_file = arg
        elif opt in ("-d", "--doutputdir"):
            output_repository_dir = arg
        elif opt in ("-r", "--recollect", "--recollect-from-source"):
            recollect = str2bool(arg)
        else:
            print("Unknown argument [%s] is discarded" % opt)

        if None not in [output_repository_dir, out_dir_tree_structure_file]:
            print("You passed both output respository, and out_dir_tree_structure_file; which is confusing; only one of them should be passed")
            raise ValueError

    return output_repository_dir, out_dir_tree_structure_file, overwrite, recollect

#
if __name__ == '__main__':

    # =====================================================================
    #                        GENERAL SETTINGS
    # =====================================================================
    show_plots = False
    use_logscale = True
    overwrite_plots = True  # overriden by command line arguments
    # =====================================================================

    output_repository_dir, out_dir_tree_structure_file, overwrite, recollect_from_source = get_args(sys.argv[1:])
    if output_repository_dir is not None:
        output_dirs_list = utility.get_list_of_subdirectories(output_repository_dir, ignore_root=True, return_abs=False, ignore_special=True, recursive=False)
        #
    elif out_dir_tree_structure_file is not None:
        output_dirs_list = [os.path.dirname(out_dir_tree_structure_file)]
    else:
        out_dir_tree_structure_file = __def_out_dir_tree_structure_file
        output_dirs_list = [os.path.dirname(out_dir_tree_structure_file)]

    if overwrite is None:
        overwrite = overwrite_plots

    # Loop over all directories in the output repository
    for out_dir in output_dirs_list:
        out_dir_tree_structure_file = os.path.join(out_dir, 'output_dir_structure.txt')
        if not os.path.isfile(out_dir_tree_structure_file):
            print("Nothing to be done here...\nThe file %s is not valid" % out_dir_tree_structure_file)
            #
        else:
            read_assimilation_results(out_dir_tree_structure_file,
                                      show_plots=show_plots,
                                      recollect_from_source=recollect_from_source,
                                      use_logscale=use_logscale,
                                      overwrite=overwrite,
                                     )
