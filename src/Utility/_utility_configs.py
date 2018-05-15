
#
# ########################################################################################## #
#                                                                                            #
#   DATeS: Data Assimilation Testing Suite.                                                  #
#                                                                                            #
#   Copyright (C) 2016  A. Sandu, A. Attia, P. Tranquilli, S.R. Glandon,                     #
#   M. Narayanamurthi, A. Sarshar, Computational Science Laboratory (CSL), Virginia Tech.    #
#                                                                                            #
#   Website: http://csl.cs.vt.edu/                                                           #
#   Phone: 540-231-6186                                                                      #
#                                                                                            #
#   This program is subject to the terms of the Virginia Tech Non-Commercial/Commercial      #
#   License. Using the software constitutes an implicit agreement with the terms of the      #
#   license. You should have received a copy of the Virginia Tech Non-Commercial License     #
#   with this program; if not, please contact the computational Science Laboratory to        #
#   obtain it.                                                                               #
#                                                                                            #
# ########################################################################################## #
#


"""
    A module providing functions that handle configurations files; this includes reading, righting, and validating them.
"""


import shutil
import os
import sys
import re

try:
    import cPickle as pickle
except:
    import pickle
    
try:  # ConfigParser is renamed to configparser in newer versions of Python
    import ConfigParser
except (ImportError):
    import configparser as ConfigParser



#
#
def write_dates_configs_template(file_name='assimilation_configs_template.inp', directory=None):
    """ 
    generate a configurations template file in the given path.
    Note that upon reading configurations later, a validation process for entries should be called.

    Args:      
         file_name: name of the file to save configurations file template to,
         directory: relative/full path of the directory to save the template in, if directory is not given, file is written in the cwd
    """

    # TODO: This list should be updated whenever new configurations are to be added. USE AS REFERENCE...
    # TODO: This list is initial and will be revised. some will stay, some will go...
    # TODO: upon creating 'read_config()' function, make sure you add a function for validating configurations.
    #       e.g. match the assimilation_scheme_name with
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create the configuration parser object
    dates_configs = ConfigParser.ConfigParser()

    # create and set configurations of the ASSIMILATION SCHEME
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1- Start with a section containing all configurations related to the assimilation scheme
    #
    dates_configs.add_section("Sec_Assimilation_Configs")
    #
    dates_configs.set("Sec_Assimilation_Configs", 'assimilation_scheme_type', 'filter')  # { 'filter', 'smoother', 'variational', 'hybrid' }
    dates_configs.set("Sec_Assimilation_Configs", 'assimilation_scheme_name', 'EnKF')  # { 'EnKF', '3DVar', '4D-En-Var', ... }

    dates_configs.set("Sec_Assimilation_Configs", 'assimilation_ensemble_size', 30)  # used of ensemble of states is to be generated

    dates_configs.set("Sec_Assimilation_Configs", 'assimilation_initial_time', 0)
    dates_configs.set("Sec_Assimilation_Configs", 'assimilation_final_time', 20)
    dates_configs.set("Sec_Assimilation_Configs", 'assimilation_cycle_length', 0.5) # length of each assimilation cycle
    dates_configs.set("Sec_Assimilation_Configs", 'assimilation_time_span', [t*0.1 for t in range(10)])  # {a comma-ceparated timespan or interval on the form (0:10:50)}

    dates_configs.set("Sec_Assimilation_Configs", 'ignore_observations_term', False)  # {"1", "yes", "true", "on" --> True; "0", "no", "false", "off" --> False}
    dates_configs.set("Sec_Assimilation_Configs", 'observation_operator_type', 'linear')
    dates_configs.set("Sec_Assimilation_Configs", 'observed_variables_jump', 1)  # 1 observe all gridpoints. 2- every second, etc...
    dates_configs.set("Sec_Assimilation_Configs", 'observation_spacing_type', 'fixed')  # {fixed/random/specified}
    dates_configs.set("Sec_Assimilation_Configs", 'observation_steps_per_assimilation_steps', 10)
    dates_configs.set("Sec_Assimilation_Configs", 'observation_chance_per_assimilation_steps', 0.5)
    dates_configs.set("Sec_Assimilation_Configs", 'observation_time_span', [t*0.2 for t in range(5)])  # {a comma-ceparated timespan or interval on the form (0:10:50)}
    #
    dates_configs.set("Sec_Assimilation_Configs", 'observation_errors_distribution', 'gaussian')
    dates_configs.set("Sec_Assimilation_Configs", 'observation_noise_level', 0.05)

    dates_configs.set("Sec_Assimilation_Configs", 'apply_localization', 'yes')
    dates_configs.set("Sec_Assimilation_Configs", 'decorrelation_radius', 4.0)
    dates_configs.set("Sec_Assimilation_Configs", 'read_decorrelation_from_file', False)
    dates_configs.set("Sec_Assimilation_Configs", 'periodic_decorrelation', 'yes')  # for periodic BC, we should use periodic decorrelation
    dates_configs.set("Sec_Assimilation_Configs", 'periodic_decorrelation_dimensions', [1,2])  # periodicity is conducted in the mentioned directions

    dates_configs.set("Sec_Assimilation_Configs", 'inflate_ensemble', 'yes')
    dates_configs.set("Sec_Assimilation_Configs", 'inflation_factor', 4.0)
    dates_configs.set("Sec_Assimilation_Configs", 'inflation_steps_per_assimilation_steps', 1)

    dates_configs.set("Sec_Assimilation_Configs", 'background_errors_covariance_method', 'nmc') # {empirical, nmc ,...}
    dates_configs.set("Sec_Assimilation_Configs", 'background_noise_level', 0.08) # {empirical, nmc ,...}
    dates_configs.set("Sec_Assimilation_Configs", 'background_noise_type', 'gaussian')
    dates_configs.set("Sec_Assimilation_Configs", 'update_B_factor', 1.0)

    dates_configs.set("Sec_Assimilation_Configs", 'use_sparse_packages', True)  # use scipy.sparse.*

    dates_configs.set("Sec_Assimilation_Configs", 'linear_system_solver', 'lu')

    dates_configs.set("Sec_Assimilation_Configs", 'optimization_package', 'scipy')
    dates_configs.set("Sec_Assimilation_Configs", 'optimization_scheme', 'Newton-CG')
    dates_configs.set("Sec_Assimilation_Configs", 'optimization_runtime_options', '--tol 1e-8 --max_iter 50')


    # 2- Add a section containing all configurations related to the MODEL
    #    > The configurations added here SHOULD be tracked and validated if the model imposes contradicting rules!
    #
    dates_configs.add_section("Sec_Model_Configs")
    #
    dates_configs.set("Sec_Model_Configs", 'model_name', 'Lorenz-96')
    dates_configs.set("Sec_Model_Configs", 'model_time_step', 0.005)
    dates_configs.set("Sec_Model_Configs", 'model_num_of_prognostic_variables', 3)
    dates_configs.set("Sec_Model_Configs", 'model_grid_type', 'cartesian')  # {'cartesian', 'spherical', etc. }
    dates_configs.set("Sec_Model_Configs", 'model_num_of_dimensions', 2)
    dates_configs.set("Sec_Model_Configs", 'model_dimensions_spacings', [0.05, 0.1])

    dates_configs.set("Sec_Model_Configs", 'model_time_integration_package', 'fat-ode')
    dates_configs.set("Sec_Model_Configs", 'model_time_integration_scheme', 'RK2')
    dates_configs.set("Sec_Model_Configs", 'model_time_integration_runtime_options', '--step 0.01 --tol 1e-8')

    dates_configs.set("Sec_Model_Configs", 'model_errors_covariance_method', 'diagonal')
    dates_configs.set("Sec_Model_Configs", 'model_errors_distribution', 'gaussian')
    dates_configs.set("Sec_Model_Configs", 'model_noise_level', 0.04)
    dates_configs.set("Sec_Model_Configs", 'model_errors_steps_per_model_steps', 10)
    #

    # 3- Add a section containing all configurations related to the outputting to screen/files
    #
    dates_configs.add_section("Sec_InOut_Configs")
    #
    dates_configs.set("Sec_InOut_Configs", 'screen_output', True)
    dates_configs.set("Sec_InOut_Configs", 'file_output_separate_files', False)
    dates_configs.set("Sec_InOut_Configs", 'screen_output_iter', '1')
    dates_configs.set("Sec_InOut_Configs", 'file_output', True)
    dates_configs.set("Sec_InOut_Configs", 'file_output_iter', 1)
    dates_configs.set("Sec_InOut_Configs", 'file_output_moment_only', True)  # {otherwise all ensembles will be written to files}
    dates_configs.set("Sec_InOut_Configs", 'file_output_moment_name', 'mean')  # {'mean', 'mode', etc.}
    dates_configs.set("Sec_InOut_Configs", 'file_output_directory', 'Assimiltion_Results')  # {'mean', 'mode', etc.}

    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    #  write the configurations to file:
    if directory is not None:
        if not os.path.isdir(directory):
            raise IOError(" ['%s'] is not a valid directory!" % directory)
        file_path = os.path.join(directory, file_name)
    else:
        file_path = file_name

    try:
        # write, save, and close the configurations file
        with open(file_path, 'w') as configs_file_ptr:
            dates_configs.write(configs_file_ptr)
    except IOError:
        err_msg = "A configuration template could not be written due to IO Error!"
        sys.exit(err_msg)
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #


#
#
def read_assimilation_configurations(config_file_name='setup.inp', config_file_relative_dir=None):
    """ 
    Properly read the assimilation configurations from passed configurations file

    Args:
        config_file_name: name of the configurations file
        config_file_relative_dir: relative directory where the configurations file <config_file_name> can be found,                                       
            If this is None, the root directory of dates [$DATES_ROOT_PATH] will be used.
    
    Returns:
        assimilation_configs:
        model_configs:
        inout_configs:
        
    """

    #  Retrieve dates root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    if config_file_relative_dir is not None:
        config_file_dir = os.path.join(DATES_ROOT_PATH, config_file_relative_dir)
    else:
        config_file_dir = DATES_ROOT_PATH

    if not os.path.isdir(config_file_dir):
        err_msg = "IOError: '%s' is not a valid directory!" \
                  % config_file_dir
        sys.exit(err_msg)

    config_file_path = os.path.join(config_file_dir, config_file_name)
    if not os.path.isfile(config_file_path):
        err_msg = "IOError: '%s' is not found or invalid file!" \
                  % config_file_path
        sys.exit(err_msg)
    else:
        dates_configs = ConfigParser.ConfigParser()
        dates_configs.read(config_file_path)

    # Initialize dictionaries corresponding to the number of sections. This has to be updated if more sections are added.
    assimilation_configs = {}
    model_configs = {}
    inout_configs = {}

    config_section_name = "Sec_Assimilation_Configs"
    #
    if dates_configs.has_section(config_section_name):
        # load all options in this section. all options' keys are loaded in lower case...
        options = dates_configs.options(config_section_name)
        #
        # properly convert configurations to valid types... I have separated lists to ease maintenance
        list_of_int_variables = ['assimilation_ensemble_size',
                                 'observed_variables_jump',
                                 'observation_steps_per_assimilation_steps',
                                 'inflation_steps_per_assimilation_steps',
                                 ]
        list_of_float_variables = ['assimilation_initial_time',
                                   'assimilation_final_time',
                                   'assimilation_cycle_length',
                                   'observation_chance_per_assimilation_steps',
                                   'observation_noise_level',
                                   'decorrelation_radius',
                                   'inflation_factor',
                                   'background_noise_level',
                                   'update_B_factor',
                                   ]
        list_of_boolean_variables = ['ignore_observations_term',
                                     'apply_localization',
                                     'read_decorrelation_from_file',
                                     'periodic_decorrelation',
                                     'use_sparse_packages',
                                     'inflate_ensemble',
                                     ]
        list_of_int_list_variables = []
        list_of_float_list_variables = ['assimilation_time_span',
                                        'observation_time_span',
                                        'periodic_decorrelation_dimensions',
                                        ]
        list_of_opt_dict_variables = ['optimization_runtime_options',
                                      ]

        for option in options:
            #
            if option in list_of_int_variables:
                option_val = dates_configs.getint(config_section_name, option)

            elif option in list_of_float_variables:
                option_val = dates_configs.getfloat(config_section_name, option)

            elif option in list_of_boolean_variables:
                option_val = dates_configs.getboolean(config_section_name, option)

            elif option in list_of_int_list_variables:
                list_str = dates_configs.get(config_section_name, option).strip('[]')
                option_val = [int(entry) for entry in list_str.split(',')]

            elif option in list_of_float_list_variables:
                list_str = dates_configs.get(config_section_name, option).strip('[]')
                option_val = [float(entry) for entry in list_str.split(',')]

            elif option in list_of_opt_dict_variables:
                options_str = dates_configs.get(config_section_name, option)
                options_val = {}

                options_lists = [m.strip().split() for m in options_str.split('--')]
                for opt in options_lists:
                    if len(opt) == 2:
                        options_val.update({opt[0]: opt[1]})
                    elif len(opt) == 1:
                        options_val.update({'nameless': opt[1]})
                    elif len(opt) == 0:
                        pass
                    else:
                        err_msg = "Couldn't recognize the option: '%s' " % opt
                        sys.exit(err_msg)

            else:
                # the rest are loaded as strings,
                option_val = dates_configs.get(config_section_name, option)

            #
            assimilation_configs.update({option: option_val})
            #
    else:
        err_msg = "The configurations file %s must contain a section named 'Sec_Assimilation_Configs' to proceed! "
        sys.exit(err_msg)

    #
    # retrieve model configurations if exist:
    config_section_name = "Sec_Model_Configs"
    #
    if dates_configs.has_section(config_section_name):
        # load all options in this section. all options' keys are loaded in lower case...
        options = dates_configs.options(config_section_name)
        #
        # properly convert configurations to valid types... I have separated lists to ease maintenance
        list_of_int_variables = ['model_num_of_prognostic_variables',
                                 'model_num_of_dimensions',
                                 'model_errors_steps_per_model_steps',
                                 ]
        list_of_float_variables = ['model_dimensions_step_sizes',
                                   'model_noise_level',
                                   ]
        list_of_boolean_variables = []
        list_of_int_list_variables = []
        list_of_float_list_variables = ['model_dimensions_spacings']
        list_of_opt_dict_variables = ['model_time_integration_runtime_options',
                                      ]

        for option in options:
            #
            if option in list_of_int_variables:
                option_val = dates_configs.getint(config_section_name, option)

            elif option in list_of_float_variables:
                option_val = dates_configs.getfloat(config_section_name, option)

            elif option in list_of_boolean_variables:
                option_val = dates_configs.getboolean(config_section_name, option)

            elif option in list_of_int_list_variables:
                list_str = dates_configs.get(config_section_name, option).strip('[]')
                option_val = [int(entry) for entry in list_str.split(',')]

            elif option in list_of_float_list_variables:
                list_str = dates_configs.get(config_section_name, option).strip('[]')
                option_val = [float(entry) for entry in list_str.split(',')]

            elif option in list_of_opt_dict_variables:
                options_str = dates_configs.get(config_section_name, option)
                options_val = {}

                options_lists = [m.strip().split() for m in options_str.split('--')]
                for opt in options_lists:
                    if len(opt) == 2:
                        options_val.update({opt[0]: opt[1]})
                    elif len(opt) == 1:
                        options_val.update({'nameless': opt[1]})
                    elif len(opt) == 0:
                        pass
                    else:
                        err_msg = "Couldn't recognize the option: '%s' " % opt
                        sys.exit(err_msg)

            else:
                # the rest are loaded as strings,
                option_val = dates_configs.get(config_section_name, option)

            #
            model_configs.update({option: option_val})
            #

    #
    config_section_name = "Sec_InOut_Configs"
    #
    if dates_configs.has_section(config_section_name):
        # load all options in this section. all options' keys are loaded in lower case...
        options = dates_configs.options(config_section_name)
        #
        # properly convert configurations to valid types... I have separated lists to ease maintenance
        list_of_int_variables = ['screen_output_iter',
                                 'file_output_iter',
                                 ]
        list_of_float_variables = []
        list_of_boolean_variables = ['screen_output',
                                     'file_output',
                                     'file_output_moment_only',
                                     ]
        list_of_int_list_variables = []
        list_of_float_list_variables = []
        list_of_opt_dict_variables = []

        # properly convert configurations to valid types...
        for option in options:
            #
            if option in list_of_int_variables:
                option_val = dates_configs.getint(config_section_name, option)

            elif option in list_of_float_variables:
                option_val = dates_configs.getfloat(config_section_name, option)

            elif option in list_of_boolean_variables:
                option_val = dates_configs.getboolean(config_section_name, option)

            elif option in list_of_int_list_variables:
                list_str = dates_configs.get(config_section_name, option).strip('[]')
                option_val = [int(entry) for entry in list_str.split(',')]

            elif option in list_of_float_list_variables:
                list_str = dates_configs.get(config_section_name, option).strip('[]')
                option_val = [float(entry) for entry in list_str.split(',')]

            elif option in list_of_opt_dict_variables:
                options_str = dates_configs.getboolean(config_section_name, option)
                options_val = {}

                options_lists = [m.strip().split() for m in options_str.split('--')]
                for opt in options_lists:
                    if len(opt) == 2:
                        options_val.update({opt[0]: opt[1]})
                    elif len(opt) == 1:
                        options_val.update({'nameless': opt[1]})
                    elif len(opt) == 0:
                        pass
                    else:
                        err_msg = "Couldn't recognize the option: '%s' " % opt
                        sys.exit(err_msg)

            else:
                # the rest are loaded as strings,
                option_val = dates_configs.get(config_section_name, option)

            #
            inout_configs.update({option:option_val})
            #

    return assimilation_configs, model_configs, inout_configs
    #


#
#
def validate_assimilation_configurations(assimilation_configs=None, def_assimilation_configs=None,
                                         model_configs=None, def_model_configs=None,
                                         inout_configs=None, def_inout_configs=None,
                                         copy_configurations=True
                                         ):
    """
    Configurations are validated against passed defaults.
    If defaults are empty, passed configurations are returned as is.
    If only default configs are passed, they are copied to corresponding configs dict

    Args:
        model_configs: a dictionary containing default model configurations. This should be obtained from a
                       configurations file
        default_model_configs: default model configurations
        assimilation_configs: a dictionary containing default assimilation configurations.
        def_assimilation_configs: default assimilation configurations
        inout_configs: a dictionary containing default input/output configurations.
        def_inout_configs: default input/output_configurations
        copy_configurations: If True, A deep copy of default dict is returned rather than a reference.
                             This is relevant only in passed configs are None.

    Returns:
        valid_model_configs: validated model configurations
        valid_assimilation_configs: validated assimilation configurations
        valid_inout_configs: validated input/output configurations

    """
    # Compare dictionaries
    # 1- model configurations:
    if model_configs is None and def_model_configs is None:
        pass
    elif model_configs is None and def_model_configs is not None:
        if copy_configurations:
            model_configs = def_model_configs.copy()
        else:
            model_configs = def_model_configs
    else:
        # start aggregating both dicts. We need to add a check to avoid any contradictions
        for key in def_model_configs:
            if key not in model_configs:
                model_configs.update({key: def_model_configs[key]})
        #

    # 2- assimilation configurations:
    if assimilation_configs is None and def_assimilation_configs is None:
        pass
    elif assimilation_configs is None and def_assimilation_configs is not None:
        if copy_configurations:
            assimilation_configs = def_assimilation_configs.copy()
        else:
            assimilation_configs = def_assimilation_configs
    else:
        # start aggregating both dicts. We need to add a check to avoid any contradictions
        for key in def_assimilation_configs:
            if key not in assimilation_configs:
                assimilation_configs.update({key: def_assimilation_configs[key]})

    # 3- input/output configurations:
    if inout_configs is None and def_inout_configs is None:
        pass
    elif inout_configs is None and def_inout_configs is not None:
        if copy_configurations:
            inout_configs = def_inout_configs.copy()
        else:
            inout_configs = def_inout_configs
    else:
        # start aggregating both dicts. We need to add a check to avoid any contradictions
        for key in def_inout_configs:
            if key not in inout_configs:
                inout_configs.update({key: def_inout_configs[key]})

    #
    # Prepare for return:
    return assimilation_configs, model_configs, inout_configs


    #


#
#
def aggregate_configurations(configs, def_configs, copy_configurations=True):
    """
    Blindly (and recursively) combine the two dictionaries. One-way copying: from def_configs to configs only.
    Add default configurations to the passed configs dictionary
    
    Args:
        configs:
        def_configs:
        copy_configurations:
        
    Returns:
        configs:
        
    """
    if configs is None and def_configs is None:
        raise ValueError("both inputs are None")
    elif configs is None:
        if copy_configurations:
            configs = dict.copy(def_configs)
        else:
            configs = def_configs
    else:
        #
        for key in def_configs:
            if key not in configs:
                configs.update({key: def_configs[key]})
            elif configs[key] is None:
                    configs[key] = def_configs[key]
            elif isinstance(configs[key], dict) and isinstance(def_configs[key], dict):
                # recursively aggregate the dictionary-valued keys
                sub_configs = aggregate_configurations(configs[key], def_configs[key])
                configs.update({key: sub_configs})
    #
    return configs


def write_dicts_to_config_file(file_name, out_dir, dicts, sections_headers):
    """
    Write one or more dictionaries (passed as a list) to a configuration file.
    
    Args:
        param file_name:
        out_dir:
        dicts:
        sections_headers:
    
    """
    configs = ConfigParser.ConfigParser()
    if isinstance(dicts, list) and isinstance(sections_headers, list):
        # configs = ConfigParser.ConfigParser()
        num_sections = len(sections_headers)
        for sec_ind in xrange(num_sections):
            #
            curr_dict = dicts[sec_ind]
            if isinstance(curr_dict, dict):
                if len(curr_dict) == 0:
                    continue
                else:
                    header = sections_headers[sec_ind]
                    configs.add_section(header)
                    for key in curr_dict:
                        configs.set(header, key, curr_dict[key])
            elif curr_dict is None:
                continue
            else:
                raise AssertionError("An entry passed is neither a dictionary nor None. passed %s" %repr(curr_dict))
    #
    elif isinstance(dicts, dict) and isinstance(sections_headers, str):
        # This is a single dictionary --> a single section in the configuration file.
        if len(dicts) == 0:
            pass
        else:
            # configs = ConfigParser.ConfigParser()
            configs.add_section(sections_headers)
            for key in dicts:
                configs.set(sections_headers, key, dicts[key])
    #
    elif dicts is None:
        pass
    else:
        print(dicts, sections_headers)
        raise AssertionError("Either 'dicts' should be a dictionary and 'sections_headers' be a string, OR, \n"
                             "both are lists of same length.")

    #  write the configurations to file:
    if len(configs.sections()) > 0:
        if not os.path.isdir(out_dir):
            raise IOError(" ['%s'] is not a valid directory!" % out_dir)
        file_path = os.path.join(out_dir, file_name)
        try:
             # write, save, and close the configurations file
             with open(file_path, 'w') as configs_file_ptr:
                configs.write(configs_file_ptr)
        except IOError:
            raise IOError("A configuration template could not be written due to IO Error!")
    #


