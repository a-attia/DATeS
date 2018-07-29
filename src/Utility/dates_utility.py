
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
#                                                                                            #
# This is like a driver for all _utility_* modules. We can think later of adding             #
# all _utility_* functions here.                                                             #
# Measillinious utility functions are defined here. Can be moved to the proper module.       #
# I expect these to be allot, this is why I started several _utility modules.                #
# We can easily add other modules and import them here if we happen to need another cateogry #
# of utility functions.                                                                      #
# ########################################################################################## #
#


#
#    A module providing utility functions.
#    This imports all utility functions from:
#        - _utility_file_IO
#        - _utility_machine_learning
#        - _utility_stat
#        - _utility_optimization
#        - _utility_configs
#        - _utility_url
#        - _utility_data_assimilation
#
#


import os
import sys
import shutil

import numpy as np

from _utility_file_IO import *
from _utility_machine_learning import *
from _utility_stat import *
from _utility_optimization import *
from _utility_configs import *
from _utility_url import *
from _utility_data_assimilation import *
from _utility_plot import *
from _utility_misc import *

#
# This group of methods handle files containing list to be maintained,
# such as models list, assimilation schemes list,etc.
#

#
def read_models_list(return_model_full_path=True,
                     models_list_relative_dir='src/Models_Forest',
                     models_list_file_name='models_list.txt'
                     ):
    """
    Retrieve the list of implemented models in DATeS.

    Args:
        model_full_path         : return the models' path as full path if true, otherwise relative paths are returned.
        models_list_relative_dir: relative path of the directory where the models_list files exists,
                                  the path must be relative to DATEeS_root_path.
        models_list_file_name   : name of the file containing models' list.

    Returns:
        Three lists:
            - the first is a list contains models names,
            - the second is a list of lists containing source code languages,
            - the third is a list containing models paths, return Full path by default.
    """
    #
    #  Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of models
    while models_list_relative_dir.startswith('/'):
        models_list_relative_dir = models_list_relative_dir[1:]
    while models_list_relative_dir.endswith('/'):
        models_list_relative_dir = models_list_relative_dir[:-1]

    #
    #  Read models' info:
    models_list_file_path = os.path.join(DATES_ROOT_PATH, models_list_relative_dir, models_list_file_name)
    try:
        models_names = []
        models_src_languages = []
        models_src_paths = []
        with open(models_list_file_path,'r') as file_iterator:
            for line in file_iterator:
                line = line.strip()
                if (line.startswith('#')) or (not line) or ('[start_table]' in line) or ('[end_table]' in line):
                    # ignore comment lines and empty lines
                    continue
                else:
                    # read model info in the current line
                    _, model_name, model_src_languages, model_relative_dir = [entry.strip(' #') for entry in line.split('|')]
                    #
                    models_names.append(model_name)
                    models_src_languages.append([entry.strip() for entry in model_src_languages.split(',')] )
                    if return_model_full_path:
                        models_src_paths.append(os.path.join(DATES_ROOT_PATH,model_relative_dir))
                    else:
                        models_src_paths.append(model_relative_dir)

    except IOError:
        err_msg = "Either the models list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "models_list_file_path: '%s' \n" % models_list_file_path
        print(err_msg)
        sys.exit(1)

    #
    return models_names, models_src_languages, models_src_paths


#
#
def read_filters_list(return_filter_full_path=True,
                      filters_list_relative_dir='src/filters',
                      filters_list_file_name='filters_list.txt'
                      ):
    """
    Retrieve the list of implemented filters in DATeS.

    Args:
        filter_full_path          : return the filters' path as full path if true, otherwise relative paths are returned.
        filters_list_relative_dir : relative path of the directory where the filters_list files exists,
                                    the path must be relative to DATEeS_root_path.
        filters_list_file_name    : name of the file containing filters' list.

    Returns:
        Three lists:
            - the first is a list contains filters names,
            - the second is a list of lists containing source code languages,
            - the third is a list containing filters paths, return Full path by default.

    """
    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of filters
    while filters_list_relative_dir.startswith('/'):
        filters_list_relative_dir = filters_list_relative_dir[1:]
    while filters_list_relative_dir.endswith('/'):
        filters_list_relative_dir = filters_list_relative_dir[:-1]

    #
    #  Read filters' info:
    filters_list_file_path = os.path.join(DATES_ROOT_PATH, filters_list_relative_dir, filters_list_file_name)
    try:
        filters_names = []
        filters_src_languages = []
        filters_src_paths = []
        with open(filters_list_file_path,'r') as file_iterator:
            for line in file_iterator:
                line = line.strip()
                if (line.startswith('#')) or (not line) or ('[start_table]' in line) or ('[end_table]' in line):
                    # ignore comment lines and empty lines
                    continue
                else:
                    # read filter info in the current line
                    _, filter_name, filter_src_languages, filter_relative_dir = [entry.strip(' #') for entry in line.split('|')]
                    #
                    filters_names.append(filter_name)
                    filters_src_languages.append([entry.strip() for entry in filter_src_languages.split(',')] )
                    if return_filter_full_path:
                        filters_src_paths.append(os.path.join(DATES_ROOT_PATH,filter_relative_dir))
                    else:
                        filters_src_paths.append(filter_relative_dir)

    except IOError:
        err_msg = "Either the filters list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "filters_list_file_path: '%s' \n" % filters_list_file_path
        print(err_msg)
        sys.exit(1)

    #
    return filters_names, filters_src_languages, filters_src_paths
#


#
#
def read_smoothers_list(return_smoother_full_path=True,
                        smoothers_list_relative_dir='src/smoothers',
                        smoothers_list_file_name='smoothers_list.txt'
                        ):
    """
    Retrieve the list of implemented smoothers in DATeS.

    Args:
        smoother_full_path          : return the smoothers' path as full path if true, otherwise relative paths are returned.
        smoothers_list_relative_dir : relative path of the directory where the smoothers_list files exists,
                                      the path must be relative to DATEeS_root_path.
        smoothers_list_file_name    : name of the file containing smoothers' list.

    Returns:
        Three lists:
            - the first is a list contains smoothers names,
            - the second is a list of lists containing source code languages,
            - the third is a list containing smoothers paths, return Full path by default.
    """
    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of smoothers
    while smoothers_list_relative_dir.startswith('/'):
        smoothers_list_relative_dir = smoothers_list_relative_dir[1:]
    while smoothers_list_relative_dir.endswith('/'):
        smoothers_list_relative_dir = smoothers_list_relative_dir[:-1]

    #
    #  Read smoothers' info:
    smoothers_list_file_path = os.path.join(DATES_ROOT_PATH, smoothers_list_relative_dir, smoothers_list_file_name)
    try:
        smoothers_names = []
        smoothers_src_languages = []
        smoothers_src_paths = []
        with open(smoothers_list_file_path,'r') as file_iterator:
            for line in file_iterator:
                line = line.strip()
                if (line.startswith('#')) or (not line) or ('[start_table]' in line) or ('[end_table]' in line):
                    # ignore comment lines and empty lines
                    continue
                else:
                    # read smoother info in the current line
                    _, smoother_name, smoother_src_languages, smoother_relative_dir = [entry.strip(' #') for entry in line.split('|')]
                    #
                    smoothers_names.append(smoother_name)
                    smoothers_src_languages.append([entry.strip() for entry in smoother_src_languages.split(',')] )
                    if return_smoother_full_path:
                        smoothers_src_paths.append(os.path.join(DATES_ROOT_PATH,smoother_relative_dir))
                    else:
                        smoothers_src_paths.append(smoother_relative_dir)

    except IOError:
        err_msg = "Either the smoothers list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "smoothers_list_file_path: '%s' \n" % smoothers_list_file_path
        print(err_msg)
        sys.exit(1)

    #
    return smoothers_names, smoothers_src_languages, smoothers_src_paths


#
#
def read_variational_schemes_list(return_variational_schemes_full_path=True,
                                  variational_schemes_list_relative_dir='src/variational_schemes',
                                  variational_schemes_list_file_name='variational_schemes_list.txt'
                                  ):
    """
    Retrieve the list of implemented variational_schemes in DATeS.

    Args:
        variational_scheme_full_path          : return the variational_schemes' path as full path if true, otherwise relative paths are returned.
        variational_schemes_list_relative_dir : relative path of the directory where the variational_schemes_list files exists,
                                      the path must be relative to DATEeS_root_path.
        variational_schemes_list_file_name    : name of the file containing variational_schemes' list.

    Returns:
        Three lists:
            - the first is a list contains variational_schemes names,
            - the second is a list of lists containing source code languages,
            - the third is a list containing variational_schemes paths, return Full path by default.
    """
    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of variational_schemes
    while variational_schemes_list_relative_dir.startswith('/'):
        variational_schemes_list_relative_dir = variational_schemes_list_relative_dir[1:]
    while variational_schemes_list_relative_dir.endswith('/'):
        variational_schemes_list_relative_dir = variational_schemes_list_relative_dir[:-1]

    #
    #  Read variational_schemes' info:
    variational_schemes_list_file_path = os.path.join(DATES_ROOT_PATH, variational_schemes_list_relative_dir, variational_schemes_list_file_name)
    try:
        variational_schemes_names = []
        variational_schemes_src_languages = []
        variational_schemes_src_paths = []
        with open(variational_schemes_list_file_path,'r') as file_iterator:
            for line in file_iterator:
                line = line.strip()
                if (line.startswith('#')) or (not line) or ('[start_table]' in line) or ('[end_table]' in line):
                    # ignore comment lines and empty lines
                    continue
                else:
                    # read variational_scheme info in the current line
                    _, variational_scheme_name, variational_scheme_src_languages, variational_scheme_relative_dir = [entry.strip(' #') for entry in line.split('|')]
                    #
                    variational_schemes_names.append(variational_scheme_name)
                    variational_schemes_src_languages.append([entry.strip() for entry in variational_scheme_src_languages.split(',')] )
                    if return_variational_schemes_full_path:
                        variational_schemes_src_paths.append(os.path.join(DATES_ROOT_PATH,variational_scheme_relative_dir))
                    else:
                        variational_schemes_src_paths.append(variational_scheme_relative_dir)

    except IOError:
        err_msg = "Either the variational_schemes list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "variational_schemes_list_file_path: '%s' \n" % variational_schemes_list_file_path
        print(err_msg)
        sys.exit(1)

    #
    return variational_schemes_names, variational_schemes_src_languages, variational_schemes_src_paths


#
#
def read_hybrid_schemes_list(return_hybrid_scheme_full_path=True,
                             hybrid_schemes_list_relative_dir='src/hybrid_schemes',
                             hybrid_schemes_list_file_name='hybrid_schemes_list.txt'
                             ):
    """
    Retrieve the list of implemented hybrid_schemes in DATeS.

    Args:
        hybrid_scheme_full_path          : return the hybrid_schemes' path as full path if true, otherwise relative paths are returned.
        hybrid_schemes_list_relative_dir : relative path of the directory where the hybrid_schemes_list files exists,
                                      the path must be relative to DATEeS_root_path.
        hybrid_schemes_list_file_name    : name of the file containing hybrid_schemes' list.

    Returns:
        Three lists:
            - the first is a list contains hybrid_schemes names,
            - the second is a list of lists containing source code languages,
            - the third is a list containing hybrid_schemes paths, return Full path by default.

    """
    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of hybrid_schemes
    while hybrid_schemes_list_relative_dir.startswith('/'):
        hybrid_schemes_list_relative_dir = hybrid_schemes_list_relative_dir[1:]
    while hybrid_schemes_list_relative_dir.endswith('/'):
        hybrid_schemes_list_relative_dir = hybrid_schemes_list_relative_dir[:-1]

    #
    #  Read hybrid_schemes' info:
    hybrid_schemes_list_file_path = os.path.join(DATES_ROOT_PATH, hybrid_schemes_list_relative_dir, hybrid_schemes_list_file_name)
    try:
        hybrid_schemes_names = []
        hybrid_schemes_src_languages = []
        hybrid_schemes_src_paths = []
        with open(hybrid_schemes_list_file_path,'r') as file_iterator:
            for line in file_iterator:
                line = line.strip()
                if (line.startswith('#')) or (not line) or ('[start_table]' in line) or ('[end_table]' in line):
                    # ignore comment lines and empty lines
                    continue
                else:
                    # read hybrid_scheme info in the current line
                    _, hybrid_scheme_name, hybrid_scheme_src_languages, hybrid_scheme_relative_dir = [entry.strip(' #') for entry in line.split('|')]
                    #
                    hybrid_schemes_names.append(hybrid_scheme_name)
                    hybrid_schemes_src_languages.append([entry.strip() for entry in hybrid_scheme_src_languages.split(',')] )
                    if return_hybrid_scheme_full_path:
                        hybrid_schemes_src_paths.append(os.path.join(DATES_ROOT_PATH,hybrid_scheme_relative_dir))
                    else:
                        hybrid_schemes_src_paths.append(hybrid_scheme_relative_dir)

    except IOError:
        err_msg = "Either the hybrid_schemes list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "hybrid_schemes_list_file_path: '%s' \n" % hybrid_schemes_list_file_path
        print(err_msg)
        sys.exit(1)

    #
    return hybrid_schemes_names, hybrid_schemes_src_languages, hybrid_schemes_src_paths


#
#
def add_model_to_models_list(new_model_name,
                             new_model_src_languages,
                             new_model_src_path,
                             new_model_dimension,
                             models_list_relative_dir='src/models_forest',
                             models_list_file_name='models_list.txt',
                             backup_old_file=True
                             ):
    """
    Add a new model to the list of implemented models in DATeS.

    Args:
        new_model_name           : name of the new model to add to list
        new_model_src_languages  : list containing source code languages,
        new_model_src_path       : directory to add the new models' source code.
        new_model_dimension      : dimension of the model (0, 1, 2, 3, ...)
        models_list_relative_dir : relative path of the directory where the models_list files exists,
                                  the path must be relative to DATEeS_root_path.
        models_list_file_name    : name of the file containing models' list.
        backup_old_file          : save a copy of the file with extension .bk appended to the end of the file name.

    Returns:
        None

    """
    #
    # validate and pre-process inputs
    new_model_name = new_model_name.strip().replace(' ','_').replace('-','_')

    while new_model_src_path.startswith('/'):
        new_model_src_path = new_model_src_path[1:]
    while new_model_src_path.endswith('/'):
        new_model_src_path = new_model_src_path[:-1]


    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of models
    while models_list_relative_dir.startswith('/'):
        models_list_relative_dir = models_list_relative_dir[1:]
    while models_list_relative_dir.endswith('/'):
        models_list_relative_dir = models_list_relative_dir[:-1]

    #
    #  Read models' info:
    models_list_file_path = os.path.join(DATES_ROOT_PATH, models_list_relative_dir, models_list_file_name)
    try:
        models_names, models_src_languages, models_src_paths = read_models_list(return_model_full_path=False)
        num_of_models = len(models_names)
        dimension_tocken = str(new_model_dimension)+'D'

        with open(models_list_file_path,'r') as file_iterator:
            file_contents = file_iterator.readlines()

        table_start_ind = 0
        table_start_found = False
        for line in file_contents:
            if not '[start_table]' in line:
                table_start_ind += 1
            else:
                table_start_found = True
                break

        table_end_ind = 1
        table_end_found = False
        for line in file_contents:
            if not '[end_table]' in line:
                table_end_ind += 1
            else:
                table_end_found = True
                break

        # remove old table.
        print(table_start_ind)
        file_contents = file_contents[:table_start_ind]
        print(file_contents)
        if table_start_found:
            table_header = '[start_table]\n'
        else:
            table_header = '\n'*4+'[start_table]\n'
        #
        table_divider = '# '+ '-'*116 + ' #'
        table_header += table_divider +'\n'
        table_header += '# ' \
                        + ' Id'.ljust(6)  \
                        + '|' + ' model name'.ljust(26) \
                        + '|' + ' Source code language(s)'.ljust(29)  \
                        + '|' + ' model relative path'.ljust(52) \
                        + ' #\n'
        table_header += table_divider +'\n'

        if table_start_found:
            file_contents.append(table_header)
        else:
            file_contents = [table_header]

        # find suitable location of the new model entry in the new table
        next_model_ind = 0
        new_model_added = False
        for model_ind in xrange(num_of_models):
            if dimension_tocken in models_src_paths[model_ind] and not new_model_added:
                new_line =  '  ' \
                        + (' %03d' % next_model_ind).ljust(6)  \
                        + '|' + (' %s' % new_model_name).ljust(26) \
                        + '|' + (' %s' % (', '.join(new_model_src_languages))).ljust(29)  \
                        + '|' + (' %s' % new_model_src_path).ljust(52) \
                        + '  \n'
                next_model_ind += 1
                new_model_added = True
            else:
                new_line = ''

            new_line += '  ' \
                        + (' %03d' % next_model_ind).ljust(6)  \
                        + '|' + (' %s' % models_names[model_ind]).ljust(26) \
                        + '|' + (' %s' % (', '.join(models_src_languages[model_ind]))).ljust(29)  \
                        + '|' + (' %s' % models_src_paths[model_ind]).ljust(52) \
                        + '  \n'
            next_model_ind +=1

            file_contents.append(new_line)

        if not new_model_added:
            new_line = '  ' \
                        + (' %03d' % next_model_ind).ljust(6)  \
                        + '|' + (' %s' % new_model_name).ljust(26) \
                        + '|' + (' %s' % (', '.join(new_model_src_languages))).ljust(29)  \
                        + '|' + (' %s' % new_model_src_path).ljust(52) \
                        + '  \n'
            file_contents.append(new_line)

        table_footer = '# '+ '-'*116 + ' #\n'
        table_footer += '[end_table]'

        file_contents.append(table_footer)

        if backup_old_file:
            models_list_file_path_bk = models_list_file_path+'.bk'
            shutil.move(models_list_file_path, models_list_file_path_bk)

        # replace old file with the new one.
        with open(models_list_file_path,'w') as file_iterator:
            file_iterator.writelines(file_contents)

    except IOError:
        err_msg = "Either the models list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "models_list_file_path: '%s' \n" % models_list_file_path
        print(err_msg)
        sys.exit(1)


#
#
def get_model_source_path(model_name, full_path=True):
    """
    retrieve the path of the source code directory given the model name

    Args:
        model_name: string containing model name
        full_path: full path if true, otherwise relative path to DATES_ROOT_PATH is returned

    Returns:
        model_src_path: path of the model source files

    """
    #
    # obtain models' list:
    models_names, models_src_languages, models_src_paths = read_models_list()
    try:
        model_src_path = models_src_paths[models_names.index(model_name)]
    except ValueError:
        err_msg = "Model name '%s' is not recognized! Check the list of implemented models\n\n" % model_name
        err_msg += "Supported models are:\n" +'-'*(len("Supported models are:")+1) + "\n"
        err_msg += '\n'.join(['   - '+txt for txt in models_names]) + "\nTerminating...\n"
        sys.exit(err_msg)

    return model_src_path


#
#
def add_filter_to_filters_list(new_filter_name,
                               new_filter_src_languages,
                               new_filter_src_path,
                               filters_list_relative_dir='src/filters_forest',
                               filters_list_file_name='filters_list.txt',
                               backup_old_file=True
                               ):
    """
    Add a new filter to the list of implemented filters in DATeS.

    Args:
        new_filter_name           : name of the new filter to add to list
        new_filter_src_languages  : list containing source code languages,
        new_filter_src_path       : directory to add the new filters' source code.
        filters_list_relative_dir : relative path of the directory where the filters_list files exists,
                                  the path must be relative to DATEeS_root_path.
        filters_list_file_name    : name of the file containing filters' list.
        backup_old_file          : save a copy of the file with extension .bk appended to the end of the file name.

    Returns:
        None

    """
    #
    # validate and pre-process inputs
    new_filter_name = new_filter_name.strip().replace(' ','_').replace('-','_')

    while new_filter_src_path.startswith('/'):
        new_filter_src_path = new_filter_src_path[1:]
    while new_filter_src_path.endswith('/'):
        new_filter_src_path = new_filter_src_path[:-1]


    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of filters
    while filters_list_relative_dir.startswith('/'):
        filters_list_relative_dir = filters_list_relative_dir[1:]
    while filters_list_relative_dir.endswith('/'):
        filters_list_relative_dir = filters_list_relative_dir[:-1]

    #
    #  Read filters' info:
    filters_list_file_path = os.path.join(DATES_ROOT_PATH, filters_list_relative_dir, filters_list_file_name)
    try:
        filters_names, filters_src_languages, filters_src_paths = read_filters_list(return_filter_full_path=False)
        num_of_filters = len(filters_names)

        with open(filters_list_file_path,'r') as file_iterator:
            file_contents = file_iterator.readlines()

        table_start_ind = 0
        table_start_found = False
        for line in file_contents:
            if not '[start_table]' in line:
                table_start_ind += 1
            else:
                table_start_found = True
                break

        table_end_ind = 1
        table_end_found = False
        for line in file_contents:
            if not '[end_table]' in line:
                table_end_ind += 1
            else:
                table_end_found = True
                break

        if table_start_found:
            file_contents = file_contents[:table_start_ind]
        else:
            file_contents = [formulate_list_file_header('FILTER')]
        #
        table_header = '[start_table]\n'
        #
        table_divider = '# '+ '-'*116 + ' #'
        table_header += table_divider +'\n'
        table_header += '# ' \
                        + ' Id'.ljust(6)  \
                        + '|' + ' filter name'.ljust(26) \
                        + '|' + ' Source code language(s)'.ljust(29)  \
                        + '|' + ' filter relative path'.ljust(52) \
                        + ' #\n'
        table_header += table_divider +'\n'

        file_contents.append(table_header)

        # insert the new filter entry in the beginning of the new table
        next_filter_ind = 0
        new_line = '  ' \
                        + (' %03d' % next_filter_ind).ljust(6)  \
                        + '|' + (' %s' % new_filter_name).ljust(26) \
                        + '|' + (' %s' % (', '.join(new_filter_src_languages))).ljust(29)  \
                        + '|' + (' %s' % new_filter_src_path).ljust(52) \
                        + '  \n'
        file_contents.append(new_line)

        next_filter_ind +=1
        for filter_ind in xrange(num_of_filters):
            new_line  = '  ' \
                        + (' %03d' % next_filter_ind).ljust(6)  \
                        + '|' + (' %s' % filters_names[filter_ind]).ljust(26) \
                        + '|' + (' %s' % (', '.join(filters_src_languages[filter_ind]))).ljust(29)  \
                        + '|' + (' %s' % filters_src_paths[filter_ind]).ljust(52) \
                        + '  \n'
            next_filter_ind +=1

            file_contents.append(new_line)


        table_footer = '# '+ '-'*116 + ' #\n'
        table_footer += '[end_table]'

        file_contents.append(table_footer)

        if backup_old_file:
            filters_list_file_path_bk = filters_list_file_path+'.bk'
            shutil.move(filters_list_file_path, filters_list_file_path_bk)

        # replace old file with the new one.
        with open(filters_list_file_path,'w') as file_iterator:
            file_iterator.write(file_contents)

    except IOError:
        err_msg = "Either the filters list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "filters_list_file_path: '%s' \n" % filters_list_file_path
        print(err_msg)
        sys.exit(1)



#
#
def add_smoother_to_smoothers_list(new_smoother_name,
                                   new_smoother_src_languages,
                                   new_smoother_src_path,
                                   smoothers_list_relative_dir='src/smoothers_forest',
                                   smoothers_list_file_name='smoothers_list.txt',
                                   backup_old_file=True
                                   ):
    """
    Add a new smoother to the list of implemented smoothers in DATeS.

    Args:
        new_smoother_name           : name of the new smoother to add to list
        new_smoother_src_languages  : list containing source code languages,
        new_smoother_src_path       : directory to add the new smoothers' source code.
        smoothers_list_relative_dir : relative path of the directory where the smoothers_list files exists,
                                  the path must be relative to DATEeS_root_path.
        smoothers_list_file_name    : name of the file containing smoothers' list.
        backup_old_file          : save a copy of the file with extension .bk appended to the end of the file name.

    Returns:
        None

    """
    #
    # validate and pre-process inputs
    new_smoother_name = new_smoother_name.strip().replace(' ','_').replace('-','_')
    while new_smoother_src_path.startswith('/'):
        new_smoother_src_path = new_smoother_src_path[1:]
    while new_smoother_src_path.endswith('/'):
        new_smoother_src_path = new_smoother_src_path[:-1]


    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of smoothers
    while smoothers_list_relative_dir.startswith('/'):
        smoothers_list_relative_dir = smoothers_list_relative_dir[1:]
    while smoothers_list_relative_dir.endswith('/'):
        smoothers_list_relative_dir = smoothers_list_relative_dir[:-1]

    #
    #  Read smoothers' info:
    smoothers_list_file_path = os.path.join(DATES_ROOT_PATH, smoothers_list_relative_dir, smoothers_list_file_name)
    try:
        smoothers_names, smoothers_src_languages, smoothers_src_paths = read_smoothers_list(return_smoother_full_path=False)
        num_of_smoothers = len(smoothers_names)

        with open(smoothers_list_file_path,'r') as file_iterator:
            file_contents = file_iterator.readlines()

        table_start_ind = 0
        table_start_found = False
        for line in file_contents:
            if not '[start_table]' in line:
                table_start_ind += 1
            else:
                table_start_found = True
                break

        table_end_ind = 1
        table_end_found = False
        for line in file_contents:
            if not '[end_table]' in line:
                table_end_ind += 1
            else:
                table_end_found = True
                break

        if table_start_found:
            file_contents = file_contents[:table_start_ind]
        else:
            file_contents = [formulate_list_file_header('SMOOTHER')]
        #
        table_header = '[start_table]\n'
        #
        table_divider = '# '+ '-'*116 + ' #'
        table_header += table_divider +'\n'
        table_header += '# ' \
                        + ' Id'.ljust(6)  \
                        + '|' + ' smoother name'.ljust(26) \
                        + '|' + ' Source code language(s)'.ljust(29)  \
                        + '|' + ' smoother relative path'.ljust(52) \
                        + ' #\n'
        table_header += table_divider +'\n'

        file_contents.append(table_header)

        # insert the new smoother entry in the beginning of the new table
        next_smoother_ind = 0
        new_line = '  ' \
                        + (' %03d' % next_smoother_ind).ljust(6)  \
                        + '|' + (' %s' % new_smoother_name).ljust(26) \
                        + '|' + (' %s' % (', '.join(new_smoother_src_languages))).ljust(29)  \
                        + '|' + (' %s' % new_smoother_src_path).ljust(52) \
                        + '  \n'
        file_contents.append(new_line)

        next_smoother_ind +=1
        for smoother_ind in xrange(num_of_smoothers):
            new_line  = '  ' \
                        + (' %03d' % next_smoother_ind).ljust(6)  \
                        + '|' + (' %s' % smoothers_names[smoother_ind]).ljust(26) \
                        + '|' + (' %s' % (', '.join(smoothers_src_languages[smoother_ind]))).ljust(29)  \
                        + '|' + (' %s' % smoothers_src_paths[smoother_ind]).ljust(52) \
                        + '  \n'
            next_smoother_ind +=1

            file_contents.append(new_line)


        table_footer = '# '+ '-'*116 + ' #\n'
        table_footer += '[end_table]'

        file_contents.append(table_footer)

        if backup_old_file:
            smoothers_list_file_path_bk = smoothers_list_file_path+'.bk'
            shutil.move(smoothers_list_file_path, smoothers_list_file_path_bk)

        # replace old file with the new one.
        with open(smoothers_list_file_path,'w') as file_iterator:
            file_iterator.write(file_contents)

    except IOError:
        err_msg = "Either the smoothers list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "smoothers_list_file_path: '%s' \n" % smoothers_list_file_path
        print(err_msg)
        sys.exit(1)


#
#
def add_scheme_to_variational_schemes_list(new_variational_scheme_name,
                                           new_variational_scheme_src_languages,
                                           new_variational_scheme_src_path,
                                           variational_schemes_list_relative_dir='src/variational_schemes_forest',
                                           variational_schemes_list_file_name='variational_schemes_list.txt',
                                           backup_old_file=True
                                           ):
    """
    Add a new variational_scheme to the list of implemented variational_schemes in DATeS.

    Args:
        new_variational_scheme_name           : name of the new variational_scheme to add to list
        new_variational_scheme_src_languages  : list containing source code languages,
        new_variational_scheme_src_path       : directory to add the new variational_schemes' source code.
        variational_schemes_list_relative_dir : relative path of the directory where the variational_schemes_list files exists,
                                  the path must be relative to DATEeS_root_path.
        variational_schemes_list_file_name    : name of the file containing variational_schemes' list.
        backup_old_file          : save a copy of the file with extension .bk appended to the end of the file name.

    Returns:
        None

    """
    #
    # validate and pre-process inputs
    new_variational_scheme_name = new_variational_scheme_name.strip().replace(' ','_').replace('-','_')
    while new_variational_scheme_src_path.startswith('/'):
        new_variational_scheme_src_path = new_variational_scheme_src_path[1:]
    while new_variational_scheme_src_path.endswith('/'):
        new_variational_scheme_src_path = new_variational_scheme_src_path[:-1]


    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of variational_schemes
    while variational_schemes_list_relative_dir.startswith('/'):
        variational_schemes_list_relative_dir = variational_schemes_list_relative_dir[1:]
    while variational_schemes_list_relative_dir.endswith('/'):
        variational_schemes_list_relative_dir = variational_schemes_list_relative_dir[:-1]

    #
    #  Read variational_schemes' info:
    variational_schemes_list_file_path = os.path.join(DATES_ROOT_PATH, variational_schemes_list_relative_dir, variational_schemes_list_file_name)
    try:
        variational_schemes_names, variational_schemes_src_languages, variational_schemes_src_paths = read_variational_schemes_list(return_variational_scheme_full_path=False)
        num_of_variational_schemes = len(variational_schemes_names)

        with open(variational_schemes_list_file_path,'r') as file_iterator:
            file_contents = file_iterator.readlines()

        table_start_ind = 0
        table_start_found = False
        for line in file_contents:
            if not '[start_table]' in line:
                table_start_ind += 1
            else:
                table_start_found = True
                break

        table_end_ind = 1
        table_end_found = False
        for line in file_contents:
            if not '[end_table]' in line:
                table_end_ind += 1
            else:
                table_end_found = True
                break

        if table_start_found:
            file_contents = file_contents[:table_start_ind]
        else:
            file_contents = [formulate_list_file_header('VARIATIONAL scheme')]
        #
        table_header = '[start_table]\n'
        #
        table_divider = '# '+ '-'*116 + ' #'
        table_header += table_divider +'\n'
        table_header += '# ' \
                        + ' Id'.ljust(6)  \
                        + '|' + ' variational_scheme name'.ljust(26) \
                        + '|' + ' Source code language(s)'.ljust(29)  \
                        + '|' + ' variational_scheme relative path'.ljust(52) \
                        + ' #\n'
        table_header += table_divider +'\n'

        file_contents.append(table_header)


        # insert the new variational_scheme entry in the beginning of the new table
        next_variational_scheme_ind = 0
        new_line = '  ' \
                        + (' %03d' % next_variational_scheme_ind).ljust(6)  \
                        + '|' + (' %s' % new_variational_scheme_name).ljust(26) \
                        + '|' + (' %s' % (', '.join(new_variational_scheme_src_languages))).ljust(29)  \
                        + '|' + (' %s' % new_variational_scheme_src_path).ljust(52) \
                        + '  \n'
        file_contents.append(new_line)

        next_variational_scheme_ind +=1
        for variational_scheme_ind in xrange(num_of_variational_schemes):
            new_line  = '  ' \
                        + (' %03d' % next_variational_scheme_ind).ljust(6)  \
                        + '|' + (' %s' % variational_schemes_names[variational_scheme_ind]).ljust(26) \
                        + '|' + (' %s' % (', '.join(variational_schemes_src_languages[variational_scheme_ind]))).ljust(29)  \
                        + '|' + (' %s' % variational_schemes_src_paths[variational_scheme_ind]).ljust(52) \
                        + '  \n'
            next_variational_scheme_ind +=1

            file_contents.append(new_line)


        table_footer = '# '+ '-'*116 + ' #\n'
        table_footer += '[end_table]'

        file_contents.append(table_footer)

        if backup_old_file:
            variational_schemes_list_file_path_bk = variational_schemes_list_file_path+'.bk'
            shutil.move(variational_schemes_list_file_path, variational_schemes_list_file_path_bk)

        # replace old file with the new one.
        with open(variational_schemes_list_file_path,'w') as file_iterator:
            file_iterator.write(file_contents)

    except IOError:
        err_msg = "Either the variational_schemes list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "variational_schemes_list_file_path: '%s' \n" % variational_schemes_list_file_path
        print(err_msg)
        sys.exit(1)


#
#
def add_scheme_to_hybrid_schemes_list(new_hybrid_scheme_name,
                                      new_hybrid_scheme_src_languages,
                                      new_hybrid_scheme_src_path,
                                      hybrid_schemes_list_relative_dir='src/hybrid_schemes_forest',
                                      hybrid_schemes_list_file_name='hybrid_schemes_list.txt',
                                      backup_old_file=True
                                      ):
    """
    Add a new hybrid_scheme to the list of implemented hybrid_schemes in DATeS.

    Args:
        new_hybrid_scheme_name           : name of the new hybrid_scheme to add to list
        new_hybrid_scheme_src_languages  : list containing source code languages,
        new_hybrid_scheme_src_path       : directory to add the new hybrid_schemes' source code.
        hybrid_schemes_list_relative_dir : relative path of the directory where the hybrid_schemes_list files exists,
                                  the path must be relative to DATEeS_root_path.
        hybrid_schemes_list_file_name    : name of the file containing hybrid_schemes' list.
        backup_old_file          : save a copy of the file with extension .bk appended to the end of the file name.

    Returns:
        None

    """
    #
    # validate and pre-process inputs
    new_hybrid_scheme_name = new_hybrid_scheme_name.strip().replace(' ', '_').replace('-', '_')
    while new_hybrid_scheme_src_path.startswith('/'):
        new_hybrid_scheme_src_path = new_hybrid_scheme_src_path[1:]
    while new_hybrid_scheme_src_path.endswith('/'):
        new_hybrid_scheme_src_path = new_hybrid_scheme_src_path[:-1]


    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    # read data list of hybrid_schemes
    while hybrid_schemes_list_relative_dir.startswith('/'):
        hybrid_schemes_list_relative_dir = hybrid_schemes_list_relative_dir[1:]
    while hybrid_schemes_list_relative_dir.endswith('/'):
        hybrid_schemes_list_relative_dir = hybrid_schemes_list_relative_dir[:-1]

    #
    #  Read hybrid_schemes' info:
    hybrid_schemes_list_file_path = os.path.join(DATES_ROOT_PATH, hybrid_schemes_list_relative_dir, hybrid_schemes_list_file_name)
    try:
        hybrid_schemes_names, hybrid_schemes_src_languages, hybrid_schemes_src_paths = read_hybrid_schemes_list(return_hybrid_scheme_full_path=False)
        num_of_hybrid_schemes = len(hybrid_schemes_names)

        with open(hybrid_schemes_list_file_path,'r') as file_iterator:
            file_contents = file_iterator.readlines()

        table_start_ind = 0
        table_start_found = False
        for line in file_contents:
            if not '[start_table]' in line:
                table_start_ind += 1
            else:
                table_start_found = True
                break

        table_end_ind = 1
        table_end_found = False
        for line in file_contents:
            if not '[end_table]' in line:
                table_end_ind += 1
            else:
                table_end_found = True
                break

        if table_start_found:
            file_contents = file_contents[:table_start_ind]
        else:
            file_contents = [formulate_list_file_header('HYBRID scheme')]
        #
        table_header = '[start_table]\n'
        #
        table_divider = '# '+ '-'*116 + ' #'
        table_header += table_divider +'\n'
        table_header += '# ' \
                        + ' Id'.ljust(6)  \
                        + '|' + ' hybrid_scheme name'.ljust(26) \
                        + '|' + ' Source code language(s)'.ljust(29)  \
                        + '|' + ' hybrid_scheme relative path'.ljust(52) \
                        + ' #\n'
        table_header += table_divider +'\n'

        file_contents.append(table_header)

        # insert the new hybrid_scheme entry in the beginning of the new table
        next_hybrid_scheme_ind = 0
        new_line = '  ' \
                        + (' %03d' % next_hybrid_scheme_ind).ljust(6)  \
                        + '|' + (' %s' % new_hybrid_scheme_name).ljust(26) \
                        + '|' + (' %s' % (', '.join(new_hybrid_scheme_src_languages))).ljust(29)  \
                        + '|' + (' %s' % new_hybrid_scheme_src_path).ljust(52) \
                        + '  \n'
        file_contents.append(new_line)

        next_hybrid_scheme_ind +=1
        for hybrid_scheme_ind in xrange(num_of_hybrid_schemes):
            new_line  = '  ' \
                        + (' %03d' % next_hybrid_scheme_ind).ljust(6)  \
                        + '|' + (' %s' % hybrid_schemes_names[hybrid_scheme_ind]).ljust(26) \
                        + '|' + (' %s' % (', '.join(hybrid_schemes_src_languages[hybrid_scheme_ind]))).ljust(29)  \
                        + '|' + (' %s' % hybrid_schemes_src_paths[hybrid_scheme_ind]).ljust(52) \
                        + '  \n'
            next_hybrid_scheme_ind +=1

            file_contents.append(new_line)


        table_footer = '# '+ '-'*116 + ' #\n'
        table_footer += '[end_table]'

        file_contents.append(table_footer)

        if backup_old_file:
            hybrid_schemes_list_file_path_bk = hybrid_schemes_list_file_path+'.bk'
            shutil.move(hybrid_schemes_list_file_path, hybrid_schemes_list_file_path_bk)

        # replace old file with the new one.
        with open(hybrid_schemes_list_file_path,'w') as file_iterator:
            file_iterator.write(file_contents)

    except IOError:
        err_msg = "Either the hybrid_schemes list directory does not exist or the files list is not available. "
        err_msg += "\n"+ "hybrid_schemes_list_file_path: '%s' \n" % hybrid_schemes_list_file_path
        print(err_msg)
        sys.exit(1)


#
#
def formulate_list_file_header(schemes_type, line_length=120 ):
    """
    Return a string containing the header of any of the files containing lists of (models, filters,...)

    Returns:
        file_header

    """
    #
    # Some constant strings for maintaining headers of list files....
    #
    divider = "#%s#\n" %('='*(line_length-2))
    header = divider + "#%s#\n" % (' '*(line_length-2))
    header += ("# This file contains a list of all %ss incorporated in DATeS." % schemes_type).ljust(line_length-1) + "#\n"
    header += "# lines starting with '#' are taken as comments lines.".ljust(line_length-1)  + "#\n"
    header += ("# You can add new %s manually, but we recommend adding new SCHEME using suitable driver." % schemes_type).ljust(line_length-1)  + "#\n"
    header += "#%s#\n" % (' '*(line_length-2))
    temp_r  = "# To add a %s to the list:" % schemes_type
    header += temp_r.ljust(line_length-1) + "#\n"
    header += ("# %s" % ('-'*(len(temp_r)+2)) ).ljust(line_length-1) + "#\n"
    header += ("#      1- Make sure the %s does not exist in the list (or chose a new name and a new directory)," % schemes_type ).ljust(line_length-1) + "#\n"
    header +=  "#      2- Copy your source code into a new directory (respect the forest structure logic),".ljust(line_length-1) + "#\n"
    header += ("#      3- Add the new %s info as described in the table below." % schemes_type).ljust(line_length-1) + "#\n"
    temp_r  = "#       Remarks:"
    header += temp_r.ljust(line_length-1) + "#\n"
    header += ("# %s" % ('~'*(len(temp_r)+2)) ).ljust(line_length-1) + "#\n"
    header += ("#           i   - Make sure you add a %s name." % schemes_type).ljust(line_length-1) + "#\n"
    header +=  "#           ii  - If the source codes incorporate different language, list all (comma-separated) e.g. C,C++,Fortran).".ljust(line_length-1) + "#\n"
    header += ("#           iii - The %s relative path must be added" % schemes_type).ljust(line_length-1) + "#\n"
    header += "#           iv  - Adding the code manually, means, you need to take care of the interfacing yourself!".ljust(line_length-1) + "#\n"
    header += "#           v   - Don't remove the line starting with '[start_table]' !".ljust(line_length-1) + "#\n"
    header += "#%s#\n" % (' '*(line_length-2))
    header += divider + "#%s#\n\n\n\n" % (' '*(line_length-2))

    return header
    #



#
#  This group of methods handle experiment preparation, and cleanup
#  this includes copying source files, deleting executables etc.
#
def prepare_model_files(model_name, working_dir_rel_path='EXPERIMENT_RUN/', subdir_name='model_src'):
    """
    Copy the necessary model files to the working directory directory.

    Args:
        model_name: name of the model. must be in the table of models in the models list file

    Returns:
         target_dir: full path of the model_source files directory for experiment run

    """
    #
    # Retrieve DATeS root directory:
    DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
    if DATES_ROOT_PATH is None:
        err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                  % ('DATES_ROOT_PATH', 'dates_setup.py')
        sys.exit(err_msg)

    run_dir_path = os.path.join(DATES_ROOT_PATH, working_dir_rel_path)

    # create new directory with folder_name and copy model files in it
    if os.path.exists(run_dir_path):
        print("Cleaning-up experiment directory||"),
        shutil.rmtree(run_dir_path)

    # find model source files
    source_model_path = get_model_source_path(model_name, full_path=True)

    # target directory:
    if subdir_name is not None:
        target_dir = os.path.join(run_dir_path, subdir_name)
    else:
        target_dir = run_dir_path

    # copy the whole contents of source folder
    print("Copying necessary model files || "),
    shutil.copytree(source_model_path, target_dir)

    return target_dir


#
#
#
def clean_executable_files(root_dir=None, rm_extensions=['.o', '.out', '.pyc', '.exe']):
    """
    remove executable files generated during execution in all subdirectories under the passed root_dir

    Args:
         root_dir:      directory to start search recursively for executable files under.
                        If None is passed, DATES_ROOT_PATH will be used...
         rm_extensions: list containing all extensions to search for and remove.
                        if only one type is passed, it can be a string.
                        Be careful with this list after making use of Python Extensions...
    Returns:
        None

    """
    #
    if type(rm_extensions) is str:
        rm_extensions = [rm_extensions]

    if root_dir is None:
        DATES_ROOT_PATH = os.getenv("DATES_ROOT_PATH")
        if DATES_ROOT_PATH is None:
            err_msg = "The variable '%s' must be exported to the environment!\n   \Please run the script '%s' in the beginning of your MAIN driver!" \
                      % ('DATES_ROOT_PATH', 'dates_setup.py')
            sys.exit(err_msg)
        else:
            root_dir = DATES_ROOT_PATH


    if not os.path.isdir(root_dir):
        err_msg = " ['%s'] is not a valid directory!" % root_dir
        sys.exit(err_msg)

    rm_extensions = [ext if ext.startswith('.') else '.'+ext for ext in rm_extensions]

    for root, _, files_list in os.walk(root_dir):
        for file_name in files_list:
            # check for any occurrences of any of the extensions passed
            if len([True for check in rm_extensions if file_name.endswith(check)]) > 0:
                os.remove(os.path.join(root,file_name))


#
#
def query_yes_no(message, default="yes"):
    """
    Terminal-based query: Y/N.
    This keeps asking until a valid yes/no is passed by user.

    Args:
        message: a string prented on the termianl to the user.
        def_answer: the answer presumed, e.g. if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (in the latter case, the answer is required of the user).

    Returns:
        The "answer" return value is True for "yes" or False for "no".

    """
    # valid = {'yes': True, 'y': True, 'ye': True, 'yeah': True, 'yep': True, 'yup': True,
    #          'no': False, 'n': False, 'nope': False, 'nop': False}

    valid_yes = {'yes': True, 'y': True, 'ye': True, 'yeah': True, 'yep': True, 'yup': True}
    valid_no = {'no': False, 'n': False, 'nope': False, 'nop': False}
    valid = valid_yes.copy()
    valid.update(valid_no)

    if default is None or isinstance(default, str):
        pass
    else:
        print("How is this possible? Not 'None', and not valid string???")
        raise TypeError

    if default is None:
        prompt = " [y/n]:? "
    elif default.lower() in valid_yes:
        prompt = " [y/n]: default [Y] "
    elif default.lower() in valid_no:
        prompt = " [y/n]: default [N] "
    else:
        raise ValueError("invalid default answer: '%s' Please input a wide yes/no answer!" % default)

    while True:
        sys.stdout.write(message + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n') or at least something close such as yep, nop, etc.\n")
