#!/usr/bin/env python

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
    Initialize DATeS:
    -----------------
        Define necessary global variables needed for access by DATeS modules, and scripts.
        An alternative to this procedure, will be to export to Environment variables.
        This module should be imported, and the method 'initialize_dates' should be executed prior to further use of DATeS by your driver script.

        Also, this module import all global variables for the package from src.dates_constants


    Example: initialize DATeS with default settings:
    ------------------------------------------------
        In your driver script:
        import dates_setup
        dates_setup.initialize_dates()
"""


import os
import sys
import numpy as np

try:
    from src.dates_constants import *
except ImportError:
    print("Failed to import DATeS constants;")
    msg = "\n\t  DATeS constnats' file 'src.dates_constants.py' is missing!\n\t\t Download it, and past it into 'src/' directory \n\n "
    sep = "\n" + "*"*81 + "\n"
    print(sep + msg + sep)
    raise


# a list of paths containing basic (manually updated) source paths in the DATeS package.
_def_rel_paths_list = ['src', 'test_cases']

#
def add_src_paths(add_all_dirs=True, relative_paths_list=None):
    """
    Add directories containing source files to the system directories.
    All package-specified paths are inserted in the beginning of the system paths' list
    for proper import and faster access of DATeS modules.

    Args:
        add_all_dirs: bool (default is True).
            * If True: all DATeS subdirectories (including models, test_cases, etc.) will be added to the sys.path.
            * If False: only necessary directories (passed to 'relative_paths_list') will be added.
              This requires later update if other directories are added to the package.

        relative_paths_list: list (default is None)
        A list of relative paths to add to sys.path to give access to DATeS modules.
        This shoudl be used only if you know all necessary paths in DATeS.
        Used only if add_all_dirs==False

    Returns:
        None

    """
    #
    # Input Assertion
    #--------------------------
    if not isinstance(add_all_dirs, bool):
        print("Input argument 'add_all_dirs' has to be a boolean. Passed %s !" % repr(add_all_dirs))
        raise AssertionError()

    if relative_paths_list is not None:
        if not isinstance(relative_paths_list, list):
            print("Input argument 'relative_paths_list' has to be None or a list. \
                   Passed %s !" % repr(relative_paths_list))
            raise AssertionError()
    #--------------------------
    #
    if add_all_dirs:
        sys.path.insert(2,DATES_ROOT_PATH)
        _add_subdirs_to_sys(os.path.join(DATES_ROOT_PATH,'src'))
        _add_subdirs_to_sys(os.path.join(DATES_ROOT_PATH,'test_cases'))
        _add_subdirs_to_sys(os.path.join(DATES_ROOT_PATH,'experimental_test_cases'))
        #

    else:
        # This is a list of source directories to be manually maintained if this route is chosen.
        if relative_paths_list is None:
            relative_paths_list = _def_rel_paths_list
        elif len(relative_paths_list) == 0:
            relative_paths_list = _def_rel_paths_list
        else:
            pass

        for rel_path in relative_paths_list:
            # validate string of the relative path
            rel_path = rel_path.strip()
            while rel_path.endswith('/') or rel_path.endswith('\\'):
                rel_path = rel_path[:-1]
            while rel_path.startswith('/') or rel_path.startswith('\\'):
                rel_path = rel_path[1:]

            # get the full path and add to sys.path if it is a valid directory
            path_to_add = os.path.join(DATES_ROOT_PATH, rel_path)
            if os.path.isdir(path_to_add):
                if not path_to_add in sys.path:
                    sys.path.insert(2, path_to_add)
            else :
                print(" [%s] is not a valid path/directory to add to sys.path. " % path_to_add)
                raise IOError()


def _add_subdirs_to_sys(path):
    """
    """
    for root, _, _ in os.walk(path):
        # '/.' insures to ignore any subdirectory of special directory such as '.git' subdirs.
        # '__' insures that the iterator ignores any cashed subdirectory.
        if os.path.isdir(root) and not ('/.' in root or '__' in root):
            # in case this is not the initial run. We don't want  to add duplicates to the system paths' list.
            if root not in sys.path:
                sys.path.insert(2, root)

#
def initialize_dates(add_all_dirs=True, relative_paths_list=None, random_seed=None, verbose=None):
    """
    Setup up Environment Variables and add source directories to PYTHONPATH.

    Args:
        add_all_dirs: bool (True/False).
            * If True: all DATeS subdirectories (including models, test_cases, etc.)
              will be added to the sys.path.
            * If False: only necessary directories (passed to 'relative_paths_list') will be added.
              This requires later update if other directories are added to the package.

        relative_paths_list: list (default is None)
        A list of relative paths to add to sys.path to give access to DATeS modules.
        This shoudl be used only if you know all necessary paths in DATeS.
        Used only if add_all_dirs==False

        random_seed: integer (default is None)
        An integer used to reset the seed of NumPy random number generator globally.

    Returns:
        None

    """
    #
    # Input Assertion
    #--------------------------
    if not isinstance(add_all_dirs, bool):
        print("Input argument 'add_all_dirs' has to be a boolean. Passed %s !" % repr(add_all_dirs))
        raise AssertionError()

    if relative_paths_list is not None:
        if not isinstance(relative_paths_list, list):
            print("Input argument 'relative_paths_list' has to be None or a list. \
                   Passed %s !" % repr(relative_paths_list))
            raise AssertionError()

    if random_seed is not None:
        if not isinstance(random_seed, int):
            print("Input argument 'random_seed' has to be an integer. \
                   Passed %s !" % repr(random_seed))
            raise AssertionError()
    #--------------------------
    #
    # Try to retrieve, then set/update the root directory of the DATeS package ('DATES_ROOT_PATH')
    # to the location of this file:
    env_DATeS_root_path = os.getenv("DATES_ROOT_PATH")
    if env_DATeS_root_path is None or env_DATeS_root_path != DATES_ROOT_PATH:
        os.environ['DATES_ROOT_PATH'] = DATES_ROOT_PATH

    # Try to retrieve, then set/update tim-precision variable
    env_DATeS_eps = os.getenv("DATES_TIME_EPS")
    if env_DATeS_eps is None or env_DATeS_eps != DATES_TIME_EPS:
        os.environ['DATES_TIME_EPS'] = str(DATES_TIME_EPS)

    # Recursively add directories containing source files to be used by DATeS
    add_src_paths(add_all_dirs=add_all_dirs,
                  relative_paths_list=relative_paths_list
                  )

    # Reset the random seed of Numpy random number generators
    if random_seed is not None:
        np.random.seed(random_seed)


    # set general DATeS verbosity if passed, otherwise it will be read from DATeS constants module
    env_DATeS_verbose = os.getenv("DATES_VERBOSE")
    if str(verbose) == 'None':
        verbose = DATES_VERBOSE
    elif str(verbose) == 'True' or str(verbose) == 'False':
        pass
    else:
        verbose = False

    if env_DATeS_verbose is None or env_DATeS_verbose != str(verbose):
        os.environ['DATES_VERBOSE'] = verbose

#
if __name__ == "__main__":
    # Initialize DATeS with default settings
    initialize_dates()
