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
    dates_constants:  (module under development); we may migrate it to dates_setup.py!
    -----------------
        Define necessary global variables needed for access by DATeS modules, and scripts.
        These variables are defined as environment variables.
            1- DATES_ROOT_PATH: Full path of the root directory of the DATeS package,
            2- DATES_TIME_EPS: time precesion; used to compare time instances t0, t1;
                e.g. if abs(t0-t1)<=DATES_TIME_EPS, then t0==t1
"""

import os
from sys import platform as _platform

#
# 1- Full path of the root directory of the DATeS package:
if _platform.lower().startswith("linux"):
    # linux
    path_sep = '/'
elif _platform.lower().startswith("darwin"):
    # MAC OS X
    path_sep = '/'
elif _platform.lower().startswith("win"):
    # Windows
    path_sep = '\\'
else:
    # Other OS!
    path_sep = '/'

DATES_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
while DATES_ROOT_PATH.endswith(path_sep):
    DATES_ROOT_PATH = DATES_ROOT_PATH[:-1]

if os.path.basename(DATES_ROOT_PATH).lower() != 'src':
    print("This module 'dates_constants' MUST be placed inside 'src/' directory!")
    raise IOError
else:
    DATES_ROOT_PATH = os.path.dirname(DATES_ROOT_PATH)

#
# 2- time precision; used to compare time instances t0, t1; e.g. if abs(t0-t1)<=DATES_TIME_EPS, then t0==t1
DATES_TIME_EPS = 1e-7

# 3- general verbosity; this will be overridden when explicityly passed to a python module
DATES_VERBOSE = 'FALSE'
