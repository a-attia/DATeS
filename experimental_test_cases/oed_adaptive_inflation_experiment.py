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

Apply Ensemble Kalman Filter With Adaptive inflation to Lorenz96 or QG1.5 model.
"""

import sys
sys.path.insert(1, "../")

import os
import re
import numpy as np
import ConfigParser

BASE_RESULTS_DIR = 'Results/CoupledLorenz_OED_EnKF'

from OED_adaptive_EnKF_test_driver import start_filtering


def write_configs_to_file(file_path, configs, section_header):
    """
    """
    parser = ConfigParser.ConfigParser()
    parser.add_section(section_header)
    for key in configs:
        parser.set(section_header, key, configs[key])
    with open(file_path, 'w') as f_id:
        parser.write(f_id)


if __name__ == '__main__':

    design_penalty_pool = np.arange(0, 0.1, 0.005)
    localization_function_pool = ['Gaspari-Cohn']
    ensemble_size_pool = [25]  # np.arange(5, 41, 5)
    regularization_norm_pool = ['l1', 'l2']
    moving_average_radius_pool = [0, 2, 4]

    localization_radius = 0.5

    # configurations file:
    settings_filename = 'oed_settings.dat'
    this_dir = os.path.abspath(os.path.dirname(__file__))
    settings_file_path = os.path.join(this_dir, settings_filename)
    model_name = 'Lorenz96'

    num_experiments = len(regularization_norm_pool) * len(ensemble_size_pool) * len(localization_function_pool) * len(design_penalty_pool)
    exp_no = 0
    #
    for moving_average_radius in moving_average_radius_pool:
        for regularization_norm in regularization_norm_pool:
            for localization_function in localization_function_pool:
                for ensemble_size in ensemble_size_pool:
                    for design_penalty in design_penalty_pool:
                        #
                        configs = dict(adaptive_inflation=True,
                                       moving_average_radius=moving_average_radius,
                                       regularization_norm=regularization_norm,
                                       inflation_factor=1.09,
                                       inflation_bounds=(1.000001, 1.5),
                                       inflation_design_penalty=design_penalty,
                                       #
                                       adaptive_localization=False,
                                       localization_space='B',
                                       loc_direct_approach=3,
                                       localization_function=localization_function,
                                       localization_radius=localization_radius,
                                       localization_bounds=(0.0, 20.0),
                                       localization_design_penalty = 0.000,
                                       ensemble_size=ensemble_size,
                                      )

                        write_configs_to_file(settings_file_path, configs, 'filter settings')

                        # Print some header
                        sep = "%s(Adaptive-Inflation EnKF Experiment)%s\n" % ('='*25, '='*25)
                        exp_no += 1
                        print("Experiment Number [%d] out of [%d]" % (exp_no, num_experiments))
                        print(sep)
                        print("Ensemble Size: %d" % ensemble_size)
                        print("Design Penalty Size: %f" % design_penalty)
                        print("Localization Function: %s" % localization_function)
                        print("%s\n" % ('='*len(sep)))

                        # prepare output path:
                        results_dir = os.path.join(BASE_RESULTS_DIR, "Adaptive_Inflation")
                        results_dir = os.path.join(results_dir, 'Reg_Norm_%s_LocRad_%f_InflPenalty_%f_MovingAvgRad_%d' %(regularization_norm, localization_radius,
                                                                                                                         design_penalty, moving_average_radius))

                        try:
                            out = start_filtering(results_dir, overwrite=False)
                        except Exception as e:
                            print("An exception is thrown... [%s]" % type(e))
                            print("Exception original message:")
                            print(str(e))
                            if True:  # TODO: Remove after debugging
                                raise
                            else:
                                print("...PASSING...")

    #
    # Clean executables and temporary modules
    # ---------------------
    # utility.clean_executable_files(rm_extensions=['.pyc'])
    #
