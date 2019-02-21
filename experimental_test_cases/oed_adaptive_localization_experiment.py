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

Apply Ensemble Kalman Filter With Adaptive covariance localization to Lorenz96 or QG1.5 model.
"""

import sys
import os
import re
import numpy as np

from oed_adaptive_inflation_experiment import write_configs_to_file
from OED_adaptive_EnKF_test_driver import start_filtering


BASE_RESULTS_DIR = 'Results/CoupledLorenz_OED_EnKF'

if __name__ == '__main__':

    # Pools of combinations to try
    design_penalty_pool        = [0.0002]  # np.arange(0, 0.00051, 0.00005)
    localization_function_pool = ['Gaspari-Cohn']
    ensemble_size_pool         = [25]
    regularization_norm_pool   = ['l1']
    moving_average_radius_pool = [3]
    localization_space_pool    = ['R1', 'R2']
    loc_direct_approach_pool   = [3]
    inflation_factor_pool      = [1.35]  # np.arange(1, 1.351, 0.05)

    # fixed settings
    initial_localization_radius = 1.0
    localization_bounds = (0.0, 5)
    #
    # configurations file:
    model_name = 'lorenz'
    settings_filename = 'oed_settings.dat'
    this_dir = os.path.abspath(os.path.dirname(__file__))
    settings_file_path = os.path.join(this_dir, settings_filename)

    # Start experiments
    num_experiments = len(ensemble_size_pool) * len(localization_function_pool) * len(design_penalty_pool) * len(regularization_norm_pool)
    num_experiments *= (len(loc_direct_approach_pool) * len(inflation_factor_pool) * len(localization_space_pool) * len(moving_average_radius_pool))
    exp_no = 0
    #
    l1_ok = False
    l2_ok = False
    for ensemble_size in ensemble_size_pool:
        for design_penalty in design_penalty_pool:
            for localization_function in localization_function_pool:
                for loc_direct_approach in loc_direct_approach_pool:
                    for moving_average_radius in moving_average_radius_pool:
                        for localization_space in localization_space_pool:
                            for inflation_factor in inflation_factor_pool:
                                for regularization_norm in regularization_norm_pool:

                                    # Avoid duplication
                                    if re.match('r\Al(-|_| )*1\Z', regularization_norm, re.IGNORECASE):
                                        if not l1_ok:
                                            l1_ok = True
                                        if l2_ok and design_penalty==0:
                                            continue
                                    if re.match('r\Al(-|_| )*2\Z', regularization_norm, re.IGNORECASE):
                                        if not l2_ok:
                                            l2_ok = True
                                        if l1_ok and design_penalty==0:
                                            continue
                                    #
                                    configs = dict(adaptive_inflation=False,
                                                   forecast_inflation_factor=inflation_factor,
                                                   inflation_factor=1.0,
                                                   inflation_design_penalty=0.0,
                                                   inflation_bounds=(1.000001, 1.5),
                                                   #
                                                   adaptive_localization=True,
                                                   localization_function=localization_function,
                                                   localization_radius=initial_localization_radius,
                                                   loc_direct_approach=loc_direct_approach,
                                                   localization_space=localization_space,
                                                   regularization_norm=regularization_norm,
                                                   localization_design_penalty=design_penalty,
                                                   localization_bounds=localization_bounds,
                                                   moving_average_radius=moving_average_radius,
                                                   #
                                                   ensemble_size=ensemble_size,
                                                  )
                                    #
                                    write_configs_to_file(settings_file_path, configs, 'filter settings')
                                    #
                                    # Print some header
                                    sep = "%s(Adaptive-Localization EnKF Experiment)%s\n" % ('='*25, '='*25)
                                    exp_no += 1
                                    print("Experiment Number [%d] out of [%d]" % (exp_no, num_experiments))
                                    print(sep)
                                    print("Ensemble Size: %d" % ensemble_size)
                                    print("Design Penalty: %f" % design_penalty)
                                    print("Localization Function: %s" % localization_function)
                                    print("Localization direction/approach: %d " % loc_direct_approach)
                                    print("Inflation factor: %f " % inflation_factor)
                                    print("Localization Space: %s " % localization_space)
                                    print("Moving-average Smoothing Radius: %d " % moving_average_radius)
                                    print("Regularization Norm: %s " % regularization_norm)
                                    print("%s\n" % ('='*len(sep)))
                                    #
                                    # prepare output path:
                                    results_dir = os.path.join(BASE_RESULTS_DIR, "Adaptive_Localization")
                                    results_dir = os.path.join(results_dir, 'Reg_Norm_%s_InflFac_%f_LocPenalty_%f_LocSpace_%s_MovingAvgRad_%d' %(regularization_norm,
                                                                                                                                                 inflation_factor,
                                                                                                                                                 design_penalty,
                                                                                                                                                 localization_space,
                                                                                                                                                 moving_average_radius))

                                    try:
                                        out = start_filtering(results_dir, overwrite=False, create_plots=False)
                                    except Exception as e:
                                        print("An exception is thrown... [%s]" % type(e))
                                        print("Exception original message:")
                                        print(str(e))
                                        if True:  # TODO: Remove after debugging
                                            raise
                                        else:
                                            print("...PASSING...")
                                    
                                    try:
                                        osf = os.path.join(results_dir, 'output_dir_structure.txt')
                                        cmd = "python filtering_results_reader_coupledlorenz.py -f %s -o True -r True" % osf
                                        print("Plotting Results:\n%s" % cmd)
                                        # os.system(cmd)
                                    except:
                                        print("Failed to generate plots!")
                                    #
    # Clean executables and temporary modules
    # ---------------------
    # utility.clean_executable_files(rm_extensions=['.pyc'])
    #
