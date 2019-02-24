
import os
import numpy as np


inflation_factors = np.arange(1.41, 1.5, 0.01)
localization_radii = [0]  # np.append(np.arange(0.1, 2, 0.1), np.infty)

for infl in inflation_factors:
    for loc in localization_radii:
        cmd = "python coupledlorenz96_enkf_test_driver.py %f %f" % (infl, loc)
        print("RUNNING 'coupledlorenz96_enkf_test_driver.py'\n\tWITH INFLATION FACTOR %f AND LOCALIZATION PARAMETER %f" % (infl, loc))
        os.system(cmd)
