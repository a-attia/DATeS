
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
    A module providing functions that handle optimization related functionalities; this includes validating gradient, etc.
"""


import shutil
import os
import sys
import numpy as np


#
#
def validate_gradient(state, gradient, func, *func_params, **fd_type_and_output):
    """
    Validate gradient of a scalar function
    
    Args:
        state: state vector at which gradient of the objective function is evaluated
        gradient: exact gradient vector of 'func', evaluated at 'state'
        func: objective function to differentiate
        func_params: (tuple) of function parameters passed to 'func'
        fd_type_and_output: (dictionary) of named arguments. Supported are:
            - 'FD_type': type of the finite difference scheme: 'left', 'right', 'central'
            - 'screen_output': If True, results are printed to screen before return.

    Returns:
        Grad_FD: Gradient vector evaluated using finite difference approximation
        Rel_ERR: vector containing relative errors of the exact vs. approximate gradient.
        
    """
    #
    # default values of un-named keyword args, issues o avoid portability ito older versions.
    if fd_type_and_output.has_key('FD_type'):
        FD_type = fd_type_and_output['FD_type']
    else:
        FD_type = 'centeral'

    if fd_type_and_output.has_key('screen_output'):
        screen_output = fd_type_and_output['screen_output']
    else:
        screen_output = False

    # Initialize finite difference approximation gradient and relative error vector:
    Grad_FD = np.zeros(state.size , dtype=np.float64)
    Rel_ERR = np.zeros(state.size , dtype=np.float64)

    eps = 1e-5

    if FD_type == 'left':
        # left finite difference approximation
        #
        F_1 = func( state , *func_params )
        #
        for i in range(state.size):
            X_2   = state.copy()
            X_2[i] = state[i] - eps
            F_2 = func( X_2 , *func_params )

            Grad_FD[i] = (F_1 - F_2)/(eps)
            Rel_ERR[i] = (gradient[i]-Grad_FD[i])/Grad_FD[i]

            if screen_output:
                sp = '    '
                print(' i=[%4d]:%sGradient=%12.9e;%s|%sFD-Grad=%12.9e;%s|%sRel-Err =%12.9e' % (i,sp, gradient[i], sp, sp, Grad_FD[i], Rel_ERR[i], sp, sp))

    elif FD_type =='right':
        # right finite difference approximation
        #
        F_1 = func( state , *func_params )
        #
        for i in range(state.size):
            X_2   = state.copy()
            X_2[i] = state[i] + eps
            F_2 = func( X_2 , *func_params )

            Grad_FD[i] = (F_2-F_1)/(eps)
            Rel_ERR[i] = (gradient[i]-Grad_FD[i])/Grad_FD[i]

            if screen_output:
                sp = '    '
                print(' i=[%4d]:%sGradient=%12.9e;%s|%sFD-Grad=%12.9e;%s|%sRel-Err =%12.9e' % (i,sp, gradient[i], sp, sp, Grad_FD[i], Rel_ERR[i], sp, sp))

    elif FD_type == 'central':
        #
        # for i in range(state.size):
        for i in range(10):
            X_1   = state.copy()
            X_2   = state.copy()
            X_1[i] = state[i] - eps
            X_2[i] = state[i] + eps
            F_1 = func( X_1 , *func_params )
            F_2 = func( X_2 , *func_params )

            Grad_FD[i] = (F_2-F_1)/(2*eps)
            Rel_ERR[i] = (gradient[i]-Grad_FD[i])/Grad_FD[i]

            if screen_output:
                sp = '    '
                print(' i=[%4d]:%sGradient=%12.9e;%s|%sFD-Grad=%12.9e;%s|%sRel-Err =%12.9e' % (i,sp, gradient[i], sp, sp, Grad_FD[i], Rel_ERR[i], sp, sp))

        # FUNC_Params:: model_Object, forecast_state, observation, HMC_Options

    elif FD_type in ['complex','complex_step','complex-step']:
        # This requires us to make sure model and integration scheme support complex numbers...
        raise NotImplementedError("Complex-step finite difference approximation of derivatives is not currently supported!")

    else:
        raise ValueError("Finite difference Approximation strategy ["+FD_type+"] is unrecognized or not supported!")
    #

    return Grad_FD, Rel_ERR

