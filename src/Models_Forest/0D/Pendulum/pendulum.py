#!/opt/local/bin/python


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
    A module providing implementations of two-variables pendulum
"""

from models_base import ModelsBase


import numpy as np
import scipy.io as sio
import os
import re

import dates_utility as utility
from explicit_runge_kutta import ExplicitRungeKutta as ExplicitTimeIntegrator
from linear_implicit_runge_kutta import LIRK as ImplicitTimeIntegrator
from scipy.integrate import ode as scipy_ode


# adjoint integrator
from fatode_erk_adjoint import FatODE_ERK_ADJ as AdjointIntegrator  # This should be made more flexible...
from fatode_erk_adjoint import initialize_adjoint_configs

from state_vector_numpy import StateVectorNumpy as StateVector
from state_matrix_numpy import StateMatrixNumpy as StateMatrix
from state_matrix_sp_scipy import StateMatrixSpSciPy as SparseStateMatrix
from observation_vector_numpy import ObservationVectorNumpy as ObservationVector
from observation_matrix_numpy import ObservationMatrixNumpy as ObservationMatrix
from observation_matrix_sp_scipy import ObservationMatrixSpSciPy as SparseObservationMatrix
from error_models_numpy import BackgroundErrorModelNumpy as BackgroundErrorModel
from error_models_numpy import ObservationErrorModelNumpy as ObservationErrorModel



class Pendulum(ModelsBase):
    """
    Class implementing a very simple 2-variables model: the pendulum model.
    """

    def __init__(self , x_init = np.array([1.5 , -1])):
        #
        self._name       = "Pendulum model"
        self._setup      = False
        self._g          = 9.81

        self._dt         = 0.001
        self._nvars      = 2

        self._Ref_IC = x_init


    #-----------------------------------------------------------------------------------------------------------------------#
    # Set the grids' parameters:
    #       - both spatial and temporal grids are set here.
    #       - the temporal grid is a must, however the spatial will be a single-entry array
    #         pointing out that the model is zero dimensional, e.g. Pendulum, Lorenz, etc.,
    #-----------------------------------------------------------------------------------------------------------------------#
    def set_grids(self):
        #
        # set spatial grid: (Only To unify work with HyPar design)
        #------------------------
        self._grids = np.array([0])
        self._Grids_dict = {'0': self._grids}
        self._spacings = np.array([1]) # used here to use index for decorrelation array




    #
    #-----------------------------------------------------------------------------------------------------------------------#




    #-----------------------------------------------------------------------------------------------------------------------#
    # Model forward function; Here the right-hand side function.
    #-----------------------------------------------------------------------------------------------------------------------#
    def step_forward_function(self, x , t ):
        """
        Function describing the dynamical model: $\dot{x}=f(x)$
               + param x: the state of the model.
               +  type x: numpy array (state vector)
             > return fx: function of the state.
        """
        fx = np.zeros(shape=x.shape)
        fx[0] = x[1]
        fx[1] = -self._g * np.sin(x[0])

        return fx
    #
    #-----------------------------------------------------------------------------------------------------------------------#




    #-----------------------------------------------------------------------------------------------------------------------#
    # Jacobian of the model (Jacobian of right-hand side). Not needed now!
    #-----------------------------------------------------------------------------------------------------------------------#
    def step_dforward_dx_function(self,x ,t ):
        """
        Jacobian of the model. Not needed now!
        """
        pass

    #
    #-----------------------------------------------------------------------------------------------------------------------#




    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    def step_forward(self, state , time_bounds ):
        """
        Steps the model forward

        :param state: the state of the model.
        :type state: list.
        :param params: parameters of the model.
        :type params: list.
        :return: list of states from the model advanced by one step.
        """
        # calculate the number of iterations based on dt in the configurations of the model
        dt = self._dt
        tspan_length = time_bounds[1] - time_bounds[0]
        n_iter = int(np.ceil(tspan_length / dt))

        #
        #
        if not(self._add_Model_Errors) or                          \
               np.isinf(self._modelError_Steps_per_Model_Steps) or \
               self._modelError_Steps_per_Model_Steps > n_iter     \
               :
            #
            # propagate directly without model errors
            Tspan = _utility.linspace(time_bounds[0], time_bounds[1], dt)
            #print Tspan


            odeSolver = DESolver(self.step_forward_function)
            for iter_stp in range(n_iter-1):
                # create a DESolver object and passing the step_forward_funciton it's constructor
                odeSolver.set_initial_condition(state)
                y , t     = odeSolver.solve(time_points=Tspan[iter_stp:iter_stp+2] , solver='RK2a')
                #y , t     = odeSolver.solve(time_points=np.asarray(time_bounds) , solver='RK2a')
                state = y[1,:]

            out_state = state


        else:
            #step forward with intermediate model errors added.
            #print 'n_iter ', n_iter
            #print 'self._modelError_Steps_per_Model_Steps', self._modelError_Steps_per_Model_Steps

            add_Q_iters = n_iter/self._modelError_Steps_per_Model_Steps
            #print 'add_Q_iters ',add_Q_iters
            more_iters  = n_iter%self._modelError_Steps_per_Model_Steps
            #print 'more_iters ',more_iters

            TSpan_W_ME = [itr*dt*self._modelError_Steps_per_Model_Steps for itr in xrange(add_Q_iters)]
            #print 'TSpan_W_ME',TSpan_W_ME

            for iter_ME in range(len(TSpan_W_ME)-1):
                #
                #print TSpan_W_ME[iter_ME]
                #print TSpan_W_ME[iter_ME+1]
                #print dt
                Tspan = _utility.linspace(TSpan_W_ME[iter_ME], TSpan_W_ME[iter_ME+1], dt , endpoint=True )
                #print Tspan
                odeSolver = DESolver(self.step_forward_function)
                for iter_stp in range(self._modelError_Steps_per_Model_Steps-1):
                    # create a DESolver object and passing the step_forward_funciton it's constructor
                    odeSolver.set_initial_condition(state)
                    #print '>>', Tspan[iter_stp:iter_stp+2]
                    y , t     = odeSolver.solve(time_points=Tspan[iter_stp:iter_stp+2] , solver='RK2a')
                    state = y[1,:]

                    # add model errors
                    model_errors_Vec = self.generate_modelNoise_Vector()
                    state = state + model_errors_Vec

                # add model errors:
                model_errors_Vec = self.generate_modelNoise_Vector()
                state = state + model_errors_Vec

            out_state = state
            #print 'out_state' , out_state



            if more_iters>0:
                # more iterations without model errors
                state = out_state
                tspan_length = time_bounds[-1] - TSpan_W_ME[-1]
                n_iter = int(np.ceil(tspan_length / dt))
                Tspan = _utility.linspace(TSpan_W_ME[-1], time_bounds[-1], dt)
                odeSolver = DESolver(self.step_forward_function)

                for iter_stp in range(n_iter-1):
                    # create a DESolver object and passing the step_forward_funciton it's constructor
                    odeSolver.set_initial_condition(state)
                    y , t     = odeSolver.solve(time_points=Tspan[iter_stp:iter_stp+2] , solver='RK2a')
                    #y , t     = odeSolver.solve(time_points=np.asarray(time_bounds) , solver='RK2a')
                    state = y[1,:]

                out_state = state


        x_out = out_state

        return x_out
    #
    #-----------------------------------------------------------------------------------------------------------------------#




#
#
#================================================================================================================================#





#
#================================================================================================================================#


#
#
#
#<><><><><><><><><><><><><><><><><><><><><><><(Driver 2 Test/Load the model in hand)><><><><><><><><><><><><><><><><><><><><><><>#
#
if __name__ == "__main__":
    #

    Pendulum_Model = Pendulum()
