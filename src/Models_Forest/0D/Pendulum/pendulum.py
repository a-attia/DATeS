#!/opt/local/bin/python

raise NotImplementedError

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


if False:
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
    import os
    plt.close('all')
    os.system('clear')
    np.random.seed(2345)
    #
    divider = "------------------------------------------------------------"
    #

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1- Clean-up temporary model directory,
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    _utility.clean_model_dir('temp_model')
    #
    #~~~~~~~~~~~~~~~~~~~~~          ~~~~~~~~          ~~~~~~~~~~~~~~~~~~~~~
    # prepare model configurationf file to unify procedure with HyPar models
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Model_Configs = {'ndims': 1 , 'nvars': 2 , 'size': np.array([1]) , 'n_iter':100 , 'model': 'Lorenz-3' }
    _utility.create_original_model_configurations(Model_Configs)
    #~~~~~~~~~~~~~~~~~~~~~          ~~~~~~~~          ~~~~~~~~~~~~~~~~~~~~~
    # 1- Read filter and model configs from the da_solver.ini
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    daFilterConfigs = _utility.read_filter_configurations()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2- Create an instance for the DensitySineWave model and build
    #    the experiment.
    #    Then, prepare the model: overwrite the configuration file,
    #    setup the grids, and create necessary matrices.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Pendulum_Model = ModelPendulum()
    _utility.prepare_model_for_filtering(Pendulum_Model)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3- Create an instance for the filter and set up the parameters.
    #    Also, the uncertainty parameters are added to the model based
    #    on the filter configurations.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Filter = DAFilter(daFilterConfigs)
    Filter.set_DA_filter(Pendulum_Model)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4- Sequential data assimilation process
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Filter.DA_filtering_process(Pendulum_Model , keep_all_in_memory=True)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5- Plot selective results:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    print 'Plotting Results...\n',divider
    da_plotter = DA_Visualization()
    da_plotter.plot_RMSE_results(Filter_Object=Filter , read_from_files=True, log_scale=True  , show_fig = True )
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot results. Reference trajectory, forecast trajectory and observations.
    # this is specific to this model.
    #
    referenceTraject  =  Filter._referenceTrajectory
    noAssimTraject    =  Filter._NOassimilationTrajectory
    forecastTraject   =  Filter._forecastTrajectory
    analysisMeans     =  Filter._analysisMeans
    Observations      =  Filter._Observations
    Fl_Timespan       =  Filter._Filter_Time_Span
    Obs_Timespan      =  Filter._Observation_Time_Span


    #
    print 'Plotting Results...\n',divider
    #
    outfig = plt.figure(1)
    FS = 14
    font = {'weight' : 'bold', 'size' : FS}
    plt.rc('font', **font)
    plt.plot(Fl_Timespan, referenceTraject[0,:] , 'k' , linewidth=2 , label='Reference')
    plt.plot(Fl_Timespan, referenceTraject[1,:] , 'k' , linewidth=2)
    plt.plot(Fl_Timespan, noAssimTraject[0,:] , 'b' , linewidth=2 , label='NO Assimilation')
    plt.plot(Fl_Timespan, noAssimTraject[1,:] , 'b' , linewidth=2)
    plt.plot(Fl_Timespan, forecastTraject[0,:] , 'r--' , linewidth=2 , label='Forecast')
    plt.plot(Fl_Timespan, forecastTraject[1,:] , 'r--' , linewidth=2)
    plt.plot(Fl_Timespan, analysisMeans[0,:] , 'c--' , linewidth=2 , label='EnKF')
    plt.plot(Fl_Timespan, analysisMeans[1,:] , 'c--' , linewidth=2)

    plt.plot(Obs_Timespan, Observations[0,:] ,'g^' , label='Observations')
    plt.plot(Obs_Timespan, Observations[1,:] ,'g^')

    plt.xlabel('Time',fontsize=FS , fontweight='bold' )
    plt.ylabel('$x_i$',fontsize=FS+10, fontweight='bold' )
    plt.title('Pendulum model experiment')
    plt.legend()
    plt.draw()
    
    
    plt.show()




    print '\nDone...Terminating...\n----------------------------------------------',
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _utility.clean_executables()
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

