
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
    FDVAR:
    ------
    A class implementing the standard Four Dimansional Variational DA scheme.
    This is a Vanilla implementation, and is intended for illustration purposes.
    Will be tuned as we go!

    Given list of observation vectors, forecast state, and forecast error covariance matrix
    (or a representative forecast ensemble), the analysis step updates the 'analysis state'.
"""


import numpy as np
import os

try:
    import cPickle as pickle
except:
    import pickle

from scipy.sparse.linalg import LinearOperator
import scipy.optimize as optimize  # may be replaced with dolfin_adjoint or PyIPOpt later!

import dates_utility as utility
from smoothers_base import SmoothersBase
from state_vector_base import StateVectorBase as StateVector
from state_matrix_base import StateMatrixBase as StateMatrix
from observation_vector_base import ObservationVectorBase as ObservationVector


class FDVAR(SmoothersBase):
    """
    A class implementing the vanilla 4D-Var DA scheme.

    Given list of observation vectors, forecast state, and forecast error covariance matrix (or a representative forecast ensemble),
    the analysis step updates the 'analysis state';

    Args:
        smoother_configs:  dict, a dictionary containing smoother configurations.
            Supported configuarations:
                * model (default None):  model object
                * smoother_name (default None): string containing name of the smoother; used for output.
                * linear_system_solver (default 'lu'): String containing the name of the system solver
                    used to solver for the inverse of internal matrices. e.g. $(HBH^T+R)^{-1}(y-Hx)$
                * smoother_name (default None): string containing name of the smoother; used for output.
                * analysis_time (default None): time at which analysis step of the smoother is carried out
                * analysis_state (default None): model.state_vector object containing the analysis state.
                    This is where the smoother output (analysis state) will be saved and returned.
                * analysis_error_covariance (default None): analysis error covariance matrix obtained
                    by the smoother.
                * forecast_time (default None): time at which forecast step of the smoother is carried out
                * forecast_state (default None): model.state_vector object containing the forecast state.
                * forecast_error_covariance (default None): forecast error covariance matrix obtained
                    by the smoother.
                * B_operator: a linear operator that stands for the background error covariance matrix.
                              If it is None, it is constructed from the background error covariance matrix.
                * observation_time (default None): time instance at which observation is taken/collected
                * observation (default None): model.observation_vector object
                * reference_time (default None): time instance at which the reference state is provided
                * reference_state(default None): model.state_vector object containing the reference/true state
                * forecast_first (default True): A bool flag; Analysis then Forecast or Forecast then Analysis
                * timespan (default None): Cycle timespan.
                                           This interval includes observation, forecast, & analysis times
                * apply_preprocessing (default False): call the pre-processing function before smoothing
                * apply_postprocessing (default False): call the post-processing function after smoothing
                * screen_output (default False): Output results to screen on/off switch

        output_configs: dict,
            A dictionary containing screen/file output configurations.
            Supported configuarations:
                * scr_output (default False): Output results to screen on/off switch
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default 'Results'): relative path (to DATeS root directory)
                    of the directory to output results in
                * file_output_separate_files (default True): save all results to a single or multiple files
                * file_output_file_name_prefix (default 'FDVAR_results'): name/prefix of output file
                * file_output_file_format (default 'mat'): file ouput format.
                    Supported formats:
                        - 'mat': matlab matrix files,
                        - 'pickle': python pickled objects,
                        - 'txt' or 'ascii': text files

                * file_output_variables (default ['smoother_statistics']): a list of variables to ouput.

    """
    _smoother_name = "4D-VAR"
    _local_def_4DVAR_smoother_configs = dict(smoother_name=_smoother_name,
                                             optimizer='lbfgs',  # TODO: Revisit this...
                                             optimizer_configs=dict(maxiter=10000,
                                                                    maxfun=1000,
                                                                    tol=1e-12,
                                                                    reltol=1e-5,
                                                                    pgtol=1e-05,
                                                                    epsilon=1e-08,
                                                                    factr=10000000.0,
                                                                    disp=1,
                                                                    maxls=50,
                                                                    iprint=-1
                                                                    ),
                                             background_error_covariance=None,
                                             B_operator=None
                                             )
    _local_def_4DVAR_output_configs = dict(scr_output=False,
                                           file_output=True,
                                           file_output_separate_files=True,
                                           file_output_file_name_prefix='FDVAR_results',
                                           file_output_file_format='mat',
                                           smoother_statistics_dir='Smoother_Statistics',
                                           model_states_dir='Model_States_Repository',
                                           observations_dir='Observations_Rpository',
                                           )

    #
    local__time_eps = 1e-7  # this can be useful to compare time instances
    try:
        #
        __time_eps = os.getenv('DATES_TIME_EPS')
        if __time_eps is not None:
            __time_eps = eval(__time_eps)
        else:
            pass
        #
    except :
        __time_eps = None
    finally:
        if __time_eps is None:
            __time_eps = local__time_eps
        elif np.isscalar(__time_eps):
            pass
        else:
            print("\n\n** Failed to get/set the value of '__time_eps'**\n setting it to %f\n " % local__time_eps)
            __time_eps = local__time_eps
            # raise ValueError


    #
    def __init__(self, smoother_configs=None, output_configs=None):

        self.smoother_configs = utility.aggregate_configurations(smoother_configs, FDVAR._local_def_4DVAR_smoother_configs)
        self.output_configs = utility.aggregate_configurations(output_configs, FDVAR._local_def_4DVAR_output_configs)

        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(smoother_configs=smoother_configs, output_configs=self.output_configs)
        else:
            # old-type class
            super(FDVAR, self).__init__(smoother_configs=smoother_configs, output_configs=self.output_configs)
        #
        self.model = self.smoother_configs['model']
        #
        try:
            self._model_step_size = self.model._default_step_size
        except:
            self._model_step_size = FDVAR.__time_eps
        self._time_eps = FDVAR.__time_eps
        # print("self.__time_eps", self._time_eps)
        # print("self._model_step_size", self._model_step_size)

        if False:  # TODO: revisit and rewrite this
            if self.smoother_configs['background_error_covariance'] is None and self.smoother_configs['B_operator'] is None:
                print("Both 'background_error_covariance', and 'B_operator' cannot be None!")
                raise AssertionError
                #
            elif B_operator is None:
                def B_mult(x, smoother):
                    model = smoother.smoother_configs['model']
                    B = smoother.smoother_configs['background_error_covariance']
                    state_vector = model.state_vector()
                    state_vector[:] = B.vector_product(x)
                    return state_vector
                self._matvec = lambda x: B_mult(x, self)
                # create B_operator from the background error covariance matrix
                state_size = self.smoother_configs['model'].state_size()
                self.B_operator = LinearOperator((state_size, state_size), matvec=self._matvec)
                #
            else:
                # Both are provided
                pass

        #
        self.__initialized = True
        #

    #
    def smoothing_cycle(self, update_reference=True):
        """
        Carry out a single smoothing cycle. Forecast followed by analysis or the other way around.
        All required variables are obtained from 'smoother_configs' dictionary.

        Args:
            update_reference (default True): bool,
                A flag to decide whether to update the reference state in the smoother or not.


        """
        # Call basic functionality from the parent class:
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().smoothing_cycle(update_reference=update_reference)
        else:
            # old-type class
            super(FDVAR, self).smoothing_cycle(update_reference=update_reference)
        #
        # Add further functionality if you wish...
        #

    #
    def forecast(self):
        """
        Forecast step of the smoother.
        Use the model to propagate the analysis state over the assimilation timespan to generate the analysis
        trajectory that is fed to the next assimilation cycle if needed.

        """
        timespan = self.smoother_configs['analysis_timespan']
        analysis_timespan = np.asarray(timespan)
        wb = self.smoother_configs['window_bounds']
        if (wb[-1]-analysis_timespan[-1]) > self._time_eps:
            np.append(analysis_timespan, wb[-1])

        initial_state = self.smoother_configs['analysis_state']
        initial_time = self.smoother_configs['analysis_time']
        if abs(initial_time - analysis_timespan[0]) > self._time_eps:
            print("The initial time has to be at the initial time of the assimilation window here!")
            raise ValueError
        #
        elif (analysis_timespan[0] - initial_time)> self._time_eps:
            # propagate the forecast state to the beginning of the assimilation window
            local_ckeckpoints = [initial_time, analysis_timespan[0]]
            tmp_trajectory = self.model.integrate_state(initial_state, local_ckeckpoints)
            if isinstance(tmp_trajectory, list):
                initial_state = tmp_trajectory[-1].copy()
            else:
                initial_state = tmp_trajectory.copy()
        else:
            # We are good to go; the initial time matches the beginning of the assimilation timespan
            pass
        #
        analysis_trajectory = self.model.integrate_state(initial_state, analysis_timespan)
        self.smoother_configs.update({'analysis_trajectory': analysis_trajectory})
        #

    #
    def analysis(self):
        """
        Analysis step of the (Vanilla 4DVar) smoother.
        In this case, the given forecast state is propagated to the observation time instances to create model
        observations H(xk) that will be used in the assimilation process...

        """
        # state and observation vector dimensions

        forecast_time = self.smoother_configs['forecast_time']
        analysis_time = self.smoother_configs['analysis_time']
        obs_checkpoints = self.smoother_configs['obs_checkpoints']
        observations_list = self.smoother_configs['observations_list']
        if abs(forecast_time-analysis_time) > 1e-12:
            print("At this point, the analysis and forecast times have to be the SAME!")
            print("Forecast Time=%f, Analysis time=%f" % (forecast_time, analysis_time))
            raise ValueError
        #
        elif (analysis_time-obs_checkpoints[0]) > 1e-12:
            print("At this point, the analysis time should be equal to or less that the time of the first observation")
            print("Analysis time=%f Time of the first observation=%f" %(analysis_time, obs_checkpoints[0]))
            raise ValueError
        else:
            if len(observations_list) == 0:
                print("An empty list of observations is detected! No observations to assimilate;")
                print("Nothing to assimilate...")
                print("Setting the analysis state to the forecast state...")

                self.smoother_configs['analysis_state'] = self.smoother_configs['forecast_state'].copy()
                return

                #
            elif len(observations_list) == 1:
                if abs(obs_checkpoints[0]-forecast_time)<=self._time_eps and \
                    abs(analysis_time-forecast_time)<=self._time_eps:
                    print("A single observation is detected, and the assimilation time coincides with observation time;")
                    print("You are advised to use 3D-Var!")  # Todo, redirect automatically to 3D-Var
                    raise ValueError
                elif (obs_checkpoints[0]-forecast_time)>self._time_eps or (obs_checkpoints[0]-analysis_time)>self._time_eps:
                    pass
                else:
                    print("This settings instances are not acceptable:")
                    print("Observation time instance: %f" % obs_checkpoints[0])
                    print("Forecast time: %f" % forecast_time)
                    print("Observation time instance: %f" % analysis_time)
                    print("Terminating ")
                    raise ValueError
            else:
                # Good to go!
                pass

        #
        # > --------------------------------------------||
        # START the 4D-VAR process:
        # > --------------------------------------------||
        #
        # set the optimizer, and it's configurations (e.g. tol, reltol, maxiter, etc.)
        # start the optimization process:
        optimizer = self.smoother_configs['optimizer'].lower()
        #
        if optimizer == 'lbfgs':  # TODO: Consider replacing with the general interface optimize.minimize...
            #
            # Retrieve/Set the objective function, and the gradient of the objective function
            fdvar_value = lambda x: self.objective_function_value(x)
            fdvar_gradient = lambda x: self.objective_function_gradient(x)
            #
            optimizer_configs = self.smoother_configs['optimizer_configs']
            x0 = self.smoother_configs['forecast_state'].get_numpy_array()
            #
            x, f, d = optimize.fmin_l_bfgs_b(fdvar_value,
                                             x0,
                                             fprime=fdvar_gradient,
                                             m=10,
                                             factr=float(optimizer_configs['factr']),
                                             pgtol=optimizer_configs['pgtol'],
                                             epsilon=optimizer_configs['epsilon'],
                                             iprint=optimizer_configs['iprint'],
                                             maxfun=optimizer_configs['maxfun'],
                                             maxiter=optimizer_configs['maxiter'],
                                             maxls=optimizer_configs['maxls'],
                                             disp=optimizer_configs['disp']
                                             )
            # print "4D-VAR RESULTS:", "optimal state:", x, "Minimum:", f, "flags:", d  # This is to be rewritten appropriately after debugging

            # Save the results, and calculate the results' statistics
            failed = d['warnflag']  # 0 flag --> converged
            if failed:
                print d
                self.smoother_configs['analysis_state'] = None
                print("The 4D-Var algorithm Miserably failed!")
                raise ValueError

            analysis_state_numpy = x
            analysis_state = self.model.state_vector()
            analysis_state[:] = analysis_state_numpy
            self.smoother_configs['analysis_state'] = analysis_state.copy()
            #
        else:
            print("The optimizer '%s' is not recognized or not yet supported!" % optimizer)
            raise ValueError
            #

        #
        # > --------------------------------------------||
        # END the 4D-VAR process:
        # > --------------------------------------------||
        #

    #
    #
    def fdvar_objective(self, state):
        """
        Return the value and the gradient of the objective function evaluated at the given state

        Args:
            state:

        Returns:
            value:
            gradient:

        """
        value = self.objective_function_value(state)
        gradient = self.objective_function_gradient(state)
        return value, gradient
        #

    #
    def objective_function_value(self, state):
        """
        Evaluate the value of the 4D-Var objective function

        Args:
            state:

        Returns:
            objective_value

        """
        #
        model = self.model
        #
        # TODO: check if copying is actually necessary after finishing the implementation...
        if isinstance(state, np.ndarray):
            local_state = self.model.state_vector()
            local_state[:] = state.copy()
        else:
            local_state = state.copy()

        #
        forecast_state = self.smoother_configs['forecast_state'].copy()
        forecast_time = self.smoother_configs['forecast_time']

        #
        # Evaluate the Background term:  # I am assuming forecast vs. analysis times match correctly here.
        state_dev = forecast_state.copy().scale(-1.0)  # <- state_dev = - forecast_state
        state_dev = state_dev.add(local_state, in_place=False)  # state_dev = x - forecast_state
        scaled_state_dev = state_dev.copy()
        scaled_state_dev = model.background_error_model.invB.vector_product(scaled_state_dev)
        background_term = scaled_state_dev.dot(state_dev)

        #
        # Evaluate the observation terms:
        # get observations list:
        observations_list = self.smoother_configs['observations_list']
        analysis_time = self.smoother_configs['analysis_time']
        if self.smoother_configs['obs_checkpoints'] is None:
            print("Couldn't find observation checkpoints in self.smoother_configs['obs_checkpoints']; None found!")
            raise ValueError
        else:
            obs_checkpoints = np.asarray(self.smoother_configs['obs_checkpoints'])

        #
        if self._verbose:
            print("In objective_function_value:")
            print("in-state", state)
            print("forecast_state", forecast_state)
            print("obs_checkpoints", obs_checkpoints)
            print("observations_list", observations_list)
            print("background_term", background_term)
            raw_input("\npress Enter to continue")


        if (analysis_time - obs_checkpoints[0] ) >= self._model_step_size:
            print("Observations MUST follow the assimilation times in this implementation!")
            raise ValueError
        else:
            pass

        #
        if (forecast_time - obs_checkpoints[0]) >= self._model_step_size:
            print("forecast time can't be after the first observation time instance!")
            print("forecast_time", forecast_time)
            print("obs_checkpoints[0]", obs_checkpoints[0])
            raise AssertionError
            #
        elif abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
            # an observation exists at the analysis time
            Hx = model.evaluate_theoretical_observation(local_state)
            obs_innov = observations_list[0].copy().scale(-1.0)
            obs_innov = obs_innov.add(Hx)
            # obs_innov = Hx.axpy(-1.0, observations_list[0], in_place=False)
            scaled_obs_innov = obs_innov.copy()
            scaled_obs_innov = model.observation_error_model.invR.vector_product(scaled_obs_innov)
            # scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
            observation_term = scaled_obs_innov.dot(obs_innov)

            #
        else:
            obs_checkpoints = np.insert(obs_checkpoints, 0, forecast_time)
            observation_term = 0.0

        #
        num_obs_points = len(observations_list)
        # Forward to observation time instances and update observation term
        for iter_ind, t0, t1 in zip(xrange(num_obs_points), obs_checkpoints[: -1], obs_checkpoints[1: ]):
            local_ckeckpoints = np.array([t0, t1])
            tmp_trajectory = model.integrate_state(initial_state=local_state, checkpoints=local_ckeckpoints)
            if isinstance(tmp_trajectory, list):
                local_state = tmp_trajectory[-1].copy()
            else:
                local_state = tmp_trajectory.copy()
            #
            Hx = model.evaluate_theoretical_observation(local_state)

            # obs_innov = Hx.axpy(-1.0, observations_list[iter_ind], in_place=False)
            obs_innov = observations_list[iter_ind].copy().scale(-1.0)
            obs_innov = obs_innov.add(Hx, in_place=False)
            scaled_obs_innov = obs_innov.copy()
            scaled_obs_innov = model.observation_error_model.invR.vector_product(scaled_obs_innov)

            observation_term += scaled_obs_innov.dot(obs_innov)
            #

            #
            if self._verbose:
                print("subinterval:" + repr(local_ckeckpoints))
                print("local_state (at end of subinterval above): ", local_state)
                print("H(x) (at end of subinterval above): ", Hx)
                print("observation:", observations_list[iter_ind])
                print("obs_innov", obs_innov)
                print("scaled_obs_innov", scaled_obs_innov)
                print("observation_term", observation_term)
                raw_input("\npress Enter to continue")
                #


        #
        # Add observation and background terms and return:
        objective_value = 0.5 * (background_term + observation_term)

        #
        if self._verbose:
            print("In objective_function_value:")
            print("bacground state_dev: ", state_dev)
            print("bacground scaled_state_dev: ", scaled_state_dev)
            print("background_term", background_term)
            print()
            print("observation_term", observation_term)
            print("objective_value", objective_value)
            raw_input("\npress Enter to continue")

            #
        #
        return objective_value
        #

    #
    def objective_function_gradient(self, state, FD_Validation=False, FD_eps=1e-7, FD_central=True):
        """
        Evaluate the gradient of the 4D-Var objective function

        Args:
            state:
            FD_Validation:
            FD_eps:
            FD_central:

        Returns:
            objective_gradient:

        """
        # get a pointer to the model object
        model = self.model
        #
        if isinstance(state, np.ndarray):
            local_state = model.state_vector()
            local_state[:] = state.copy()
        else:
            local_state = state.copy()

        #
        # Start Evaluating the gradient:

        #
        forecast_state = self.smoother_configs['forecast_state'].copy()
        forecast_time = self.smoother_configs['forecast_time']

        #
        # Evaluate the Background term:  # I am assuming forecast vs. analysis times match correctly here.
        # state_dev = local_state.axpy(-1.0, forecast_state, in_place=False)
        state_dev = forecast_state.copy().scale(-1.0)
        state_dev = state_dev.add(local_state)

        background_term = model.background_error_model.invB.vector_product(state_dev, in_place=False)

        #
        # Evaluate the observation terms:
        # get observations list:
        observations_list = self.smoother_configs['observations_list']
        analysis_time = self.smoother_configs['analysis_time']
        if self.smoother_configs['obs_checkpoints'] is None:
            print("Couldn't find observation checkpoints in self.smoother_configs['obs_checkpoints']; None found!")
            raise ValueError
        else:
            obs_checkpoints = np.asarray(self.smoother_configs['obs_checkpoints'])

        #
        if (analysis_time - obs_checkpoints[0] ) >= self._model_step_size:
            print("Observations have to follow the assimilation times in this implementation!")
            print("analysis_time", analysis_time)
            print("obs_checkpoints[0]", obs_checkpoints[0])
            raise ValueError
        else:
            pass

        #
        if self._verbose:
            print("In objective_function_gradient:")
            print("in-state", state)
            print("forecast_state", forecast_state)
            print("obs_checkpoints", obs_checkpoints)
            print("observations_list", observations_list)
            print("background_term", background_term)
            raw_input("\npress Enter to continue")

        #
        # 1- forward checkpointing:
        checkpointed_state = local_state.copy()
        checkpointed_states = []  # this holds states only at observation time instances
        checkpointed_times = []

        if abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
            checkpointed_states.append(local_state)
            checkpointed_times.append(forecast_time)
            #
        elif (obs_checkpoints[0] - forecast_time) >= self._model_step_size:
            local_ckeckpoints = [forecast_time, obs_checkpoints[0]]

            tmp_trajectory = model.integrate_state(initial_state=checkpointed_state.copy(), checkpoints=local_ckeckpoints)
            #
            if isinstance(tmp_trajectory, list):
                checkpointed_state = tmp_trajectory[-1].copy()
            else:
                checkpointed_state = tmp_trajectory.copy()
            checkpointed_states.append(checkpointed_state)
            checkpointed_times.append(obs_checkpoints[0])
        else:
            print("forecast time can't be after the first observation time!")
            print("obs_checkpoints[0]", obs_checkpoints[0])
            print("forecast_time", forecast_time)
            raise ValueError

        #
        for t0, t1 in zip(obs_checkpoints[:-1], obs_checkpoints[1:]):
            local_checkpoints = np.array([t0, t1])
            local_trajectory = model.integrate_state(initial_state=checkpointed_state, checkpoints=local_checkpoints)
            if isinstance(local_trajectory, list):
                checkpointed_state = local_trajectory[-1].copy()
            else:
                checkpointed_state = local_trajectory.copy()
            checkpointed_states.append(checkpointed_state.copy())
            checkpointed_times.append(t1)
            #

        if self._verbose:
            print("checkpointed_times", checkpointed_times)
            print("checkpointed_states", checkpointed_states)

        #
        # 2- backward propagation, and sensitivity update:
        last_obs_ind = len(observations_list) - 1
        #
        if len(checkpointed_states) != len(observations_list):
            print("Checkpointed states don't match observation time indexes!")
            raise ValueError


        # Initialize the sensitivity matrix:
        Hx = model.evaluate_theoretical_observation(checkpointed_states[-1])
        obs = observations_list[-1]
        obs_innov = Hx.axpy(-1.0, obs, in_place=False)  # innov = H(x) - y
        scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
        lambda_ = model.observation_operator_Jacobian_T_prod_vec(checkpointed_states[-1], scaled_obs_innov)

        #
        if self._verbose:
            print("\nIn objective_function_gradient:")
            print("checkpointed_states[-1]", checkpointed_states[-1])
            print("final obs", obs)
            print("obs_innov", obs_innov)
            print("scaled_obs_innov", scaled_obs_innov)
            print("Initial lambda_", lambda_)
            raw_input("\npress Enter to continue")


        # backward propagation, and update sensitivity matrix (lambda)
        adjoint_integrator = model._adjoint_integrator
        for t0_ind, t1_ind in zip(xrange(last_obs_ind-1, -1, -1), xrange(last_obs_ind, 0, -1)):
            t0, t1 = obs_checkpoints[t0_ind], obs_checkpoints[t1_ind]
            #
            lambda_k = adjoint_integrator.integrate_adj(y=checkpointed_states[t0_ind],
                                                        lambda_=lambda_,
                                                        tin=t0,
                                                        tout=t1
                                                        )
            #
            if isinstance(lambda_k, np.ndarray):
                try:
                    lambda_k = model.state_vector(lambda_k)
                except():
                    tmp_lamda = model.state_vector()
                    tmp_lamda[:] = lambda_k.copy()
                    lambda_k = tmp_lamda
            elif isinstance(lambda_k, (StateVector, StateMatrix)):
                pass
            else:
                print("Returned Sensitivity matrix is of unrecognized Type: %s" % repr(type(lambda_k)) )
                raise TypeError

            Hx = model.evaluate_theoretical_observation(checkpointed_states[t0_ind])
            obs = observations_list[t0_ind]
            obs_innov = Hx.axpy(-1.0, obs, in_place=False)  # innov = H(x) - y
            scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
            obs_correction_term = model.observation_operator_Jacobian_T_prod_vec(checkpointed_states[t0_ind], scaled_obs_innov)
            #
            lambda_ = lambda_k.add(obs_correction_term, in_place=False)
            #
            #
            if self._verbose:
                print("\nIn objective_function_gradient: Iteration %s" % repr([t0_ind, t1_ind]))
                print("t0 = %f;\t t1=%f" % (t0, t1))
                print("local_state", local_state)
                print("checkpointed_states[%d]:" %t0_ind, checkpointed_states[t0_ind])
                print("observations_list[%d]:" %t0_ind, observations_list[t0_ind])
                print("obs_innov", obs_innov)
                print("obs_correction_term", obs_correction_term)
                print("lambda_k", lambda_k)
                print("lambda_", lambda_)
                raw_input("\npress Enter to continue")

        #
        if (obs_checkpoints[0] - forecast_time) >= self._model_step_size:
            lambda_ = adjoint_integrator.integrate_adj(y=local_state,
                                                       lambda_=lambda_,
                                                       tin=forecast_time,
                                                       tout=obs_checkpoints[0]
                                                       )
            if self._verbose:
                print("(obs_checkpoints[0] - forecast_time) >= self._model_step_size")
                print("Propagating back from time %f to time %f" % (obs_checkpoints[0], forecast_time))
            if isinstance(lambda_, np.ndarray):
                try:
                    observation_term = model.state_vector()
                    observation_term[:] = lambda_[:]
                except:
                    observation_term = model.state_vector()
                    observation_term[:] = lambda_.copy()
            elif isinstance(lambda_, (StateVector, StateMatrix)):
                observation_term = lambda_
            else:
                print("Returned Sensitivity matrix is of unrecognized Type: %s" % repr(type(lambda_)) )
                raise TypeError
            #
        elif abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
            if self._verbose:
                print("abs(forecast_time - obs_checkpoints[0]) <= self._time_eps")
            observation_term = lambda_
        #
        else:
            print("Forecast time point can't be after first observation time!")
            raise ValueError
        #
        if self._verbose:
            print("|||||||| FINAL LAMBDA (observation term) >>>>> ", observation_term)
            raw_input("\npress Enter to continue")
        #
        # Add observation and background terms and return:
        objective_gradient = background_term.add(observation_term, in_place=False)


        #
        if FD_Validation:
            self.__validate_gradient(state, objective_gradient, FD_eps=FD_eps, FD_central=FD_central)
        #
        if isinstance(state, np.ndarray):
            objective_gradient = objective_gradient.get_numpy_array()

        if self._verbose:
            print("^"*100+"\n"+"^"*100)
            print("\nIn objective_function_gradient: ")
            print("forecast_state", forecast_state)
            print("local_state", local_state)
            print("state_dev", state_dev)
            print("background_term: ", background_term)

            print("observation_term", observation_term)
            print("lambda_", lambda_)

            print("Gradient:", objective_gradient)
            print("v"*100+"\n"+"v"*100)

        #
        return objective_gradient
        #

    #
    #
    def __validate_gradient(self, state, gradient, FD_eps=1e-5, FD_central=False):
        """
        Use Finite Difference to validate the gradient

        Args:
            state:
            gradient:
            FD_eps:
            FD_central:

        """
        #
        # Use finite differences to validate the Gradient (if needed):
        state_size = self.model.state_size()
        # Finite Difference Validation:
        eps = FD_eps
        grad_array = gradient.get_numpy_array()
        if isinstance(state, np.ndarray):
            state_array = state.copy()
        elif isinstance(state, (StateVector, StateMatrix)):
            state_array = state.get_numpy_array()
        else:
            print("Passed state is of unrecognized Type: %s" % repr(type(state)) )
            raise TypeError

        sep = "\n"+"~"*80+"\n"
        # print some statistics for monitoring:
        print(sep + "FD Validation of the Gradient" + sep)
        print("  + Maximum gradient entry :", grad_array.max())
        print("  + Minimum gradient entry :", grad_array.min())

        #
        perturbed_state = self.model.state_vector()
        state_perturb = np.zeros_like(grad_array)
        fd_grad = np.zeros_like(grad_array)
        #
        if not FD_central:
            f0 = self.objective_function_value(state)
        #

        for i in xrange(state_size):
            state_perturb[:] = 0.0
            state_perturb[i] = eps
            #
            if FD_central:
                perturbed_state[:] = state_array - state_perturb
                f0 = self.objective_function_value(perturbed_state)
            #
            perturbed_state[:] = state_array + state_perturb
            f1 = self.objective_function_value(perturbed_state)

            if FD_central:
                fd_grad[i] = (f1-f0)/(2.0*eps)
            else:
                fd_grad[i] = (f1-f0)/(eps)

            err = (grad_array[i] - fd_grad[i]) / fd_grad[i]
            print(">>>>Gradient/FD>>>> %4d| Grad = %+8.6e\t FD-Grad = %+8.6e\t Rel-Err = %+8.6e <<<<" % (i, grad_array[i], fd_grad[i], err))


    #
    def print_cycle_results(self):
        """
        Print smoothing results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        """
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().print_cycle_results()
        else:
            # old-type class
            super(FDVAR, self).print_cycle_results()
        pass  # Add more...
        #

    #
    def save_cycle_results(self, output_dir=None, cleanup_out_dir=False, save_err_covars=False):
        """
        Save smoothing results from the current cycle to file(s).
        A check on the correspondidng options in the configurations dictionary is made to make sure
        saving is requested.

        Args:
            out_dir (default None): directory to put results in.
                The output_dir is created (with all necessary parent paths) if it is not on disc.
                The directory is relative to DATeS root directory.

            cleanup_out_dir (default None): bool,
                Takes effect if the output directory is not empty. True: remove directory contents.


        """
        #
        # The first code block that prepares the output directory can be moved to parent class later...
        # Retrieve output configurations
        output_configs = self.output_configs
        file_output = output_configs['file_output']
        if not file_output:
            raise ValueError("The output flag is turned of. The method 'save_cycle_results' is called though!")

        # We are good to go! --> Start preparing directories (if necessary) then save results...
        if output_dir is not None:
            file_output_directory = output_dir
        else:
            file_output_directory = output_configs['file_output_dir']
        # clean-up output directory; this is set to true only if the smoother is called once, otherwise smoothing_process should handle it.
        if cleanup_out_dir:
            parent_path, out_dir = os.path.split(file_output_directory)
            utility.cleanup_directory(directory_name=out_dir, parent_path=parent_path)
        # check the output sub-directories...
        smoother_statistics_dir = os.path.join(file_output_directory, output_configs['smoother_statistics_dir'])
        model_states_dir = os.path.join(file_output_directory, output_configs['model_states_dir'])
        observations_dir = os.path.join(file_output_directory, output_configs['observations_dir'])
        file_output_variables = output_configs['file_output_variables']  # I think it's better to remove it from the smoother base...

        if not os.path.isdir(smoother_statistics_dir):
            os.makedirs(smoother_statistics_dir)
        if not os.path.isdir(model_states_dir):
            os.makedirs(model_states_dir)
        if not os.path.isdir(observations_dir):
            os.makedirs(observations_dir)

        # check if results are to be saved to separate files or appended on existing files.
        # This may be overridden if not adequate for some output (such as model states), we will see!
        file_output_separate_files = output_configs['file_output_separate_files']

        # This is useful for saving smoother statistics but not model states or observations as models should handle both
        file_output_file_name_prefix = output_configs['file_output_file_name_prefix']  # this is useless!

        # Format of the ouput files
        file_output_file_format = output_configs['file_output_file_format'].lower()
        if file_output_file_format not in ['mat', 'pickle', 'txt', 'ascii']:
            print("The file format ['%s'] is not supported!" % file_output_file_format )
            raise ValueError()

        # Retrieve smoother and ouput configurations needed to be saved
        smoother_configs = self.smoother_configs  # we don't need to save all configs
        smoother_conf= dict(smoother_name=smoother_configs['smoother_name'],
                            apply_preprocessing=smoother_configs['apply_preprocessing'],
                            apply_postprocessing=smoother_configs['apply_postprocessing'],
                            analysis_time=smoother_configs['analysis_time'],
                            forecast_time=smoother_configs['forecast_time'],
                            window_bounds=smoother_configs['window_bounds'],
                            reference_time=smoother_configs['reference_time'],
                            obs_checkpoints=smoother_configs['obs_checkpoints'],
                            analysis_timespan=smoother_configs['analysis_timespan']
                            )
        io_conf = output_configs

        # Start writing cycle settings, and reuslts:
        # 1- write model configurations configurations:
        model_conf = self.model.get_model_configs()
        if file_output_file_format == 'pickle':
            pickle.dump(model_conf, open(os.path.join(file_output_directory, 'model_configs.pickle')))
        elif file_output_file_format in ['txt', 'ascii', 'mat']:  # 'mat' here has no effect.
            utility.write_dicts_to_config_file('model_configs.txt', file_output_directory,
                                                model_conf, 'Model Configs'
                                                )

        # 2- get a proper name for the folder (cycle_*) under the model_states_dir path
        suffix = 0
        while True:
            cycle_dir = 'cycle_' + str(suffix)
            cycle_states_out_dir = os.path.join( model_states_dir, cycle_dir)  # full path where states will be saved for the current cycle
            if not os.path.isdir(cycle_states_out_dir):
                cycle_observations_out_dir = os.path.join( observations_dir, cycle_dir)
                if os.path.isdir(cycle_observations_out_dir):
                    raise IOError("There is inconsistency problem. Naming mismatch in cycles folders for states and observations!")
                os.makedirs(cycle_states_out_dir)
                os.makedirs(cycle_observations_out_dir)
                break
            else:
                suffix += 1

        # 3- save reference, forecast, and analysis states; use model to write states to file(s)
        # i- save reference state
        reference_state = self.smoother_configs['reference_state'].copy()
        self.model.write_state(state=reference_state, directory=cycle_states_out_dir, file_name='reference_state')
        # ii- save forecast state
        forecast_state = self.smoother_configs['forecast_state']
        self.model.write_state(state=forecast_state, directory=cycle_states_out_dir, file_name='forecast_state')
        # iii- save analysis state
        analysis_state = self.smoother_configs['analysis_state'].copy()
        self.model.write_state(state=analysis_state, directory=cycle_states_out_dir, file_name='analysis_state')

        # 4- Save observations to files; use model to write observations to file(s)
        for observation in self.smoother_configs['observations_list']:
            file_name = utility.try_file_name(directory=cycle_observations_out_dir, file_prefix='observation')
            self.model.write_observation(observation=observation, directory=cycle_observations_out_dir, file_name=file_name, append=False)

        # 4- Save smoother configurations and statistics to file,
        # i- Output the configurations dictionaries:
        assim_cycle_configs_file_name = 'assim_cycle_configs'
        if file_output_file_format in ['txt', 'ascii', 'mat']:
            # Output smoother and model configurations; this goes under state directory
            assim_cycle_configs_file_name += '.txt'
            utility.write_dicts_to_config_file(assim_cycle_configs_file_name, cycle_states_out_dir,
                                                   [smoother_conf, io_conf], ['Smoother Configs', 'Output Configs'])

        elif file_output_file_format in ['pickle']:
            #
            # Output smoother and model configurations; this goes under state directory
            assim_cycle_configs_file_name += '.pickle'
            assim_cycle_configs = dict(smoother_configs=smoother_conf, output_configs=io_conf)
            pickle.dump(assim_cycle_configs, open(os.path.join(cycle_states_out_dir, assim_cycle_configs_file_name)))

        else:
            raise ValueError("Unsupported output format for configurations dictionaries: '%s' !" % file_output_file_format)
            #

        # ii Output the RMSE results; it's meaningless to create a new file for each cycle:
        rmse_file_name = 'rmse.txt'  # RMSE are always saved in text files
        rmse_file_path = os.path.join(smoother_statistics_dir, rmse_file_name)
        # Create a header for the file if it is newely created
        if not os.path.isfile(rmse_file_path):
            # rmse file does not exist. create file and add header.
            header = "RMSE Results: Smoother: '%s' \n %s \t %s \t %s \t %s \n" % (self._smoother_name,
                                                                                  'Forecast-Time'.rjust(20),
                                                                                  'Analysis-Time'.rjust(20),
                                                                                  'Forecast-RMSE'.rjust(20),
                                                                                  'Analysis-RMSE'.rjust(20),
                                                                                  )
            if False:
                # get the initial RMSE and add it if forecast is done first...
                initial_time = self.smoother_configs['forecast_time']
                initial_rmse = self.output_configs['smoother_statistics']['initial_rmse']
                header += " %20s \t %20.14e \t %20.14e \t %20.14e \t %20.14e \n" % ('0000000',
                                                                                    initial_time,
                                                                                    initial_time,
                                                                                    initial_rmse,
                                                                                    initial_rmse
                                                                                    )
            # dump the header to the file
            with open(rmse_file_path, mode='w') as file_handler:
                file_handler.write(header)
        else:
            # rmse file does exist. Header should be already there!
            pass

        # Now rmse results file exists --> Append rmse results to the file:
        forecast_time = self.smoother_configs['forecast_time']
        analysis_time = self.smoother_configs['analysis_time']
        #
        forecast_rmse = self.output_configs['smoother_statistics']['forecast_rmse'][0]
        analysis_rmse = self.output_configs['smoother_statistics']['analysis_rmse'][0]
        if self._verbose:
            print("forecast_time, forecast_rmse, analysis_time, analysis_rmse")
            print(forecast_time, forecast_rmse, analysis_time, analysis_rmse)
        output_line = u" {0:20.14e} \t {1:20.14e} \t {2:20.14e} \t {3:20.14e} \n".format(forecast_time,
                                                                                         analysis_time,
                                                                                         forecast_rmse,
                                                                                         analysis_rmse
                                                                                         )
        # now write the rmse results to file
        with open(rmse_file_path, mode='a') as file_handler:
            file_handler.write(output_line)
        #

        # save error covariance matrices if requested; these will go in the state output directory
        if save_err_covars:
            Pf = self.smoother_configs['forecast_error_covariance']
            Pa = self.smoother_configs['forecast_error_covariance']
            print("Saving covariance matrices is not supported yet. CDF will be considered soon!")
            raise NotImplementedError()
        else:
            pass


    #
    def read_cycle_results(self, output_dir, read_err_covars=False):
        """
        Read smoothing results from file(s).
        Check the output directory first. If the directory does not exist, raise an IO error.
        If the directory, and files exist, Start retrieving the results properly

        Args:
            output_dir: directory where FDVAR results are saved.
                We assume the structure is the same as the structure created by the FDVAR implemented here
                in 'save_cycle_results'.
            read_err_covars:

        Returns:
            reference_state:
            forecast_state:
            analysis_state:
            observation:
            forecast_err_covar:
            analysis_err_covar:

        """
        # TODO: To be written!
        raise NotImplementedError
        #
