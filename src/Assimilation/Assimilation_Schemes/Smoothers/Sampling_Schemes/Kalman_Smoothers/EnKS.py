
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
    EnKS:
    ------
    A class implementing the Ensemble Kalman Smoother (EnKS)

    Given list of observation vectors, and forecast ensemble,
    the analysis step updates the 'analysis ensemble'.
"""


import numpy as np
from scipy import linalg
from scipy import sparse
import os

try:
    import cPickle as pickle
except:
    import pickle

from scipy.sparse.linalg import LinearOperator
import scipy.optimize as optimize  # may be replaced with dolfin_adjoint or PyIPOpt later!

import dates_utility as utility
from smoothers_base import SmootherBase
from state_vector_base import StateVectorBase as StateVector
from state_matrix_base import StateMatrixBase as StateMatrix
from observation_vector_base import ObservationVectorBase as ObservationVector


class EnKS(SmootherBase):
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
                * analysis_ensemnble (default None): a list of model state objects representing the posterior
                    distribution of the model state at the initial time of the assimilation window
                * forecast_time (default None): time at which forecast step of the smoother is carried out
                * forecast_ensemble (default None): a list of model state objects representing the prior
                    distribution of the model state at the initial time of the assimilation window
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
                * file_output_file_name_prefix (default 'EnKS_results'): name/prefix of output file
                * file_output_file_format (default 'mat'): file ouput format.
                    Supported formats:
                        - 'mat': matlab matrix files,
                        - 'pickle': python pickled objects,
                        - 'txt' or 'ascii': text files

                * file_output_variables (default ['smoother_statistics']): a list of variables to ouput.

    """
    _smoother_name = "EnKS"
    _local_def_EnKS_smoother_configs = dict(smoother_name=_smoother_name,
                                            forecast_ensemble=None,
                                            analysis_ensemble=None,
                                            inflation_factor=1.05,
                                             )
    _local_def_4DVAR_output_configs = dict(scr_output=False,
                                           file_output=True,
                                           file_output_separate_files=True,
                                           file_output_file_name_prefix='EnKS_results',
                                           file_output_file_format='mat',
                                           smoother_statistics_dir='Smoother_Statistics',
                                           model_states_dir='Model_States_Repository',
                                           observations_dir='Observations_Rpository',
                                           save_update_kernels=False,
                                           update_kernels_dir='EnKF_Update_Kernels'
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

        self.smoother_configs = utility.aggregate_configurations(smoother_configs, EnKS._local_def_4DVAR_smoother_configs)
        self.output_configs = utility.aggregate_configurations(output_configs, EnKS._local_def_4DVAR_output_configs)

        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(smoother_configs=smoother_configs, output_configs=self.output_configs)
        else:
            # old-type class
            super(EnKS, self).__init__(smoother_configs=smoother_configs, output_configs=self.output_configs)
        #
        self.model = self.smoother_configs['model']
        #
        try:
            self._model_step_size = self.model._default_step_size
        except:
            self._model_step_size = EnKS.__time_eps
        self._time_eps = EnKS.__time_eps
        # print("self.__time_eps", self._time_eps)
        # print("self._model_step_size", self._model_step_size)

        # TODO: Get prior info
        pass
        # raise NotImplementedError

        # Prepare output directories:
        output_configs = self.output_configs
        file_output = output_configs['file_output']
        if file_output:
            output_dir = output_configs['file_output_dir']
            output_dirs = self._prepare_output_paths(output_dir=output_dir, cleanup_out_dir=True)
            self.output_configs.update({'file_output_dir':output_dirs[0],
                                        'smoother_statistics_dir': output_dirs[1],
                                        'model_states_dir':output_dirs[2],
                                        'observations_dir':output_dirs[3]},
                                        'EnKF_Update_Kernels'output_dirs[4])  # in case None was give

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
            super(EnKS, self).smoothing_cycle(update_reference=update_reference)
        #
        # Add further functionality if you wish...
        #

    #
    def forecast(self, checkpoint_states=False, checkpoint_initial_state=True):
        """
        Forecast step of the smoother.
        Use the model to propagate the analysis state over the assimilation timespan to generate the analysis
        trajectory that is fed to the next assimilation cycle if needed.
        
        This updates/creates an entry in the smoother_configs dictionary named 'analysis_ensemble_trajectories'
        args:
            checkpoint_states: if True, the result is an ensmble of model trajectories, rather than ensemble of states
            checkpoint_initial_state: save the analysis state at the initial time of the window to the analysis trajectory

        """
        model = self.model
        #
        timespan = self.smoother_configs['analysis_timespan']
        analysis_timespan = np.asarray(timespan)
        wb = self.smoother_configs['window_bounds']
        if (wb[-1]-analysis_timespan[-1]) > self._time_eps:
            np.append(analysis_timespan, wb[-1])

        analysis_ensemble = self.smoother_configs['analysis_ensemble']
        if analysis_ensmble is None:
            print("Failed to retrieve the analysis ensemble; found None!")
            print("The forecast step is carried out after the analysis ensemble is generated at the beginning of the assimilation window!")
            raise ValueError
        for initial_state in analysis_ensemble:
            pass
            initial_time = self.smoother_configs['analysis_time']
            if abs(initial_time - analysis_timespan[0]) > self._time_eps:
                print("The initial time has to be at the initial time of the assimilation window here!")
                raise ValueError
            #
            elif (analysis_timespan[0] - initial_time)> self._time_eps:
                # propagate the state to the beginning of the assimilation window
                local_ckeckpoints = [initial_time, analysis_timespan[0]]
                tmp_trajectory = model.integrate_state(initial_state, local_ckeckpoints)
                if isinstance(tmp_trajectory, list):
                    initial_state = tmp_trajectory[-1].copy()
                else:
                    initial_state = tmp_trajectory.copy()
            else:
                # We are good to go; the initial time matches the beginning of the assimilation timespan
                pass

            # Now, propagate each of the states in the analysis ensemble:
            if checkpoint_states:
                analysis_trajectory = model.integrate_state(initial_state, analysis_timespan)
            else:
                analysis_trajectory = model.integrate_state(initial_state, [analysis_timespan[0], analysis_timespan[-1]])
            if not checkpoint_initial_state:
                analysis_trajectory = analysis_trajectory[1: ]
            analysis_trajectories.append(analysis_trajectory)
        #
        self.smoother_configs.update({'analysis_ensemble_trajectories': analysis_trajectories})
        #

    #
    def analysis(self):
        """
        Analysis step of the (Vanilla Ensemble Kalman Smoother (EnKS).
        In this case, the given forecast ensemble is propagated to the observation time instances to create model
        observations H(xk) that will be used in the assimilation process...

        """
        model = self.model

        # Timsespan info:
        forecast_time = self.smoother_configs['forecast_time']
        analysis_time = self.smoother_configs['analysis_time']
        if self.smoother_configs['obs_checkpoints'] is None:
            print("Couldn't find observation checkpoints in self.smoother_configs['obs_checkpoints']; None found!")
            raise ValueError
        else:
            obs_checkpoints = np.asarray(self.smoother_configs['obs_checkpoints'])

        if (analysis_time - obs_checkpoints[0] ) >= self._model_step_size:
            print("Observations MUST follow the assimilation times in this implementation!")
            raise ValueError

        if (analysis_time - obs_checkpoints[0] ) >= self._model_step_size:
            print("Observations MUST follow the assimilation times in this implementation!")
            raise ValueError

        # Get observations list:
        observations_list = self.smoother_configs['observations_list']
        #
        assim_flags = [True] * len(obs_checkpoints)
        if (forecast_time - obs_checkpoints[0]) >= self._model_step_size:
            print("forecast time can't be after the first observation time instance!")
            print("forecast_time", forecast_time)
            print("obs_checkpoints[0]", obs_checkpoints[0])
            raise AssertionError
            #
        elif abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
            # an observation exists at the analysis time
            pass
            #
        else:
            obs_checkpoints = np.insert(obs_checkpoints, 0, forecast_time)
            assim_flags.insert(0, False)

        # state and observation vector dimensions
        forecast_ensemble = self.smoother_configs['forecast_ensemble']
        if forecast_ensemble is None:
            print("Can't carry out the analysis step without a forecast ensemble; Found None!")
            raise ValueError

        if abs(forecast_time-analysis_time) > self._time_eps:
            print("At this point, the analysis and forecast times have to be the SAME!")
            print("Forecast Time=%f, Analysis time=%f" % (forecast_time, analysis_time))
            raise ValueError
        #
        elif (analysis_time-obs_checkpoints[0]) > self._time_eps:
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
                    print("You are advised to use EnKF!")  # Todo, redirect automatically to 3D-Var
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
        # START the Smoothing  process:
        # > --------------------------------------------||
        # Forward propagation:
        # 1- Apply EnKF at each observation timepoint,
        # 2- Save Kalman gain matrix
        #
        cwd = os.getcwd()
        saved_kernels = []
        
        # Copy the initial forecast ensemble,
        moving_ensemble = [state.copy() for state in forecast_ensemble]
        ensemble_size = len(moving_ensemble)
        num_obs_points = len(observations_list)
        if self._verbose:
            print("Started the Analysis step of EnKS; with [num_obs_points] observation/assimilation time points;")
        inflation_factor = self.smoother_configs['inflation_factor']
        #
        # Forward to observation time instances and update observation term
        obs_ind = 0
        for iter_ind, t0, t1 in zip(xrange(num_obs_points), obs_checkpoints[: -1], obs_checkpoints[1: ]):
            local_ckeckpoints = np.array([t0, t1])

            # For initial subwindow only, check the flag on both and final assimilation flags
            if iter_ind == 0:
                assim_flg = assim_flags[iter_ind]
                if assim_flg:
                    observation = observations_list[obs_ind]
                    obs_ind += 1
                    print("TODO: Applying EnKF step at the initial time of the whole assimilation window")
                    moving_ensemble, X5 = self.EnKF_update(moving_ensemble, observation, inflation_factor, in_place=True)
                    saved_kernels = self._save_EnKF_update_kernel(X5, saved_kernels)
                # Copy initial ensemble; could be saved to file instead
                analysis_ensemble_np = utility.ensemble_to_np_array(moving_ensemble)

            # Now, let's do forecast/analysis step for each subwindow:
            for ens_ind in ensemble_size:
                state = moving_ensemble[ens_ind]
                tmp_trajectory = model.integrate_state(initial_state=local_state, checkpoints=local_ckeckpoints)
                if isinstance(tmp_trajectory, list):
                    moving_ensemble[ens_ind] = tmp_trajectory[-1].copy()
                else:
                    moving_ensemble[ens_ind] = tmp_trajectory.copy()
            #
            assim_flg = assim_flags[iter_ind+1]
            if assim_flag:
                observation = observations_list[obs_ind]
                obs_ind += 1
                moving_ensemble, X5 = self.EnKF_update(moving_ensemble, observation, inflation_factor, in_place=True)
            saved_kernels = self._save_EnKF_update_kernel(X5, saved_kernels)

        #
        # Backward propagation:
        for iter_ind in xrange(num_obs_points-2, -1, -1):
            assim_flag = assim_flags[iter_ind]
            try:
                next_assim_ind = iter_ind + 1 + assim_flags[iter_ind+1: ].index(True)
            except ValueError:
                # This is the latest assimilation cycle; proceed
                continue

            if assim_flag:
                kernel_file = saved_kernels[iter_ind]
                X5 = np.load(kernel_file)

                next_kernel = saved_kernels[next_assim_ind]
                X6 = np.load(next_kernel)

                X5 = X5.dot(X6)
                del X6

                # save/overwrite the updated kernel; no need to keep both versions!
                np.save(kernel_file, X5)

        # Update analysis ensemble at the initial time of the window:
        X5 = np.load(saved_kernels[0])
        analysis_ensemble_np = analysis_ensemble_np.dot(X5)
        del X5

        analysis_enesmble = moving_ensemble
        for ens_ind in xrange(ensemble_size):
            analysis_ensemble[ens_ind][:] = analysis_ensemble[:, ens_ind].copy()  # Need to copy?!

        self.smoother_config.update({'analysis_ensemble': analysis_ensemble})
        #
        # > --------------------------------------------||
        # END the Smoothing process:
        # > --------------------------------------------||
        #


    def EnKF_update(self, forecast_ensemble, observation, inflation_factor=1.0, in_place=False):
        """
        Calculate the kalman updated matrix, i.e. the update matrix of ensemble anomalies
            and calculate the (moving) analysis ensemble
        Args:
            state:

        Returns:
            analysis_ensemble: a list of model.state_vector objects
            X5: the state anomalies update matrix

        """
        model = self.smoother_configs['model']
        ensemble_size = len(forecast_ensemble)
        forecast_mean = utility.ensemble_mean(forecast_ensemble)
        forecast_anomalies = []

        if inflation_factor is None:
            inflation_factor = 1.0
        #
        for state in forecast_ensemble:
            if inflation_factor != 1:
                forecast_anomalies.append(state.axpy(-1, forecast_mean, in_place=False).scale(inflation_factor))
            else:
                forecast_anomalies.append(state.axpy(-1, forecast_mean, in_place=False))
        A_prime = utility.ensemble_to_np_array(forecast_anomalies)

        observation_perturbations = [model.evaluate_theoretical_observation(state) for state in forecast_anomalies]
        S = utility.ensemble_to_np_array(observation_perturbations)

        C = S.T.dot(S)
        try:
            C += (ensemble_size - 1) * model.observation_error_model.R
        except:
            C += (ensemble_size - 1) * model.observation_error_model.R.get_numpy_array()

        C_eigs, Z = linalg.eig(C)
        C_eigs = 1.0 / C_eigs

        obs_innov = observation.axpy(-1, model.evaluate_theoretical_observation(forecast_mean))
        y = Z.T.dot(obs_innov.get_numpy_array())
        y *= C_eigs
        y = Z.dot(y)
        y = S.T.dot(y)
        
        # Update ensemble mean
        analysis_mean = model.state_vector(A_prime.dot(y))
        analysis_mean = analysis_mean.add(forecast_mean)
        xa = analysis_mean.get_numpy_array()

        inv_sqrt_Ceig = sparse.spdiags(np.sqrt(C_eigs), 0, C_eigs.size, C_eigs.size)
        X2 = inv_sqrt_Ceig.dot(Z.T.dot(S))
        _, Sig2, V2 = linalg.svd(X2)

        # Calculate the square root of the matrix I-Sig2'*Sig2*theta
        Sqrtmat = sparse.spdiags(np.sqrt(1.0 - Sig2 * Sig2), 0, Sig2.size, Sig2.size)
        Aa_prime = A_prime.dot(V2.dot(Sqrtmat.dot(V2.T)))
        if self.verbose:
            # For debugging
            print(A_prime, type(A_prime))
        
        # Update analysis ensemble:
        analysis_ensemble = [model.state_vector(xa+A_prime[:, j]) for j in xrange(np.size(A_prime, 1)) ]
        
        # Build the update kernel X5
        xx = V2.dot(Sqrtmat.dot(V2.T))
        for j in xrange(np.size(xx, 1)):
            xx[:, j] += y
        IN = np.ones((ensemble_size, ensemble_size), dtype=np.float) / ensemble_size
        X5  = IN + inflation_factor * (sparse.spdiags((np.ones(ensemble_size), 0, ensemble_size, ensemble_size)) - 1.0)

        return analysis_ensemble, X5


    def _save_EnKF_update_kernel(self, X5, saved_kernels=[]):
        """
        """
        assert isinstance(X5, np.ndarray), "The upadate kernel has to be an np array; for now!"

        save_update_kernels = self.output_configs['save_update_kernels']
        if save_update_kernels:
            pass
        else:
            pass
        kernel_file_prefix = 'EnKF_kernel'
        update_kernels_dir= os.path.join(file_output_directory, output_configs['EnKF_Update_Kernels'])
        kernel_file_name = utility.try_file_name(update_kernels_dir, file_prefix=kernel_file_prefix, ignore_base_name=True)
        kernel_file_path = os.path.join(update_kernel_dirs, kernel_file_name)
        
        if isinstance(X5, np.ndarray):
            np.save(kernel_file_path, X5)
        else:
            print("This file type is not yet suuported. Expected %s, received %s!" % ("np.ndarray", type(X5)))
            raise TypeError

        saved_kernels.append(kernel_file_path)
        return saved_kernels

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
            super(EnKS, self).print_cycle_results()
        pass  # Add more...
        #

    #
    def _prepare_output_paths(self, output_dir=None, cleanup_out_dir=False):
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
        update_kernels_dir= os.path.join(file_output_directory, output_configs['EnKF_Update_Kernels'])
        file_output_variables = output_configs['file_output_variables']  # I think it's better to remove it from the smoother base...

        if not os.path.isdir(smoother_statistics_dir):
            os.makedirs(smoother_statistics_dir)
        if not os.path.isdir(model_states_dir):
            os.makedirs(model_states_dir)
        if not os.path.isdir(observations_dir):
            os.makedirs(observations_dir)
        if not os.path.isdir(update_kernels_dir):
            os.makedirs(update_kernel_dir)
        #
        return file_output_directory, smoother_statistics_dir, model_states_dir, observations_dir, update_kernel_dir

    #
    def save_cycle_results(self, output_dir=None, save_err_covars=False):
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

        file_output_directory, smoother_statistics_dir, model_states_dir, observations_dir = self._prepare_output_paths(output_dir=output_dir, cleanup_out_dir=False)

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
            output_dir: directory where EnKS results are saved.
                We assume the structure is the same as the structure created by the EnKS implemented here
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
