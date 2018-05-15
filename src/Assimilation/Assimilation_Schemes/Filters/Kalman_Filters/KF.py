
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
    KalmanFilter:
    -------------
    A class implementing the Vanilla kalman filter [cite].
    This carries out two steps: analysis, then forecast (or forecast then analysis).
    
    Given observation vector, analysis state, and analysis error covariance matrix at a previous time step, 
    the forecast state and the forecast error covariance matrix are generated at the next time step (where observation vector is taken),
    then the analysis step updates the analysis state and the analysis error covariance matrix.
"""


import os
import shutil
try:
    import cPickle as pickle
except:
    import pickle
    
import dates_utility as utility
from filters_base import FiltersBase
from state_vector_base import StateVectorBase as StateVector
from state_matrix_base import StateMatrixBase as StateMatrix
from observation_vector_base import ObservationVectorBase as ObservationVector


class KalmanFilter(FiltersBase):
    """
    A class implementing the vanilla kalman filter.
    This carries out two steps: analysis, then forecast (or forecast then analysis).

    Given observation vector, analysis state, and analysis error covariance matrix at a previous time step,
    the forecast state and the forecast error covariance matrix are generated at the next time step (where
    observation vector is taken) then the analysis step updates the analysis state and the analysis error covariance matrix.
    
    Kalman filter constructor:
    
    Args:
        filter_configs:  dict,
            A dictionary containing filter configurations.
            Supported configuarations:
                * model (default None):  model object
                * filter_name (default None): string containing name of the filter; used for output.
                * linear_system_solver (default 'lu'): String containing the name of the system solver 
                    used to solver for the inverse of internal matrices. e.g. $(HBH^T+R)^{-1}(y-Hx)$
                * filter_name (default None): string containing name of the filter; used for output.
                * analysis_time (default None): time at which analysis step of the filter is carried out
                * analysis_state (default None): model.state_vector object containing the analysis state.
                    This is where the filter output (analysis state) will be saved and returned.
                * analysis_error_covariance (default None): analysis error covariance matrix obtained 
                    by the filter. 
                * forecast_time (default None): time at which forecast step of the filter is carried out
                * forecast_state (default None): model.state_vector object containing the forecast state.
                * forecast_error_covariance (default None): forecast error covariance matrix obtained 
                    by the filter. 
                * observation_time (default None): time instance at which observation is taken/collected
                * observation (default None): model.observation_vector object
                * reference_time (default None): time instance at which the reference state is provided
                * reference_state(default None): model.state_vector object containing the reference/true state
                * forecast_first (default True): A bool flag; Analysis then Forecast or Forecast then Analysis
                * timespan (default None): Cycle timespan. 
                                           This interval includes observation, forecast, & analysis times
                * apply_preprocessing (default False): call the pre-processing function before filtering
                * apply_postprocessing (default False): call the post-processing function after filtering
                * screen_output (default False): Output results to screen on/off switch
                
        output_configs: dict,
            A dictionary containing screen/file output configurations.
            Supported configuarations:
                * scr_output (default False): Output results to screen on/off switch
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default 'Assimilation_Results'): relative path (to DATeS root directory) 
                    of the directory to output results in
                * file_output_separate_files (default True): save all results to a single or multiple files
                * file_output_file_name_prefix (default 'KF_results'): name/prefix of output file
                * file_output_file_format (default 'mat'): file ouput format.
                    Supported formats:
                        - 'mat': matlab matrix files,
                        - 'pickle': python pickled objects,
                        - 'txt' or 'ascii': text files
                        
                * file_output_variables (default ['filter_statistics']): a list of variables to ouput. 
            
    Returns:
        None
    
    """
    _filter_name = "KF"
    _local_def_kf_filter_configs = dict(model=None, 
                                        filter_name=_filter_name,
                                        linear_system_solver='lu',
                                        analysis_time=None,
                                        analysis_state=None,
                                        analysis_error_covariance=None,
                                        forecast_time=None,
                                        forecast_state=None,
                                        forecast_error_covariance=None,
                                        observation_time=None,
                                        observation=None,
                                        reference_time=None,
                                        reference_state=None,
                                        forecast_first=True,
                                        timespan=None,
                                        apply_preprocessing=False,
                                        apply_postprocessing=False,
                                        screen_output=False,
                                        file_output=False,
                                        file_output_directory='Assimilation_Results',
                                        files_output_separate_files=True,
                                        output_file_name='KF_results.dat'
                                        )
    _local_def_kf_output_configs = dict(scr_output=False,
                                        file_output=True,
                                        file_output_dir='Assimilation_Results',
                                        file_output_separate_files=True,
                                        file_output_file_name_prefix='KF_results',
                                        file_output_file_format='mat',
                                        file_output_variables=['filter_statistics']
                                        )
    
    #
    def __init__(self, filter_configs=None, output_configs=None):
        
        filter_configs = utility.aggregate_configurations(filter_configs, KF._local_def_kf_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, KF._local_def_kf_output_configs)
        
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(filter_configs=filter_configs, output_configs=output_configs)
        else:
            # old-stype class
            super(KF, self).__init__(filter_configs=filter_configs, output_configs=output_configs)
        #
        self.model = self.filter_configs['model']
        #
        self.__initialized = True
        #

    #
    def filtering_cycle(self, update_reference=True):
        """
        Carry out a single filtering cycle. Forecast followed by analysis or the other way around.
        All required variables are obtained from 'filter_configs' dictionary.
        
        Args:
            update_reference (default True): bool,
                A flag to decide whether to update the reference state in the filter or not.
                  
        Returns:
            None
           
        """
        # Call basic functionality from the parent class:
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().filtering_cycle(update_reference=update_reference)
        else:
            # old-stype class
            super(EnKF, self).filtering_cycle(update_reference=update_reference)
        #
        # Add further functionality if you wish...
        #
     
    #
    def forecast(self):
        """
        Forecast step of the filter. 
        Use the model object to propagate the analysis state to the forecast time
        
        Args:
                  
        Returns:
            None
            
        """
        # generate the forecast state
        xa = self.filter_configs['analysis_state']
        timespan = self.filter_configs['timespan']
        trajectory = self.model.integrate_state(initial_state=xa, checkpoints=time_span)
        xf = trajectory[-1].copy()
        if isinstance(trajectory, list):
            xf = trajectory[-1].copy()
        elif isinstance(trajectory, StateVector):
            xf = trajectory.copy()
        else:
            print("The time integrator returned unrecognized data type. \
                   It should either return a list of state vectors, or a single state vector object.")
            raise TypeError()
        
        # generate the forecast error covariance matrix
        Pa = self.filter_configs['analysis_error_covariance']
        tangent_linear_model = self.model.step_forward_function_Jacobian(time_span[0], xa)
        Pa = Pa.matrix_product_matrix_transpose(tangent_linear_model)
        Pf = tangent_linear_model.matrix_product(Pa)
        
        # Update forecast information to filter_configs
        self.filter_configs['forecast_state'] = xf
        self.filter_configs['forecast_error_covariance'] = Pf
    
    #
    def analysis(self):
        """
        Analysis step of the (Vanilla Kalman) filter.
        
        Args:
        
        Returns:
            None
            
        """
        # state and observation vector dimensions
        state_size = self.model.state_size()
        observation_size = self.model.observation_vector_size()
        
        # get the forecast state, and the forecast error covariance matrix.
        xf = self.filter_configs['forecast_state']
        Pf = self.filter_configs['forecast_error_covariance']
        
        # get the measurements vector
        observation = self.filter_configs['observation']
        
        # Innovation or measurement residual
        theo_obs = self.model.evaluate_theoretical_observation(xf)
        innov = observation.axpy(-1.0, theo_obs, in_place=False)
        
        try:
            obs_oper_jacobian = self.model.evaluate_observation_operator_Jacobian(xf)
        except(NotImplementedError):
            obs_oper_jacobian = None
        
        #
        # construct the jacobian of the observation operator the hard way if not provided by the model
        if obs_oper_jacobian is None:
            # this is the jacobian transpose
            HT = np.empty((state_size, observation_size))
            tmp_obs = self.model.observation_vector()
            for obs_ind in xrange(observation_size):
                tmp_obs[:] = 0.0; tmp_obs[obs_ind] = 1.0
                HT[:, obs_ind] = self.model.observation_operator_Jacobian_T_prod_vec(model_forecast_state, tmp_obs).get_numpy_array()
        else:
            HT = obs_oper_jacobian.transpose()
        
        #
        # Innovation (or residual) covariance
        Pb_HT = np.dot(Pf, HT)
        H_Pf_HT = np.dot(HT.T, Pb_HT)
        S = H_Pb_HT + self.model.observation_error_model.R.get_numpy_array()
        
        # Optimal Kalman gain
        K = np.dot(HT, np.linalg.inv(S))
        K = np.dot(Pf, K)
        
        # Updated (a posteriori) state estimate
        xa = xf + np.dot(K, innov.get_numpy_array())
        
        # Updated (a posteriori) estimate covariance
        K = np.dot(K, HT.T)
        np.fill_diagonal(K, 1.0 - np.diag(K))
        Pa = np.dot(Pf, K)
        
        # Update filter configs with the analysis results
        analysis_state = self.model.state_vector()
        analysis_state[:] = xa[:]
        self.filter_configs['analysis_state'] = analysis_state
        
        analysis_err_covar = self.model.state_matrix()
        analysis_err_covar[:, :] = Pa[:, :]
        self.filter_configs['analysis_error_covariance'] = analysis_err_covar
        #
    
    #
    def print_cycle_results(self):
        """
        Print filtering results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        """
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().print_cycle_results()
        else:
            # old-stype class
            super(EnKF, self).print_cycle_results()
        pass  # Add more...
        #
    
    #
    def save_cycle_results(self, output_dir=None, cleanup_out_dir=False, save_err_covars=False):
        """
        Save filtering results from the current cycle to file(s).
        A check on the correspondidng options in the configurations dictionary is made to make sure
        saving is requested.
        
        Args:
            out_dir (default None): directory to put results in. 
                The output_dir is created (with all necessary parent paths) if it is not on disc.
                The directory is relative to DATeS root directory.
            
            cleanup_out_dir (default None): bool,
                Takes effect if the output directory is not empty. True: remove directory contents.
            
        Returns:
            None
        
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
        # clean-up output directory; this is set to true only if the filter is called once, otherwise filtering_process should handle it.
        if cleanup_out_dir:
            parent_path, out_dir = os.path.split(file_output_directory)
            utility.cleanup_directory(directory_name=out_dir, parent_path=parent_path)
        # check the output sub-directories...
        filter_statistics_dir = os.path.join(file_output_directory, output_configs['filter_statistics_dir'])
        model_states_dir = os.path.join(file_output_directory, output_configs['model_states_dir'])
        observations_dir = os.path.join(file_output_directory, output_configs['observations_dir'])
        file_output_variables = output_configs['file_output_variables']  # I think it's better to remove it from the filter base...

        if not os.path.isdir(filter_statistics_dir):
            os.makedirs(filter_statistics_dir)
        if not os.path.isdir(model_states_dir):
            os.makedirs(model_states_dir)
        if not os.path.isdir(observations_dir):
            os.makedirs(observations_dir)

        # check if results are to be saved to separate files or appended on existing files.
        # This may be overridden if not adequate for some output (such as model states), we will see!
        file_output_separate_files = output_configs['file_output_separate_files']
        
        # This is useful for saving filter statistics but not model states or observations as models should handle both
        file_output_file_name_prefix = output_configs['file_output_file_name_prefix']  # this is useless!
        
        # Format of the ouput files
        file_output_file_format = output_configs['file_output_file_format'].lower()
        if file_output_file_format not in ['mat', 'pickle', 'txt', 'ascii']:
            print("The file format ['%s'] is not supported!" % file_output_file_format )
            raise ValueError()
        
        # Retrieve filter and ouput configurations needed to be saved
        filter_configs = self.filter_configs  # we don't need to save all configs
        filter_conf= dict(filter_name=filter_configs['filter_name'],
                          prior_distribution=filter_configs['prior_distribution'],
                          gmm_prior_settings=filter_configs['gmm_prior_settings'],
                          ensemble_size=filter_configs['ensemble_size'],
                          apply_preprocessing=filter_configs['apply_preprocessing'],
                          apply_postprocessing=filter_configs['apply_postprocessing'],
                          timespan=filter_configs['timespan'],
                          analysis_time=filter_configs['analysis_time'],
                          observation_time=filter_configs['observation_time'],
                          forecast_time=filter_configs['forecast_time'],
                          forecast_first=filter_configs['forecast_first']
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
        reference_state = self.filter_configs['reference_state'].copy()
        self.model.write_state(state=reference_state, directory=cycle_states_out_dir, file_name='reference_state')
        # ii- save forecast state
        forecast_state = self.filter_configs['forecast_state']
        self.model.write_state(state=forecast_state, directory=cycle_states_out_dir, file_name='forecast_mean')
        # iii- save analysis state
        analysis_state = self.filter_configs['analysis_state']
        self.model.write_state(state=analysis_state, directory=cycle_states_out_dir, file_name='analysis_mean')
        
        # 4- Save observation to file; use model to write observations to file(s)
        observation = self.filter_configs['observation'].copy()
        self.model.write_observation(observation=observation, directory=cycle_observations_out_dir, file_name='observation', append=False)

        # 4- Save filter configurations and statistics to file,
        # i- Output the configurations dictionaries:
        assim_cycle_configs_file_name = 'assim_cycle_configs'
        if file_output_file_format in ['txt', 'ascii', 'mat']:
            # Output filter and model configurations; this goes under state directory
            assim_cycle_configs_file_name += '.txt'
            utility.write_dicts_to_config_file(assim_cycle_configs_file_name, cycle_states_out_dir,
                                                   [filter_conf, io_conf], ['Filter Configs', 'Output Configs'])
            
        elif file_output_file_format in ['pickle']:
            #
            # Output filter and model configurations; this goes under state directory
            assim_cycle_configs_file_name += '.pickle'
            assim_cycle_configs = dict(filter_configs=filter_conf, output_configs=io_conf)
            pickle.dump(assim_cycle_configs, open(os.path.join(cycle_states_out_dir, assim_cycle_configs_file_name)))
        
        else:
            raise ValueError("Unsupported output format for configurations dictionaries: '%s' !" % file_output_file_format)
            #
        
        # ii Output the RMSE results; it's meaningless to create a new file for each cycle:
        rmse_file_name = 'rmse.txt'  # RMSE are always saved in text files
        rmse_file_path = os.path.join(filter_statistics_dir, rmse_file_name)
        # Create a header for the file if it is newely created
        if not os.path.isfile(rmse_file_path):
            # rmse file does not exist. create file and add header.
            header = "RMSE Results: Filter: '%s' \n %s \t %s \t %s \t %s \t %s \n" % (self._filter_name,
                                                                                      'Observation-Time'.rjust(20),
                                                                                      'Forecast-Time'.rjust(20),
                                                                                      'Analysis-Time'.rjust(20),
                                                                                      'Forecast-RMSE'.rjust(20),
                                                                                      'Analysis-RMSE'.rjust(20),
                                                                                      )
            # get the initial RMSE and add it if forecast is done first...
            if self.filter_configs['forecast_first']:
                initial_time = self.filter_configs['timespan'][0]
                initial_rmse = self.output_configs['filter_statistics']['initial_rmse']
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
        forecast_time = self.filter_configs['forecast_time']
        analysis_time = self.filter_configs['analysis_time']
        observation_time = self.filter_configs['observation_time']
        #
        forecast_rmse = self.output_configs['filter_statistics']['forecast_rmse']
        analysis_rmse = self.output_configs['filter_statistics']['analysis_rmse']
        output_line = u" {0:20.14e} \t {1:20.14e} \t {2:20.14e} \t {3:20.14e} \t {4:20.14e} \n".format(observation_time,
                                                                                                       forecast_time,
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
            Pf = self.filter_configs['forecast_error_covariance']
            Pa = self.filter_configs['forecast_error_covariance']
            print("Saving covariance matrices is not supported yet. CDF will be considered soon!")
            raise NotImplementedError()        
        else:
            pass
    
    #
    def read_cycle_results(self, output_dir, read_err_covars=False):
        """
        Read filtering results from file(s).
        Check the output directory first. If the directory does not exist, raise an IO error.
        If the directory, and files exist, Start retrieving the results properly
        
        Args:
            output_dir: directory where KF results are saved. 
                We assume the structure is the same as the structure created by the KF implemented here
                in 'save_cycle_results'.
                
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
        
        
#
#
if __name__ == '__main__':
    """
    This is a test procedure
    """
    pass
    
