
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
    Vanilla particle filter. Resampling step is supported as well -->(SIR). 
"""


import numpy as np
import os
import shutil

import dates_utility as utility
from filters_base import FiltersBase
from state_vector_base import StateVectorBase as state_vector
from observation_vector_base import ObservationVectorBase as observation_vector



class PF(FiltersBase):
    """
    A class implementing the vanilla particle filter.
    Resampling step can be carried out as well.

    filter_configs:  dict,
            A dictionary containing PF filter configurations.
            Supported configuarations:
            --------------------------
                * model (default None):  model object
                * filter_name (default None): string containing name of the filter; used for output.
                * likelihood_function (default 'gaussian'): PDF of the observation likelihood
                * normalize_weights (default False): rescale weights such that the sum is 1
                * resampling_scheme: used to recover from degeneracy
                    supported are:
                        1- 'systematic'
                        2- 'stratified'
                * effective_sample_size
                * 
                * inflation_factor (default 1.0): covariance inflation factor
                * localize_weights (default False): bool,
                * ensemble_size (default None): size of the ensemble; this has to be set e.g. in a driver
                * analysis_ensemble (default None): a placeholder of the analysis ensemble.
                    All ensembles are represented by list of model.state_vector objects
                * analysis_state (default None): model.state_vector object containing the analysis state.
                    This is where the filter output (analysis state) will be saved and returned.
                * forecast_ensemble (default None): a placeholder of the forecast/background ensemble.
                    All ensembles are represented by list of model.state_vector objects
                * forecast_state (default None): model.state_vector object containing the forecast state.
                * filter_statistics: dict,
                    A dictionary containing updatable filter statistics. This will be updated by the filter.

        output_configs: dict,
            A dictionary containing screen/file output configurations.
            Supported configuarations:
            --------------------------
                * scr_output (default False): Output results to screen on/off switch
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default 'Assimilation_Results'): relative path (to DATeS root directory)
                    of the directory to output results in

                * filter_statistics_dir (default 'Filter_Statistics'): directory where filter statistics (such as RMSE, ESS,...) are saved
                * model_states_dir (default 'Model_States_Repository'): directory where model-states-like objects (including ensemble) are saved
                * observations_dir (default 'Observations_Rpository'): directory where observations and observations operators are saved
                * file_output_moment_only (default True): output moments of the ensembles (e.g. ensemble mean) or the full ensembles.
                * file_output_moment_name (default 'mean'): Name of the first order moment to save.
                    used only if file_output_moment_only is True
                * file_output_file_name_prefix (default 'PF_results'): name/prefix of output file
                * file_output_file_format (default 'mat'): file ouput format.
                    Supported formats:
                        - 'mat': matlab matrix files,
                        - 'pickle': python pickled objects,
                        - 'txt' or 'ascii': text files
                * file_output_separate_files (default True): save all results to a single or multiple files

    """
    _filter_name = "PF"

    _def_local_filter_configs = dict(model=None,
                                     filter_name=_filter_name,
                                     resampling_scheme=None,
                                     likelihood_function='gaussian',
                                     normalize_weights=True,
                                     effective_sample_size=None,
                                     inflation_factor=1.0,  # applied to forecast ensemble
                                     localize_weights=False,
                                     ensemble_size=None,
                                     analysis_ensemble=None,
                                     analysis_state=None,
                                     forecast_ensemble=None,
                                     forecast_state=None,
                                     filter_statistics=dict(forecast_rmse=None,
                                                            analysis_rmse=None,
                                                            initial_rmse=None
                                                            )
                                    )
    #
    _def_local_output_configs = dict(scr_output=True,
                                     file_output=False,
                                     filter_statistics_dir='Filter_Statistics',
                                     model_states_dir='Model_States_Repository',
                                     observations_dir='Observations_Rpository',
                                     file_output_moment_only=True,
                                     file_output_moment_name='mean',
                                     file_output_file_name_prefix='PF_results',
                                     file_output_file_format='txt',
                                     file_output_separate_files=False
                                     )



    def __init__(self, filter_configs=None, output_configs=None):

        # aggregate configurations, and attach filter_configs, output_configs to the filter object.
        filter_configs = utility.aggregate_configurations(filter_configs, PF._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, PF._def_local_output_configs)
        #
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(filter_configs=filter_configs, output_configs=output_configs)
        else:
            # old-stype class
            super(PF, self).__init__(filter_configs=filter_configs, output_configs=output_configs)

        #
        # the following configuration are filter-specific.
        # validate the ensemble size
        if self.filter_configs['ensemble_size'] is None:
            try:
                forecast_ensemble_size = len(self.filter_configs['forecast_ensemble'])
            except (ValueError, AttributeError, TypeError):
                forecast_ensemble_size = 0
            try:
                analysis_ensemble_size = len(self.filter_configs['analysis_ensemble'])
            except (ValueError, AttributeError, TypeError):
                analysis_ensemble_size = 0

            self.sample_size = max(forecast_ensemble_size, analysis_ensemble_size)
            #
        else:
            self.sample_size = self.filter_configs['ensemble_size']

        # retrieve the observation vector size from model:
        self.observation_size = self.filter_configs['model'].observation_vector_size()

        if self.filter_configs['localize_weights']:
            print("Weight Localization in Particle Filter is not yet implemented.")
            raise NotImplementedError

        #
        #
        try:
            self._verbose = self.output_configs['verbose']
        except(AttributeError, NameError):
            self._verbose = False

        # the following configuration are filter-specific.
        likelihood_function = self.filter_configs['likelihood_function'].lower().strip()
        resampling_scheme = self.filter_configs['resampling_scheme'].lower().strip()
        #
        if resampling_scheme is not None:
            self._resample = True
            if resampling_scheme in ['systematic', 'stratified']:
                pass
            else:
                print("Ruesampling scheme %s is not supported!" % self._resampling_scheme)
                raise ValueError
        else:
            self._resample = False

        # Initialized successfuly;
        self.__initialized = True
        # 

    #
    def filtering_cycle(self, update_reference=False):
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
            super(PF, self).filtering_cycle(update_reference=update_reference)
        #
        # Add further functionality if you wish...

    #
    def forecast(self):
        """
        Forecast step of the filter.
        Use the model object to propagate each ensemble member to the end of the given checkpoints to
        produce and ensemble of forecasts.
        Filter configurations dictionary is updated with the new results.
        If the prior is assumed to be a Gaussian, we are all set, otherwise we have to estimate it's
        parameters based on the provided forecast ensemble (e.g. in the case of 'prior_distribution' = GMM ).

        Args:
            None

        Returns:
            None

        """
        # generate the forecast states
        analysis_ensemble = self.filter_configs['analysis_ensemble']
        timespan = self.filter_configs['timespan']
        forecast_ensemble = utility.propagate_ensemble(ensemble=analysis_ensemble,
                                                       model=self.filter_configs['model'],
                                                       checkpoints=timespan,
                                                       in_place=False)
        self.filter_configs['forecast_ensemble'] = forecast_ensemble
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        self.filter_configs['forecast_state'] = forecast_state.copy()

        # Add more functionality after building the forecast ensemble if desired!
        # Obtain information about the prior distribution from the forecast ensemble
        # Covariance localization and hybridization are carried out if requested
        self.generate_prior_info()
        #

    #
    def generate_prior_info(self):
        """
        Generate the statistics of the approximation of the prior distribution.
            - if the prior is Gaussian, the covariance (ensemble/hybrid) matrix is generated and
              optionally localized.

        Args:
            None

        Returns:
            None

        """
        #
        # Read the forecast ensemble...
        forecast_ensemble = self.filter_configs['forecast_ensemble']
        #
        # Localization is now moved to the analysis step
        if self.filter_configs['localize_weights']:
            print("Localization in particle filters is not yet implemented...")
            raise NotImplementedError
        else:
            pass

        #


    def analysis(self):
        """
        Analysis step:
        """

        #
        model = self.filter_configs['model']
        state_size = model.state_size()
        #

        # Check if the forecast ensemble should be inflated;
        f = self.filter_configs['inflation_factor']
        forecast_ensemble = utility.inflate_ensemble(self.filter_configs['forecast_ensemble'], f, in_place=False)
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        ensemble_size = len(forecast_ensemble)

        # get the measurements vector
        observation = self.filter_configs['observation']
        observation_size = model.observation_vector_size()

        #
        normalized_weights = None
        self.filter_configs.update({'normalized_weights':normalized_weights})
        # Evaluate the likelihood for all particles
        particles_likelihood = np.asarray(self.evaluate_likelihood(model_states=forecast_ensemble, observation=observation))
        likelihood_sum = particles_likelihood.sum()
        if likelihood_sum == 0:  # This should not happen as long as all particles are passed to the likelihood
            # print particles_likelihood
            print('All particles received negligible weights rounded down to zero!. Filter diverged; Ensemble is not '
                  'updated!')
            return None
        else:
            normalized_weights = particles_likelihood/likelihood_sum

        # Monitors the effective particle number (effective sample size)
        ess = 1.0 / (np.asarray(normalized_weights)**2).sum()
        self.filter_configs['effective_sample_size'] = ess
        self.filter_configs.update({'particles_likelihood':particles_likelihood,
                                    'normalized_weights':normalized_weights})
        
        # Check the resampling flag and resampling scheme
        # Resampling step:
        if self._resample:
            resampling_scheme = self.filter_configs['resampling_scheme']
            normalize_weights = self.filter_configs['normalize_weights']
            resampling_indexes = self.resample_states(weights=normalized_weights,
                                                      ensemble_size=ensemble_size,
                                                      strategy=resampling_scheme,
                                                      normalize_weights=normalize_weights)
            # Generate Analysis ensemble by weighted resampling
            print("Resampling...")
            if self.filter_configs['analysis_ensemble'] is None:
                analysis_ensemble = []
                for ens_ind in xrange(ensemble_size):
                    res_ind = resampling_indexes[ens_ind]
                    # print("Ens ind: %d ;  resampling from index %d" %(ens_ind, res_ind))
                    analysis_ensemble.append(forecast_ensemble[res_ind].copy())
            else:
                analysis_ensemble = self.filter_configs['analysis_ensemble']
                for ens_ind in xrange(ensemble_size):
                    res_ind = resampling_indexes[ens_ind]
                    # print("Ens ind: %d ;  resampling from index %d" %(ens_ind, res_ind))
                    analysis_ensemble[ens_ind] = forecast_ensemble[res_ind].copy()
            #
        else:
            resampling_indexes = np.arange(ensemble_size)
            analysis_ensemble = [member.copy() for member in forecast_ensemble]
            

        self.filter_configs['resampling_indexes'] = resampling_indexes
        self.filter_configs['analysis_ensemble'] = analysis_ensemble
        self.filter_configs['analysis_state'] = utility.ensemble_mean(analysis_ensemble)
        #

    def evaluate_likelihood(self, model_states, observation, model_observations=None, likelihood_function='gaussian',
                   normalize_likelihood=True):
        """
        Given a model state or a list of model states (an ensemble), evaluate the likelihood of the observation using
        the likelihood function.
        if normalize_likelihood is True, the scaling factor of the likelihood function will be incorporated so that
        the likelihood value is in [0, 1]
        Args:
            model_states:
            observation:
            model_observations:
            likelihood_function:
            normalize_weights
        Returns:
            likelihood_vals: a list containing the likelihood of each ensemble member. 
                If one state is passed, the likelihood is a number.

        """
        # print(model_states, observation)
        model = self.filter_configs['model']
        observation_error_model = model.observation_error_model
        observation_vector_size = model.observation_vector_size()

        # Keep this test here for when/if we add more likelihood functions
        if likelihood_function.lower() not in ['gaussian', 'normal']:
            raise ValueError("Likelihood function '%s' is not supported" % likelihood_function)

        # check the number of ensembles passed but don't start evaluating likelihoods
        if isinstance(model_states, state_vector):
            num_states = 1
        elif isinstance(model_states, list):
            num_states = len(model_states)
            if num_states == 1:
                model_states = model_states[0]
        else:
            raise TypeError("Type of passed model_states is unrecognized!")

        if model_observations is not None:
            if isinstance(model_states, observation_vector):
                num_model_obs = 1
            elif isinstance(model_states, list):
                num_model_obs = len(model_observations)
                if num_model_obs == 1:
                    num_model_obs = model_states[0]
            else:
                raise TypeError("Type of passed model_states is unrecognized!")
        if model_states is not None and model_observations is not None and num_model_obs != num_states:
            raise ValueError("Number of model observations must be equal to model states")

        if likelihood_function.lower() in ['gaussian', 'normal']:
            # normalize the likelihood values using the likelihood function scaling factor.
            if normalize_likelihood:
                try:
                    detR = observation_error_model.detR
                except:
                    raise ValueError("determinant of observation error covariance matrix is not available!")
                scaling_factor = 2*np.pi **( -(observation_vector_size / 2.0)) / (np.abs(detR))**(0.5)
            else:
                scaling_factor = 1.0

            if num_states == 1:
                # likelihood to be returned is just a number for the given state.
                # check for theoretical observation(s)
                if model_observations is None:
                    model_observations = model.evaluate_theoretical_observation(model_states)
                innovation = model_observations.axpy(-1, observation, in_place=False)
                scaled_innovation = observation_error_model.invR.vector_product(innovation, in_place=False)
                likelihood_vals = np.exp(-0.5*(innovation.dot(scaled_innovation))) * scaling_factor
            else:
                # loop over all states, create observation vector then evaluate the likelihoods.
                likelihood_vals = np.empty(num_states, dtype=np.float128)
                neg_log_likelihood = np.empty(num_states, dtype=np.float128)
                for state_ind in xrange(num_states):
                    if model_observations is not None:
                        model_observation = model_observations[state_ind]
                    else:
                        state = model_states[state_ind]
                        model_observation = model.evaluate_theoretical_observation(state)
                    
                    # print("model_observation", model_observation)
                    innovation = model_observation.axpy(-1, observation, in_place=False)
                    scaled_innovation = observation_error_model.invR.vector_product(innovation, in_place=False)
                    neg_log_likelihood[state_ind] = 0.5*(innovation.dot(scaled_innovation))
                # print("neg_log_likelihood", neg_log_likelihood)
                min_neg_log_likelihood = np.min(np.abs(neg_log_likelihood))  # there shouldn't be negative values
                # the most favorable particle will receive a likelihood set to one.
                likelihood_vals = np.exp(-(neg_log_likelihood-min_neg_log_likelihood))
        else:
            raise ValueError("Likelihood function '%s' is not supported" % likelihood_function)

        return likelihood_vals


    def cycle_preprocessing(self):
        """
        PreProcessing on the passed data before applying the data assimilation filter cycle.
        Applied if needed based on the passed options in the configurations dictionary...
        """
        # project the prior (joint-state) ensemble onto the singular vector (columns of U).
        pass

    def cycle_postprocessing(self):
        """
        PostProcessing on the passed data before applying the data assimilation filter cycle.
        Applied if needed based on the passed options in the configurations dictionary...
        """
        pass


    def print_cycle_results(self):
        """
        Print filtering results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        
        Args:
            None
        
        Returns:
            None
        """
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().print_cycle_results()
        else:
            # old-stype class
            super(PF, self).print_cycle_results()
        pass  # Add more...
        #
    
    
    
    #
    def save_cycle_results(self, output_dir=None, cleanup_out_dir=False):
        """
        Save filtering results from the current cycle to file(s).
        Check the output directory first. If the directory does not exist, create it.

        Args:
            output_dir: full path of the directory to save results in

        Returns:
            None

        """
        model = self.filter_configs['model']
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
        file_output_file_format = output_configs['file_output_file_format'].lower()
        file_output_file_name_prefix = output_configs['file_output_file_name_prefix']  # this is useless!

        # SAVING MODEL STATES (Either Moments Only or Full Ensembles)
        # write cycle configurations:
        model_conf = model.get_model_configs()
        utility.write_dicts_to_config_file('setup.dat', file_output_directory,
                                           model_conf, 'Model Configs')
        # get a proper name for the folder (cycle_*) under the model_states_dir path
        suffix = 0
        cycle_prefix = 'cycle_'
        while True:
            cycle_dir = cycle_prefix + str(suffix)
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

        # Now we have all directories cleaned-up and ready for outputting.
        output_dir_structure_file = os.path.join(file_output_directory, 'output_dir_structure.txt')
        if not os.path.isfile(output_dir_structure_file):
            # First, we need to save the output paths info to a file to be used later by results' reader
            # print('writing output directory structure to config file \n \t%s \n' % output_dir_structure_file)
            out_dir_tree_structure = dict(file_output_separate_files=file_output_separate_files,
                                          file_output_directory=file_output_directory,
                                          model_states_dir=model_states_dir,
                                          observations_dir=observations_dir,
                                          filter_statistics_dir=filter_statistics_dir,
                                          cycle_prefix=cycle_prefix
                                          )
            utility.write_dicts_to_config_file(file_name='output_dir_structure.txt',
                                               out_dir=file_output_directory,
                                               dicts=out_dir_tree_structure,
                                               sections_headers='out_dir_tree_structure'
                                               )

        #  save states
        file_output_moment_only = output_configs['file_output_moment_only']
        if file_output_moment_only:
            file_output_moment_name = output_configs['file_output_moment_name'].lower()
            if file_output_moment_name in ['mean', 'average']:
                # start outputting ensemble means... (both forecast and analysis of course).
                # save forecast mean
                forecast_state = self.filter_configs['forecast_state']
                model.write_state(state=forecast_state, directory=cycle_states_out_dir, file_name='forecast_mean')
                # save analysis mean
                analysis_state = self.filter_configs['analysis_state']
                model.write_state(state=analysis_state, directory=cycle_states_out_dir, file_name='analysis_mean')
            else:
                raise ValueError("Unsupported ensemble moment: '%s' !" % (file_output_moment_name))
        else:
            # start outputting the whole ensemble members (both forecast and analysis ensembles of course).
            # check if all ensembles are to be saved or just one of the supported ensemble moments
            for ens_ind in xrange(self.sample_size):
                if file_output_separate_files:
                    # print('saving ensemble member to separate files: %d' % ens_ind)
                    forecast_ensemble_member = self.filter_configs['forecast_ensemble'][ens_ind]
                    model.write_state(state=forecast_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='forecast_ensemble_'+str(ens_ind),
                                            append=False
                                            )
                    #
                    analysis_ensemble_member = self.filter_configs['analysis_ensemble'][ens_ind]
                    model.write_state(state=analysis_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='analysis_ensemble_'+str(ens_ind),
                                            append=False
                                            )
                else:
                    # print('saving ensemble member to same file with resizing: %d' % ens_ind)
                    # save results to different files. For moments
                    forecast_ensemble_member = self.filter_configs['forecast_ensemble'][ens_ind]
                    model.write_state(state=forecast_ensemble_member.copy(),
                                            directory=cycle_states_out_dir,
                                            file_name='forecast_ensemble',
                                            append=True
                                            )
                    #
                    analysis_ensemble_member = self.filter_configs['analysis_ensemble'][ens_ind]
                    model.write_state(state=analysis_ensemble_member.copy(),
                                            directory=cycle_states_out_dir,
                                            file_name='analysis_ensemble',
                                            append=True
                                            )
        # save reference state
        reference_state = self.filter_configs['reference_state']
        model.write_state(state=reference_state, directory=cycle_states_out_dir, file_name='reference_state')

        #
        # Save observation to file; use model to write observations to file(s)
        # save analysis mean
        observation = self.filter_configs['observation']
        model.write_observation(observation=observation, directory=cycle_observations_out_dir, file_name='observation', append=False)

        # Save filter statistics to file
        # 1- Output filter RMSEs: RMSEs are saved to the same file. It's meaningless to create a new file for each cycle
        rmse_file_name = 'rmse'
        if file_output_file_format in ['txt', 'ascii']:
            rmse_file_name += '.dat'
            rmse_file_path = os.path.join(filter_statistics_dir, rmse_file_name)
            if not os.path.isfile(rmse_file_path):
                # rmse file does not exist. create file and add header.
                filter_name = self.filter_configs['filter_name']
                header = "RMSE Results: Filter: '%s' \n %s \t %s \t %s \t %s \t %s \n" % (filter_name,
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
                pass
            # rmse file exists --> Append rmse results to the file.
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
            #
            with open(rmse_file_path, mode='a') as file_handler:
                file_handler.write(output_line)

            # save filter and model configurations (a copy under observation directory and another under state directory)...
            filter_configs = self.filter_configs
            filter_conf= dict(filter_name=filter_configs['filter_name'],
                              ensemble_size=filter_configs['ensemble_size'],
                              apply_preprocessing=filter_configs['apply_preprocessing'],
                              apply_postprocessing=filter_configs['apply_postprocessing'],
                              timespan=filter_configs['timespan'],
                              analysis_time=filter_configs['analysis_time'],
                              observation_time=filter_configs['observation_time'],
                              forecast_time=filter_configs['forecast_time'],
                              forecast_first=filter_configs['forecast_first'],
                              resampling_weights=filter_configs['normalized_weights'],
                              resampling_indexes=filter_configs['resampling_indexes'],
                              particles_likelihood=filter_configs['particles_likelihood'],
                              effective_sample_size=filter_configs['effective_sample_size'],
                              )
            io_conf = output_configs
            #
            utility.write_dicts_to_config_file('setup.dat', cycle_observations_out_dir,
                                               [filter_conf, io_conf], ['Filter Configs', 'Output Configs'])
            utility.write_dicts_to_config_file('setup.dat', cycle_states_out_dir,
                                               [filter_conf, io_conf], ['Filter Configs', 'Output Configs'])
        else:
            print("Unsupported output format: '%s' !" % file_output_file_format)
            raise ValueError()
            #

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



    @staticmethod
    def resample_states(weights, ensemble_size=None, strategy='stratified', normalize_weights=False):
        """
         Resampling step for the particle filter.
            Args:
                Weights:  np.array of (particles') weights (normalized or not)
                ensemble_size: number of samples to generate indexes for
                Strategy: Resampling strategy;
                    Implemented schemes:
                        1- Systematic Resampling,
                        2- Stratified Resampling,
                        3- ...
            Returns:
                Indexes:  np.array of size ensemble_size containing integer indexes generated based on the given weights.
        """
        resampling_strategies = ['systematic', 'stratified']
        if not(strategy.lower().strip() in resampling_strategies):
            print("Strategies implemented so far are: %s" % repr(resampling_strategies))
            raise ValueError
        else:
            loc_weights = np.squeeze(np.asarray(weights, dtype=np.float128))
            weight_vec_len = loc_weights.size
            weight_vec_sum = loc_weights.sum()
            #
            if weight_vec_sum == 0:
                print("Weights sum to zeros!!")
                raise ValueError
            #
            if ensemble_size is None:
                ensemble_size = weight_vec_len

            if weight_vec_sum != 1 and normalize_weights:
                # Normalize weights if necessary
                loc_weights = loc_weights / weight_vec_sum

        if strategy.lower() == 'systematic':
            #
            weights_cumsum = loc_weights.cumsum()   # cumulative sum of weights
            indexes = np.zeros(ensemble_size, dtype=np.int)

            T = np.linspace(0, 1-1/ensemble_size, num=ensemble_size, endpoint=True)+((np.random.rand(1)[0])/ensemble_size )
            i = 0
            j = 0
            while i < ensemble_size and j < weight_vec_len:
                while weights_cumsum[j]<T[i]:
                    if j >= weight_vec_len-1:
                        break
                    else:
                        j+=1
                indexes[i] = j
                i += 1

        elif strategy.lower() == 'stratified':
            #
            weights_cumsum = weights.cumsum()
            indexes = np.zeros(ensemble_size, dtype=np.int)

            T = np.linspace(0, 1-1/ensemble_size, num=ensemble_size, endpoint=True) + (np.random.rand(ensemble_size)/ensemble_size)
            i = 0
            j = 0
            while i < ensemble_size:
                while weights_cumsum[j]<T[i] and j < weight_vec_len-1:
                    j += 1
                indexes[i] = j
                i += 1
        else:
            print("Strategies implemented so far are: %s" % repr(resampling_strategies))
            raise ValueError  # unecessary!


        # print('Indexes :', indexes)
        return indexes

