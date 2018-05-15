
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
    
    Particle filter constructor.
    Input:
        model_object: a reference to the model object. Error statistics are loaded from the model object.
                      Forecast is carried out by calling time integrator attached to the model object.
        filter_configs: a dictionary containing filter configurations.
    
    
    """
    _filter_name = "PF"

    _def_filter_configs = dict(resampling_scheme=None,
                               likelihood_function='gaussian',
                               normalize_weights=False,
                               effective_sample_size=None,
                               checkpoints=None,
                               analysis_ensemble=None,
                               forecast_ensemble=None,
                               observation=None,
                               refernce_initial_condition=None,
                               apply_forecast_first=True,
                               regression_model_type='global',
                               refernce_state=None,
                               forecast_rmse=None,
                               analysis_rmse=None,
                               apply_preprocessing=False,
                               apply_postprocessing=False,
                               screen_output=True,
                               file_output=False,
                               file_output_moment_only=True,
                               file_output_moment_name='mean',
                               file_output_directory='Assimilation_Results',
                               files_output_file_name_prefix='KF_results',
                               files_output_file_format='text',
                               files_output_separate_files=True
                               )

    def __init__(self, model_object, filter_configs=None):
        
        FiltersBase.__init__(self, model_object, filter_configs)
        #
        # the following configuration are filter-specific.
        self._likelihood_function = self._filter_configs['likelihood_function']
        self._resampling_scheme = self._filter_configs['resampling_scheme']
        #
        if self._resampling_scheme is not None:
            self._resample = True
            if self._resampling_scheme.lower() in ['systematic', 'stratified']:
                self._resampling_scheme = self._resampling_scheme.lower()
            else:
                raise ValueError("Resampling scheme %s is not supported!" % self._resampling_scheme)
        else:
            self._resample = False

    def filtering_cycle(self, checkpoints=None, obs=None):
        """
        Apply the filtering step. Forecast, then Analysis...
        All arguments are accessed and updated in the configurations dictionary.
        """

        # Apply pre-processing if needed. This can be ignored if called in the assimilation_process class
        if self._apply_preprocessing:
            self.cycle_preprocessing()

        # Filter core...
        # ~~~~~~~~~~~~~~~~~
        if checkpoints is None:
            checkpoints = self._checkpoints
        else:
            self._checkpoints = checkpoints
        if self._filter_configs['observation'] is None and obs is not None:
            self._filter_configs['observation'] = obs
        elif self._filter_configs['observation'] is None and obs is None:
            raise ValueError("Observation vector is not passed!")
        else:
            pass

        if self._apply_forecast_first:
            self._analysis_time = checkpoints[-1]
            # # Forecast step
            self.forecast()
            # Analysis step
            self.analysis()
            reference_state = self._filter_configs['refernce_state']
            tmp_trajectory = self._model.integrate_state(initial_state=reference_state, checkpoints=checkpoints)
            reference_state = tmp_trajectory[-1]
        else:
            self._analysis_time = checkpoints[0]
            # Forecast step
            self.analysis()
            # Analysis step
            self.forecast()
            reference_state = self._filter_configs['refernce_state']

        # Calculate filter diagnostics (if posible)
        # RMSE (forecast and analysis)
        state_size = self._model._state_size
        # Forecast RMSE
        tmp_state = utility.ensemble_mean(self._filter_configs['forecast_ensemble'])
        update_vec = (tmp_state.scale(-1)).add(reference_state)
        self._filter_configs['forecast_rmse'] = update_vec.norm2() / np.sqrt(state_size)
        # Analysis RMSE
        tmp_state = utility.ensemble_mean(self._filter_configs['analysis_ensemble'])
        update_vec = (tmp_state.scale(-1)).add(reference_state)
        self._filter_configs['analysis_rmse'] = update_vec.norm2() / np.sqrt(state_size)


        # Apply pre-processing if needed. This can be ignored if called in the assimilation_process class
        if self._apply_postprocessing:
            self.cycle_postprocessing()

        if self._screen_output:
            # print results to screen.
            self.print_cycle_results()

        if self._file_output:
            self.save_cycle_results()
        #

    def forecast(self):
        """
        Forecast step: propagate each ensemble member to the end of the given checkpoints to produce and ensemble of
                       forecasts. Filter configurations dictionary is updated with the new results.
        """
        # generate the forecast state
        analysis_ensemble = self._filter_configs['analysis_ensemble']
        if analysis_ensemble is None or len(analysis_ensemble)<1:
            raise ValueError("Either no analysis ensemble is initialized or it is an empty list!")
        else:
            time_span = self._checkpoints
            # print('time_span', time_span)
            self._filter_configs['forecast_ensemble'] = []
            for ens_member in analysis_ensemble:
                # print('ens_member', ens_member)
                trajectory = self._model.integrate_state(initial_state=ens_member, checkpoints=time_span)
                # print('forecast_state', trajectory[-1])
                self._filter_configs['forecast_ensemble'].append(trajectory[-1].copy())

        # print('analysis_ensemble', self._filter_configs['analysis_ensemble'])
        # print('forecast_ensemble', self._filter_configs['forecast_ensemble'])

    def analysis(self):
        """
        Analysis step:
        """
        # Check the resampling flag and resampling scheme
        apply_resampling = self._resample
        if apply_resampling:
            resampling_scheme = self._resampling_scheme

        # get the measurements vector
        observation = self._filter_configs['observation']
        obs_vec_size = self._model._observation_vector_size

        # retrieve the ensemble of forecast states
        forecast_ensemble = self._filter_configs['forecast_ensemble']
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        ensemble_size = len(forecast_ensemble)
        #
        # this is how list of objects should be copied...
        if self._filter_configs['analysis_ensemble'] is None:
            analysis_ensemble = [member.copy() for member in forecast_ensemble]
        else:
            analysis_ensemble = self._filter_configs['analysis_ensemble']
        #
        # Evaluate the likelihood for all particles
        particles_likelihood = np.asarray(self.likelihood(model_states=forecast_ensemble, observation=observation))
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
        self._filter_configs['effective_sample_size'] = ess

        # Resampling step:
        if apply_resampling:
            normalize_weights = self._filter_configs['normalize_weights']
            resampling_indexes = self.resample_states(weights=normalized_weights,
                                                      ensemble_size=ensemble_size,
                                                      strategy=resampling_scheme,
                                                      normalize_weights=normalize_weights)
            # Generate Analysis ensemble by weighted resampling
            for ens_ind in xrange(ensemble_size):
                res_ind = resampling_indexes[ens_ind]
                analysis_ensemble[ens_ind] = forecast_ensemble[res_ind]
        else:
            analysis_ensemble = forecast_ensemble

        self._filter_configs['analysis_ensemble'] = analysis_ensemble
        #

    def cycle_preprocessing(self):
        """
        PreProcessing on the passed data before applying the data assimilation filter cycle.
        Applied if needed based on the passed options in the configurations dictionary...
        """
        # project the prior (joint-state) ensemble onto the singular vector (columns of U).
        raise NotImplementedError()

    def cycle_postprocessing(self):
        """
        PostProcessing on the passed data before applying the data assimilation filter cycle.
        Applied if needed based on the passed options in the configurations dictionary...
        """
        raise NotImplementedError()

    def save_cycle_results(self, full_out_dir=None, relative_out_dir=None, separate_files=None):
        """
        Save filtering results from the current cycle to file(s).
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        Input:
            out_dir: directory to put results in. The directory
        """
        screen_out = self._screen_output
        if not screen_out:
            # print("Options in the configurations dictionary indicate that no file output required.")
            pass
        else:
            # Start saving parameters to be saved
            # This has to be tailored for each filter/smoother differently. We can not guarantee that all schemes
            # will spit out the same output.
            if full_out_dir is None and relative_out_dir is None:
                relative_out_dir = self._file_output_directory
                env_DATeS_root_path = os.getenv("DATES_ROOT_PATH")
                full_out_dir = os.path.join(env_DATeS_root_path, relative_out_dir)
            elif full_out_dir is not None:
                pass
            elif relative_out_dir is not None:
                env_DATeS_root_path = os.getenv("DATES_ROOT_PATH")
                full_out_dir = os.path.join(env_DATeS_root_path, relative_out_dir)
            else:
                # Remove after debugging
                raise ValueError("This shouldn't be visited!")

            # check if full_out_dir exists, otherwise create it:
            if not os.path.exists(full_out_dir):
                # create directory with all parent directories if they do not exist...
                os.makedirs(full_out_dir)

            # Start outputting results to file(s) based on the other configurations such as file_output_iterations, etc.
            pass

    def print_cycle_results(self):
        """
        Print filtering results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        """
        if self._screen_output:
            # print RMSE
            try:
                forecast_rmse = self._filter_configs['forecast_rmse']
                analysis_rmse = self._filter_configs['analysis_rmse']
            except ValueError:
                print("RMSEs are not calculated! No printable outputs...")

            try:
                effective_sample_size = self._filter_configs['effective_sample_size']
            except ValueError:
                print("Effective sample size is not evaluated")
                effective_sample_size = -1

            analysis_time = self._analysis_time
            print("Time[%5.4e]: Forecast RMSE=%9.6e,  Analysis RMSE=%9.6e, ESS=%5.3f" %(analysis_time,
                                                                                      forecast_rmse,
                                                                                      analysis_rmse,
                                                                                      effective_sample_size)
                  )

    def likelihood(self, model_states, observation, model_observations=None, likelihood_function='gaussian',
                   normalize_likelihood=True):
        """
        Given a model state or a list of model states (an ensemble), evaluate the likelihood of the observation using
        the likelihood function.
        if normalize_likelihood is True, the scaling factor of the likelihood function will be incorporated so that
        the likelihood value is in [0, 1]
        Input:
            model_states:
            observation:
            model_observations:
            likelihood_function:
            normalize_weights
        Output:
            likelihood: a list containing the likelihood of each ensemble member. If one state is passed,
            the likelihood is a number.
        """
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
        # print('num_states=', num_states)

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
                    detR = self._model.observation_error_model.detR
                except:
                    raise ValueError("determinant of observation error covariance matrix is not available!")
                observation_vector_size = self._model._observation_vector_size
                scaling_factor = 2*np.pi **( -(observation_vector_size / 2)) / (np.abs(detR))**(0.5)
            else:
                scaling_factor = 1.0

            if num_states == 1:
                # likelihood to be returned is just a number for the given state.
                # check for theoretical observation(s)
                if model_observations is None:
                    model_observations = self._model.evaluate_theoretical_observation(model_states)
                innovation = model_observations.axpy(-1, observation)
                scaled_innovation = self._model.observation_error_model.invR.vector_product(innovation, in_place=False)
                likelihood = np.exp(-0.5*(innovation.dot(scaled_innovation))) * scaling_factor
            else:
                # loop over all states, create observation vector then evaluate the likelihoods.
                likelihood = np.empty(num_states, dtype=np.float128)
                neg_log_likelihood = np.empty(num_states, dtype=np.float128)
                for state_ind in xrange(num_states):
                    state = model_states[state_ind]
                    if model_observations is None:
                        model_observations = self._model.evaluate_theoretical_observation(state)
                    innovation = model_observations.axpy(-1, observation)
                    scaled_innovation = self._model.observation_error_model.invR.vector_product(innovation,
                                                                                                in_place=False)
                    neg_log_likelihood[state_ind] = 0.5*(innovation.dot(scaled_innovation))
                min_neg_log_likelihood = np.min(np.abs(neg_log_likelihood))  # there shouldn't be negative values
                # the most favorable particle will receive a likelihood set to one.
                likelihood = np.exp(-(neg_log_likelihood-min_neg_log_likelihood))
        else:
            raise ValueError("Likelihood function '%s' is not supported" % likelihood_function)

        return likelihood


    @staticmethod
    def resample_states(weights, ensemble_size=None, strategy='stratified', normalize_weights=False):
        """
         Resampling step for the particle filter.
            Inputs:
                Weights:  np.array of (particles') weights (normalized or not)
                ensemble_size: number of samples to generate indexes for
                Strategy: Resampling strategy;
            Implemented schemes:
                1- Systematic Resampling,
                2- Stratified Resampling,
                3- ...
            Outputs:
                Indexes:  np.array of size ensemble_size containing integer indexes generated based on the given
                weights.
        """
        resampling_strategies = ['systematic', 'stratified']
        if not(strategy.lower() in resampling_strategies):
            raise ValueError("Strategies implemented so far are: %s" % repr(resampling_strategies))
        else:
            weights = np.squeeze(np.asarray(weights))
            weight_vec_len = weights.size
            weight_vec_sum = weights.sum()
            #
            if weight_vec_sum == 0:
                raise ValueError("Weights sum to zeros!!")
            #
            if ensemble_size is None:
                ensemble_size = weight_vec_len

            if weight_vec_sum != 1 and normalize_weights:
                # Normalize weights if necessary
                weights = weights / weight_vec_sum

        if strategy.lower() == 'systematic':
            #
            weights_cumsum = weights.cumsum()   # cumulative sum of weights
            indexes = np.zeros(ensemble_size)

            T = np.linspace(0, 1-1/ensemble_size, num=ensemble_size,
                            endpoint=True)+((np.random.rand(1)[0])/ensemble_size )
            i = 0
            j = 0
            while i < ensemble_size and j < weight_vec_len:
                while weights_cumsum[j]<T[i]:
                    j += 1
                indexes[i] = j
                i += 1

        elif strategy.lower() == 'stratified':
            #
            weights_cumsum = weights.cumsum()
            indexes = np.zeros(ensemble_size, dtype=np.int)

            T = np.linspace(0, 1-1/ensemble_size, num=ensemble_size, endpoint=True) \
                            + (np.random.rand(ensemble_size)/ensemble_size)
            i = 0
            j = 0
            while i < ensemble_size:
                while weights_cumsum[j]<T[i] and j < weight_vec_len-1:
                    j += 1
                indexes[i] = j
                i += 1
        else:
            raise NotImplementedError


        #print 'Indexes :',Indexes
        return indexes
        
