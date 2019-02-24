
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
    AssimilationProcess.FilteringProcess:
    A class implementing functionalities of a filtering process.
    A filtering process here refers to repeating a filtering cycle over a specific observation/assimilation timespan

    Remarks:
    --------
        - The main methods here update the DA items and update the appropriate configs dictionaries in the filter object.
        - The filter object itself is responsible for input/output.
"""

import numpy as np
import os
import shutil

import dates_utility as utility
from assimilation_process_base import AssimilationProcess
from filters_base import FiltersBase
from models_base import ModelsBase


class FilteringProcess(AssimilationProcess):
    """
    A class implementing the steps of a filtering process.
    This recursively apply filtering on several assimilation cycles given assimilation/observation timespan.

    Filtering process class constructor.
    The input configurarions are used to control the assimilation process behavious over several cycles.

    Args:
        assimilation_configs (default None); a dictionary containing assimilation configurations.
        Supported configuarations:
            * filter (default None): filter object
            * obs_checkpoints (default None): iterable containing an observation timespan
               - thsese are the time instances at which observation (synthetic or not) are given/generated
            * da_checkpoints (default None): : iterable containing an assimilation timespan
               - thsese are the time instances at which filtering is carried out
               - If same as obs_checkpoints, assimilation is synchronous.
            * da_time_spacing (default None): Type of spacing between consecutive assimilation cycles.
               - Used (along with (num_filtering_cycles) if both obs_checkpoints, da_checkpoints are None.
            * num_filtering_cycles (default None): Used with 'da_time_spacing'only
            * synchronous (default None): Now it has no effect. this will be removed.
            * ref_initial_condition (default None): model.state_vector object used as a true initial state
            * ref_initial_time (default 0):  the time at which 'ref_initial_condition' is defined
            * forecast_first (default True): Forecast then Analysis or Analysis then Forecast steps
            * random_seed (default None): An integer to reset the random seed of Numpy.random
              - resets the seed of random number generator before carrying out any filtering steps.
            * callback (default None): A function to call after each assimilation (filtering) cycle)
            * callback_args (default None): arguments to pass to the callback function.


        output_configs (default None); a dictionary containing screen/file output configurations:
        Supported configuarations:
            * scr_output (default True): Output results to screen on/off switch
            * scr_output_iter (default 1): number of iterations/windows after which outputs are printed
            * file_output (default False): save results to file on/off switch
            * file_output_iter (default 1): number of iterations/windows after which outputs are saved to files
            * file_output_dir (default None): defaul directory in which results are written to files
            * file_output_variables: which data structures to try to save to files

    Returns:
        None

    """
    # Default filtering process configurations
    _def_assimilation_configs = dict(filter=None,
                                     obs_checkpoints=None,
                                     observations=None,
                                     da_checkpoints=None,
                                     da_time_spacing=None,  # Used if both obs_checkpoints,da_checkpoints are None.
                                     num_filtering_cycles=None,  # Used with 'da_time_spacing'only.
                                     synchronous=None,
                                     ref_initial_condition=None,
                                     ref_initial_time=0,  # the time at which 'ref_initial_condition' is passed.
                                     forecast_first=True,  # Forecast then Analysis (default).
                                     random_seed=None,  # Reset the random seed before carrying out any filtering steps.
                                     callback=None,
                                     callback_args=None
                                     )

    _def_output_configs = dict(scr_output=True,
                               scr_output_iter=1,
                               file_output=False,
                               file_output_iter=1,
                               file_output_dir=None,
                               file_output_variables=['model_states',  # we will see if this is adequate or not!
                                                      'observations',
                                                      'rmse',
                                                      'filter_statistics']
                               )

    def __init__(self, assimilation_configs=None, output_configs=None):

        self.assimilation_configs = self.validate_assimilation_configs(assimilation_configs,
                                                                       FilteringProcess._def_assimilation_configs)
        # extract assimilation configurations for easier access
        self.filter = self.assimilation_configs['filter']
        try:
            self.model = self.filter.filter_configs['model']
        except(KeyError, NameError, AttributeError):
            self.model = self.filter.model
        self.da_checkpoints = self.assimilation_configs['da_checkpoints']
        self.obs_checkpoints = self.assimilation_configs['obs_checkpoints']
        self.synchronous = self.assimilation_configs['synchronous']
        self.ref_initial_time = self.assimilation_configs['ref_initial_time']
        self.ref_initial_condition = self.assimilation_configs['ref_initial_condition']
        self.forecast_first = self.assimilation_configs['forecast_first']
        #
        self._callback = self.assimilation_configs['callback']
        self._callback_args = self.assimilation_configs['callback_args']

        # extract output configurations for easier access
        self.output_configs = self.validate_output_configs(output_configs, FilteringProcess._def_output_configs)
        self.scr_output = self.output_configs['scr_output']
        self.scr_output_iter = self.output_configs['scr_output_iter']
        self.file_output = self.output_configs['file_output']
        self.file_output_iter = self.output_configs['file_output_iter']
        self.file_output_dir = self.output_configs['file_output_dir']
        self.file_output_variables = self.output_configs['file_output_variables']

        self.random_seed = self.assimilation_configs['random_seed']

        # Make sure the directory is created and cleaned up at this point...
        self.set_filtering_output_dir(self.file_output_dir, rel_to_root_dir=True)
        #
        model_conf = self.model.model_configs.copy()
        filter_conf = self.filter.filter_configs.copy()
        filter_conf.update({'model':None})
        assim_conf = self.assimilation_configs.copy()
        assim_conf.update({'filter':None})
        file_output_dir= self.output_configs['file_output_dir']
        utility.write_dicts_to_config_file('setup.pickle', file_output_dir,
                                           [model_conf, filter_conf, assim_conf],
                                           ['Model Configs', 'Filter Configs', 'Assimilation Configs'],
                                           True
                                          )
        #

    #
    def recursive_assimilation_process(self, observations=None, obs_checkpoints=None, da_checkpoints=None, update_ref_here=False):
        """
        Loop over all assimilation cycles and output/save results (forecast, analysis, observations)
        for all the assimilation cycles.

        Args:
            observations (default None): list of obs.observation_vector objects,
                A list containing observaiton vectors at specific obs_checkpoints to use for sequential filtering
                If not None, len(observations) must be equal to len(obs_checkpoints).
                If it is None, synthetic observations should be created sequentially

            obs_checkpoints (default None): iterable containing an observation timespan
                Thsese are the time instances at which observation (synthetic or not) are given/generated

            da_checkpoints (default None): iterable containing an assimilation timespan
                Thsese are the time instances at which filtering is carried out.
                If same as obs_checkpoints, assimilation is synchronous.

            update_ref_here (default False): bool,
                A flag to decide to whether to update the reference state here,
                or request to updated it inside the filter.

        Returns:
            None

        """
        filter_obj = self.filter
        model_obj = self.model
        #

        # Check availability of true/reference solution:
        reference_state = self.assimilation_configs['ref_initial_condition']
        if reference_state is None:
            print("No reference soltution found; No synthetic observations can be created, nor statistics e.g. RMSE can be calculated!")
            reference_time = None
        else:
            reference_time = self.assimilation_configs['ref_initial_time']

        # Check availability of observations and observations' and assimilations checkpoints
        if observations is not None and obs_checkpoints is not None:
            # override process configurations and use given obs_checkpoints and observations list.
            # This will be useful if non-synthetic observations are to be used.
            if len(obs_checkpoints) != len(observations):
                print("The number of observations %d does not match the number of observation checkpoints %d" % (len(obs_checkpoints), len(observations)))
                raise ValueError
            else:
                self.assimilation_configs['obs_checkpoints'] = [obs for obs in obs_checkpoints]
                self.assimilation_configs['observations'] = [obs for obs in observations]

            # validate da_checkpoints
            if da_checkpoints is None:
                if self.assimilation_configs['da_checkpoints'] is not None:
                    da_checkpoints = self.assimilation_configs['da_checkpoints']
                else:
                    da_checkpoints = [d for d in obs_checkpoints]
                    self.assimilation_configs['da_checkpoints'] = da_checkpoints
            else:
                self.assimilation_configs['da_checkpoints'] = [d for d in da_checkpoints]

            # No need to create synthetic observations, as real observations exist
            create_synthetic_observations = False
        else:
            #
            if observations is None:
                if self.assimilation_configs['observations'] is None:
                    create_synthetic_observations = True
                else:
                    observations = [obs for obs in self.assimilation_configs['observations']]
                    create_synthetic_observations = False

            if obs_checkpoints is None:
                if self.assimilation_configs['obs_checkpoints'] is None:
                    raise ValueError
                else:
                    obs_checkpoints = [o for o in self.assimilation_configs['obs_checkpoints']]

            if da_checkpoints is None:
                if self.assimilation_configs['da_checkpoints'] is None:
                    da_checkpoints = obs_checkpoints
                else:
                    da_checkpoints = [o for o in self.assimilation_configs['da_checkpoints']]
                    #
        if create_synthetic_observations and reference_state is None:
            print("synthetic observations are to be created, yet the reference state couldn't be found in the assimilation_configs dictionary!")
            raise ValueError

        if reference_state is not None:
            self.filter.filter_configs['reference_state'] = reference_state
            self.filter.filter_configs['reference_time'] = reference_time

            if da_checkpoints[0] == reference_time:
                # TODO: needs to be handled more carefully!
                ignore_first_obs = True
            elif da_checkpoints[0] > reference_time:
                ignore_first_obs = False
                da_checkpoints = np.insert(self.da_checkpoints, 0, self.ref_initial_time)
                self.assimilation_configs['da_checkpoints'] = da_checkpoints
            else:
                raise ValueError("first da_checkpoint is less than the reference initial time!")
        else:
            ignore_first_obs = False

        # Initialize the output variable for callback returns:
        if self._callback is not None:
            callback_output = dict()
        else:
            callback_output = None

        # Random-State: to guarantee same sequence of observations, if Numpy is used to generte random numbers.
        if self.random_seed is not None:
            self._random_state = np.random.get_state()
            np.random.seed(self.random_seed)
        else:
            self._random_state = None

        # Loop over each two consecutive da_points to create cycle limits
        for time_ind in xrange(len(da_checkpoints)-1):
            #
            if create_synthetic_observations:
                reference_state = filter_obj.filter_configs['reference_state'].copy()
                reference_time = filter_obj.filter_configs['reference_time']

            local_da_checkpoints = da_checkpoints[time_ind: time_ind+2]
            local_obs_checkpoints = obs_checkpoints[time_ind: time_ind+2]
            local_timespan = local_da_checkpoints

            # get assimilation, forecast, and observation time and timespan (for flexibility and external calls)
            if self.assimilation_configs['forecast_first']:
                analysis_time = local_da_checkpoints[-1]  # based on forecast_first
                forecast_time = analysis_time  # based on forecast_first
                observation_time = local_obs_checkpoints[-1]
            else:
                analysis_time = local_da_checkpoints[0]  # based on forecast_first
                forecast_time = local_da_checkpoints[-1]  # based on forecast_first
                observation_time = local_obs_checkpoints[0]
            
            # retrieve/create observation at the current observation time
            if create_synthetic_observations:
                if self._random_state is not  None:
                    np_state = np.random.get_state()
                    np.random.set_state(self._random_state)
                else:
                    np_state = None

                reference_traject = self.model.integrate_state(initial_state=reference_state,
                                                               checkpoints=local_obs_checkpoints
                                                               )
                if isinstance(reference_traject, list):
                    # print('REFERENC STATE:', reference_traject[-1])
                    observation = self.model.evaluate_theoretical_observation(reference_traject[-1])
                else:
                    # print('REFERENC STATE:', reference_traject)
                    observation = self.model.evaluate_theoretical_observation(reference_traject)

                # print('Theoritical REFERENCE Observation', observation)
                observation = observation.add(self.model.observation_error_model.generate_noise_vec())
                # print('SCALED Observation', observation)

                if np_state is not  None:
                    self._random_state = np.random.get_state()
                    np.random.set_state(np_state)

            else:
                # use real observations
                if ignore_first_obs:
                    observation = observations[time_ind+1]
                else:
                    observation = observations[time_ind]

            if self.output_configs['scr_output']:
                if (time_ind % self.output_configs['scr_output_iter']) == 0:
                    scr_output = True  # switch on/off screen output in the current filtering cycle
                else:
                    scr_output = False
            else:
                scr_output = False  # switch on/off screen output in the current filtering cycle

            if self.output_configs['file_output']:
                if (time_ind % self.output_configs['file_output_iter']) == 0:
                    file_output = True  # switch on/off file output in the current filtering cycle
                else:
                    file_output = False
            else:
                file_output = False  # switch on/off screen output in the current filtering cycle

            # Control where the reference state is updated. It has to be propagated only once :)
            # update_ref_here = True   # this is moved to the input arguments

            self.assimilation_cycle(analysis_time=analysis_time,
                                    forecast_time=forecast_time,
                                    observation_time=observation_time,
                                    assimilation_timespan=local_timespan,
                                    observation=observation,
                                    scr_output=scr_output,
                                    file_output=file_output,
                                    update_reference=not update_ref_here
                                    )

            if update_ref_here:
                if filter_obj.filter_configs['reference_state'] is not None:
                    reference_state = filter_obj.filter_configs['reference_state'].copy()
                    tmp_traject = self.model.integrate_state(initial_state=reference_state,
                                                             checkpoints=local_timespan
                                                             )
                    if isinstance(tmp_traject, list):
                        self.filter.filter_configs['reference_state'] = tmp_traject[-1].copy()
                    else:
                        self.filter.filter_configs['reference_state'] = tmp_traject.copy()
                    self.filter.filter_configs['reference_time'] = local_timespan[-1]

            # attempt to udpate the model's observation operator:
            try:
                self.model.update_observation_operator()
            except (NotImplementedError):
                print("Model does not support updating the observation oprator or the observational grid.")

            # save/output initial results based on scr_output and file_output flags
            pass

            # Call the callback function if provided:
            if self._callback is not None:
                _cout = self._callback(self._callback_args)
                callback_output.update({str(time_ind): _cout})
        #
        return callback_output
        #
    # aliasing
    start_assimilation_process = recursive_assimilation_process

    #
    def assimilation_cycle(self,
                           analysis_time,
                           forecast_time,
                           observation_time,
                           assimilation_timespan,
                           observation,
                           scr_output=False,
                           file_output=False,
                           update_reference=True
                           ):
        """
        Carry out filtering for the next assimilation cycle.
        Given the assimilation cycle is carried out by applying filtering using filter_object.filtering_cycle().
        filter_object.filtering_cycle() itself should carry out the two steps: forecast, analysis.
        This function modifies the configurations dictionary of the filter object to carry out a single
        filtering cycle.

        Args:
            analysis_time: scalar,
                The time instance at which analysis step of the filter is carried out

            forecast_time: scalar,
                The time instance at which forecast step of the filter is carried out

            observation_time: scalar,
                The time instance at which observation is available

            assimilation_timespan: an iterable containing the beginning and the end of the assimilation cycle i.e. len=2
                A timespan for the filtering process on the current assimilation cycle.
                - analysis_time, obs_time, and forecast_time all must be within in that timespan limits
                - The reference_state is always given at the beginning of the current timespan and
                  updated here or inside the filter cycle while calculating RMSE.


            observation: model.observation_vector object,
                The observaiton vector to use for assimilation/filtering

            scr_output (default False): bool,
                A boolean flag to cotrol whether to output results to the screen or not
                The per-cycle screen output options are set by the filter configs.

            file_output (default False): bool,
                A boolean flag to cotrol whether to save results to file or not.
                The per-cycle file output options are set by the filter configs.

            update_reference (default False): bool,
                A flag to decide to whether to request updating the reference state in by the filter.

        Returns:
            None

        """
        # get and update the configurations of the filter. Both output and filter configs are handled here
        # Update filter configs
        if self.filter.filter_configs['forecast_first'] != self.assimilation_configs['forecast_first']:
            self.filter.filter_configs['forecast_first'] != self.assimilation_configs['forecast_first']
        self.filter.filter_configs['observation_time'] = observation_time
        self.filter.filter_configs['observation'] = observation
        self.filter.filter_configs['timespan'] = assimilation_timespan
        self.filter.filter_configs['analysis_time'] = analysis_time
        self.filter.filter_configs['forecast_time'] = forecast_time

        # Update output configs (based on flag and iterations)
        if self.filter.output_configs['scr_output'] != scr_output:
            self.filter.output_configs['scr_output'] = scr_output
        if self.filter.output_configs['file_output'] != file_output:
            self.filter.output_configs['file_output'] = file_output

        # Now start the filtering cycle.
        self.filter.filtering_cycle(update_reference=update_reference)

    #
    @staticmethod
    def validate_assimilation_configs(assimilation_configs, def_assimilation_configs):
        """
        Aggregate the passed dictionaries with default configurations then make sure parameters are consistent.
        The first argument (assimilation_configs) is validated, updated with missing entries, and returned.

        Args:
            assimilation_configs: dict,
                A dictionary containing assimilation configurations. This should be the assimilation_configs dict
                passed to the constructor.

            def_assimilation_configs: dict,
                A dictionary containing the default assimilation configurations.

        Returns:
            assimilation_configs: dict,
                Same as the first argument (assimilation_configs) but validated, adn updated with missing entries.

        """
        assimilation_configs = utility.aggregate_configurations(assimilation_configs, def_assimilation_configs)
        # Since aggregate never cares about the contents, we need to make sure now all parameters are consistent
        for key in assimilation_configs:
            if key.lower() not in def_assimilation_configs:
                print("Caution: Unknown key detected: '%s'. Ignored and defaults are restored if necessary" % key)
        if assimilation_configs['filter'] is None:
            raise ValueError("You have to create a filter object and attach it here so that "
                             "I can use it for sequential DA!")
        else:
            if not isinstance(assimilation_configs['filter'], FiltersBase):
                raise ValueError("Passed filter is not an instance of 'FiltersBase'!. Passed: %s" %
                                 repr(assimilation_configs['filter']))

        try:
            model = assimilation_configs['filter'].filter_configs['model']
        except(AttributeError, KeyError, NameError):
            try:
                model = assimilation_configs['filter'].model
            except(AttributeError, KeyError, NameError):
                print("Could not retrieve a reference to a valid model object from the passed filter object!")
                raise AttributeError
        #
        if not isinstance(model, ModelsBase):
            raise ValueError("The model retrieved from the passed filter object is not an instance of 'ModelsBase'!. Passed: %s" %
                             repr(assimilation_configs['model'])
                             )

        # check the reference initial condition and the reference initial time
        if assimilation_configs['ref_initial_condition'] is None or assimilation_configs['ref_initial_time'] is None:
            print("You didn't pass a reference initial state, and reference initial time.")
            print("This indicates, you will provide a list of observations to use as real data.")
            print("Please call filtering_process.recursive_assimilation_process() with the following arguments:")
            print("observations, obs_checkpoints, da_checkpoints")
            # raise ValueError("Both the reference initial condition and the initial time must be passed!")

        # Check for observation and assimilation checkpoints and update synchronous accordingly
        # if assimilation_configs['da_checkpoints'] is not None and assimilation_configs['obs_checkpoints'] is not None:
        if assimilation_configs['obs_checkpoints'] is not None:
            # observation checkpoints is built in full.
            if isinstance(assimilation_configs['obs_checkpoints'], int) or isinstance(assimilation_configs['obs_checkpoints'], float):
                assimilation_configs['obs_checkpoints'] = [assimilation_configs['obs_checkpoints']]
            try:
                obs_checkpoints = np.asarray(assimilation_configs['obs_checkpoints'])
                assimilation_configs['obs_checkpoints'] = obs_checkpoints
                num_observations = np.size(obs_checkpoints)
            except:
                raise ValueError("Couldn't cast the observation checkpoints into np.ndarray. "
                                 "This mostly means you didn't pass an iterable!"
                                 "Passed: %s" % str(assimilation_configs['obs_checkpoints']))

            # Now check the assimilation checkpoints
            if assimilation_configs['da_checkpoints'] is not None:
                if isinstance(assimilation_configs['da_checkpoints'], int) or isinstance(assimilation_configs['da_checkpoints'], float):
                    assimilation_configs['da_checkpoints'] = [assimilation_configs['da_checkpoints']]
                try:
                    da_checkpoints = np.asarray(assimilation_configs['da_checkpoints'])
                    assimilation_configs['da_checkpoints'] = da_checkpoints
                    num_assimilation_cycles = np.size(da_checkpoints)
                except:
                    raise ValueError("Couldn't cast the assimilation checkpoints into np.ndarray. "
                                     "This mostly means you didn't pass an iterable!"
                                     "Passed: %s" % repr(assimilation_configs['da_checkpoints']))

                if num_assimilation_cycles != num_observations:
                    raise ValueError("Number of observations and number of assimilation cycles must match!\n"
                                     "Number of assimilation cycles passed = %d\n"
                                     "Number of observation time points = %d" % (num_assimilation_cycles, num_observations)
                                     )
                else:
                    # We are all good to go now. now check if the assimilation should be synchronous or not
                    test_bool = assimilation_configs['obs_checkpoints'] != assimilation_configs['da_checkpoints']
                    if isinstance(test_bool, list) or isinstance(test_bool, np.ndarray):
                        if (test_bool).any():
                            assimilation_configs['synchronous'] = False
                        else:
                            assimilation_configs['synchronous'] = True
                    elif isinstance(test_bool, bool):  # this was supposed to handle single entries. I made sure all are converted to arrays. REM...
                        if test_bool:
                            assimilation_configs['synchronous'] = False
                        else:
                            assimilation_configs['synchronous'] = True
                    else:
                        raise AssertionError(" Unexpected comparison results!")
            else:
                assimilation_configs['da_checkpoints'] = assimilation_configs['obs_checkpoints']
                assimilation_configs['synchronous'] = True  # No matter what!

            assimilation_configs['da_time_spacing'] = None
            if assimilation_configs['ref_initial_time'] > np.min(assimilation_configs['da_checkpoints']):
                raise ValueError("Some observation times or assimilation times are set before "
                                 "the time of the reference initial condition!")
            elif assimilation_configs['ref_initial_time'] == assimilation_configs['da_checkpoints'][0]:
                assimilation_configs['num_filtering_cycles'] = np.size(assimilation_configs['da_checkpoints']) - 1
            else:
                assimilation_configs['num_filtering_cycles'] = np.size(assimilation_configs['da_checkpoints'])

            # check the first observation time against the reference initial time...
            if assimilation_configs['ref_initial_time'] > np.min(assimilation_configs['da_checkpoints']):
                raise ValueError("Some observation times or assimilation times are set before "
                                 "the time of the reference initial condition!")
        #
        else:
            # No valid checkpoints are passed.
            # All are created based on the given spacing and assume filtering is synchronous
            if assimilation_configs['num_filtering_cycles'] is None or assimilation_configs['da_time_spacing'] is None:
                raise ValueError("da_checkpoints and obs_checkpoints are not provided. "
                                 "The alternatives: da_time_spacing and num_filtering_cycles are not provided either!"
                                 "Filtering process needs one of the two alternatives!")
            else:
                ref_initial_time = assimilation_configs['ref_initial_time']
                da_time_spacing = assimilation_configs['da_time_spacing']
                num_da_cycles = assimilation_configs['num_filtering_cycles']
                eps = 1e-16
                assimilation_configs['obs_checkpoints'] = np.arange(ref_initial_time,
                                                                    ref_initial_time+da_time_spacing*num_da_cycles+eps,
                                                                    da_time_spacing)
                assimilation_configs['da_checkpoints'] = assimilation_configs['obs_checkpoints']
                assimilation_configs['synchronous'] = True

        return assimilation_configs

    #
    @staticmethod
    def validate_output_configs(output_configs, def_output_configs):
        """
        Aggregate the passed dictionaries with default configurations then make sure parameters are consistent.
        The first argument (output_configs) is validated, updated with missing entries, and returned.

        Args:
            output_configs: dict,
                A dictionary containing output configurations. This should be the output_configs dict
                passed to the constructor.

            def_output_configs: dict,
                A dictionary containing the default output configurations.

        Returns:
            output_configs: dict,
                Same as the first argument (output_configs) but validated, adn updated with missing entries.

        """
        output_configs = utility.aggregate_configurations(output_configs, def_output_configs)
        # screen output
        if output_configs['scr_output']:
            # screen output is turned on, make sure iterations are positive
            if output_configs['scr_output_iter'] is not None:
                if isinstance(output_configs['scr_output_iter'], float):
                    output_configs['scr_output_iter'] = np.int(output_configs['scr_output_iter'])
                if output_configs['scr_output_iter'] <= 0:
                    output_configs['scr_output'] = 1
                if not isinstance(output_configs['scr_output_iter'], int):
                    output_configs['scr_output'] = 1
        else:
            output_configs['scr_output_iter'] = np.infty  # just in case

        # file output
        if output_configs['file_output']:
            if output_configs['file_output_iter'] is not None:
                if isinstance(output_configs['file_output_iter'], float):
                    output_configs['file_output_iter'] = np.int(output_configs['file_output_iter'])
                if output_configs['file_output_iter'] <= 0:
                    output_configs['file_output'] = 1
                if not isinstance(output_configs['file_output_iter'], int):
                    output_configs['file_output'] = 1
            #
            if output_configs['file_output_dir'] is None:
                output_configs['file_output_dir'] = 'Results/Filtering_Results'  # relative to DATeS directory of course
            # Create the full path of the output directory ( if only relative dir is passed)
            dates_root_dir = os.getenv('DATES_ROOT_PATH')
            if not str.startswith(output_configs['file_output_dir'], dates_root_dir):
                output_configs['file_output_dir'] = os.path.join(dates_root_dir, output_configs['file_output_dir'])
            output_configs['file_output_dir'] = output_configs['file_output_dir'].rstrip('/ ')

            if output_configs['file_output_variables'] is None:
                output_configs['file_output'] = False
            for var in output_configs['file_output_variables']:
                if var not in def_output_configs['file_output_variables']:
                    raise ValueError("Unrecognized variable to be saved is not recognized!"
                                     "Received: %s" % var)
        else:
            output_configs['file_output_iter'] = np.infty  # just in case
            output_configs['file_output_dir'] = None
            output_configs['file_output_variables'] = None

        return output_configs
        #

    #
    def set_filtering_output_dir(self, file_output_dir, rel_to_root_dir=True, backup_existing=True):
        """
        Set the output directory of filtering results.

        Args:
            file_output_dir_path: path/directory to save results under
            rel_to_root_dir (default True): the path in 'output_dir_path' is relative to DATeS root dir or absolute
            backup_existing (default True): if True the existing folder is archived as *.zip file.

        Returns:
            None

        """
        # Make sure the directory is created and cleaned up at this point...
        if self.file_output:
            #
            dates_root_path = os.getenv('DATES_ROOT_PATH')
            if rel_to_root_dir:
                file_output_dir = os.path.join(dates_root_path, file_output_dir)
            #
            parent_path, out_dir = os.path.split(file_output_dir)
            utility.cleanup_directory(directory_name=out_dir, parent_path=parent_path, backup_existing=backup_existing)
            # Override output configurations of the filter:
            self.file_output_dir = file_output_dir
            self.output_configs['file_output_dir'] = file_output_dir
        else:
            file_output_dir = None
        #
        # Override output configurations of the filter:
        self.filter.output_configs['file_output'] = self.file_output
        self.filter.output_configs['file_output_dir'] = file_output_dir
        #
