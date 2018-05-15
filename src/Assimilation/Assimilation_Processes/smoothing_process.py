
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
    AssimilationProcess.SmoothingProcess:
    A class implementing functionalities of a smoothing process.
    A smoothing process here refers to repeating a smoothing cycle several times over identical or different assimilation windows.
    
    Here is how it works;
    The configurations dictionary must provide vlid values of the following keys:

    Mandatory keys in 'assimilation_configs':
    -----------------------------------------
        1-  'smoother': smoother (object of SmoothersBase or a derived class)
            'experiment_timespan': an iterable with two entries [t0, tf] where the assimilation windows are subintervals of these bounds.
        2-  'obs_checkpoints': an iterable with time instances at which observations are made
        3-  'da_checkpoints': beginning of assimilation windows (all entries must be exclusively between t0, tf).
        4-  'initial_forecast': forecast state (model.StateVector instance) at the initial time of the experiment time span, i.e. t0.
            

    Optional keys in 'assimilation_configs':
    -----------------------------------------    
        1-  'ref_initial_condition': reference/true state (model.StateVector instance) at the initial time of the experiment time span, i.e. t0.
            As explained next, this becomes mandatory if 'observations_list' is None.
        2-  'observations_list': None, or a list of model.ObservationVector instances. 
                Its length must match the lenght of 'obs_checkpoints'.
                If this is None, synthetic observations are calculated from the reference solution, which in this case becomes mandatory.
        3-  'random_seed': used as an initial random seed for numpy.random; this is set at the beginning of the experiment.
    
    
    Remarks:
    --------
        1-  The main methods here update the DA items and update the appropriate configs dictionaries in the smoother object.
        2-  The smoother object itself is responsible for input/output.
"""


import numpy as np
import os
# import shutil

import dates_utility as utility
from assimilation_process_base import AssimilationProcess
from smoothers_base import SmoothersBase
from models_base import ModelsBase


#
#
class SmoothingProcess(AssimilationProcess):
    """
    A class implementing the steps of a smoothing process.
    This recursively apply smoothing on several assimilation windows to update the initial condition.
    
    Smoothing process class constructor:
    The input configurarions are used to control the assimilation process behavious over several cycles/windows.
    
    Args:
        assimilation_configs: dict,
        A dictionary containing assimilation configurations.
        Supported configuarations:
            * smoother (default None): smoother/variational object
            * experiment_timespan (default None): this is the timespan of the whole experiment;
                based on the assimilation time instances, this timespan is subdivided into subsequent assimilation windows.
                    It has to be an iterable with two entries [t0, tf] where the assimilation windows are subintervals of these bounds.
            * obs_checkpoints (default None): iterable containing an observation timespan 
               - thsese are the time instances at which observation (synthetic or not) are given/generated
            * observations_list (default None): a list of observation vectors to be assimilated;
                this is useful when true observations are made
            * da_checkpoints (default None): an iterable containing an assimilation timespan;
                - thsese are the time instances at which analysis step is carried out;
                  i.e. these specify the bounds of the assimilation windows.
            * initial_forecast (default None): the background/forecast state given at the initial time of the experiment timespan
            * ref_initial_condition (default None): true state at some initial reference time,
            
            * random_seed (defaul None): reset the numpy.random seed before carrying out any smoothing steps.
            
        output_configs (default None); a dictionary containing screen/file output configurations:
        Supported configuarations:
        --------------------------
            * scr_output (default True): Output results to screen on/off switch
            * scr_output_iter (default 1): number of iterations/windows after which outputs are printed
            * file_output (default False): save results to file on/off switch
            * file_output_iter (default 1): number of iterations/windows after which outputs are saved to files
            * file_output_dir (default None): defaul directory in which results are written to files
            
    Returns:
        None
        
    """
    # Default smoothing process configurations
    _def_assimilation_configs = dict(smoother=None,
                                     experiment_timespan=None,
                                     obs_checkpoints=None,
                                     observations_list=None,
                                     da_checkpoints=None,
                                     initial_forecast=None,
                                     initial_ensemble=None,
                                     ref_initial_condition=None,
                                     random_seed=None  # Reset the random seed before carrying out any smoothing steps.
                                     )

    _def_output_configs = dict(scr_output=True,
                               scr_output_iter=1,
                               file_output=False,
                               file_output_iter=1,
                               file_output_dir=None,
                               file_output_variables=['model_states',  # we will see if this is adequate or not!
                                                      'observations',
                                                      'rmse',
                                                      'smoother_statistics'],
                               verbose=False
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
    def __init__(self, assimilation_configs=None, output_configs=None):
        
        # Validate and attach assimilation, and input configurations:
        self.assimilation_configs = self.validate_assimilation_configs(assimilation_configs, SmoothingProcess._def_assimilation_configs)
        self.output_configs = self.validate_output_configs(output_configs, SmoothingProcess._def_output_configs)
        #
        
        # Retrieve references to the model, and the smoother
        self.smoother = self.assimilation_configs['smoother']
        self.model = self.smoother.model
        
        # Compare timespan vs reference initial time, and integrate reference forward if needed!
        experiment_timespan = self.assimilation_configs['experiment_timespan']
        ref_initial_condition = self.assimilation_configs['ref_initial_condition']
        
        # Create (running) states and times for reference, forecast, and analysis;
        if ref_initial_condition is not None:
            
            # reference and forecast state information:
            self._running_reference_time = experiment_timespan[0]  # This is a forced-assumption
            self._running_reference_state = ref_initial_condition.copy()
            #
        else:
            self._running_reference_time = None
            self._running_reference_state = None
        
        # 
        self._running_forecast_time = experiment_timespan[0]
        self._running_forecast_state = self.assimilation_configs['initial_forecast'].copy()
        #
        self._running_analysis_time = self._running_forecast_time
        self._running_analysis_state = self.assimilation_configs['initial_forecast'].copy()
        
        #
        # a list of dictionaries that will hold essential information from each assimilation cycle
        self.assimilation_configs.update({'assimilation_windows_stats':[]})
        
        # Make sure the directory is created and cleaned up at this point...
        self.set_smoothing_output_dir(self.output_configs['file_output_dir'], rel_to_root_dir=True)
        #
        self._time_eps = SmoothingProcess.__time_eps
        self._verbose = self.output_configs['verbose']
        self.__initialized = True
        #
    
    #
    def assimilation_cycle(self, window_bounds=None,
                                 observations_list=None,
                                 obs_checkpoints=None,
                                 analysis_time=None,
                                 forecast_time=None,
                                 forecast_state=None,
                                 forecast_ensemble=None,
                                 reference_time=None,
                                 reference_state=None,
                                 analysis_timespan=None,
                                 scr_output=False,
                                 file_output=False,
                                 update_reference=True):
        """
        Carry out smoothing for the next assimilation cycle.
        The assimilation cycle is carried out by applying smoothing using smoother_object.smoothing_cycle().
        This function modifies the configurations dictionary of the smoother object to carry out a single 
        smoothing cycle.
        
        Args:
            window_bounds: beginning, and end of the assimilation window
            observations_list: iterable that contains the list of observations to be assimilated over this assimilation window
            obs_checkpoints: iterable that contains observation time instances
            analysis_time: scalar, 
                the time at which state update (analysis) is generated
            forecast_time: scalar,
                The time instance at which forecast step of the smoother is carried out. 
                This is supposed to be same as analysis time.
            forecast_state: model.state_vector,
                The background state (at the assimilation time
            
            reference_time: time at which the reference/true solution is given (if any)
            reference_state: reference/trues state at the reference time 'reference_time' (if any)
            analysis_timespan: an iterable that contains checkpoints at whcih analysis trajectory is returned
            scr_output (default False): bool,
                A boolean flag to cotrol whether to output results to the screen or not
                The per-cycle screen output options are set by the smoother configs.
            file_output (default False): bool,
                A boolean flag to cotrol whether to save results to file or not. 
                The per-cycle file output options are set by the smoother configs.
            update_reference (default False): bool,
                A flag to decide to whether to request updating the reference state in by the smoother.
                                       
        Returns:
            analysis_trajectory: a list of analysis states (evaluated at the time instances described by 'analysis_timespan'
                This list will contain only the analysis state if 'analysis_timespan' isn't given, i.e. is None.
        
        """
        # Update the configurations of the smoothing/variational object:                  
        # 1- update smoother assimilations configs (based on flag and iterations) prioritize local values:
        smoothing_configs_dict = {'reference_time':reference_time,
                                  'reference_state': reference_state,
                                  'window_bounds':window_bounds,
                                  'obs_checkpoints':obs_checkpoints,
                                  'observations_list':observations_list,
                                  'forecast_time':forecast_time,
                                  'forecast_state': forecast_state,
                                  'forecast_ensemble': forecast_ensemble,
                                  'analysis_time':analysis_time,
                                  'analysis_timespan':analysis_timespan,
                                  }
        self.smoother.smoother_configs.update(smoothing_configs_dict)
        #
        # 2- update output configs (based on flag and iterations) prioritize local values:
        self.smoother.output_configs.update({'scr_output':scr_output, 'file_output':file_output})
        
        #
        # Now start the assimilation cycle.
        self.smoother.smoothing_cycle(update_reference=update_reference)
        
        #
        # retrieve/construct and return analysis trajectory
        try:
            analysis_trajectory = self.smoother.smoother_configs['analysis_trajectory']
        except:
            analysis_trajectory = None
        finally:
            if analysis_trajectory is None:
                #
                analysis_state = self.smoother.smoother_configs['analysis_state']
                if analysis_timespan is not None:            
                    analysis_trajectory = self.model.integrate_state(initial_state=analysis_state, checkpoints=analysis_timespan)
                else:
                    analysis_trajectory = [analysis_state.copy()]
                self.smoother.smoother_configs.update({'analysis_trajectory':analysis_trajectory})
                #
            
        #
        return analysis_trajectory
        #
    
    #
    def recursive_assimilation_process(self, update_ref_here=True):
        """
        Loop over all assimilation cycles (consequtive assimilation windows) and output/save results 
        (forecast, analysis, observations) for all the assimilation cycles if requested.
        
        Args:
            update_ref_here (default False): bool,
                A flag to decide to whether to update the reference state here, 
                or request to updated it inside the smoother.
                  
        Returns:
            None
        
        """
        #
        sepp = "\n"*2 + "="*100 + "\n" + "*"*100 + "\n"*2
        #
        
        # > --------------------------------------------------------------------------------- < #
        # >                 START the squential assimilation/smoothing process                < #
        # > --------------------------------------------------------------------------------- < #
        
        # Extract assimilation elements from the configurations dictionary:
        obs_checkpoints = np.asarray(self.assimilation_configs['obs_checkpoints'])
        observations_list = self.assimilation_configs['observations_list']
        da_checkpoints = np.asarray(self.assimilation_configs['da_checkpoints'])
        experiment_timespan = self.assimilation_configs['experiment_timespan']
        
        # Check for observations; synthetic vs. real/actual: 
        if observations_list is None:
            observations_list = []
        elif not isinstance(observations_list, list):
            print("observations_list' must be a list, not %s" % type(observations_list))
            raise TypeError
        else:
            pass
        #
        if len(observations_list) == 0 :
            # Synthetic observations will be created (this requires the reference state to be available
            if self._running_reference_state is not None:
                synthetic_obs = True
            else:
                print("It is not possible to create synthetic observations without a referece/true solution!")
                raise AssertionError
        else:
            # Real/Actual observations are intended here;
            synthetic_obs = False
        #
        # Options/configurations are acceptable; proceed...
        #
        
        #
        # Retrieve the state of the random number generator, and set the seed
        random_seed = self.assimilation_configs['random_seed']
        if random_seed is not None:
            np_random_state = np.random.get_state()
            np.random.seed(random_seed)
        
        
        # Loop over all assimilation windows, and call the smoother for each:
        # * For each cycle, do the following:
        #   1) retrieve assimilation window time bounds
        #   2) retrieve or create observations over this assimilation window
        #   3) update the forecast state, and forecast time
        #   4) Check/update file/screen output settings for the current cycle
        #   5) Carry out a single smoothing cycle (analysis and forecast time are at the beginning of each window)
        #   6) Update the running analysis state, and running analysis time
        
        #
        # Number of assimilation cycles (windows):
        num_assim_windows = da_checkpoints.size
        #
        for wind_ind in xrange(num_assim_windows):
            #
            # 1) retrieve assimilation window time bounds: ([t0, tf])
            if wind_ind == num_assim_windows-1:
                # this is the last assimilation window:
                tf = experiment_timespan[-1]
                if (obs_checkpoints>=da_checkpoints[wind_ind]).any():
                    tf = max(obs_checkpoints[-1], tf)                    
                
                window_bounds = [da_checkpoints[wind_ind], tf]
                
            else:                
                # this is not the last assimilation window:
                window_bounds = [da_checkpoints[wind_ind], da_checkpoints[wind_ind+1]]
            
            if window_bounds[1] < window_bounds[0]:
                print("An error occured while setting window_bounds!")
                print("window_bounds[1] = %f < window_bounds[0] = %f" % (window_bounds[1], window_bounds[0]))
                raise ValueError
            
            #
            
            #
            # 2) retrieve or create observations over this assimilation window
            s0 = set(np.where((obs_checkpoints-window_bounds[0])>self._time_eps)[0])
            s1 = set(np.where((window_bounds[-1]-obs_checkpoints)>=self._time_eps)[0])
            local_obs_indexes = set.intersection(s0, s1)
            local_obs_indexes = np.asarray(list(local_obs_indexes))
            local_obs_indexes.sort()  # This shouldn't be needed!
            #
            local_obs_checkpoints = np.asarray([obs_checkpoints[i] for i in local_obs_indexes])
            
            #
            print(sepp + "\t Assimilating window # %d with bounds: " % (wind_ind+1) + str(window_bounds) + "\n")
            print("\t Assimilation observations at times:")
            print("\t\t %s \n " % str(local_obs_checkpoints) + sepp)
            #
            
            # Update the reference state if found; propagate to beginning of this window:
            # If 'update_ref_here' is False, the reference state is propagated to end of window inside smoother
            if update_ref_here and self._running_reference_time is not None:
                #
                tf = window_bounds[0]
                ref_state = self._running_reference_state
                ref_time = self._running_reference_time
                #
                if abs(ref_time-tf) > self._time_eps:
                    ref_trajectory = self.model.integrate_state(ref_state,[ref_time, tf])
                    if isinstance(ref_trajectory, list):
                        ref_state = ref_trajectory[-1]
                    else:
                        ref_state = ref_trajectory
                        ref_time = tf
                    self._running_reference_state = ref_state
                    self._running_reference_time = tf
                else:
                    # reference state is already at the beginning of this window...
                    pass
            
            #
            if synthetic_obs:
                # Synthesize observations given the reference solution:
                local_observations = []
                #
                if len(local_obs_checkpoints) > 0:
                    # 
                    ref_state = self._running_reference_state.copy()
                    ref_time = self._running_reference_time
                    #
                    for t in local_obs_checkpoints:
                        if (ref_time-t) > self._time_eps:
                            print(sepp + "WARNING: You are trying to propagate state backward in time")
                            print("Reference time: %f" % initial_time)
                            print("first observatio time: %f" % local_obs_checkpoints[0])
                            print(sepp)
                            tf = t
                            # raise ValueError  # should we let the model time-integrator handle it?!
                        elif (t-ref_time) > self._time_eps:
                            tf = t
                        else:
                            # there is an observation at the at ref_time already
                            tf = ref_time
                        #
                        if abs(ref_time-tf) > self._time_eps:
                            ref_trajectory = self.model.integrate_state(ref_state,[ref_time, tf])
                            if isinstance(ref_trajectory, list):
                                ref_state = ref_trajectory[-1]
                            else:
                                ref_state = ref_trajectory
                            ref_time = tf
                            #
                        else:
                            # Don't propagate forawrd; ref_time == observation time in turn
                            pass
                            
                        if self._verbose:
                            print("Generating synthetic observations at time: %f" % tf)
                            
                        obs = self.model.evaluate_theoretical_observation(ref_state)
                        obs = obs.add(self.model.observation_error_model.generate_noise_vec())
                        local_observations.append(obs)                        
                    #
            else:
                # retrieve actual/real observations:                
                local_observations = [observations_list[i] for i in local_obs_indexes]
                #
            
            
            # 3) update the forecast state (forward propagation of the previous analysis state), and forecast time
            # The analysis state is propagated forward to the beginning of the current window (if needed) 
            # to generate a forecast state...
            #
            if (window_bounds[0]-self._running_analysis_time) > self._time_eps:
                #
                self._running_forecast_time = window_bounds[0]
                
                # Propagate the analysis state (from previous cycle) to the beginning of this window:
                tmp_trajectory = self.model.integrate_state(self._running_analysis_state,
                                                            [self._running_analysis_time, window_bounds[0]]
                                                            )
                if isinstance(tmp_trajectory, list):
                    self._running_forecast_state = tmp_trajectory[-1]
                else:
                    self._running_forecast_state = tmp_trajectory
                
                #
            elif (self._running_analysis_time-window_bounds[0]) > self._time_eps:
                print("The forecast/analysis time must be at (or before) the beginning of the assimilation window!")
                raise ValueError
            else:
                # This should happen only in the first assimilation cycle
                if wind_ind != 0:
                    #
                    msg = """ %s 
                        \rWARNING: beginning of the window matches the running analysis time before the cycle begins!\n
                        \rRunning analysis at this stage should be equal to beginning of the previous  window!\n
                        \rThis should happen only in the first assimilation cycle %s\n
                        \r""" % (sepp, sepp)
                    print(msg)
                    #
                else:
                    pass
            
            #
            # 4) Check/update file/screen output settings for the current cycle:
            #
            if self.output_configs['scr_output']:
                if (wind_ind % self.output_configs['scr_output_iter']) == 0:
                    scr_output = True  # switch on/off screen output in the current smoothing cycle
                else:
                    scr_output = False
            else:
                scr_output = False  # switch on/off screen output in the current smoothing cycle

            if self.output_configs['file_output']:
                if (wind_ind % self.output_configs['file_output_iter']) == 0:
                    file_output = True  # switch on/off file output in the current smoothing cycle
                else:
                    file_output = False
            else:
                file_output = False  # switch on/off screen output in the current smoothing cycle
            
            #
            # =========================================== 
            # 5) Carry out a single smoothing cycle:
            # ===========================================
            
            # Discard observations at the beginning of the window for windows after the initial one
            # This is necessary to avoid reassimilating observations
            if wind_ind > 0 and len(local_obs_checkpoints) >= 1:
                if abs(local_obs_checkpoints[0] - window_bounds[0]) < self._time_eps:
                    local_obs_checkpoints = local_obs_checkpoints[1:]
                    local_observations.pop(0)
            else:
                # On the first window, it is OK to use initial observation...
                pass
            
            # get an analysis timespan for bookkeeping:
            analysis_checkpoints = np.asarray(list(set.union(set(window_bounds), set(local_obs_checkpoints))))
            analysis_checkpoints.sort()
            
            local_forecast_time = local_analysis_time = window_bounds[0]
            if len(local_observations) > 0:
                # Found some observations; start Assimilation over this window:
                
                if wind_ind == 0:
                    forecast_ensemble = self.assimilation_configs['initial_ensemble']
                else:
                    forecast_ensemble = self.smoother.smoother_configs['forecast_ensemble']
                
                analysis_trajectory = self.assimilation_cycle(window_bounds=window_bounds,
                                                              observations_list=local_observations,
                                                              obs_checkpoints=local_obs_checkpoints,
                                                              analysis_time=local_analysis_time,
                                                              forecast_time=local_forecast_time,
                                                              forecast_state=self._running_forecast_state,
                                                              forecast_ensemble=forecast_ensemble,
                                                              reference_time=self._running_reference_time,
                                                              reference_state=self._running_reference_state,
                                                              analysis_timespan=analysis_checkpoints,
                                                              scr_output=scr_output,
                                                              file_output=file_output,
                                                              update_reference=not update_ref_here)
                
                
                analysis_state = self.smoother.smoother_configs['analysis_state']
                #
            else:
                # No observations on this window, no assimilation needed; updated analysis state
                print(" No observations to assimilate over window # %d \n" % wind_ind)
                analysis_state = self._running_forecast_state
                analysis_trajectory = []
                analysis_trajectory.append(analysis_state)
                #
            #
            # Analysis step is done;
            # update running analysis state and time to be used by the next cycle
            self._running_analysis_time = window_bounds[0]
            self._running_analysis_state = analysis_state.copy()
            #
            
            
            if not update_ref_here:
                # get udated reference state and time from the smoother configs
                self._running_reference_time = self.smoother.smoother_configs['reference_time']
                self._running_reference_state = self.smoother.smoother_configs['reference_state'].copy()
            else:
                # reference state is already at the beginning of this window...
                pass
                    
            
            # Update the assimilation statistics/diagnostic list:
            # this is added for convenience (will be removed if it creates redundancy!)
            d = self._construct_time_settings_dict()
            d.update(dict(window_bounds = window_bounds,
                          obs_checkpoints = local_obs_checkpoints,
                          observations_list = local_observations,
                          forecast_time = local_forecast_time,
                          forecast_state = self._running_forecast_state,
                          analysis_time = local_analysis_time,
                          analysis_state = self._running_analysis_state,
                          analysis_timespan=analysis_checkpoints,
                          analysis_trajectory=analysis_trajectory,
                          reference_time = self._running_reference_time,
                          reference_state = self._running_reference_state,
                          synthetic_obs=synthetic_obs,
                          update_ref_here=update_ref_here
                         ))
            self.assimilation_configs['assimilation_windows_stats'].append(d)
            
            
            # print diagnostics if needed!
            if self._verbose:
                sep = "\n" + "="*100 + "\n"
                print("%s \t Diagnostings Assimilation/Smoothing on Window number [%d] %s" % (sep, wind_ind, sep))
                for key in d:
                    print(key),
                    print(" :")
                    print(d[key])
                    print("----------------------------------")
                print("%s \t\t\t ...End of Diagnostics... %s" % (sep, sep))
            
            #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            #   End looping over assimilation windows   #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            #
        
        #
        # Retrieve the state of the random number generator
        if random_seed is not None:
            np.random.set_state(np_random_state)
            #
            
        #    
        # > --------------------------------------------------------------------------------- < #
        # >                 END the squential assimilation/smoothing process                  < # 
        # > --------------------------------------------------------------------------------- < #
            
            
    #
    @staticmethod    
    def _construct_time_settings_dict():
        d = dict(window_bounds=None,  # beginning, and end of the assimilation window
                 synthetic_obs=None,
                 observations_list=None,
                 obs_checkpoints=None,
                 analysis_time=None,  # this is the analysis time
                 update_ref_here=None
                 )
        return d
        #
       
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
        
        # Assumptions (to be validated by the validation static methods down here)... 
        #   1- Assimilation timepoints are unique and are within the timespan bounds. 
        #      
        #   3- 
        
        
        # Now, validate the configurations, e.g. check for consistency of timespans:
        mandatory_keys = ['smoother', 
                          'experiment_timespan', 
                          'obs_checkpoints', 
                          'da_checkpoints', 
                          'initial_forecast']
        missing_keys = []
        for key in mandatory_keys:
            if key not in assimilation_configs:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            print("The following keys are missing from the dictionary 'assimilation_configs':\n \
                   \r%s \n\n \tTerminating...\n" % str(missing_keys))
            raise AssertionError
        
        if assimilation_configs['smoother'] is None:
            print("A smoother object must be passed to the constructor; None is found!")
            print("Terminating...")
            raise AssertionError
        
        if assimilation_configs['initial_forecast'] is None:
            if 'initial_ensemble' in assimilation_configs:
                initial_ensemble = assimilation_configs['initial_ensemble']
            else:
                initial_ensemble = None
            
            if initial_ensemble is None:
                print("The initial forecast has to be passed to the constructor;")
                print("Alternatively, an 'initial_ensemble' should be passed, and the average is taken!")
                print("Terminating...")
                raise AssertionError
            else:
                initial_forecast = utility.ensemble_mean(initial_ensemble)
                assimilation_configs.update({'initial_forecast': initial_forecast})
        
        #    
        # Check the experiment timespan vs observation timespan:
        experiment_timespan = assimilation_configs['experiment_timespan']
        obs_checkpoints = assimilation_configs['obs_checkpoints']
        da_checkpoints = assimilation_configs['da_checkpoints']
        
        if experiment_timespan is None:
            print("experiment_timespan is not passed! You have to specify at least the bounds of the experiment timespan")
            print("Terminating...")
            raise AssertionError
        else:
            if len(experiment_timespan) > 2:
                print("WARNING: The 'experiment_timespan' key in the assimilation_configs dictionary must be an iterable of length 2, with [t0, tf] being the bounds of the whole experiment timespan")
                print("Since you have longer timespan, I will take the firs, and last entries as t0, tf respectively!")
                experiment_timespan = [experiment_timespan[0], experiment_timespan[1]]
            else:
                print(" The 'experiment_timespan' key in the assimilation_configs dictionary must be an iterable of length 2,\n with [t0, tf] being the bounds of the whole experiment timespan")
                print("If you choose to give longer timespan [t0, t1, ..., tf], the bounds will be taken as t0, tf\n")
                print("\nExtracting from da_checkpoints, obs_checkpoints")
                #
                tspan = np.asarray(list( set.union(set(da_checkpoints), set(obs_checkpoints)) ))
                t0, tf = tspan.min(), tspan.max()
                experiment_timespan = [t0, tf]
                #
                print("Experiment bounds are set to: t0=%f, tf=%f \n" %(t0, tf))
                print("NOTE THAHT: in this case, the reference state, forecast state, and analysis state are all considered to be available at t0=%f\n" % t0)
                # raise AssertionError
            
        if obs_checkpoints is None:
            print("obs_checkpoints is not passed! This must be provided to know where (in time) measurements are collected")
            print("Terminating...")
            raise AssertionError
            
        if da_checkpoints is None:
            print("da_checkpoints is not passed! This must be provided to know where smoothings/analysis is carried out!")
            print("Terminating...")
            raise AssertionError
        
        #
        if da_checkpoints[-1] > obs_checkpoints[-1]:
            print("The last assimilation time point has to be below, or at most at, the last observation time instance!")
            print("Terminating...")
            raise AssertionError
                    
        if da_checkpoints[0] > obs_checkpoints[0]:
            print("da_checkpoints[0] > obs_checkpoints[0]")
            print("Adjusting the first assimilation time point to make use of available observation!")
            da_checkpoints[0] = obs_checkpoints[0]
            assimilation_configs.update({'da_checkpoints': da_checkpoints})
        
        #   
        adj = False         
        # experiment timespan vs observation timespan:
        if experiment_timespan[0] > obs_checkpoints[0]:
            experiment_timespan[0] = obs_checkpoints[0]
            adj = True
        if experiment_timespan[-1] < obs_checkpoints[-1]:
            experiment_timespan[-1] = obs_checkpoints[-1]
            adj = True
        if experiment_timespan[0] > da_checkpoints[0]:
            experiment_timespan[0] = da_checkpoints[0]
            adj = True
        if adj:
            print("The experiment timespan bounds are updated to accomodate obs_checkpoints, and da_checkpoints")
            print("experiment_timespan now is: "),
            print(experiment_timespan)
        
        # Now update the experiment_timspan in the configurations dictionary
        assimilation_configs.update({'experiment_timespan': experiment_timespan})
        
        #
        if 'observations_list' not in assimilation_configs:
            assimilation_configs.update({'observations_list':None})
        
        observations_list = assimilation_configs['observations_list']
        if observations_list is None:
            pass
        elif isinstance(observations_list, list):
            if len(observations_list) == 0:
                assimilation_configs.update({'observations_list':None})
            else:
                if len(observations_list) != len(obs_checkpoints):
                    print("Number of observation time instances must match passed observations!")
                    raise AssertionError
        else:
            print("This shouldn't happen!\n\t'observations_list' has to be a LIST of observations vectors or None! ")
            raise TypeError
        
        #    
        # check reference state, and reference time:
        if 'ref_initial_condition' not in assimilation_configs:
            # This shouldn't happen actually!
            assimilation_configs.update({'ref_initial_condition': None})
        
        # check availability of either observations list, or reference/true solution
        if observations_list is None and assimilation_configs['ref_initial_condition'] is None:
            print("If the observations list is None, you must provide an initial true/reference solution to be used for creating synthetic observations!")
            raise AssertionError
        #
        
        return assimilation_configs
        #
    
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
                output_configs['file_output_dir'] = 'Results/Smoothing_Results'  # relative to DATeS directory of course
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
    def set_smoothing_output_dir(self, file_output_dir, rel_to_root_dir=True, backup_existing=True):
        """
        Set the output directory of smoothing results. 
        
        Args:
            file_output_dir_path: path/directory to save results under
            rel_to_root_dir (default True): the path in 'output_dir_path' is relative to DATeS root dir or absolute
            backup_existing (default True): if True the existing folder is archived as *.zip file.

        Returns:
            None
            
        """
        # Make sure the directory is created and cleaned up at this point...
        if self.output_configs['file_output']:
            #
            dates_root_path = os.getenv('DATES_ROOT_PATH')
            if rel_to_root_dir:
                file_output_dir = os.path.join(dates_root_path, file_output_dir)
            #
            parent_path, out_dir = os.path.split(file_output_dir)
            utility.cleanup_directory(directory_name=out_dir, parent_path=parent_path, backup_existing=True)
            # Override output configurations of the smoother:
            self.output_configs['file_output_dir'] = file_output_dir
        else:
            file_output_dir = None
        #
        # Override output configurations of the smoother:
        self.smoother.output_configs['file_output'] = self.output_configs['file_output']
        self.smoother.output_configs['file_output_dir'] = file_output_dir
        #


