
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



import os
import numpy as np

import dates_utility as utility
from models_base import ModelsBase


class SmootherBase(object):
    """
    A base class implementing common features of data assimilation variational smoothers (with one or more observation)
    
    The implementation in classes inheriting this base class should carry out smoothing over a single assimilation cycle.
    A single assimilation cycle may contain:
       - One observation --> 3D-Var
       - More than one observation --> 4D-Var
    
    Args:
        smoother_configs: dict, A dictionary containing smoother configurations.
            Supported configuarations:
                * smoother_name (default None): string containing name of the smoother; used for output.
                * model (default None):  model object
                * reference_time (default None): time instance at which the reference state is provided
                * reference_state(default None): model.state_vector object containing the reference/true state
                * window_bounds(default None): bounds of the assimilation window (should be iterable of lenght 2)
                * obs_checkpoints (default None): time instance at which observations are taken/collected
                * observations_list (default None): a list of model.observation_vector objects
                * analysis_timespan (default None): Cycle analysis_timespan. 
                                           This interval includes [t0, ..., tf], with lower and upper bound  equal to 
                                           the initial and final times of the assimilation window.
                                           This timespan is also used to evaluate RMSE over the window
                                           Observation timespan should lie withing this interval.
                * forecast_state: the forecast state of the model at the forecast time;
                * forecast_time (default None): time at which forecast state of the smoother is given, and used to 
                                                generate the analysis state.
                * analysis_time (default None): time at which analysis state of the smoother is generated.
                                                This should be equal to (or after) the forecast time at which the 
                                                forecast state is given.
                * analysis_state: placeholder of the analysis state. This refers to the first entry of the analysis trajectory
                * analysis_trajectory: a list of model states generated at the analysis_timespan
                * apply_preprocessing  (default False): call the pre-processing function before smoothing
                * apply_postprocessing (default False): call the post-processing function after smoothing
           
        output_configs: dict,
            A dictionary containing screen/file output configurations.
            Supported configuarations:
                * scr_output (default False): Output results to screen on/off switch
                * verbose (default False): This is used for extensive outputting e.g. while debugging
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default True): full path of the directory to output results in
                * file_output_variables (default ['smoother_statistics']): a list of variables to ouput. 
                        This is gonna be very much dependent on the smoother in hand.
              
    """
    # Default smoother configurations
    _def_smoother_configs = dict(smoother_name=None,
                                 model=None,
                                 reference_time=None,
                                 reference_state=None,
                                 window_bounds=None,
                                 obs_checkpoints=None,
                                 observations_list=None,
                                 forecast_time=None,
                                 forecast_state=None,
                                 analysis_time=None,
                                 analysis_state=None,
                                 analysis_timespan=None,
                                 analysis_trajectory=None,
                                 apply_preprocessing=False,
                                 apply_postprocessing=False
                                 )
    _def_output_configs = dict(scr_output=True,
                               verbose=False,
                               file_output=False,
                               file_output_dir=None,
                               file_output_variables=['smoother_statistics']
                               )
    local__time_eps = 1e-7
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
    
    
    def __init__(self, smoother_configs, output_configs=None):
        
        self.smoother_configs = self.validate_smoother_configs(smoother_configs, SmootherBase._def_smoother_configs)
        self.output_configs = self.validate_output_configs(output_configs, SmootherBase._def_output_configs)
        # self.model = self.smoother_configs['model']
        if self.output_configs['file_output'] and not os.path.isdir(self.output_configs['file_output_dir']):
                os.mkdir(self.output_configs['file_output_dir'])
        # After these basic steps, the specific smoother should continue with it's constructor steps
        self.model = self.smoother_configs['model']
        #
        self.smoother_configs['analysis_timespan'] = np.asarray(self.smoother_configs['analysis_timespan']).flatten()
        self.smoother_configs['obs_checkpoints'] = np.asarray(self.smoother_configs['obs_checkpoints']).flatten()
        #
        self._time_eps = SmootherBase.__time_eps
        
        try:
            self._verbose = self.output_configs['verbose']
        except(KeyError):
            self._verbose = False
        #
        
        
    #
    #
    def smoothing_cycle(self, update_reference=True):
        """
        Carry out a single assimilation cycle. The analysis has to be carried out first given the forecast 
        at the assimilation time, and the observations (Real observations), or synthetic created using 
        the reference state of the model.
        All required variables are obtained from 'smoother_configs' dictionary.
        This base method is designed to work in both ensemble, and variational framework. 
        You can override it for sure in your smoother.
        
        Args:
            update_reference (default True): bool,
                A flag to decide whether to update the reference state in the smoother or not.
                  
        """
        #
        # > ===========================( Assimilation Cycle Starts )================================ <
        #
        model = self.model
        #
        try:
            # Retrieve the reference state and evaluate initial root-mean-squared error
            reference_state = self.smoother_configs['reference_state'].copy()
            reference_time = self.smoother_configs['reference_time']
        except(NameError, KeyError, AttributeError):
            reference_state = reference_time = None
        finally:
            if reference_state is None or reference_time is None:
                reference_state = reference_time = None
                if self._verbose:
                    print("Couldn't retrieve the reference state/time! ")
                    print("True observations must be present in this case, and RMSE can't be evaluated!")
            
        #
        try:
            forecast_state = self.smoother_configs['forecast_state'].copy()
        except(NameError, KeyError, AttributeError):
            print("Couldn't retrieve the forecast state! This can't work; Aborting!")
            raise AssertionError
        #
        try:
            observations_list = self.smoother_configs['observations_list']
        except(NameError, KeyError, AttributeError):
            observations_list = None
        finally:
            if observations_list is None:
                print("Neither the reference state is passed, nor true observations list is found!")
                raise AssertionError
        # 
        try:
            obs_checkpoints = self.smoother_configs['obs_checkpoints']
        except:
            print("Either analysis_time or obs_checkpoints or both is/are missing! Aborting!")
        
        try:
            analysis_timespan = self.smoother_configs['analysis_timespan']
        except(NameError, KeyError, AttributeError):
            analysis_timespan = None
        finally:
            if analysis_timespan is None:
                obs_checkpoints = self.smoother_configs['obs_checkpoints']
                bounds = self.smoother_configs['window_bounds']
                t0, t1 = bounds[0], bounds[-1]
                tf, ta = self.smoother_configs['forecast_time'], self.smoother_configs['analysis_time']
                analysis_timespan = np.asarray(list(set.union(set(obs_checkpoints), {t0, t1, tf, ta})))
                analysis_timespan.sort()
                self.smoother_configs.update({'analysis_timespan':analysis_timespan})
                
        #
        # Forecast state/ensemble should be given:
        if forecast_state is None:
            try:
                forecast_state = utility.ensemble_mean(self.smoother_configs['forecast_ensemble'])
            except:
                print("Couldn't find either forecast state or forecast ensemble while analysis should be done first!")
                raise AssertionError
        
        # Check forecast, and analysis times:
        forecast_time = self.smoother_configs['forecast_time']
        analysis_time = self.smoother_configs['analysis_time']
        #
        if forecast_time is not None and analysis_time is not None:
            pass
            #
        elif forecast_time is None and analysis_time is None:
            print("Both forecast and analysis time are not set. At least specify the forecast time!")
            raise ValueError
            #
        elif analysis_time is None:
            if analysis_timespan is not None:
                analysis_time = analysis_timespan[0]
            else:
                # Analysis is carried out in the same place where forecast state is provided!
                analysis_time = forecast_time
        else:
            print("You have to specify where the provided forecast state is calculated!")
            raise ValueError
        
        #
        if analysis_time > forecast_time:
            print("Analysis time > forecast time. Forward propagation of the forecast state is taking place now.")
            #
            local_checkpoints = [forecast_time, analysis_time]
            tmp_trajectory = model.integrate_state(initial_state=forecast_state, checkpoints=local_checkpoints)
            if isinstance(tmp_trajectory, list):
                forecast_state = tmp_trajectory[-1].copy()
            else:
                forecast_state = tmp_trajectory.copy()
            forecast_time = analysis_time
            self.smoother_configs['forecast_time'] = local_checkpoints[-1]
            self.smoother_configs['forecast_state'] = forecast_state
            #
        elif analysis_time < forecast_time:
            print("Analysis time < Forecast time! Backward propagation of the state is Needed!")
            raise AssertionError
            #
        else:
            # All is Good; analysis_time = forecast_time
            pass
        
        #
        if forecast_time is not None and reference_time is not None:
            #
            if reference_time < forecast_time:
                local_checkpoints = [reference_time, forecast_time]
                tmp_trajectory = model.integrate_state(initial_state=reference_state, checkpoints=local_checkpoints)
                if isinstance(tmp_trajectory, list):
                    reference_state = tmp_trajectory[-1].copy()
                else:
                    reference_state = tmp_trajectory.copy()
                    #
                self.smoother_configs['reference_state'] = reference_state
                self.smoother_configs['reference_time'] = local_checkpoints[-1]
                #
            elif reference_time == forecast_time:
                pass
            else:
                print("Time settings are not conformable!")
                print("Reference time:", reference_time)
                print("Forecast time:", forecast_time)
                print("Analysis time:", analysis_time)
                raise AssertionError
                #
                
        #
        # calculate the initial RMSE: this is the RMSE before assimilation at the first entry of the time-span
        if reference_time is not None:
            # print("analysis_timespan", analysis_timespan)
            # print("reference_time", reference_time)
            if ( analysis_timespan[0] - reference_time) > self._time_eps:
                print("Reference time < initial time of the assimilation time-span. Forward propagation of the reference is taking place now.")
                print("reference_time", reference_time)
                print("analysis_timespan[0]", analysis_timespan[0])
                #
                local_checkpoints = [reference_time, analysis_timespan[0]]
                tmp_trajectory = model.integrate_state(initial_state=reference_state, checkpoints=local_checkpoints)
                if isinstance(tmp_trajectory, list):
                    reference_state = tmp_trajectory[-1].copy()
                else:
                    reference_state = tmp_trajectory.copy()
                reference_time = local_checkpoints[-1]
                reference_time = analysis_timespan[0]
                self.smoother_configs['reference_time'] = reference_time
                self.smoother_configs['reference_state'] = reference_state
                #
            elif( reference_time - analysis_timespan[0]) > self._time_eps:
                print("Reference time > the initial time of the assimilation time-span!")
                raise ValueError
            else:
                # All good...
                pass
        else:
            self.smoother_configs['reference_time'] = None
            self.smoother_configs['reference_state'] = None
        
        #
        if abs(analysis_time - analysis_timespan[0]) > self._time_eps:
            print("*"*100 + "\nWARNING: The analysis time is not at the beginning of the assimilation timespan!\n" + "*"*100)
            print("analysis_time", analysis_time)
            print("analysis_timespan[0]", analysis_timespan[0])
            print("\n" + "*"*100 + "\n")
        
        #
        if (analysis_time - obs_checkpoints[0] ) > self._time_eps:
            print("First observation has to be at, or after, the analysis time")
            raise AssertionError
        if len(obs_checkpoints) != len(observations_list):
            print("Observaions timespan must of be same length as the observations list!")
            raise AssertionError
            
        #
        if forecast_state is not None and reference_state is not None:
            if abs(analysis_time - reference_time) > self._time_eps:
                print("While evaluating initial RMSE; analysis_time != reference_time")
                print("reference_time: %f" % reference_time)
                print("Analysis time: %f" % analysis_time)
                raise ValueError
            else:
                initial_rmse = utility.calculate_rmse(forecast_state, reference_state)
        else:
            initial_rmse = 0
            
        #
        # Start the smoothing process: preprocessing -> smoothing(forecast->+<-anslsysis) -> postprocessing
        if self.smoother_configs['apply_preprocessing']:
            self.cycle_preprocessing()
        
        #
        # Analysis step (This calls the smoother's analysis step.)
        # > --------------------------------------------||
        sep = "\n" + "*"*80 + "\n"
        if self._verbose:
            print("%s ANALYSIS STEP %s" % (sep, sep) )
        self.analysis()
        # > --------------------------------------------||
        #
        
        
        # Evaluate RMSEs: This is potential redundancy; just read analysis trajectory from the smoother)
        reference_state = self.smoother_configs['reference_state'].copy()
        reference_time = self.smoother_configs['reference_time']
        #
        forecast_state = self.smoother_configs['forecast_state'].copy()
        forecast_time = self.smoother_configs['forecast_time']
        #
        analysis_state = self.smoother_configs['analysis_state'].copy()        
        analysis_time = self.smoother_configs['analysis_time']
        #
        analysis_trajectory = []
        analysis_trajectory.append(analysis_state)
        #
        
        state_size = model.state_size()
        
        # Forecast and analysis RMSE (at the beginning of the assimilation window)
        if abs(analysis_time - reference_time) > self._time_eps:
            print("While evaluating initial RMSE; analysis_time != reference_time")
            print("reference_time: %f" % reference_time)
            print("Analysis time: %f" % analysis_time)
            raise ValueError
        else:
            f_rmse = utility.calculate_rmse(reference_state, forecast_state, state_size)
            forecast_rmse_list = [f_rmse]
            forecast_times_list = [forecast_time]
            a_rmse = utility.calculate_rmse(reference_state, analysis_state, state_size)
            analysis_rmse_list = [a_rmse]
            analysis_times_list = [analysis_time]
        
        #
        if self._verbose:
            print("Initial (f_rmse, a_rmse) :", (f_rmse, a_rmse))
            print("analysis_state", analysis_state)
            print("forecast_state", forecast_state)
            print("reference_state", reference_state)
        
        #        
        #
        for t0, t1 in zip(analysis_timespan[:-1], analysis_timespan[1:]):
            local_checkpoints = np.array([t0, t1])
            #
            # Propagate forecast
            local_trajectory = model.integrate_state(forecast_state, local_checkpoints)
            if isinstance(local_trajectory, list):
                forecast_state = local_trajectory[-1].copy()
            else:
                forecast_state = local_trajectory.copy()
            # Propagate analysis
            local_trajectory = model.integrate_state(analysis_state, local_checkpoints)
            if isinstance(local_trajectory, list):
                analysis_state = local_trajectory[-1].copy()
            else:
                analysis_state = local_trajectory.copy()
            analysis_trajectory.append(analysis_state)
            #
            
            #
            if reference_state is not None:
                if self._verbose:
                    print("\n Evaluating RMSE, inside smoothers_base...:")
                    print("Subinterval: ", [t0, t1])
                    print("reference_time: ", reference_time)
                    print("reference_state: ", reference_state)
                
                # Propagate reference
                local_trajectory = model.integrate_state(reference_state, [reference_time, t1])
                reference_time = t1
                if isinstance(local_trajectory, list):
                    reference_state = local_trajectory[-1].copy()
                else:
                    reference_state = local_trajectory.copy()
                    #
                #
                f_rmse = utility.calculate_rmse(reference_state, forecast_state, state_size)
                a_rmse = utility.calculate_rmse(reference_state, analysis_state, state_size)
                forecast_rmse_list.append(f_rmse)
                analysis_rmse_list.append(a_rmse)
                forecast_times_list.append(t1)
                analysis_times_list.append(t1)
                #
                
                if self._verbose:
                    print("forecast_state: ", forecast_state)
                    print("analysis_state: ", analysis_state)
                
            else:
                # No reference state is provided. RMSE will be an empty list
                if self._verbose:
                    print("WARNING: No reference state is provided. RMSE will be an empty list!")
        
        #
        if reference_state is not None and update_reference:
            # If the reference time is below window final limit, march it forward:
            if (self.smoother_configs['window_bounds'][-1] - reference_time) > self._time_eps:
                tf = self.smoother_configs['window_bounds'][-1]
                local_trajectory = model.integrate_state(reference_state, [reference_time, tf])
                reference_time = tf
                if isinstance(local_trajectory, list):
                    reference_state = local_trajectory[-1]
                else:
                    reference_state = local_trajectory
                #                
            self.smoother_configs.update({'reference_state': reference_state})
            self.smoother_configs.update({'reference_time': reference_time})
            #
        
        #
        # Apply post-processing if required
        if self.smoother_configs['apply_postprocessing']:
            self.cycle_postprocessing()
        
        #
        # forecast now  (This calls the smoother's forecast step):
        # > --------------------------------------------||
        if self._verbose:
            print("%s ANALYSIS STEP %s" % (sep, sep) )
        self.forecast()
        # > --------------------------------------------||
        
        
        #
        # Update smoother statistics (including RMSE)
        if 'smoother_statistics' not in self.output_configs:
            self.output_configs.update(dict(smoother_statistics=dict(initial_rmse=None,forecast_rmse=None, analysis_rmse=None)))
        else:
            if 'analysis_rmse' not in self.output_configs['smoother_statistics']:
                self.output_configs['smoother_statistics'].update(dict(analysis_rmse=None))
            if 'forecast_rmse' not in self.output_configs['smoother_statistics']:
                self.output_configs['smoother_statistics'].update(dict(forecast_rmse=None))
            if 'initial_rmse' not in self.output_configs['smoother_statistics']:
                self.output_configs['smoother_statistics'].update(dict(initial_rmse=None))

        # now update the RMSE's
        self.output_configs['smoother_statistics'].update({'initial_rmse': initial_rmse})
        self.output_configs['smoother_statistics'].update({'forecast_rmse': forecast_rmse_list})
        self.output_configs['smoother_statistics'].update({'forecast_times': forecast_times_list})
        #
        self.output_configs['smoother_statistics'].update({'analysis_rmse': analysis_rmse_list})
        self.output_configs['smoother_statistics'].update({'analysis_times': analysis_times_list})

        # output and save results if requested
        if self.output_configs['scr_output']:
            self.print_cycle_results()
        if self.output_configs['file_output']:
            self.save_cycle_results()
        #
        # > ===========================( Assimilation Cycle End )=================================== <
        #

    
    #
    def forecast(self, *argv, **kwargs):
        """
        Forecast step of the smoother.
        
        Args:
                  
            
        """
        raise NotImplementedError

    #
    #
    def analysis(self, *argv, **kwargs):
        """
        Analysis step of the smoother.
        
        Args:
                  
            
        """
        raise NotImplementedError
    
    #
    #
    def cycle_preprocessing(self, *argv, **kwargs):
        """
        PreProcessing on the passed data before applying the data assimilation smoother cycle.
        Applied if needed based on the passed options in the configurations dictionary.
        
        Args:
                  
            
        """
        raise NotImplementedError()
    
    #
    #
    def cycle_postprocessing(self, *argv, **kwargs):
        """
        PostProcessing on the passed data before applying the data assimilation smoother cycle.
        Applied if needed based on the passed options in the configurations dictionary.
        
        Args:
                  
            
        """
        raise NotImplementedError()
    
    #
    #
    def save_cycle_results(self, *argv, **kwargs):
        """
        Save smoothing results from the current cycle to file(s).
        Check the output directory first. If the directory does not exist, create it.
        If the directory, and files exist, either new files will be created, 
        or the existing files will be appended. Implementation from smoother to another is expected to vary.
        
        Args:
                  
            
        """
        raise NotImplementedError
    
    #
    #
    def read_cycle_results(self, *argv, **kwargs):
        """
        Read smoothing results from file(s).
        Check the output directory first. If the directory does not exist, raise an IO error.
        If the directory, and files exist, Start retrieving the results properly
        This method will be useful for analyzing, printing, and plotting assimilation cycle results.
        
        Args:
                  
            
        """
        raise NotImplementedError
    
    #
    #
    def print_cycle_results(self):
        """
        Print smoothing results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested. Implementation from smoother to another is expected to vary.
        
        Args:
                  
            
        """
        forecast_time = self.smoother_configs['forecast_time']
        analysis_time = self.smoother_configs['analysis_time']
        forecast_rmse_list = self.output_configs['smoother_statistics']['forecast_rmse']
        analysis_rmse_list = self.output_configs['smoother_statistics']['analysis_rmse']
        
        
        # Get the analysis and forecat rmse (at the analsyis and forecast time instances)
        forecast_rmse = None
        analysis_rmse = None
        frcsterr_fnd = False
        anlsyserr_fnd = False
        prec = 1e-12
        #
        try:
            analysis_timespan = self.smoother_configs['analysis_timespan']
        except(NameError, KeyError, AttributeError):
            analysis_timespan = None
        finally:
            if analysis_timespan is None:
                print("Failed to retrieve analysis timespan form the smoother object!")
                raise ValueError
        #
        for t_ind in xrange(len(analysis_timespan)):
            t = analysis_timespan[t_ind]
            if (t-forecast_time)<prec:
                if frcsterr_fnd:
                    print("Something dangerous happend!")
                    print("Forecast time matches more than one instance in the analysis timespan with precision: %s" %repr(prec))
                else:
                    forecast_rmse = forecast_rmse_list[t_ind]
                    frcsterr_fnd = True
            if (t-analysis_time)<prec:
                if anlsyserr_fnd:
                    print("Something dangerous happend!")
                    print("Analysis time matches more than one instance in the analysis timespan with precision: %s" %repr(prec))
                else:
                    analysis_rmse = analysis_rmse_list[t_ind]
                    anlsyserr_fnd = True
            #
            if frcsterr_fnd and anlsyserr_fnd:
                break
            #
        
        if not (frcsterr_fnd and anlsyserr_fnd):
            sep = "=" * 40
            print(sep*2)
            print("Smoothing using: %s" % self.smoother_configs['smoother_name'])
            print(sep)
            
            print("TIMESPAN:", analysis_timespan)
            print("FORECAST RMSEs", forecast_rmse_list)
            print("ANALYSIS RMSEs", analysis_rmse_list)
            
            print(sep*2)
        else:
            print("Smoothing:%s: FORECAST[time:%5.3e > RMSE:%8.5e]  :: ANALYSIS[time:%5.3e > RMSE:%8.5e]"
                  % (self.smoother_configs['smoother_name'],
                     forecast_time, forecast_rmse,
                     analysis_time, analysis_rmse
                     )
                  )
    
    #
    # TODO: Revisit this too...
    @staticmethod
    def validate_smoother_configs(smoother_configs, def_smoother_configs):
        """
        Aggregate the passed dictionaries with default configurations then make sure parameters are consistent.
        The first argument (smoother_configs) is validated, updated with missing entries, and returned.
        
        Args:
            smoother_configs: dict,
                A dictionary containing smoother configurations. This should be the smoother_configs dict 
                passed to the constructor.
                
            def_smoother_configs: dict,
                A dictionary containing the default smoother configurations. 

        Returns:
            smoother_configs: dict,
                Same as the first argument (smoother_configs) but validated, adn updated with missing entries.
            
        """
        smoother_configs = utility.aggregate_configurations(smoother_configs, def_smoother_configs)
        if smoother_configs['smoother_name'] is None:
            smoother_configs['smoother_name'] = 'Unknown_'

        # Since aggregate never cares about the contents, we need to make sure now all parameters are consistent
        if smoother_configs['model'] is None:
            raise ValueError("You have to pass a reference to the model object so that"
                             "model observations can be created!")
        else:
            if not isinstance(smoother_configs['model'], ModelsBase):
                raise ValueError("Passed model is not an instance of 'ModelsBase'!. Passed: %s" %
                                 repr(smoother_configs['model']))
        return smoother_configs

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
        # Validating file output
        if output_configs['file_output']:
            if output_configs['file_output_dir'] is None:
                dates_root_dir = os.getenv('DATES_ROOT_PATH')
                directory_name = '_smoother_results_'
                tmp_dir = os.path.join(dates_root_dir, directory_name)
                # print("Output directory of the smoother is not set. Results are saved in: '%s'" % tmp_dir)
                output_configs['file_output_dir'] = tmp_dir
            else:
                dates_root_dir = os.getenv('DATES_ROOT_PATH')
                if not str.startswith(output_configs['file_output_dir'], dates_root_dir):
                    output_configs['file_output_dir'] = os.path.join(dates_root_dir, output_configs['file_output_dir'])

            if output_configs['file_output_variables'] is None:
                output_configs['file_output'] = False
            for var in output_configs['file_output_variables']:
                if var not in def_output_configs['file_output_variables']:
                    raise ValueError("Unrecognized variable to be saved is not recognized!"
                                     "Received: %s" % var)
        else:
            output_configs['file_output_dir'] = None
            output_configs['file_output_variables'] = None

        return output_configs


