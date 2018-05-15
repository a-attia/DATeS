
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
    FiltersBase:
    A base class for filtering schemes.
    The implementation in classes inheriting this base class should carry out filtering over a single assimilation cycle.
"""


import os
import numpy as np

import dates_utility as utility
from models_base import ModelsBase


class FiltersBase(object):
    """
    A base class implementing common features of data assimilation filters.
    
    Basic portion of the filter constructor.
    This should be inserted/called at the beginning of the constructor of each filter class.
    
    Args:
        filter_configs: dict,
            A dictionary containing filter configurations.
            Supported configuarations:
                * filter_name (default None): string containing name of the filter; used for output.
                * model (default None):  model object
                * reference_time (default None): time instance at which the reference state is provided
                * reference_state(default None): model.state_vector object containing the reference/true state
                * observation_time (default None): time instance at which observation is taken/collected
                * observation (default None): model.observation_vector object
                * timespan (default None): Cycle timespan. 
                                           This interval includes observation, forecast, & analysis times
                * analysis_time (default None): time at which analysis step of the filter is carried out
                * forecast_time (default None): time at which forecast step of the filter is carried out
                * forecast_first (default True): A bool flag; Analysis then Forecast or Forecast then Analysis
                * apply_preprocessing (default False): call the pre-processing function before filtering
                * apply_postprocessing (default False): call the post-processing function after filtering
           
        output_configs: dict,
            A dictionary containing screen/file output configurations.
            Supported configuarations:
                * scr_output (default False): Output results to screen on/off switch
                * verbose (default False): This is used for extensive outputting e.g. while debugging
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default True): full path of the directory to output results in
                * file_output_variables (default ['filter_statistics']): a list of variables to ouput. 
                        This is gonna be very much dependent on the filter in hand.
              
    Returns:
        None
        
    """
    # Default filter configurations
    _def_filter_configs = dict(filter_name=None,
                               model=None,
                               reference_time=None,
                               reference_state=None,
                               observation_time=None,
                               observation=None,
                               timespan=None,
                               analysis_time=None,
                               forecast_time=None,
                               forecast_first=True,
                               apply_preprocessing=False,
                               apply_postprocessing=False
                               )
    _def_output_configs = dict(scr_output=True,
                               verbose=False,
                               file_output=False,
                               file_output_dir=None,
                               file_output_variables=['filter_statistics']
                               )

    def __init__(self, filter_configs, output_configs=None):
        
        self.filter_configs = self.validate_filter_configs(filter_configs, FiltersBase._def_filter_configs)
        self.output_configs = self.validate_output_configs(output_configs, FiltersBase._def_output_configs)
        # self.model = self.filter_configs['model']
        if self.output_configs['file_output'] and not os.path.isdir(self.output_configs['file_output_dir']):
            os.mkdir(self.output_configs['file_output_dir'])
        # After these basic steps, the specific filter should continue with it's constructor steps
        
        try:
            self._verbose = self.output_configs['verbose']
        except(KeyError):
            self._verbose = False

    #
    #
    def filtering_cycle(self, update_reference=True):
        """
        Carry out a single assimilation cycle. Forecast followed by analysis or the other way around.
        All required variables are obtained from 'filter_configs' dictionary.
        This base method is designed to work in both ensemble, and standard framework. 
        You can override it for sure in your filter.
        
        Args:
            update_reference (default True): bool,
                A flag to decide whether to update the reference state in the filter or not.
                  
        Returns:
            None
            
        """
        model = self.filter_configs['model']
        # reference_state = self.filter_configs['reference_state']  # always given at the initial time of the timespan
        # print('reference_state', reference_state)
        timespan = self.filter_configs['timespan']
        observation_time = self.filter_configs['observation_time']
        forecast_time = self.filter_configs['forecast_time']
        analysis_time = self.filter_configs['analysis_time']
        forecast_first = self.filter_configs['forecast_first']
        apply_preprocessing = self.filter_configs['apply_preprocessing']
        apply_postprocessing = self.filter_configs['apply_postprocessing']

        # calculate the initial RMSE: this is the RMSE before assimilation at the first entry of the time-span
        try:
            reference_time = self.filter_configs['reference_time']
        except(KeyError, ValueError, AttributeError, NameError):
            raise ValueError("Couldn't find the reference time in the configurations dictionary")
        else:
            if reference_time != timespan[0]:
                raise ValueError("Reference time does not match the initial time of the reference time-span!")
        #
        #
        if forecast_first:
            # Analysis ensemble should be given first in this case
            try:
                initial_state = self.filter_configs['analysis_state']
            except(KeyError, ValueError, AttributeError, NameError):
                try:
                    initial_state = utility.ensemble_mean(self.filter_configs['analysis_ensemble'])
                except:
                    raise ValueError("Couldn't find either analysis state or analysis ensemble while forecast should be done first!")
            finally:
                if initial_state is None:
                    try:
                        initial_state = utility.ensemble_mean(self.filter_configs['analysis_ensemble'])
                    except:
                        raise ValueError("Couldn't find either analysis state or analysis ensemble while forecast should be done first!")
        else:
            # Forecast ensemble should be given first in this case
            try:
                initial_state = self.filter_configs['forecast_state']
            except:
                try:
                    initial_state = utility.ensemble_mean(self.filter_configs['forecast_ensemble'])
                except:
                    raise ValueError("Couldn't find either forecast state or forecast ensemble while analysis should be done first!")
            finally:
                if initial_state is None:
                    try:
                        initial_state = utility.ensemble_mean(self.filter_configs['forecast_ensemble'])
                    except:
                        raise ValueError("Couldn't find either forecast state or forecast ensemble while analysis should be done first!")

        # Retrieve the reference state and evaluate initial root-mean-squared error
        reference_state = self.filter_configs['reference_state'].copy()
        initial_rmse = utility.calculate_rmse(initial_state, reference_state)

        #
        # Start the filtering process: preprocessing -> filtering(forecast->+<-anslsysis) -> postprocessing
        if apply_preprocessing:
            self.cycle_preprocessing()

        
        if not forecast_first and analysis_time != min(forecast_first, analysis_time, observation_time):
            # this is a double check!
            raise ValueError("While ANALYSIS should be done first, confusion occurred with times given!\n"
                             "\tCycle timespan:%s"
                             "\tObservation time: %f\n"
                             "\tForecast time: %f\n"
                             "\tAnalysis time: %f" % (repr(timespan), observation_time, forecast_time, analysis_time))
        elif not forecast_first:
            #
            # print("\n\n\n\n\n\n ANALYSIS FIRTS \n\n\n\n\n")
            # analysis should be carried out first in this case
            state_size = self.filter_configs['model'].state_size()
            analysis_rmse = utility.calculate_rmse(reference_state, analysis_state, state_size)
            analysis_time = self.filter_configs['analysis_time']
            
            # Analysis step
            self.analysis()
            #

            # update the reference state
            try:
                reference_time = self.filter_configs['reference_time']
            except:
                raise ValueError("Couldn't find reference time in the configurations dictionary")
            else:
                if reference_time != timespan[0] or reference_time != analysis_time:
                    raise ValueError("Reference time does not match the initial time of the reference time-span!")
            local_checkpoints = [analysis_time, forecast_time]

            tmp_trajectory = model.integrate_state(initial_state=reference_state, checkpoints=local_checkpoints)
            if isinstance(tmp_trajectory, list):
                reference_state = tmp_trajectory[-1].copy()
            else:
                reference_state = tmp_trajectory.copy()
            reference_time = local_checkpoints[-1]
            
            # forecast now:
            self.forecast()
            
            forecast_rmse = utility.calculate_rmse(reference_state, forecast_state, state_size)

            # update the reference state Moved to the process
            if update_reference:
                self.filter_configs['reference_state'] = reference_state.copy()
                self.filter_configs['reference_time'] = reference_time
                
        else:
            # forecast should be done first
            # print("\n\n\n\n\n\n FORECAST FIRTS \n\n\n\n\n")
            state_size = self.filter_configs['model'].state_size()
            try:
                reference_time = self.filter_configs['reference_time']
            except:
                raise ValueError("Couldn't find reference time in the configurations dictionary")
            else:
                if reference_time != timespan[0]:
                    raise ValueError("Reference time does not match the initial time of the reference time-span!")
            local_checkpoints = [reference_time, timespan[-1]]
            
            # Forecast step:
            self.forecast()
            # 
            try:
                forecast_state = self.filter_configs['forecast_state']
            except (NameError, AttributeError):
                raise NameError("forecast_state must be updated by the filter "
                                "and added to 'self.filter_configs'!")
            
            tmp_trajectory = model.integrate_state(initial_state=reference_state, checkpoints=timespan)
            if isinstance(tmp_trajectory, list):
                up_reference_state = tmp_trajectory[-1].copy()
            else:
                up_reference_state = tmp_trajectory.copy()
            reference_time = local_checkpoints[-1]
            
            # Update the reference state
            if update_reference:
                self.filter_configs['reference_state'] = up_reference_state.copy()
                self.filter_configs['reference_time'] = reference_time
                
            # observation = self.filter_configs['observation']
            #
            forecast_rmse = utility.calculate_rmse(up_reference_state, forecast_state, state_size)
            
            # Analysis step:
            self.analysis()
            #
            try:
                analysis_state = self.filter_configs['analysis_state']
            except (NameError, AttributeError):
                raise NameError("analysis_state must be updated by the filter "
                                "and added to 'self.filter_configs'!")
            
            analysis_rmse = utility.calculate_rmse(up_reference_state, analysis_state, state_size)

            #

        # Apply post-processing if required
        if apply_postprocessing:
            self.cycle_postprocessing()

        # Update filter statistics (including RMSE)
        if 'filter_statistics' not in self.output_configs:
            self.output_configs.update(dict(filter_statistics=dict(initial_rmse=None,forecast_rmse=None, analysis_rmse=None)))
        else:
            if 'analysis_rmse' not in self.output_configs['filter_statistics']:
                self.output_configs['filter_statistics'].update(dict(analysis_rmse=None))
            if 'forecast_rmse' not in self.output_configs['filter_statistics']:
                self.output_configs['filter_statistics'].update(dict(forecast_rmse=None))
            if 'initial_rmse' not in self.output_configs['filter_statistics']:
                self.output_configs['filter_statistics'].update(dict(initial_rmse=None))

        # now update the RMSE's
        self.output_configs['filter_statistics']['initial_rmse'] = initial_rmse
        self.output_configs['filter_statistics']['forecast_rmse'] = forecast_rmse
        self.output_configs['filter_statistics']['analysis_rmse'] = analysis_rmse

        # output and save results if requested
        if self.output_configs['scr_output']:
            self.print_cycle_results()
        if self.output_configs['file_output']:
            self.save_cycle_results()

    #
    #
    def forecast(self, *argv, **kwargs):
        """
        Forecast step of the filter.
        
        Args:
                  
        Returns:
            None
            
        """
        raise NotImplementedError

    #
    #
    def analysis(self, *argv, **kwargs):
        """
        Analysis step of the filter.
        
        Args:
                  
        Returns:
            None
            
        """
        raise NotImplementedError
    
    #
    #
    def cycle_preprocessing(self, *argv, **kwargs):
        """
        PreProcessing on the passed data before applying the data assimilation filter cycle.
        Applied if needed based on the passed options in the configurations dictionary.
        
        Args:
                  
        Returns:
            None
            
        """
        raise NotImplementedError()
    
    #
    #
    def cycle_postprocessing(self, *argv, **kwargs):
        """
        PostProcessing on the passed data before applying the data assimilation filter cycle.
        Applied if needed based on the passed options in the configurations dictionary.
        
        Args:
                  
        Returns:
            None
            
        """
        raise NotImplementedError()
    
    #
    #
    def save_cycle_results(self, *argv, **kwargs):
        """
        Save filtering results from the current cycle to file(s).
        Check the output directory first. If the directory does not exist, create it.
        If the directory, and files exist, either new files will be created, 
        or the existing files will be appended. Implementation from filter to another is expected to vary.
        
        Args:
                  
        Returns:
            None
            
        """
        raise NotImplementedError
    
    #
    #
    def read_cycle_results(self, *argv, **kwargs):
        """
        Read filtering results from file(s).
        Check the output directory first. If the directory does not exist, raise an IO error.
        If the directory, and files exist, Start retrieving the results properly
        This method will be useful for analyzing, printing, and plotting assimilation cycle results.
        
        Args:
                  
        Returns:
            None
            
        """
        raise NotImplementedError
    
    #
    #
    def print_cycle_results(self):
        """
        Print filtering results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested. Implementation from filter to another is expected to vary.
        
        Args:
                  
        Returns:
            None
            
        """
        forecast_time = self.filter_configs['forecast_time']
        analysis_time = self.filter_configs['analysis_time']
        forecast_rmse = self.output_configs['filter_statistics']['forecast_rmse']
        analysis_rmse = self.output_configs['filter_statistics']['analysis_rmse']
        print("Filtering:%s: FORECAST[time:%5.3e > RMSE:%8.5e]  :: ANALYSIS[time:%5.3e > RMSE:%8.5e]"
              % (self.filter_configs['filter_name'],
                 forecast_time, forecast_rmse,
                 analysis_time, analysis_rmse
                 )
              )
    
    #
    #
    @staticmethod
    def validate_filter_configs(filter_configs, def_filter_configs):
        """
        Aggregate the passed dictionaries with default configurations then make sure parameters are consistent.
        The first argument (filter_configs) is validated, updated with missing entries, and returned.
        
        Args:
            filter_configs: dict,
                A dictionary containing filter configurations. This should be the filter_configs dict 
                passed to the constructor.
                
            def_filter_configs: dict,
                A dictionary containing the default filter configurations. 

        Returns:
            filter_configs: dict,
                Same as the first argument (filter_configs) but validated, adn updated with missing entries.
            
        """
        filter_configs = utility.aggregate_configurations(filter_configs, def_filter_configs)
        if filter_configs['filter_name'] is None:
            filter_configs['filter_name'] = 'Unknown_'

        # Since aggregate never cares about the contents, we need to make sure now all parameters are consistent
        if filter_configs['model'] is None:
            raise ValueError("You have to pass a reference to the model object so that"
                             "model observations can be created!")
        else:
            if not isinstance(filter_configs['model'], ModelsBase):
                raise ValueError("Passed model is not an instance of 'ModelsBase'!. Passed: %s" %
                                 repr(filter_configs['model']))
        return filter_configs

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
                directory_name = '_filter_results_'
                tmp_dir = os.path.join(dates_root_dir, directory_name)
                # print("Output directory of the filter is not set. Results are saved in: '%s'" % tmp_dir)
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


