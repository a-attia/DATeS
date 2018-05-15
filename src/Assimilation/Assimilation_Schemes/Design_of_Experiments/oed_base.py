
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



class PriorBase(object):
    """
    A base class for prior distribution of a model state...
    Derived classes construct a prior-distribution object. This should inherit the model-based background error model!
    It is used in the Bayesian approach for regularization
    
    """
    def __init__():
        raise NotImplementedError
        
        

class OEDBase(object):
    """
    A base class implementing common features of optimal experimental design problems (optimal design of experiments).
        
    Args:
        oed_configs: dict, A dictionary containing configurations of the OED object.
            Supported configuarations:
                * oed_name (default None): string containing name of the OED object; used for output.
                * model (default None):  model object
                * window_bounds(default None): bounds of the experimental design window (should be iterable of lenght 2)
                
                * observations_list (default None): a list of model.observation_vector objects;
                * obs_checkpoints (default None): time instance at which observations are taken/collected.
                    These are necessaryt for adjoint calculation
                * reference_time (default None): time instance at which the reference state is provided (if available);
                * reference_state(default None): model.state_vector object containing the reference/true state;
                    this is provided only if observations_list is not available, and is used to create synthetic observations!
                * initial_design: the initial design vector
                * optimal_design: the optimal experimental design
                
        output_configs: dict,
            A dictionary containing screen/file output configurations.
            Supported configuarations:
                * scr_output (default False): Output results to screen on/off switch
                * verbose (default False): This is used for extensive outputting e.g. while debugging
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default True): full path of the directory to output results in
              
    """
    # Default configurations of the OED object
    _def_oed_configs = dict(oed_name=None,
                            model=None,
                            oed_criterion=None,
                            initial_time=None,
                            ref_initial_state=None,
                            forecast_state=None,
                            observations_list=None,
                            obs_checkpoints=None,
                            )
    _def_output_configs = dict(scr_output=True,
                               verbose=False,
                               file_output=False,
                               file_output_dir=None,
                               file_output_variables=['oed_statistics'],
                               verbose=False
                               )
    #
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
    
    
    #
    def __init__(self, oed_configs=None, output_configs=None):
        
        self.oed_configs = self.validate_oed_configs(oed_configs, oedsBase._def_oed_configs)
        self.output_configs = self.validate_output_configs(output_configs, SmoothersBase._def_output_configs)
        # 
        if self.output_configs['file_output'] and not os.path.isdir(self.output_configs['file_output_dir']):
                os.mkdir(self.output_configs['file_output_dir'])
        # After these basic steps, the specific oed should continue with it's constructor steps
        self.model = self.oed_configs['model']
        #
        self.oed_configs['obs_checkpoints'] = np.asarray(self.oed_configs['obs_checkpoints']).flatten()
        #
        self._time_eps = SmoothersBase.__time_eps
        self._verbose = self.output_configs['verbose']
        #
        self.__initialized = True
        
    
    #
    def oed_objective_value(self, state, design=None, oed_criterion=None, kwargs):
        """
        Evaluate the OED optimality objective at the given model state, and the design.
        
        Args:
            state:
            design:
            oed_criterion:
            kwArgs:
        
        Returns:
            fun: the value of the OED ovjective function
        
        """
        raise NotImplementedError
        
    
    #
    def oed_objective_gradient(self, state, design=None, oed_criterion=None, kwargs):
        """
        Construct the gradient of the OED optimality objective at the given model state, and the design.
        
        Args:
            state:
            design:
            oed_criterion:
            kwArgs:
        
        Returns:
            grad: the derivative of the OED ovjective function
            
        """
        raise NotImplementedError
            
    #
    def print_results(self, kwargs):
        """
        Print smoothing results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        
        Args:
            kwArgs:
            
        Returns:
            
        """
        pass
        
        
    #
    @staticmethod
    def validate_oed_configs(oed_configs, def_oed_configs):
        """
        Aggregate the passed dictionaries with default configurations then make sure parameters are consistent.
        The first argument (oed_configs) is validated, updated with missing entries, and returned.
        
        Args:
            oed_configs: dict,
                A dictionary containing oed configurations. This should be the oed_configs dict 
                passed to the constructor.
                
            def_oed_configs: dict,
                A dictionary containing the default oed configurations. 

        Returns:
            oed_configs: dict,
                Same as the first argument (oed_configs) but validated, adn updated with missing entries.
            
        """
        oed_configs = utility.aggregate_configurations(oed_configs, def_oed_configs)
        if oed_configs['oed_name'] is None:
            oed_configs['oed_name'] = 'Unknown_'

        # Since aggregate never cares about the contents, we need to make sure now all parameters are consistent
        if oed_configs['model'] is None:
            raise ValueError("You have to pass a reference to the model object so that"
                             "model observations can be created!")
        else:
            if not isinstance(oed_configs['model'], ModelsBase):
                raise ValueError("Passed model is not an instance of 'ModelsBase'!. Passed: %s" %
                                 repr(oed_configs['model']))
        return oed_configs

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
                directory_name = '_oed_results_'
                tmp_dir = os.path.join(dates_root_dir, directory_name)
                # print("Output directory of the oed object is not set. Results are saved in: '%s'" % tmp_dir)
                output_configs['file_output_dir'] = tmp_dir
            else:
                dates_root_dir = os.getenv('DATES_ROOT_PATH')
                if not str.startswith(output_configs['file_output_dir'], dates_root_dir):
                    output_configs['file_output_dir'] = os.path.join(dates_root_dir, output_configs['file_output_dir'])

            if output_configs['file_output_variables'] is None:
                output_configs['file_output'] = False
            for var in output_configs['file_output_variables']:
                if var not in def_output_configs['file_output_variables']:
                    raise ValueError("Unrecognized variable to be saved! \n Received: %s" % var)
        else:
            output_configs['file_output_dir'] = None
            output_configs['file_output_variables'] = None

        return output_configs


