
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


import dates_utility as utility


class TimeIntegratorBase(object):
    """
    Base class defining the time integration process.
    
    Time Integrator constructor.

    Args:
        model: Model Object
        options_dict: Dictionary object with all time integration objects
            Support configurations:
                * 'model': None,
                * 'initial_state': None,
                * 'checkpoints': None,
                * 'initial_time': None,
                * 'final_time': None,
                * 'step_size': None,
                * 'rel_tol': 1e-3,
                * 'abs_tol': 1e-3,
                * 'int_method': 0,
                * 'adaptive_stepping': False,
                * 'fixed_stepping': True
    
    """
    # TODO: This should be updated after adding several fwd/adj wrappers
    _def_options_dict = {'model': None,
                         'initial_state': None,
                         'checkpoints': None,
                         'initial_time': None,
                         'final_time': None,
                         'step_size': None,
                         'rel_tol': 1e-3,
                         'abs_tol': 1e-3,
                         'int_method': 0,
                         'adaptive_stepping': False,
                         'fixed_stepping': True
                         }
                         
    def __init__(self, options_dict=None):
        
        # Default step size will be defined in the model constructor.
        try:
            # update default step size to match model default stepsize if available!
            model_def_step_size = options_dict['model']._default_step_size
            self._def_options_dict['step_size'] = model_def_step_size
        except ValueError:
            pass
        
        _configs = utility.aggregate_configurations(options_dict, TimeIntegratorBase._def_options_dict)
        
        self._model = _configs['model']
        self._initial_state = _configs['initial_state']
        self._checkpoints = _configs['checkpoints']
        self._initial_time = _configs['initial_time']
        self._final_time = _configs['final_time']
        self._step_size = _configs['step_size']
        self._rel_tol = _configs['rel_tol']
        self._abs_tol = _configs['abs_tol']
        self._int_method = _configs['int_method']
        self._adaptive_stepping = _configs['adaptive_stepping']
        self._fixed_stepping = _configs['fixed_stepping']

        self._options_dict = _configs
        self.set_options(options_dict)
        self.set_coeffs()

    def integrate(self, initial_state=None, checkpoints=None, step_size=None, rel_tol=None, abs_tol=None):
        """
        Integrates forward in time with a variable timestep with relative, rel_tol, and absolute, abs_tol, tolerances
        between checkpoints(i) and checkpoints(i+1).

        Args:
            initial_state: initial state of the model
            checkpoints: out times for the model states.  This list of model states corresponds to the times in checkpoints.
            step_size: time integration step size.
            rel_tol: relative tolerance for adaptive integration
            abs_tol: absolute tolerance for adaptive integration

        Returns:
            soln_state: A list of solution vectors corresponding to the times specified in checkpoints or self._checkpoints
        """
        # make sure that stepping strategies are consistent
        if self._adaptive_stepping is True and self._fixed_stepping is True:
            raise Exception('inconsistent stepping strategies')
        if self._adaptive_stepping is False and self._fixed_stepping is False:
            raise Exception('no stepping strategy is specified')

        # Store in a temporary variable all currently stored values for _initial_state, self._checkpoints,
        # self._step_size, self._rel_tol, and self._abs_tol.  Temporarily set given values for these parameters.
        if initial_state is not None:
            initial_state_temp = self._initial_state
            self._initial_state = initial_state
        elif self._initial_state is None:
            raise Exception('No initial state specified')
        else:
            initial_state_temp = self._initial_state

        if checkpoints is not None:
            checkpoints_temp = self._checkpoints
            self._checkpoints = checkpoints
        elif self._checkpoints is None:
            if self._initial_time is None or self._final_time is None:
                raise Exception('No timespan specified')
            else:
                self._checkpoints =  [self._initial_time, self._final_time]
                self._checkpoints_temp = self._checkpoints
        else:
            checkpoints_temp = self._checkpoints

        if step_size is not None:
            step_size_temp = self._step_size
            self._step_size = step_size
        elif self._step_size is None and self._fixed_stepping is True:
            raise Exception('No step_size specified')
        else:
            step_size_temp = self._step_size

        if rel_tol is not None:
            rel_tol_temp = self._rel_tol
            self._rel_tol = rel_tol
        elif self._rel_tol is None and self._adaptive_stepping is True:
            raise Exception('No relative tolerance specified')
        else:
            rel_tol_temp = self._rel_tol

        if abs_tol is not None:
            abs_tol_temp = abs_tol
            self._abs_tol = abs_tol
        elif self._abs_tol is None and self._adaptive_stepping is True:
            raise Exception('No absolute tolerance specified')
        else:
            abs_tol_temp = self._abs_tol

        # Construct output variable, and compute solutions at times corresponding to entries
        # in checkpoints
        soln_state = list()
        soln_state.append(self._initial_state)
        for i in xrange(len(self._checkpoints)-1):
            temp_state = self.sub_iterate(self._checkpoints[i], self._checkpoints[i+1])
            soln_state.append(temp_state)
            self._initial_state = temp_state

        self._initial_state = initial_state_temp
        self._checkpoints = checkpoints_temp
        self._step_size = step_size_temp
        self._rel_tol = rel_tol_temp
        self._abs_tol = abs_tol_temp

        # Paul: please return corresponding checkpoints as a second output
        return soln_state

    def sub_iterate(self, local_t0, local_tf):
        """
        Integrates across a single subinterval of the time integration window.

        Args:
            local_t0: beginning time of local integration window.
            local_tf: end time of local integration window.

        Returns:
            soln_state: solution at local_tf
        """
        raise NotImplementedError

    def set_options(self, options_dict):
        """
        Set the options of the time integrators.  If only a single dictionary is provided, update current dictionary with new values, otherwise reconcile with the default options dictionary.  Then update class variables with the provided / reconciled dictionary values.

        Args:
            options_dict: a dictionary object with the following keys:
            model: The model object to be integrated forward in time.
            initial_state: initial state of the model.
            initial_time: initial model time.
            final_time: final model time.
            step_size: time integration step size.
            rel_tol: relative tolerance for adaptive integration.
            abs_tol: absolute tolerance for adaptive integration.
            int_method: a key corresponding to the integration method chosen.
            adaptive_stepping: A boolean value, that forces use of adaptive time stepping.
            fixed_stepping: A boolean value, that forces use of fixed time stepping.
            def_options_dict(optional): a dictionary object with same keys as above.
        """
        if options_dict is not None:
            for key in options_dict:
                self._options_dict[key] = options_dict[key]

        if self._options_dict['model'] is not None:
            self._model = self._options_dict['model']
        if self._options_dict['initial_state'] is not None:
            self._initial_state = self._options_dict['initial_state']
        if self._options_dict['checkpoints'] is not None:
            self._checkpoints = self._options_dict['checkpoints']
        if self._options_dict['initial_time'] is not None:
            self._initial_time = self._options_dict['initial_time']
        if self._options_dict['final_time'] is not None:
            self._final_time = self._options_dict['final_time']
        if self._options_dict['step_size'] is not None:
            self._step_size = self._options_dict['step_size']
        if self._options_dict['rel_tol'] is not None:
            self._rel_tol = self._options_dict['rel_tol']
        if self._options_dict['abs_tol'] is not None:
            self._abs_tol = self._options_dict['abs_tol']
        if self._options_dict['int_method'] is not None:
            self._int_method = self._options_dict['int_method']
        if self._options_dict['adaptive_stepping'] is not None:
            self._adaptive_stepping = self._options_dict['adaptive_stepping']
        if self._options_dict['fixed_stepping'] is not None:
            self._fixed_stepping = self._options_dict['fixed_stepping']

    def set_coeffs(self):
        """
        set method coefficients
        """
        pass
        
        
