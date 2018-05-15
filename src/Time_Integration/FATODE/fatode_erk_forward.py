
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


# TODO: We need to discuss how to UNIFY the approaches and build fwd/adj wrappers
# TODO: Yeah, why don't we import the FATODE in full, and wrap all subpackages at once!
# 

import numpy as np
import functools
import numpy.random as nr

import re 

import dates_utility as utility

from time_integration_base import TimeIntegratorBase
import erkfwd_setup  # wrapper built on the fly



class FatODE_ERK_FWD(TimeIntegratorBase):
    #
    # ERk-FWD configurations' default values are set here
    _def_options_dict = {'tin':None,                   # input float
                         'tout':None,                  # input float
                         'var':None,                   # in/output rank-1 array('d') with bounds (nvar)
                         'rtol':None,                  # input rank-1 array('d') with bounds (nvar)
                         'atol':None,                  # input rank-1 array('d') with bounds (nvar)
                         'fun':None,                   # call-back function
                         'nvar':None,                  # input int, optional Default: len(var)
                         'fun_extra_args':tuple(),     # input tuple, optional Default: ()
                         'icntrl_u':None,              # input rank-1 array('i') with bounds (20)
                         'rcntrl_u':None,              # input rank-1 array('d') with bounds (20)
                         'istatus_u':None,             # input rank-1 array('i') with bounds (20)
                         'rstatus_u':None,             # input rank-1 array('d') with bounds (20)
                         'ierr_u':None,                # input int
                         'verbose':False,
                         }

    _necessary_config_keys = ('model', 'y', 'tin', 'tout', 'atol', 'rtol', 'fun')  # these have to be either passed upon initialization or on calling the method 'integrate'
    _mandatory_config_keys = ('model', 'fun')  # these have to be passed upon initialization
    _optional_configs_keys = ('nvar', 'fun_extra_args', 'icntrl_u', 'rcntrl_u', 'istatus_u', 'rstatus_u', 'ierr_u')


    def __init__(self, options_dict=None):
        #
        integration_configs = utility.aggregate_configurations(options_dict, FatODE_ERK_FWD._def_options_dict)
        integration_configs = utility.aggregate_configurations(integration_configs, TimeIntegratorBase._def_options_dict)
        #
        # validate and set configurations of the forward integrator:
        integration_configs, status, info = self._validate_configs(integration_configs)
        self.integration_configs = integration_configs
        if status !=0:
            print("Failed to configure the FATODE time integrator!")
            print("The configuration dictionary 'options_dict' is not Valid; Check the necessary keys:")
            print(FatODE_ERK_FWD._necessary_config_keys)
            raise ValueError
            #
        
        # Configurations are OK, we can proceed...
        self._verbose = self.integration_configs['verbose']
        
        TimeIntegratorBase.__init__(self, integration_configs)  # TODO: revisit when unification starts...
        
        #
        # ==================================================================
        # For testing, we are currently using FATODE-ERK here.
        # After testing, we shall decide where to put the forward(here vs.
        #   inside model wrappers/classes).
        #
        # Create a wrapper for the FATODE ERK-FWD:
        # ==================================================================
        # -----------------<Online WRAPPER GENERATION>----------------------
        # ==================================================================
        erkfwd_setup.create_wrapper(verbose=self._verbose)
        from erkfwd import erk_f90_integrator as integrate  # CAN-NOT be moved up.
        self.__integrate = integrate
        # ==================================================================
        # ----------------------------<Done!>-------------------------------
        # ==================================================================

        #
        self._initialized = True
        #

    def integrate(self, y=None, checkpoints=None, atol=None, rtol=None):
        """
        Forward solver.
        Integrates a state y over the 'checkpoints' that specifies time integration window
        This is a wrapper for the method 'erk_f90_integrator' in FATODE
        
        Parameters:
            y: in/output rank-1 array('d') with bounds (nvar); 
                y is assumed to be given at the first entry of checkpoints.
                If y is None, the state is set to self.integration_configs['var'].
            checkpoints: an iterable e.g. a list or numpy 1d array with checkpoints to save states at;
                If checkpoints is None, it is set to [tin, tfin] from 'self.integration_configs'
            atol: input rank-1 array('d') with bounds (nvar)
            rtol: input rank-1 array('d') with bounds (nvar)
            
        Returns:
            traject: y integrated forward at the passed checkpoints. 
                Entries of traject are StateVector instances.
            
        """
        # get a reference to the model object
        model = self.integration_configs['model']
        
        try:
            # get the model state vector type
            exec("from %s import StateVector" % model.__class__.__module__)
        except:
            StateVector = type(model.state_vector())
        
        # check the type of the passed state. (Numpy/C/Fortran wrapped objects)
        # based on the type of the wrapped data structure, the underlying state is handled.
        # One approach (followed here) is to pass the necessary functions (e.g. QFUN) to
        # a vlidator to check the type of input/output data structures, and copy to
        # the right objects if necessary (this is just an idea to start with)...
        # extract a numpy 1d array out of y as local_y:
        if isinstance(y, StateVector):
            # The trivial approach is copying (for a starter only!)...
            local_y = y.get_numpy_array().squeeze()
            #
        elif isinstance(y, np.ndarray):
            local_y = y.copy().squeeze()
            #
        else:
            print("y is of unsupported data type.")
            print("'y' must be an instance of StateVector or a 1d Numpy arry!")
            raise TypeError
        
        # Reshape local_y, and get state size
        local_y = np.reshape(local_y, local_y.shape, order='F')
        nvar = local_y.size
        state_size = model.state_size()
        if nvar != state_size:
            print("Model state does not match the size of the passed state?!")
            raise ValueError
        
        # initialize parameters dictionary to be passed to the forward integrator
        parameters = dict()
        for key in self._optional_configs_keys:
            key_val = self.integration_configs[key]
            if key_val is not None:
                parameters.update({key:key_val})
            else:
                pass

        #
        # Check (and validate) input args, and kwargs
        if y is None:
            y = self.integration_configs['var']
        if checkpoints is None:
            tin = self.integration_configs['tin']
            tout = self.integration_configs['tout']
            checkpoints = [tin, tout]
        if atol is None:
            atol = self.integration_configs['atol']
        if rtol is None:
            rtol = self.integration_configs['rtol']
        
        checkpoints = np.asarray(checkpoints)
        if checkpoints.size <= 1:
            print("'checkpoints' must be an iterable of lenght > 1!")
            print("\tcheckpoints:", checkpoints)
            raise ValueError
        
        #
        # TODO: Let's discuss that...
        if self.integration_configs['fun'] is None:
            # self.integration_configs['fun'] = __null_fun
            self.integration_configs['fun'] = model.step_forward_function

        # Check if anything is still missing:
        if y is None or checkpoints is None or atol is None or rtol is None:
            print("Some of the arguments are set to None. Please initialize all argumens properly!")
            raise ValueError

        #
        # update the essential funcitons (to handle various data structures appropriately.
        # TODO: If this turns out to be a good idea, we will repeat it for other (optional)
        #       functions after testing.
        #
        parameters.update({'fun':self.__fun})

        # We are good. Survived these checks; good to go
        parameters.update({'atol':atol,
                           'rtol':rtol,
                           'nvar':nvar
                           })
        
        # Initialize the trajectory
        traject = []
        if isinstance(y, StateVector):
            traject.append(y)
        else:
            state = model.state_vector()
            state[:] = local_y.copy()
            traject.append(state)
        
        # Loop over all checkpoints, propagate forward over subintervals, and append final states
        for t_ind in xrange(len(checkpoints)-1):
            # update parameters based on the current subinterval
            tin = checkpoints[t_ind]
            tout = checkpoints[t_ind+1]
            parameters.update({'var':local_y,
                               'tin':tin,
                               'tout':tout,
                               })
            # Call forward integrator
            local_y = self.sub_iterate(**parameters)

            # propagate the local state forward in time, and append to traject
            state = model.state_vector()
            try:
                state[:] = local_y.copy()
            except:
                state[:] = local_y[:].copy()
                
            traject.append(state)

        # sanity check:
        if checkpoints.size != len(traject):
            print("length of passed timespan does not match the length of generated traject!")
            print("len(checkpoints):", checkpoints.size)
            print("len(trajcet):", len(traject))
            raise ValueError
        else:
            pass
        #
        return traject
        #


    # The next three methods hadle the in/out of the passed functions from/to FATODE side,
    # and makes sure everything is seen as Fortran data structure...
    def __fun(self, t, y, f, n):
        """
        RHS of the model

        we may use a lambda function if self is not allowed to be called externally (e.g. from Fortran side)
        integer, optional,intent(in),check(len(y)>=n),depend(y) :: n=len(y)
        double precision, intent(in) :: t
        double precision, dimension(n), intent(in) :: y
        double precision, dimension(n), intent(inout) :: f

        """

        model = self.integration_configs['model']
        state = model.state_vector()
        state[:] = y[:]

        fun = self.integration_configs['fun']
        rhs = fun(t, state)

        f[:] = rhs[:]
        return f


    def sub_iterate(self, **kwargs): # Should we make it a staticmethod? this should be integrated with integrate
        """
        Integrates across a single subinterval of the time integration window.

        Input:

        Output:
            soln_state: y
        """
        # Evaluate (and overwrite), the sensitivity matrix lambda values
        if False:
            print "Here are the arguments passed to sub_iterate:"
            for key in kwargs:
                print key, kwargs[key]

        self.__integrate.integrate(**kwargs)
        
        if self._verbose:
            print("R"*30+ " PASSED" + ">"*30)
        
        y = kwargs['var']
        #
        return y

    def _validate_configs(self, configs):
        """
        validate, and set parameters and coefficients based on the passed configurations dictionary 'configs'

        Parameters:
            configs: a dictionary that contains parameters of the forward integrator

        Returns:
            Configs:
                # Necessary Parameters
                # ----------------
                var=None,                   # in/output rank-1 array('d') with bounds (nvar)
                tin=None,                   # input float
                tout=None,                  # input float
                atol=None,                  # input rank-1 array('d') with bounds (nvar)
                rtol=None,                  # input rank-1 array('d') with bounds (nvar)
                fun=None,                   # call-back function
                #
                # Other (Op) Parameters
                # ----------------
                nvar=None,                  # input int, optional Default: len(y)
                fun_extra_args=tuple(),     # input tuple, optional Default: ()
                icntrl_u=None,              # input rank-1 array('i') with bounds (20)
                rcntrl_u=None,              # input rank-1 array('d') with bounds (20)
                istatus_u=None,             # input rank-1 array('i') with bounds (20)
                rstatus_u=None,             # input rank-1 array('d') with bounds (20)
                ierr_u=None,                # input int
            
            status: flag; describes the status of the initialization
                - 0: successful
                - 1:
                
            info: message that describes the code in 'status'

        """

        # Check and validate the configs
        info = None

        # Check validity of mandatory configurations
        mandatory_conf_keys = self._mandatory_config_keys  # class level attribute
        try:
            mandatory_conf = [configs[key] for key in mandatory_conf_keys]
            status = 0
            #
        except:
            status = 1
            #
        finally:
            if status != 0:
                # return with failure flag
                info = "Configurations are invalid"
                # return status, info
                #
            else:
                # We are good to go, now check Values of the configs
                for key in mandatory_conf:
                    if key is None:
                        status = 1
                        info = "Configuration parameter %s CANNOT be None!" % key
                        break
                    else:
                        pass

        #
        if status == 0:
            info = "Successfully initialized the parameters"
            #
            # Retrieve the type of the input, and output date structures (C/Fortran/Numpy), and direct all operations accordingly...

            # Update configs
            if configs['nvar'] is None:
                model = self.integration_configs['model']
                configs['nvar'] = model.state_vector_size()
            #
            if configs['icntrl_u'] is None:
                pass
            if configs['rcntrl_u'] is None:
                pass
            if configs['ierr_u'] is None:
                pass

            #
        elif info is not None:
            pass
            #
        else:
            info = "Configurations are invalid"
            #
        #
        return configs, status, info
        #

    
# ============================================================================================ #
#        Define a set of functions those can be used if necessary arguments are missing        #
# ============================================================================================ #
def __null_fun(t, y, f, n):
    """
    """
    # Do not modify Fortran memory as a whole,
    # modify at element level
    for i in xrange(n):
        f[i] = y[i]


#
def initialize_forward_configs(model, forward_scheme):
    """
    A function to initialize a configurations dictionary for the forward integrator
    
    Inputs:
        model: a model object
        forward_scheme: Scheme used for forward model integration
        Schemes supported:
            1- erk-fwd 
            2- 
                
    Returns:
        
    """
    assert isinstance(forward_scheme, str), " 'forward_scheme' is INVALD string representation of forward method!" 
    if re.match(r'\Afwd(_|-)*erk\Z', forward_scheme, re.IGNORECASE) or  \
        re.match(r'\Aerk(_|-)*fwd\Z', forward_scheme, re.IGNORECASE)or  \
        re.match(r'\Aerk\Z', forward_scheme, re.IGNORECASE):
        #
        state_size = model.state_size()
       
        # create the forward integrator instance
        state = model.state_vector()
        try:
            state[:] = 0.0
        except:
            pass
        options_dict = {'var':state,
                        'model':model,
                        'tin':0.0,
                        'tout':0.0,
                        'atol':np.ones(state_size)*1e-8,
                        'rtol':np.ones(state_size)*1e-11,
                        'fun':model.step_forward_function,
                        'nvar':state_size,
                        'fun_extra_args':tuple(),
                        'icntrl_u':np.zeros(20, dtype=np.int32),
                        'rcntrl_u':np.zeros(20),
                        'istatus_u':np.zeros(20, dtype=np.int32),
                        'rstatus_u':np.zeros(20),
                        'ierr_u':np.zeros(1, dtype=np.int32),
                        }
        #
    else:
        print("The forward integration/solver scheme %s is not supported by this Model!" % forward_scheme)
        raise ValueError
    #
    return options_dict
    #

