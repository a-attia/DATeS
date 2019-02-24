
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

import numpy as np
import functools
import numpy.random as nr

import sys
import re

import dates_utility as utility

from time_integration_base import TimeIntegratorBase
import erkadj_setup  # wrapper built on the fly
#


class FatODE_ERK_ADJ(TimeIntegratorBase):
    #
    # ERk-ADJ configurations' default values are set here
    _def_options_dict = {'y':None,                     # in/output rank-1 array('d') with bounds (nvar)
                         'lambda_':None,              # (we may need other name!) in/output rank-2 array('d') with bounds (nvar,nadj)
                         'tin':None,                   # input float
                         'tout':None,                  # input float
                         'atol':None,                  # input rank-1 array('d') with bounds (nvar)
                         'rtol':None,                  # input rank-1 array('d') with bounds (nvar)
                         'fun':None,                   # call-back function
                         'jac':None,                   # call-back function
                         'adjinit':None,               # call-back function
                         'nvar':None,                  # input int, optional Default: len(y)
                         'nadj':None,                  # input int, optional Default: shape(lambda_,1)
                         'fun_extra_args':tuple(),     # input tuple, optional Default: ()
                         'jac_extra_args':tuple(),     # input tuple, optional Default: ()
                         'adjinit_extra_args':tuple(), # input tuple, optional Default: ()
                         'icntrl_u':None,              # input rank-1 array('i') with bounds (20)
                         'rcntrl_u':None,              # input rank-1 array('d') with bounds (20)
                         'istatus_u':None,             # input rank-1 array('i') with bounds (20)
                         'rstatus_u':None,             # input rank-1 array('d') with bounds (20)
                         'ierr_u':None,                # input int
                         'verbose':False,
                         }

    _necessary_config_keys = ('model', 'y', 'lambda_', 'tin', 'tout', 'atol', 'rtol', 'fun', 'jac', 'adjinit')  # these have to be either passed upon initialization or on calling the method 'integrate_adj'
    _mandatory_config_keys = ('model', 'fun', 'jac', 'adjinit')  # these have to be passed upon initialization
    _optional_configs_keys = ('nvar', 'nadj', 'fun_extra_args', 'jac_extra_args', 'adjinit_extra_args',
                              'icntrl_u', 'rcntrl_u', 'istatus_u', 'rstatus_u', 'ierr_u')

    #'np','nnz','mu', 'jacp', 'jacp_extra_args',
    #'drdy', 'drdy_extra_args', 'drdp', 'drdp_extra_args', 'qfun', 'qfun_extra_args', 'q'


    #
    def __init__(self, options_dict=None):
        #
        integration_configs = utility.aggregate_configurations(options_dict, FatODE_ERK_ADJ._def_options_dict)
        integration_configs = utility.aggregate_configurations(integration_configs, TimeIntegratorBase._def_options_dict)
        #
        # validate and set configurations of the adjoint integrator:
        integration_configs, status, info = self._validate_configs(integration_configs)
        self.integration_configs = integration_configs
        if status !=0:
            print("Failed to configure the FATODE time integrator!")
            raise ValueError
            #
        
        # Configurations are OK, we can proceed...
        self._verbose = self.integration_configs['verbose']
        
        TimeIntegratorBase.__init__(self, integration_configs)  # TODO: revisit when unification starts...
        
        #
        # ==================================================================
        # For testing, we are currently using FATODE-ERK here.
        # After testing, we shall decide where to put the adjoint (here vs.
        #   inside model wrappers/classes).
        #
        # Create a wrapper for the FATODE ERK-ADJOINT:
        # ==================================================================
        # -----------------<Online WRAPPER GENERATION>----------------------
        # ==================================================================
        try:
            from erkadj import erk_adj_f90_integrator as integrate_adjoint  # CAN-NOT be moved up.
        except(ImportError):
            print("Recreating the FATODE Adjoint Integrator..."),
            sys.stdout.flush()
            erkadj_setup.create_wrapper(verbose=self._verbose)
            from erkadj import erk_adj_f90_integrator as integrate_adjoint  # CAN-NOT be moved up.
            print("done...")
        self.__integrate_adjoint = integrate_adjoint
        # ==================================================================
        # ----------------------------<Done!>-------------------------------
        # ==================================================================

        #
        self._initialized = True
        #

    def integrate_adj(self, y=None, lambda_=None, tin=None, tout=None, atol=None, rtol=None):
        """
        Adjoint solver.
        Integrates across a single subinterval of the time integration window
        This is a wrapper for the method 'erk_adj_f90_integrator' in FATODE
        The arguments of the wrapped "integrate_adj" are described in the help of "_validate_configs".
        
        Parameters:
            
            y: in/output rank-1 array('d') with bounds (nvar)
            lambda_: in/output rank-2 array('d') with bounds (nvar,nadj)
            tin: input float
            tout: input float
            atol: input rank-1 array('d') with bounds (nvar)
            rtol: input rank-1 array('d') with bounds (nvar)
            
        Returns:
            lambda_: sensitivity matrix
            
        """
        # initialize parameters dictionary to be passed to the adjoint integrator
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
            y = self.integration_configs['y']
        if lambda_ is None:
            lambda_ = self.integration_configs['lambda_']
        if tin is None:
            tin = self.integration_configs['tin']
        if tout is None:
            tout = self.integration_configs['tout']
        if atol is None:
            atol = self.integration_configs['atol']
        if rtol is None:
            rtol = self.integration_configs['rtol']
        
        # TODO: Let's discuss that...
        if self.integration_configs['fun'] is None:
            model = self.integration_configs['model']
            self.integration_configs['fun'] = model.step_forward_function
        if self.integration_configs['jac'] is None:
            model = self.integration_configs['model']
            jac_handle = functools.partial(model.step_forward_function_Jacobian, create_sparse=False)
            self.integration_configs['jac'] = jac_handle
            #
        if self.integration_configs['adjinit'] is None:
            self.integration_configs['adjinit'] = __null_adjinit
        if self.integration_configs['jacp'] is None:
            self.integration_configs['jacp'] = __null_jacp
        if self.integration_configs['drdy'] is None:
            self.integration_configs['drdy'] = __null_drdy
        if self.integration_configs['drdp'] is None:
            self.integration_configs['drdp'] = __null_drdp
        if self.integration_configs['qfun'] is None:
            self.integration_configs['qfun'] = __null_qfun
            
        
        #
        # Check if anything is still missing:
        if y is None or tin is None or tout is None or atol is None or rtol is None:
            print("Some of the arguments are set to None. Please initialize all argumens properly!")
            raise ValueError

        """
        keep track of the type of original y, and lambda vectors/matrices, 
        and make sure the return type is valid
        maybe we should get the types of the wrapped data structures as well.
        """
        orig_y = y
        orig_lambda_ = lambda_

        # The trivial approach is copying (for a starter only!)...
        local_y = orig_y.get_numpy_array().squeeze()
        local_y = np.reshape(local_y, local_y.shape, order='F')
        nvar = local_y.size
        #
        if isinstance(orig_lambda_, np.ndarray):
            local_lambda = orig_lambda_.copy().squeeze()
        else:
            local_lambda = orig_lambda_.get_numpy_array().squeeze()

        shp = local_lambda.shape
        if len(shp) == 1:
            local_lambda = np.reshape(local_lambda, shp[0], order='F')
            nadj = 1
        else:
            local_lambda = np.reshape(local_lambda, shp, order='F')
            nadj = shp[1]

        # We are good. Survived these checks; good to go
        parameters.update({'y':local_y,
                           'lambda_':local_lambda,  # only here we use 'lambda as key to avoid conflict with lambda function'
                           'tin':tin,
                           'tout':tout,
                           'atol':atol,
                           'rtol':rtol,
                           'nvar':nvar,
                           'nadj':nadj,
                           })

        """
        update the essential funcitons (to handle various data structures appropriately.
        TODO: If this turns out to be a good idea, we will repeat it for other (optional)
            functions after testing.
        """
        parameters.update({'fun':self.__fun,
                           'jac':self.__jac,
                           'adjinit':self.__adjinit,
                           })

        # check the type of the passed state. (Numpy/C/Fortran wrapped objects)
        # based on the type of the wrapped data structure, the underlying state is handled.
        # One approach (followed here) is to pass the necessary functions (e.g. QFUN) to
        # a vlidator to check the type of input/output data structures, and copy to
        # the right objects if necessary (this is just an idea to start with)...

        # Call adjoint integrator
        lambda_ = self.sub_iterate(**parameters)

        # This is dirty; should be optimized
        if lambda_.__class__ == orig_lambda_.__class__:
            out_lambda = lambda_
        else:
            out_lambda = orig_lambda_
            nadj = self.integration_configs['nadj']
            if nadj == 1:
                out_lambda[:] = lambda_[:]
            else:
                for col in xrange(nadj):
                    out_lambda[:, col] = lambda_[:, col]

        return lambda_
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

    def __jac(self, t, y, fjac, n):
        """
        Jacobian of the RHS of the model

        integer, optional,intent(in),check(len(y)>=n),depend(y) :: n=len(y)
        double precision,intent(in) :: t
        double precision, dimension(n), intent(in) :: y
        double precision, dimension(n,n), intent(inout) :: fjac

        """

        model = self.integration_configs['model']
        state = model.state_vector()
        state[:] = y[:]

        jac_fun = self.integration_configs['jac']
        jac_val = jac_fun(t, state)

        # This will only work for
        # numpy and fortran
        # Neet to pay attention to fjac which is a fortran
        # object passed by the adjoint integrator
        # and jac_val which is a model state_matrix.
        # we might not be able to do the following all
        # the time.        
        
        fjac[:, :] = jac_val.get_numpy_array()
        return fjac

    def __adjinit(self, t, y, lambda_, n, nadj):
        """

        integer, optional,intent(in),check(len(y)>=n),depend(y) :: n=len(y)
        integer, optional,intent(in),check(shape(mu,0)==np),depend(mu) :: np=shape(mu,0)
        integer, optional,intent(in),check(shape(lambda,1)==nadj),depend(lambda) :: nadj=shape(lambda,1)
        double precision, intent(in) :: t
        double precision, dimension(n),intent(in) :: y
        double precision, dimension(n,nadj),intent(inout),depend(n) :: lambda
        double precision, optional, dimension(np,nadj),intent(inout),depend(nadj) :: mu

        """
        adjinit_fun = self.integration_configs['adjinit']
        # print("\n%s\t\tINSIDE __adjinit doint nothing for now...%s\n" %('='*80, '='*80))
        adjinit_fun(t, y, lambda_, n, nadj)
        # return lambda_, mu
        

    def sub_iterate(self, **kwargs): # Should we make it a staticmethod? this should be integrated with integrate_adj
        """
        Integrates (Backward) across a single subinterval of the time integration window.

        Input:

        Output:
            soln_state: adjoint (lambda_)
        """
        # Evaluate (and overwrite), the sensitivity matrix lambda values
        if False:
            print "Here are the arguments passed to sub_iterate:"
            for key in kwargs:
                print key, kwargs[key]

        self.__integrate_adjoint.integrate_adj(**kwargs)
        
        if self._verbose:
            print("R"*30+ " PASSED" + ">"*30)
        
        lambda_ = kwargs['lambda_']
        #
        return lambda_

    def update_forcing_term(self, lambda_in, copy_mat=False):
        """
        update the 'lambda' value based on the input lambda_in
        """
        if copy_mat:
            self.integration_configs['lambda_'] = lambda_in.copy()
        else:
            self.integration_configs['lambda_'] = lambda_in
        #

    def _validate_configs(self, configs):
        """
        validate, and set parameters and coefficients based on the passed configurations dictionary 'configs'

        Parameters:
            configs: a dictionary that contains parameters of the adjoint integrator

        Return:
            status: flag; describes the status of the initialization
                - 0: successful
                - 1:
            info; message that describes the code in 'status'



        Configs:
            # Necessary Parameters
            # ----------------
            y=None,                     # in/output rank-1 array('d') with bounds (nvar)
            lambda_=None                # in/output rank-2 array('d') with bounds (nvar,nadj)
            tin=None,                   # input float
            tout=None,                  # input float
            atol=None,                  # input rank-1 array('d') with bounds (nvar)
            rtol=None,                  # input rank-1 array('d') with bounds (nvar)
            fun=None,                   # call-back function
            jac=None,                   # call-back function
            adjinit=None,               # call-back function
            #
            # Other (Op) Parameters
            # ----------------
            nvar=None,                  # input int, optional Default: len(y)
            np=None,                    # input int, optional Default: shape(mu,0)
            nadj=None,                  # input int, optional Default: shape(lambda_,1)
            nnz=None,                   # input int
            fun_extra_args=tuple(),     # input tuple, optional Default: ()
            jac_extra_args=tuple(),     # input tuple, optional Default: ()
            adjinit_extra_args=tuple()  # input tuple, optional Default: ()
            icntrl_u=None,              # input rank-1 array('i') with bounds (20)
            rcntrl_u=None,              # input rank-1 array('d') with bounds (20)
            istatus_u=None,             # input rank-1 array('i') with bounds (20)
            rstatus_u=None,             # input rank-1 array('d') with bounds (20)
            ierr_u=None,                # input int
            mu=None,                    # in/output rank-2 array('d') with bounds (np,nadj)
            jacp=None,                  # call-back function
            jacp_extra_args=tuple(),    # input tuple, optional Default: ()
            drdy=None,                  # call-back function
            drdy_extra_args=tuple(),    # input tuple, optional Default: ()
            drdp=None,                  # call-back function
            drdp_extra_args=tuple(),    # input tuple, optional Default: ()
            qfun=None,                  # call-back function
            qfun_extra_args=tuple(),    # input tuple, optional Default: ()
            q=None,                     # in/output rank-1 array('d') with bounds (nadj)
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
            if configs['np'] is None:
                pass
            if configs['nadj'] is None:
                lambda_ = configs['lambda_']
                pass # TODO
            if configs['nnz'] is None:
                pass
            #
            if configs['icntrl_u'] is None:
                pass
            if configs['rcntrl_u'] is None:
                pass
            if configs['ierr_u'] is None:
                pass
            if configs['mu'] is None:
                pass
            #
            if configs['jacp'] is None:
                configs['jacp_extra_args'] = tuple()
            if configs['drdy'] is None:
                configs['drdy_extra_args'] = tuple()
            if configs['drdp'] is None:
                configs['drdp_extra_args'] = tuple()
            if configs['qfun'] is None:
                configs['qfun_extra_args'] = tuple()
            #
            if configs['q'] is None:
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

def __null_jac(t, y, fjac, n):
    """
    """
    # Do not modify Fortran memory as a whole,
    # modify at element level
    for i in xrange(n):
        for j in xrange(n):
            fjac[i,j] = 0.0
        fjac[i,i] = 1.0

def __null_adjinit(t, y, lambda_, n, nadj):
    """
    """
    # Do not modify Fortran memory as a whole,
    # modify at element level
    for i in xrange(n):
        lambda_[i] = lambda_[i]

def __null_jacp(t, y, fpjac, n, np):  # n,np,t,y,fpjac
    """
    """
    fpjac = np.zeros((n, np), order='F')


def __null_drdy(t, y, nadj, n, nr):
    """
    """
    ry = np.zeros((nr, nadj), order='F')
    return ry


def __null_drdp(t, y, rp, nadj, n, nr):  # nadj,n,nr,t,y,rp
    """
    """
    rp = np.zeros((nr, nadj), order='F')


def __null_qfun(t, y, r, n, nr):
    """
    """
    r = np.zeros(nr)
    


#
def initialize_adjoint_configs(model, adjoint_scheme):
    """
    Initialize a configurations dictionary for the adjoint integrator
    
    Parameters:
        model: a model object
        adjoint_scheme: Scheme used for adjoint model integration, i.e. propagates sensitivity. 
        Schemes supported:
            1- erk-adj 
            2- 
                
    Returns:
        options_dict: a configurations dictionary, with the following keys:
            y=None,                     : in/output rank-1 array('d') with bounds (nvar)
            lambda_=None                : in/output rank-2 array('d') with bounds (nvar,nadj)
            tin=None,                   : input float
            tout=None,                  : input float
            atol=None,                  : input rank-1 array('d') with bounds (nvar)
            rtol=None,                  : input rank-1 array('d') with bounds (nvar)
            fun=None,                   : call-back function
            jac=None,                   : call-back function
            adjinit=None,               : call-back function
            nvar=None,                  : input int, optional Default: len(y)
            np=None,                    : input int, optional Default: shape(mu,0)
            nadj=None,                  : input int, optional Default: shape(lambda_,1)
            nnz=None,                   : input int
            fun_extra_args=tuple(),     : input tuple, optional Default: ()
            jac_extra_args=tuple(),     : input tuple, optional Default: ()
            adjinit_extra_args=tuple()  : input tuple, optional Default: ()
            icntrl_u=None,              : input rank-1 array('i') with bounds (20)
            rcntrl_u=None,              : input rank-1 array('d') with bounds (20)
            istatus_u=None,             : input rank-1 array('i') with bounds (20)
            rstatus_u=None,             : input rank-1 array('d') with bounds (20)
            ierr_u=None,                : input int
            mu=None,                    : in/output rank-2 array('d') with bounds (np,nadj)
            jacp=None,                  : call-back function
            jacp_extra_args=tuple(),    : input tuple, optional Default: ()
            drdy=None,                  : call-back function
            drdy_extra_args=tuple(),    : input tuple, optional Default: ()
            drdp=None,                  : call-back function
            drdp_extra_args=tuple(),    : input tuple, optional Default: ()
            qfun=None,                  : call-back function
            qfun_extra_args=tuple(),    : input tuple, optional Default: ()
            q=None,                     : in/output rank-1 array('d') with bounds (nadj)
        
    """
    assert isinstance(adjoint_scheme, str), " 'adjoint_scheme' is INVALD string representation of ajoint method!" 
    if re.match(r'\Aadj(_|-)*erk\Z', adjoint_scheme, re.IGNORECASE) or  \
        re.match(r'\Aerk(_|-)*adj\Z', adjoint_scheme, re.IGNORECASE)or  \
        re.match(r'\Aerk\Z', adjoint_scheme, re.IGNORECASE):
        #
        state_size = model.state_size()

        # Create non_sparse jacobian
        jac_handle = functools.partial(model.step_forward_function_Jacobian, create_sparse=False)
        
        # create the adjoint integrator instance
        state = model.state_vector()
        try:
            state[:] = 0.0
        except:
            pass
            
        options_dict = {'y':state,
                        'model':model,
                        'lambda_':None,
                        'tin':0.0,
                        'tout':0.0,
                        'atol':np.ones(state_size)*1e-8,
                        'rtol':np.ones(state_size)*1e-11,
                        'fun':model.step_forward_function,
                        'jac':jac_handle,
                        'adjinit':__null_adjinit,
                        'nvar':state_size,
                        'np':None,
                        'nadj':1,
                        'nnz':None,
                        'fun_extra_args':tuple(),
                        'jac_extra_args':tuple(),
                        'adjinit_extra_args':tuple(),
                        'icntrl_u':np.zeros(20, dtype=np.int32),
                        'rcntrl_u':np.zeros(20),
                        'istatus_u':np.zeros(20, dtype=np.int32),
                        'rstatus_u':np.zeros(20),
                        'ierr_u':np.zeros(1, dtype=np.int32),
                        'mu':None,
                        'jacp':__null_jacp,
                        'jacp_extra_args':tuple(),
                        'drdy':__null_drdy,
                        'drdy_extra_args':tuple(),
                        'drdp':__null_drdp,
                        'drdp_extra_args':tuple(),
                        'qfun':__null_qfun,
                        'qfun_extra_args':tuple(),
                        'q':None
                        }
        #
    else:
        print("The adjoint integration/solver scheme %s is not supported by this Model!" % adjoint_scheme)
        raise ValueError
    #
    return options_dict
    #

