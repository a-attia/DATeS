
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
    HMCFilter:
    ----------
    A class implementing the Hamiltonian/Hybrid Monte-Carlo Sampling family for filtering developed by Ahmed Attia, and Adrian Sandu.
        - Publications:                                                                       
            + Ahmed Attia and Adrian Sandu (2015). A Hybrid Monte Carlo Sampling Filter for non-Gaussian Data Assimilation. AIMS Geosciences, 1(geosci-01-00041):41-78. http://www.aimspress.com/article/10.3934/geosci.2015.1.41
                                                                                          
    Momentum:                                                                             
    ---------
    A class implementing the functionalities of the synthetic momentum.
    The momentum variable P in the HMC context is an essential component of the sampling process.
    Here we model it giving it all the necessary functionality to ease working with it.   
"""


import numpy as np
from scipy.linalg import lu_factor, lu_solve
import os

import dates_utility as utility
from filters_base import FiltersBase
from state_vector_base import StateVectorBase as StateVector  # for validation


class HMCFilter(FiltersBase):
    """
    A class implementing the Hamiltonian/Hybrid Monte-Carlo Sampling family for filtering developed by Ahmed Attia, and Adrian Sandu (2015).
    
    HMC filter constructor:
    
    Args:
        filter_configs:  dict,
            A dictionary containing HMC filter configurations.
            Supported configuarations:
                * model (default None):  model object
                * filter_name (default None): string containing name of the filter; used for output.
                * hybrid_background_coeff (default 0.0): used when hybrid background errors are used,
                    this multiplies the modeled Background error covariance matrix.
                * forecast_inflation_factor (default 1.0): forecast-ensemble covariance inflation factor
                * analysis_inflation_factor (default 1.0):  analysis-ensemble covariance inflation factor
                * localize_covariances (default True): bool,
                    apply covariance localization to ensemble covariances.
                    This is done by default using Shur product, and is requested from the model.
                    This is likely to be updated in future releases to be carried out here with more options.
                * localization_radius (default np.infty): radius of influence of covariance decorrelation
                * localization_function ('gaspari-cohn'): the covariance localization function
                    'gaspari-cohn', 'gauss', etc.
                    These functions has to be supported by the model to be used here.
                * prior_distribution (default 'gaussian'): prior probability distribution;
                    this should be either 'gaussian' or 'GMM'.
                    - 'Gaussian': the prior distribution is approximated based on the forecast ensemble,
                    - 'GMM': the prior distribution is approximated by a GMM constructed using EM algorithm
                             using the forecast ensemble.
                * prior_variances_threshold: (default 1e-5): this is a threshold being put over the prior variances.
                    This mainly handles cases where ensemble members are constant in a specific direction,
                    for example due to dirichlet boundary condition.
                    This is added to all variances, e.g. if some variances are below
                * gmm_prior_settings: dict,
                    This is a configurations dictionary of the GMM approximation to the prior.
                    This will be used only if the prior is assumed to be non-Gaussian, and better estimate is 
                    needed, i.e. it is used only if 'prior_distribution' is set to 'GMM'.
                    The implementation in this case follows the cluster EnKF described by [cite].
                    The configurations supported are:
                       - clustering_model (default 'gmm'): 'gmm', 'vbgmm' are supported
                       - cov_type (default 'diag'): covariance matrix(es) shape of the mixture components,
                            Supported shapes: 'diag', 'tied', 'spherical', 'full'
                       - localize_covariances: (default True): bool,
                            apply covariance localization to the covariance matrices of the mixture components.
                            This is done by default using Shur product, and is requested from the model.
                       - localization_radius (default np.infty): radius of influence of covariance decorrelation
                       - localization_function ('gaspari-cohn'): the covariance localization function
                            'gaspari-cohn', 'gauss', etc.
                       - inf_criteria (default 'aic'): Model information selection criterion;
                            Supported criteria: 
                                + 'aic': Akaike information criterion
                                + 'bic': Bayesian information criterion
                       - number_of_components (default None): The number of components in the GMM. 
                            This overrides the inf_criteria and forces a specific number of components to be 
                            fitted by the EM algorithm.
                       - min_number_of_components (default None): If not None, and number_of_components is 
                            None, and a selection criterion is used for model selection, the number of 
                            components is forced to be at least equal to the min_number_of_components.
                       - max_number_of_components (default None): If not None, and number_of_components is 
                            None, and a selection criterion is used for model selection, the number of 
                            components is forced to be at most equal to the max_number_of_components.
                       - min_number_of_points_per_component (default 1): If number_of_components is None,
                            and a selection criterion is used for model selection, the number of 
                            components is decreased untill the number of sample points under each component
                            is at least equal to the min_number_of_points_per_component.
                       - invert_uncertainty_param (default True): From Covariances obtain Precisions, and 
                            vice-versa.
                       - approximate_covariances_from_comp (default False): use parameters/statistics of the
                            mixture components to evaluate/approximate the joint mean, and covariance matrix
                            of the ensemble.
                            If False, the ensemble itself is used to evaluate the joint mean, and covariances.
                       - use_oringinal_hmc_for_one_comp (default False): Fall back to originial HMC if 
                            one mixture component (Gaussian prior) is detected in the ensemble.
                       - initialize_chain_strategy (default 'forecast_mean'): strategy for initializing the 
                            Markov chain.
                            Supported initialization strategies:
                                + 'highest_weight': initialize the chain to the mean of the mixture component
                                    with highst mixing weight.
                                + 'forecast_mean': initialize the chain to the (joint) ensemble mean.                         
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
                * file_output_file_name_prefix (default 'EnKF_results'): name/prefix of output file
                * file_output_file_format (default 'mat'): file ouput format.
                    Supported formats:
                        - 'mat': matlab matrix files,
                        - 'pickle': python pickled objects,
                        - 'txt' or 'ascii': text files
                * file_output_separate_files (default True): save all results to a single or multiple files
                                  
    Returns:
        None
     
    """
    #
    _filter_name = "HMC-F"
    #
    _def_local_filter_configs = dict(model=None,
                                     filter_name=_filter_name,
                                     hybrid_background_coeff=0.0,
                                     forecast_inflation_factor=1.0,
                                     analysis_inflation_factor=1.0,
                                     localize_covariances=False,
                                     localization_radius=np.infty,
                                     localization_function='gaspari-cohn',
                                     prior_distribution='gmm',
                                     prior_variances_threshold=1e-15,
                                     gmm_prior_settings=dict(clustering_model='gmm',
                                                             cov_type='diag',
                                                             localize_covariances=True,
                                                             localization_radius=np.infty,
                                                             localization_function='gaspari-cohn',
                                                             inf_criteria='aic',
                                                             number_of_components=None,
                                                             min_number_of_components=None,
                                                             max_number_of_components=None,
                                                             min_number_of_points_per_component=1,
                                                             invert_uncertainty_param=True,
                                                             approximate_covariances_from_comp=False,
                                                             use_oringinal_hmc_for_one_comp=False,
                                                             initialize_chain_strategy='forecast_mean'
                                                             ),                                     
                                     chain_parameters=dict(Initial_state=None,
                                                           Symplectic_integrator='3-stage',
                                                           Hamiltonian_num_steps=30,
                                                           Hamiltonian_step_size=0.05,
                                                           Mass_matrix_type='prior_precisions',
                                                           Mass_matrix_scaling_factor=1.0,
                                                           Burn_in_num_steps=100,
                                                           Mixing_steps=10,
                                                           Convergence_diagnostic_scheme=None,
                                                           Burn_by_optimization=False,
                                                           Optimization_parameters=dict(scheme=None,
                                                                                        tol=0,
                                                                                        maxiter=0),
                                                           Automatic_tuning_scheme=None,
                                                           Tempering_scheme=None,
                                                           Tempering_parameters=None
                                                           ),
                                     ensemble_size=None,
                                     analysis_ensemble=None,
                                     analysis_state=None,
                                     forecast_ensemble=None,
                                     forecast_state=None,
                                     filter_statistics=dict(forecast_rmse=None,
                                                            analysis_rmse=None,
                                                            initial_rmse=None,
                                                            rejection_rate=None
                                                            )
                                     )
    _local_def_output_configs = dict(scr_output=True, 
                                     file_output=False,
                                     filter_statistics_dir='Filter_Statistics',
                                     model_states_dir='Model_States_Repository',
                                     observations_dir='Observations_Rpository',
                                     file_output_moment_only=True,
                                     file_output_moment_name='mean',
                                     file_output_file_name_prefix='HMC_results',
                                     file_output_file_format='text',
                                     file_output_separate_files=False
                                     )
    _supported_tempering_schemes = ['simulated-annealing', 'parallel-tempering', 'equi-energy']
    _supported_prior_distribution = ['gaussian', 'normal', 'gmm', 'gaussian-mixture', 'gaussian_mixture']

    # 
    def __init__(self, filter_configs=None, output_configs=None):
        
        filter_configs = utility.aggregate_configurations(filter_configs, HMCFilter._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, HMCFilter._local_def_output_configs)
        FiltersBase.__init__(self, filter_configs=filter_configs, output_configs=output_configs)
        #
        self.model = self.filter_configs['model']
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

        self.symplectic_integrator = self.filter_configs["chain_parameters"]['Symplectic_integrator'].lower()
        self.symplectic_integrator_parameters = {
            'automatic_tuning_scheme': self.filter_configs["chain_parameters"]['Automatic_tuning_scheme'],
            'step_size': self.filter_configs["chain_parameters"]['Hamiltonian_step_size'],
            'num_steps': int(self.filter_configs["chain_parameters"]['Hamiltonian_num_steps'])
        }

        # supported: [prior_precisions, prior_variances, identity]
        self.mass_matrix_strategy = self.filter_configs["chain_parameters"]['Mass_matrix_type'].lower()
        self.mass_matrix_scaling_factor = self.filter_configs["chain_parameters"]['Mass_matrix_scaling_factor']

        self.chain_parameters = {
            'initial_state': self.filter_configs["chain_parameters"]['Initial_state'],
            'burn_in_steps': int(self.filter_configs["chain_parameters"]['Burn_in_num_steps']),
            'mixing_steps': int(self.filter_configs["chain_parameters"]['Mixing_steps']),
            'convergence_diagnostic': self.filter_configs["chain_parameters"]['Convergence_diagnostic_scheme']
        }

        self.optimize_to_converge = self.filter_configs["chain_parameters"]['Burn_by_optimization']
        if self.optimize_to_converge:
            try:
                self.optimization_parameters = self.filter_configs["chain_parameters"]['Optimization_parameters']
            except(AttributeError, NameError, ValueError):
                # set default parameters...
                print("Optimization for convergence is yet to be implemented in full")
                raise NotImplementedError()

        if self.filter_configs["chain_parameters"]['Tempering_scheme'] is not None:
            self.tempering_scheme = self.filter_configs["chain_parameters"]['Tempering_scheme'].lower()
            self.tempering_parameters = self.filter_configs["chain_parameters"]['Tempering_parameters']
        else:
            self.tempering_scheme = None

        self.prior_distribution = self.filter_configs['prior_distribution'].lower()
        if self.prior_distribution not in self._supported_prior_distribution:
            print("Unrecognized prior distribution [%s]!" % self.prior_distribution)
            raise ValueError()

        if self.prior_distribution == 'gaussian':
            #
            forecast_ensemble = self.filter_configs['forecast_ensemble']
            if forecast_ensemble is not None:
                self.forecast_state = utility.ensemble_mean(forecast_ensemble)
            else:
                self.forecast_state = None
            # Add filter statistics to the output configuration dictionary for proper outputting.
            if 'filter_statistics' not in self.output_configs:
                self.output_configs.update(dict(filter_statistics={}))

        elif self.prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            # update the filter name
            self._filter_name = "ClHMC-F"
            self.filter_configs['filter_name'] = self._filter_name
            # Generate GMM parameters... May be it is better to move this to a method to generate prior info.
            # It might be needed in the forecast step in case FORECAST is carried out first, and forecast ensemble is empty!
            self._gmm_prior_settings = self.filter_configs['gmm_prior_settings']

            # Add filter statistics to the output configuration dictionary for proper outputting.
            if 'filter_statistics' not in self.output_configs:
                self.output_configs.update(dict(filter_statistics=dict(gmm_prior_statistics=None)))

            # Generate the forecast state only for forecast RMSE evaluation. It won't be needed by the GMM+HMC sampler
            forecast_ensemble = self.filter_configs['forecast_ensemble']
            if forecast_ensemble is not None:
                self.forecast_state = utility.ensemble_mean(forecast_ensemble)
            else:
                self.forecast_state = None
        else:
            print("Unrecognized prior distribution [%s]!" % self.prior_distribution)
            raise ValueError()

        # Setup a flag to overwrite the prior distribution in the case where:
        # If GMM is applied but only one compoenent is detected from the ensemble
        # Default is False. It will be switched to True only if GMM is converted to Gaussian
        self.switch_back_to_GMM = False
        self.switched_to_Gaussian_prior = False

        # create a momentum object
        if self.filter_configs['forecast_ensemble'] is None:
            self._momentum = None
        else:
            self._momentum = self.initialize_momentum()
        #
        try:
            self._verbose = self.output_configs['verbose']
        except(AttributeError, NameError, KeyError):
            self._verbose = False
        #
        self.__initialized = True
        #

    #
    def initialize_momentum(self):
        """
        Create the momentum object once with the first forecast cycle.
        It should be updated at the beginning of each assimilation cycle for proper and consistent performance.
        The mass matrix here is assumed to be always diagonal. The values on the diagonal are set based on the
        strategy given in self.mass_matrix_strategy.
        
        Args:
                  
        Returns:
            None
           
        """
        # create a momentum object with a predefined mass matrix based on filter parameters.
        if self.mass_matrix_strategy == 'identity':
            momentum_obj = Momentum(self.mass_matrix_scaling_factor, mass_matrix_shape='diagonal', model=self.model)
        elif self.mass_matrix_strategy in ['prior_variances', 'prior_precisions']:
            # Calculate ensemble-based variances or the hybrid version of B (to be used to construct the mass matrix)
            try:
                prior_variances = self.prior_variances
            except (AttributeError, ValueError, NameError):
                self.generate_prior_info()
                prior_variances = self.prior_variances

            if self.mass_matrix_strategy == 'prior_variances':
                momentum_obj = Momentum(prior_variances, mass_matrix_shape='diagonal', model=self.model, verbose=self._verbose)
            elif self.mass_matrix_strategy == 'prior_precisions':
                prior_precisions = prior_variances.copy()
                prior_precisions[prior_precisions < 1e-15] = 1e-15
                prior_precisions = 1.0/prior_precisions
                momentum_obj = Momentum(prior_precisions, mass_matrix_shape='diagonal', model=self.model, verbose=self._verbose)
            else:
                print("This is not even possible!\n Unrecognized mass matrix strategy!")
                raise ValueError()
        else:
            print("Unrecognized mass matrix strategy!")
            raise ValueError()
            #
        return momentum_obj
        #
    
    #
    def update_momentum(self, mass_matrix=None, mass_matrix_shape=None, dimension=None, model=None, diag_val_thresh=1e-15):
        """
        Update the statistics of the momentum object.
        
        Args:
            mass_matrix: scalar, one-D or 2-D numpy array, or a model.state_matrix object
                - if it is a scaler, it is used to fill the diagonal of a diagonal mass matrix
                - if it is one-dimensional numpy array, the mass matrix should be diagonal which values are set
                    according to the values passed in this 1D array
                - if it is two-dimensional numpy array, the mass matrix is full and is set to this mass_matrix
            mass_matrix_shape (default None): shape of the mass matrix,
                This should really be 'diagonal'
            dimension: space dimension of the momentum variable
            model: model object
            diag_val_thresh: a lower threshold value of the diagonal of the mass matrix to aviod overflow on 
                finding the mass matrix inverse.
                  
        Returns:
            None
        
        """
        # Check the input arguments and retrieve the defaults for Nones
        if model is None:
            model = self.model

        if mass_matrix_shape is None:
            mass_matrix_shape = mass_matrix_shape='diagonal'

        if dimension is None:
            dimension = self.model.state_size()

        if mass_matrix is None:
            if self.mass_matrix_strategy == 'identity':
                mass_matrix = self.mass_matrix_scaling_factor
            elif self.mass_matrix_strategy in ['prior_variances', 'prior_precisions']:
                # Calculate ensemble-based variances (to be used to construct the mass matrix)
                # retrieve the ensemble of forecast states
                try:
                    # Try to retrieve the diagonal of the prior covariance matrix.
                    # If these are not available, generate prior info then retrieve the diagonal of B
                    prior_variances = self.prior_variances
                except (AttributeError, ValueError, NameError):
                    self.generate_prior_info()
                    prior_variances = self.prior_variances
                # The values on the diagonal of the mass matrix can't be very small to avoid overflow when inverting the mass matrix                
                if isinstance(prior_variances, StateVector):
                    # slice it to convert to numpy array if not
                    prior_variances = prior_variances.get_numpy_array()
                    prior_variances = prior_variances.copy()
                prior_variances[prior_variances<diag_val_thresh] = diag_val_thresh
                #
                if self.mass_matrix_strategy == 'prior_variances':
                    mass_matrix = prior_variances
                elif self.mass_matrix_strategy == 'prior_precisions':
                    mass_matrix = 1.0 / prior_variances
                else:
                    print("This is not even possible!\n Unrecognized mass matrix strategy!")
                    raise ValueError()
            #
            else:
                print("Unrecognized mass matrix strategy!")
                raise ValueError()

        # Now update the mass matrix and the associated square root and inverse
        self._momentum.update_mass_matrix(mass_matrix=mass_matrix, mass_matrix_shape=mass_matrix_shape, model=model)
        #

    #
    def filtering_cycle(self, update_reference=False):
        """
        Apply the filtering step. Forecast, then Analysis...
        All arguments are accessed and updated in the configurations dictionary.
        
        Args:
            update_reference (default True): bool,
                A flag to decide whether to update the reference state in the filter or not.
                  
        Returns:
            None
            
        """
        # Call basic functionality from the parent class:
        FiltersBase.filtering_cycle(self, update_reference=update_reference)
        #
        # Add further functionality if you wish...
        #

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
        analysis_ensemble = list(self.filter_configs['analysis_ensemble'])
        timespan = self.filter_configs['timespan']
        forecast_ensemble = utility.propagate_ensemble(ensemble=analysis_ensemble,
                                                       model=self.filter_configs['model'],
                                                       checkpoints=timespan,
                                                       in_place=False)
        self.filter_configs['forecast_ensemble'] = list(forecast_ensemble)
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        self.filter_configs['forecast_state'] = forecast_state.copy()

        # Add more functionality after building the forecast ensemble if desired!
        # Obtain information about the prior distribution from the forecast ensemble
        # Covariance localization and hybridization are carried out if requested
        self.generate_prior_info()
        
        # If the momentum parameters are not yet initialized, then do so.
        # This can be the case if the analysis is conducted before the forecast, and the momentum
        # is not initialized in the filter constructor.
        if self._momentum is None:
            self._momentum = self.initialize_momentum()
        elif isinstance(self._momentum, Momentum):
            self.update_momentum()
        else:
            print("The current momentum is not an instance of the right object!")
            raise ValueError()
            #
            
    #
    def generate_prior_info(self):
        """
        Generate the statistics of the approximation of the prior distribution.
            - if the prior is Gaussian, the covariance (ensemble/hybrid) matrix is generated and 
              optionally localized.
            - if the prior is GMM, the parameters of the GMM (weights, means, covariances) are generated 
              based on the gmm_prior_settings. 
        
        Args:
            None
                  
        Returns:
            None
        
        """
        # Read the forecast ensemble...
        forecast_ensemble = self.filter_configs['forecast_ensemble']
        #
        # Inflate the ensemble if required.
        # This should not be required in the context of HMC, but we added it for testing...
        inflation_fac=self.filter_configs['forecast_inflation_factor']
        if inflation_fac > 1.0:
            if self._verbose:
                print('Inflating the forecast ensemble...')
            if self._verbose:
                in_place = False
            else:
                in_place = True
            #
            inflated_ensemble = utility.inflate_ensemble(ensemble=forecast_ensemble, inflation_factor=inflation_fac, in_place=in_place)
            #
            if self._verbose:
                print('inflated? : ', (forecast_ensemble[0][:]!=inflated_ensemble[0][:]).any())
            #
            self.filter_configs['forecast_ensemble'] = inflated_ensemble
        else:
            pass
        
        #
        prior_variances_threshold=self.filter_configs['prior_variances_threshold']
        #
        # Now update/calculate the prior covariances and formulate a presolver from the prior covariances.
        # Prior covariances here are found as a linear combination of both static and flow-dependent versions of the B.
        if self.prior_distribution in ['gaussian', 'normal']:
            try:
                balance_factor = self.filter_configs['hybrid_background_coeff']
            except (NameError, ValueError, AttributeError, KeyError):
                # default value is 0 in case it is missing from the filter configurations
                balance_factor = self._def_local_filter_configs['hybrid_background_coeff']
            #
            fac = balance_factor  # this multiplies the modeled Background error covariance matrix
            if 0.0 < fac < 1.0: 
                if self._verbose:
                    print("Hyberidizing the background error covariance matrix")
                model_covariances = self.model.background_error_model.B.copy().scale(fac)
                ensemble_covariances = self.model.ensemble_covariance_matrix(forecast_ensemble, localize=self.filter_configs['localize_covariances'])
                model_covariances = model_covariances.add(ensemble_covariances.scale(1-fac))
            elif fac == 0.0:
                model_covariances = self.model.ensemble_covariance_matrix(forecast_ensemble, localize=self.filter_configs['localize_covariances'])
            elif fac == 1.0:
                model_covariances = self.model.background_error_model.B.copy()  # this should be already localized
            else:
                print("balance factor for hybrid background error covariance matrix has to be a scalar in \
                       the interval [0, 1].\n  \
                       Factor received [%s]" % repr(fac)
                       )
                raise ValueError
            #
                                
            # Apply localization on the full background error covariance matrix.
            if self.filter_configs['localize_covariances']:
                if self._verbose:
                    print('Localizing the background error covariance matrix...')
                loc_func = self.filter_configs['localization_function']
                loc_radius = self.filter_configs['localization_radius']
                if loc_radius is not None and loc_radius is not np.infty:
                    # localization radius seems legit; apply covariance localization now
                    try:
                        model_covariances = self.model.apply_state_covariance_localization(model_covariances,
                                                                                            localization_function=loc_func, 
                                                                                            localization_radius=loc_radius
                                                                                            )
                    except(TypeError):
                        if self._verbose:
                            print("Covariance localization with the given settings failed. \n \
                                   Trying covariance localization with default model settings..."
                                   )
                        # Try the localization with default settings in the model
                        model_covariances = self.model.apply_state_covariance_localization(model_covariances)
            #
            model_covariances_diag = model_covariances.diag()
            tiny_covar_indices =  model_covariances_diag < prior_variances_threshold
            if tiny_covar_indices.any():
                model_covariances = model_covariances.addAlphaI(prior_variances_threshold)
            model_variances = model_covariances.diag().copy()  # this returns a 1D numpy array.
            self.prior_variances = model_variances
            #
            if self._verbose:
                print("Forecast covariances:", model_covariances)
                print("Prior Variances", self.prior_variances)
            
            #
            # print("******************* Attempting to do LU decomposition ******************")
            #
            # lu_and_piv = model_covariances.lu_factor()  # this returns two 2D numpy arrays
            lu, piv = lu_factor(model_covariances.get_numpy_array())  # this returns two 2D numpy arrays
            
            # Cleanup = priovious forecast statistics if they exist 
            # (e.g. due to switching from GMM to Gaussian prior if one compoenent is detected)
            self.prior_distribution_statistics = dict()
            
            # Avoid the following first step if 'B' is not required in full in later calculations
            self.prior_distribution_statistics['B'] = model_covariances
            self.prior_distribution_statistics['B_lu'] = lu
            self.prior_distribution_statistics['B_piv'] = piv
            #

        # Construction of a Gaussian Mixture Model representation of the prior distribution
        elif self.prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            #
            gmm_prior_settings = self.filter_configs['gmm_prior_settings']
            #
            ensemble_size = self.sample_size
            state_size = self.model.state_size()
            forecast_ensemble = self.filter_configs['forecast_ensemble']
            forecast_ensemble_numpy = np.empty((ensemble_size, state_size))  # This formulation is for GMM inputs
            # Create a numpy.ndarray containing the forecast ensemble to be used by GMM generator
            for ens_ind in xrange(ensemble_size):
                # GMM builder requires the forecast ensemble to be Nens x Nvar
                forecast_ensemble_numpy[ens_ind, :] = np.squeeze(forecast_ensemble[ens_ind].get_numpy_array())
            # Now generate the GMM model
            gmm_optimal_model, gmm_converged, gmm_lables, gmm_weights, gmm_means, gmm_covariances, gmm_precisions, gmm_optimal_covar_type \
                = utility.generate_gmm_model_info(ensemble=forecast_ensemble_numpy,
                                                  clustering_model=gmm_prior_settings['clustering_model'],
                                                  cov_type=gmm_prior_settings['cov_type'],
                                                  inf_criteria=gmm_prior_settings['inf_criteria'],
                                                  number_of_components=gmm_prior_settings['number_of_components'],
                                                  min_number_of_components=gmm_prior_settings['min_number_of_components'],
                                                  max_number_of_components=gmm_prior_settings['max_number_of_components'],
                                                  min_number_of_points_per_component=gmm_prior_settings['min_number_of_points_per_component'],
                                                  invert_uncertainty_param=gmm_prior_settings['invert_uncertainty_param'],
                                                  verbose=self._verbose
                                                  )
            if not gmm_converged:
                print("The GMM model construction process failed. GMM did NOT converge!")
                print('Switching to Gaussian prior[Single chain]')
                # Fall to Gaussian prior, recalculate the prior distribution info, then apply traditional HMC filter
                self.switch_back_to_GMM = True # set the flag first
                self.prior_distribution = 'gaussian'  # now update the prior distribution in the configurations dict.
                self.filter_configs['prior_distribution'] = 'gaussian'
                #
                try:  # cleanup the GMM statistics attached to the output config if they are still there
                    output_configs['filter_statistics']['gmm_prior_statistics'] = {}
                except(KeyError):
                    # if it is not there, then we are good anyways!
                    pass
                #
                # proceed with prior info generation
                self.generate_prior_info()
                return
                # raise ValueError
            else: 
                #
                # Cleanup = priovious forecast statistics if they exist 
                # (e.g. due to switching from GMM to Gaussian prior if one compoenent is detected)
                self.prior_distribution_statistics = dict()
                
                if isinstance(gmm_weights, np.ndarray):
                    gmm_num_components = gmm_weights.size
                else:
                    gmm_num_components = len(gmm_weights)
                if (max(gmm_lables)+1) > gmm_num_components:
                    print('gmm_lables', gmm_lables)
                    print('gmm_num_components', gmm_num_components)
                    print('gmm_weights', gmm_weights)
                    print("GMM number of components is nonsense!")
                    raise ValueError()
                
                #
                if gmm_num_components > 1:
                    gmm_weights = np.asarray(gmm_weights)
                elif gmm_num_components == 1:
                    # gmm_weights = np.asarray([gmm_weights])
                    #
                    if self.filter_configs['gmm_prior_settings']['use_oringinal_hmc_for_one_comp']:
                        print('Switching to Gaussian prior[Single chain]. One component detected!')
                        # Fall to Gaussian prior, recalculate the prior distribution info, then apply traditional HMC filter
                        self.switch_back_to_GMM = True # set the flag first
                        self.prior_distribution = 'gaussian'  # now update the prior distribution in the configurations dict.
                        self.filter_configs['prior_distribution'] = 'gaussian'
                        self.generate_prior_info()  # proceed with prior info generation
                        return
                    else:
                        pass
                #
                else:
                    print("How is the number of mixture components negative???")
                    raise ValueError()
                #
                # GMM successfully generated. attach proper information to the filter object
                self.prior_distribution_statistics['gmm_optimal_model'] = gmm_optimal_model
                self.prior_distribution_statistics['gmm_optimal_covar_type'] = gmm_optimal_covar_type
                self.prior_distribution_statistics['gmm_weights'] = gmm_weights
                self.prior_distribution_statistics['gmm_lables'] = gmm_lables
                self.prior_distribution_statistics['gmm_num_components'] = gmm_num_components
                self.prior_distribution_statistics['gmm_inf_criteria'] = gmm_prior_settings['inf_criteria']
                #
                # create a list of state vectors to store components' means
                means_list = []
                for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                    mean_vector = self.model.state_vector()
                    mean_vector[:] = gmm_means[comp_ind, :].copy()
                    means_list.append(mean_vector)
                self.prior_distribution_statistics['gmm_means'] = means_list
                #
                prior_variances_threshold = self.filter_configs['prior_variances_threshold']
                #
                # Proper objects are to be created for covariances and/or precisions matrix(es)...
                # Also the logarithm of the determinant of the covariance matrices are evaluated and stored once.
                if gmm_optimal_covar_type in ['diag', 'spherical']:
                    #
                    # create a list of state vectors/matrices to store components' covariances and/or precisions
                    if gmm_covariances is None:
                        self.prior_distribution_statistics['gmm_covariances'] = None
                        self.prior_distribution_statistics['gmm_covariances_det_log'] = None
                        self.prior_variances = None
                    else:
                        covariances_list = []
                        variances_list = []
                        covariances_det_log_list = []
                        prior_variances = self.model.state_vector(); prior_variances[:] = 0.0
                        joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                        mean_vector = self.model.state_vector()  # temporary state vector overwritten each iteration
                        #
                        for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                            # retrieve and store the covariance matrix for each component.
                            variances_vector = self.model.state_vector()
                            gmm_variances = gmm_covariances[comp_ind, :]
                            #
                            # Make sure variances are not small to avoid destroying precisions...
                            tiny_covar_indices =  gmm_variances < prior_variances_threshold
                            if tiny_covar_indices.any():
                                gmm_variances[tiny_covar_indices] += prior_variances_threshold
                            #
                            variances_vector[:] = gmm_variances.copy()
                            covariances_list.append(variances_vector)
                            variances_list.append(variances_vector)
                            # Evaluate the determinant logarithm of the diagonal covariance matrix
                            covariances_det_log = np.sum(np.log(gmm_variances))
                            covariances_det_log_list.append(covariances_det_log)
                            #
                            # find the joint variance and the joint mean for momentum construction
                            prior_variances = prior_variances.add(variances_vector.scale(gmm_weights[comp_ind]))
                            mean_vector[:] = gmm_means[comp_ind, :].copy()
                            joint_mean = joint_mean.add(mean_vector.scale(gmm_weights[comp_ind]))
                        #
                        self.prior_distribution_statistics['gmm_covariances'] = covariances_list
                        self.prior_distribution_statistics['gmm_covariances_det_log'] = np.asarray(covariances_det_log_list)
                        #
                        # proceed with evaluating joint variances and the joint mean
                        if self._gmm_prior_settings['approximate_covariances_from_comp']:
                            # use means and covariances of the components to calculate/approximate mean and covariance matrix of the mixture.
                            mean_vector = self.model.state_vector()
                            for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                                mean_vector[:] = gmm_means[comp_ind, :].copy()
                                # deviation = mean_vector.scale(-1.0).add(joint_mean); deviation.scale(-1.0)
                                deviation = mean_vector.axpy(-1.0, joint_mean, in_place=False)
                                # prior_variances = prior_variances.add(deviation.multiply(deviation, in_place=True).scale(gmm_weights[comp_ind]))
                                prior_variances = prior_variances.add(deviation.square().scale(gmm_weights[comp_ind]))
                            self.prior_variances = prior_variances.get_numpy_array()
                            self.prior_mean = joint_mean
                            # print(self.prior_variances)
                        else:
                            # Evaluate exact mean and covariance from the combined ensemble.
                            joint_mean = self.model.state_vector()
                            joint_mean[:] = np.mean(forecast_ensemble_numpy, 0)
                            self.prior_variances = np.var(forecast_ensemble_numpy, 0)

                    #
                    if gmm_precisions is None:
                        self.prior_distribution_statistics['gmm_precisions'] = None
                    else:
                        if self.prior_variances is None:  # construct prior variances here
                            prior_variances_from_precisions = True
                        else:
                            prior_variances_from_precisions = False
                        if self.prior_distribution_statistics['gmm_covariances_det_log'] is None:  # calculate the det-log of covariances
                            calculate_det_log = True
                            covariances_det_log_list = []
                            variances_list = []
                            joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                        else:
                            calculate_det_log = False
                        gmm_precisions_list = []
                        prior_variances = self.model.state_vector(); prior_variances[:] = 0.0
                        for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                            precisions_vector = self.model.state_vector()
                            precisions_vector[:] = gmm_precisions[comp_ind, :].copy()
                            gmm_precisions_list.append(precisions_vector)
                            variances_list.append(precisions_vector.reciprocal())
                            #
                            if calculate_det_log:
                                # Evaluate the determinant logarithm of the diagonal covariance matrix from precision matrix
                                covariances_det_log = - np.sum(np.log(gmm_precisions[comp_ind, :].copy()))
                                covariances_det_log_list.append(covariances_det_log)
                            #
                            if prior_variances_from_precisions:
                                # find the joint variance and the joint mean for momentum construction
                                prior_variances = prior_variances.add(precisions_vector.reciprocal().scale(gmm_weights[comp_ind]))
                                mean_vector = self.model.state_vector()
                                mean_vector[:] = gmm_means[comp_ind, :].copy()
                                joint_mean = joint_mean.add(mean_vector.scale(gmm_weights[comp_ind]))
                        #
                        self.prior_distribution_statistics['gmm_precisions'] = gmm_precisions_list
                        #
                        if calculate_det_log:
                            self.prior_distribution_statistics['gmm_covariances_det_log'] = np.asarray(covariances_det_log_list)
                        #
                        if prior_variances_from_precisions:
                            # proceed with evaluating joint variances and the joint mean
                            if self._gmm_prior_settings['approximate_covariances_from_comp']:
                                # use means and covariances of the components to calculate/approximate mean and covariance matrix of the mixture.
                                mean_vector = self.model.state_vector()
                                for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                                    mean_vector[:] = gmm_means[comp_ind, :].copy()
                                    # deviation = mean_vector.scale(-1.0).add(joint_mean); deviation.scale(-1.0)
                                    deviation = mean_vector.axpy(-1.0, joint_mean, in_place=False)
                                    # prior_variances = prior_variances.add(deviation.multiply(deviation, in_place=True).scale(gmm_weights[comp_ind]))
                                    prior_variances = prior_variances.add(deviation.square().scale(gmm_weights[comp_ind]))
                                self.prior_variances = prior_variances.get_numpy_array()
                                self.prior_mean = joint_mean
                                # print(self.prior_variances)
                            else:
                                # Evaluate exact mean and covariance from the combined ensemble.
                                joint_mean = self.model.state_vector()
                                joint_mean[:] = np.mean(forecast_ensemble_numpy, 0)
                                self.prior_variances = np.var(forecast_ensemble_numpy, 0)
                                self.prior_mean = joint_mean
                    self.prior_variances_list = variances_list
                    
                #
                elif gmm_optimal_covar_type == 'tied':
                    if gmm_covariances is None:
                        self.prior_distribution_statistics['gmm_covariances'] = None
                        self.prior_distribution_statistics['gmm_covariances_det_log'] = None
                        self.prior_variances = None
                    else:
                        covariances_matrix = self.model.state_matrix()
                        covariances_matrix[:, :] = gmm_covariances[:, :].copy()
                        # 
                        # validate variances:
                        model_covariances_diag = covariances_matrix.diag()
                        tiny_covar_indices =  model_covariances_diag < prior_variances_threshold
                        if tiny_covar_indices.any():
                            covariances_matrix = covariances_matrix.addAlphaI(prior_variances_threshold)
                        #
                         # Spatial decorrelation of the covariance matrix of each compoenent of required
                        if gmm_prior_settings['localize_covariances']:
                            # Apply localization on the full background error covariance matrix.
                            if self._verbose:
                                print('Localizing the tied covariance matrix the mixture components...')
                            loc_func = gmm_prior_settings['localization_function']
                            loc_radius = gmm_prior_settings['localization_radius']
                            if loc_radius is not None and loc_radius is not np.infty:
                                # localization radius seems legit; apply covariance localization now
                                try:
                                    covariances_matrix = self.model.apply_state_covariance_localization(covariances_matrix,
                                                                                                         localization_function=loc_func, 
                                                                                                         localization_radius=loc_radius
                                                                                                         )
                                except(TypeError):
                                    if self._verbose:
                                        print("Covariance localization with the given settings failed. \n \
                                               Trying covariance localization with default model settings..."
                                               )
                                    # Try the localization with default settings in the model
                                    covariances_matrix = self.model.apply_state_covariance_localization(covariances_matrix)
                            #
                        singular_vals = np.linalg.svd(covariances_matrix.get_numpy_array(), compute_uv=False)
                        self.prior_distribution_statistics['gmm_covariances'] = covariances_matrix
                        covariances_det_log = np.sum(np.log(singular_vals))
                        # Evaluate and replicate the logarithm of the covariances matrix's determinant for all components.
                        self.prior_distribution_statistics['gmm_covariances_det_log'] \
                            = np.asarray([covariances_det_log for i in xrange(self.prior_distribution_statistics['gmm_num_components'])])

                        # find the joint variance and the joint mean for momentum construction
                        prior_variances = self.model.state_vector()
                        prior_variances[:] = covariances_matrix.diag().copy()
                        joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                        for comp_ind in xrange(gmm_num_components):
                            mean_vector = self.model.state_vector()
                            mean_vector[:] = gmm_means[comp_ind, :].copy()
                            joint_mean = joint_mean.add(mean_vector.scale(gmm_weights[comp_ind]))
                        #
                        # proceed with evaluating joint variances and the joint mean
                        if self._gmm_prior_settings['approximate_covariances_from_comp']:
                            # use means and covariances of the components to calculate/approximate mean and covariance matrix of the mixture.
                            mean_vector = self.model.state_vector()
                            for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                                mean_vector[:] = gmm_means[comp_ind, :].copy()
                                # deviation = mean_vector.scale(-1.0).add(joint_mean); deviation.scale(-1.0)
                                deviation = mean_vector.axpy(-1.0, joint_mean, in_place=False)
                                # prior_variances = prior_variances.add(deviation.multiply(deviation, in_place=True).scale(gmm_weights[comp_ind]))
                                prior_variances = prior_variances.add(deviation.square().scale(gmm_weights[comp_ind]))
                            self.prior_variances = prior_variances.get_numpy_array()
                            self.prior_mean = joint_mean
                            # print(self.prior_variances)
                        else:
                            # Evaluate exact mean and covariance from the combined ensemble.
                            joint_mean = self.model.state_vector()
                            joint_mean[:] = np.mean(forecast_ensemble_numpy, 0)
                            self.prior_variances = np.var(forecast_ensemble_numpy, 0)
                            self.prior_mean = joint_mean
                    #
                    if gmm_precisions is None:
                        self.prior_distribution_statistics['gmm_precisions'] = None
                    else:
                        if self.prior_variances is None:  # construct prior variances here
                            prior_variances_from_precisions = True
                        else:
                            prior_variances_from_precisions = False
                        if self.prior_distribution_statistics['gmm_covariances_det_log'] is None:  # calculate the det-log of covariances
                            calculate_det_log = True
                        else:
                            calculate_det_log = False
                        #
                        precisions_matrix = self.model.state_matrix()
                        precisions_matrix[:, :] = gmm_precisions[:, :].copy()
                        if gmm_prior_settings['localize_covariances']:
                            # Apply localization on the full background error covariance matrix.
                            # quite clear this is really a bad idea!
                            if self._verbose:
                                print('Localizing the tied covariance matrix the mixture components...')
                            loc_func = gmm_prior_settings['localization_function']
                            loc_radius = gmm_prior_settings['localization_radius']
                            if loc_radius is not None and loc_radius is not np.infty:
                                # localization radius seems legit; apply covariance localization now
                                try:
                                    precisions_matrix = self.model.apply_state_covariance_localization(precisions_matrix.inv(),
                                                                                                        localization_function=loc_func, 
                                                                                                        localization_radius=loc_radius
                                                                                                        ).inv()  # quite clear this is bad
                                except(TypeError):
                                    if self._verbose:
                                        print("Covariance localization with the given settings failed. \n \
                                               Trying covariance localization with default model settings..."
                                               )
                                    # Try the localization with default settings in the model
                                    precisions_matrix = self.model.apply_state_covariance_localization(precisions_matrix.inv()).inv()
                            #
                            
                        self.prior_distribution_statistics['gmm_precisions'] = precisions_matrix
                        
                        # calculate covariance_det_log if necessary
                        if calculate_det_log:
                            # Evaluate and replicate the logarithm of the covariances matrix's determinant for all components.
                            if gmm_prior_settings['localize_covariances']:
                                singular_vals = np.linalg.svd(precisions_matrix.get_numpy_array(), compute_uv=False)
                            else:
                                singular_vals = np.linalg.svd(gmm_precisions[:, :], compute_uv=False)

                            covariances_det_log = - np.sum(np.log(singular_vals))
                            self.prior_distribution_statistics['gmm_covariances_det_log'] \
                                = np.asarray([covariances_det_log for i in xrange(self.prior_distribution_statistics['gmm_num_components'])])

                        # calculate variances_from precisions if necessary
                        if prior_variances_from_precisions:
                            prior_variances = self.model.state_vector()
                            prior_variances[:] = precisions_matrix.diag().copy()
                            prior_variances = prior_variances.reciprocal()
                            joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                            for comp_ind in xrange(gmm_num_components):
                                mean_vector = self.model.state_vector()
                                mean_vector[:] = gmm_means[comp_ind, :].copy()
                                joint_mean = joint_mean.add(mean_vector.scale(gmm_weights[comp_ind]))
                            #
                            # proceed with evaluating joint variances and the joint mean
                            if self._gmm_prior_settings['approximate_covariances_from_comp']:
                                # use means and covariances of the components to calculate/approximate mean and covariance matrix of the mixture.
                                mean_vector = self.model.state_vector()
                                for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                                    mean_vector[:] = gmm_means[comp_ind, :].copy()
                                    # deviation = mean_vector.scale(-1.0).add(joint_mean); deviation.scale(-1.0)
                                    deviation = mean_vector.axpy(-1.0, joint_mean, in_place=False)
                                    # prior_variances = prior_variances.add(deviation.multiply(deviation, in_place=True).scale(gmm_weights[comp_ind]))
                                    prior_variances = prior_variances.add(deviation.square().scale(gmm_weights[comp_ind]))
                                self.prior_variances = prior_variances.get_numpy_array()
                                self.prior_mean = joint_mean
                                # print(self.prior_variances)
                            else:
                                # Evaluate exact mean and covariance from the combined ensemble.
                                joint_mean = self.model.state_vector()
                                joint_mean[:] = np.mean(forecast_ensemble_numpy, 0)
                                self.prior_variances = np.var(forecast_ensemble_numpy, 0)
                                self.prior_mean = joint_mean
                    self.prior_variances_list = [prior_variances] 

                #
                elif gmm_optimal_covar_type == 'full':
                    if gmm_covariances is None:
                        self.prior_distribution_statistics['gmm_covariances'] = None
                        self.prior_distribution_statistics['gmm_covariances_det_log'] = None
                        self.prior_variances = None
                    else:
                        covariances_list = []
                        variances_list = []
                        covariances_det_log_list = []
                        prior_variances = self.model.state_vector(); prior_variances[:] = 0.0
                        joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                        for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                            covariances_matrix = self.model.state_matrix()
                            covariances_matrix[:, :] = np.squeeze(gmm_covariances[comp_ind, :, :]).copy()
                            #
                            # validate covariances:
                            model_covariances_diag = covariances_matrix.diag()
                            tiny_covar_indices =  model_covariances_diag < prior_variances_threshold
                            if tiny_covar_indices.any():
                                covariances_matrix = covariances_matrix.addAlphaI(prior_variances_threshold)
                            #
                            covariances_list.append(covariances_matrix)
                            # retrieve and store the covariance matrix for each component.
                            variances_vector = self.model.state_vector()
                            variances_vector[:] = covariances_matrix.diag().copy()
                            variances_list.append(variances_vector)
                            #
                            # Evaluate and replicate the logarithm of the covariances matrix's determinant for all components.
                            singular_vals = np.linalg.svd(covariances_matrix.get_numpy_array(), compute_uv=False)
                            covariances_det_log = np.sum(np.log(singular_vals))
                            covariances_det_log_list.append(covariances_det_log)
                            #
                            # find the joint variance and the joint mean for momentum construction
                            prior_variances = prior_variances.add(variances_vector.scale(gmm_weights[comp_ind]))
                            mean_vector = self.model.state_vector()
                            mean_vector[:] = gmm_means[comp_ind, :].copy()
                            joint_mean = joint_mean.add(mean_vector.scale(gmm_weights[comp_ind]))

                        self.prior_distribution_statistics['gmm_covariances'] = covariances_list
                        self.prior_distribution_statistics['gmm_covariances_det_log'] = np.asarray(covariances_det_log_list)
                        #
                        # proceed with evaluating joint variances and the joint mean
                        if self._gmm_prior_settings['approximate_covariances_from_comp']:
                            # use means and covariances of the components to calculate/approximate mean and covariance matrix of the mixture.
                            mean_vector = self.model.state_vector()
                            for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                                mean_vector[:] = gmm_means[comp_ind, :].copy()
                                # deviation = mean_vector.scale(-1.0).add(joint_mean); deviation.scale(-1.0)
                                deviation = mean_vector.axpy(-1.0, joint_mean, in_place=False)
                                # prior_variances = prior_variances.add(deviation.multiply(deviation, in_place=True).scale(gmm_weights[comp_ind]))
                                prior_variances = prior_variances.add(deviation.square().scale(gmm_weights[comp_ind]))
                            self.prior_variances = prior_variances.get_numpy_array()
                            self.prior_mean = joint_mean
                            # print(self.prior_variances)
                        else:
                            # Evaluate exact mean and covariance from the combined ensemble.
                            joint_mean = self.model.state_vector()
                            joint_mean[:] = np.mean(forecast_ensemble_numpy, 0)
                            self.prior_variances = np.var(forecast_ensemble_numpy, 0)
                            self.prior_mean = joint_mean
                    #
                    if gmm_precisions is None:
                        self.prior_distribution_statistics['gmm_precisions'] = None
                    else:
                        if gmm_prior_settings['localize_covariances'] and self.prior_distribution_statistics['gmm_covariances'] is None:
                            print("This is really disadvantageous! \n \
                                   Applying localization given only precisions requires many many inverse calculations!\n \
                                   I am saving you from yourself!")
                            raise NotImplementedError()
                        #
                        if self.prior_variances is None:  # construct prior variances here
                            prior_variances_from_precisions = True
                        else:
                            prior_variances_from_precisions = False
                        if self.prior_distribution_statistics['gmm_covariances_det_log'] is None:  # calculate the det-log of covariances
                            calculate_det_log = True
                            covariances_det_log_list = []
                            joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                        else:
                            calculate_det_log = False
                        #
                        gmm_precisions_list = []
                        variances_list = []
                        prior_variances = self.model.state_vector(); prior_variances[:] = 0.0
                        for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                            precisions_matrix = self.model.state_matrix()
                            precisions_matrix[:, :] = gmm_precisions[comp_ind, :, :].copy()
                            gmm_precisions_list.append(precisions_matrix)
                            precisions_vector = self.model.state_vector()
                            precisions_vector[:] = precisions_matrix.diag().copy()
                            variances_list.append(precisions_vector.reciprocal())
                            #
                            if calculate_det_log:
                                # Evaluate the determinant logarithm of the diagonal covariance matrix from precision matrix
                                singular_vals = np.linalg.svd(gmm_precisions[comp_ind, :, :], compute_uv=False)
                                covariances_det_log = - np.sum(np.log(singular_vals))
                                covariances_det_log_list.append(covariances_det_log)
                            #
                            if prior_variances_from_precisions:
                                # find the joint variance and the joint mean for momentum construction
                                prior_variances = prior_variances.add(precisions_vector.reciprocal().scale(gmm_weights[comp_ind]))
                                mean_vector = self.model.state_vector()
                                mean_vector[:] = gmm_means[comp_ind, :].copy()
                                joint_mean = joint_mean.add(mean_vector.scale(gmm_weights[comp_ind]))
                        #
                        self.prior_distribution_statistics['gmm_precisions'] = gmm_precisions_list
                        #
                        if calculate_det_log:
                            self.prior_distribution_statistics['gmm_covariances_det_log'] = np.asarray(covariances_det_log_list)
                        #
                        if prior_variances_from_precisions:
                            # proceed with evaluating joint variances and the joint mean
                            if self._gmm_prior_settings['approximate_covariances_from_comp']:
                                # use means and covariances of the components to calculate/approximate mean and covariance matrix of the mixture.
                                mean_vector = self.model.state_vector()
                                for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                                    mean_vector[:] = gmm_means[comp_ind, :].copy()
                                    # deviation = mean_vector.scale(-1.0).add(joint_mean); deviation.scale(-1.0)
                                    deviation = mean_vector.axpy(-1.0, joint_mean, in_place=False)
                                    # prior_variances = prior_variances.add(deviation.multiply(deviation, in_place=True).scale(gmm_weights[comp_ind]))
                                    prior_variances = prior_variances.add(deviation.square().scale(gmm_weights[comp_ind]))
                                self.prior_variances = prior_variances.get_numpy_array()
                                self.prior_mean = joint_mean
                                # print(self.prior_variances)
                            else:
                                # Evaluate exact mean and covariance from the combined ensemble.
                                joint_mean = self.model.state_vector()
                                joint_mean[:] = np.mean(forecast_ensemble_numpy, 0)
                                self.prior_variances = np.var(forecast_ensemble_numpy, 0)
                                self.prior_mean = joint_mean
                    
                    self.prior_variances_list = variances_list
                    
                #
                else:
                    print("This is unexpected!. optimal_covar_type = '%s' " % gmm_optimal_covar_type)
                    raise ValueError()

            # Add GMM statistics to filter statistics
            gmm_statistics = dict(gmm_num_components=self.prior_distribution_statistics['gmm_num_components'],
                                  gmm_optimal_covar_type=self.prior_distribution_statistics['gmm_optimal_covar_type'],
                                  gmm_weights=self.prior_distribution_statistics['gmm_weights'],
                                  gmm_lables=self.prior_distribution_statistics['gmm_lables'],
                                  gmm_covariances_det_log=self.prior_distribution_statistics['gmm_covariances_det_log'],
                                  gmm_inf_criteria=self.prior_distribution_statistics['gmm_inf_criteria']
                                  )
            self.output_configs['filter_statistics']['gmm_prior_statistics'] = gmm_statistics
        #
        else:
            print("Prior distribution [%s] is not yet supported!" % self.filter_configs['prior_distribution'])
            raise ValueError()


    #
    def analysis(self):
        """
        Analysis step.
        
        Args:
        
        Returns:
            chain_diagnostics: a dictionary containing the diagnostics of the chain such as acceptance rate, and
                effective sample size, etc.
            
        """
        #
        # get the forecast state as the mean of the forecast ensemble. Will not be used for GMM!
        initialize_chain_strategy = self.filter_configs['gmm_prior_settings']['initialize_chain_strategy']
        if initialize_chain_strategy is None:
            initialize_chain_strategy = 'forecast_mean'
        else:
            assert isinstance(initialize_chain_strategy, str)
            initialize_chain_strategy = initialize_chain_strategy.lower()
            initialize_chain_strategies = ['forecast_mean', 'highest_weight']
            if initialize_chain_strategy not in initialize_chain_strategies:
                raise ValueError("Chain initialization policy is not recognized. \n"
                                 "You have to pass on of the values:\n %s" % repr(initialize_chain_strategies))
        if initialize_chain_strategy == 'forecast_mean':
            try:
                initial_state = self.prior_mean
            except(ValueError, NameError, AttributeError):
                initial_state = None
            finally:
                if initial_state is None:
                    try:
                        initial_state = self.forecast_state
                    except(ValueError, NameError, AttributeError):
                        initial_state = None
            if initial_state is None:
                initial_state = utility.ensemble_mean(self.filter_configs['forecast_ensemble'])

        elif initialize_chain_strategy == 'highest_weight':
            gmm_prior_weights = self.prior_distribution_statistics['gmm_weights']
            winner_index = np.argmax(gmm_prior_weights)  # for multiple occurrences, the first winner is taken
            initial_state = self.prior_distribution_statistics['gmm_means'][winner_index].copy()
        else:
            raise ValueError("Chain initialization policy is not recognized. \n"
                             "You have to pass on of the values:\n %s" % repr(initialize_chain_strategies))

        # print('forecast_state', forecast_state)
        # print('obs', observation)
        # print('ref_s', self.filter_configs['reference_state'])
        #
        # TODO: After finishing Traditional HMC implementation, update this branching to call:
        # TODO:     1- Traditional HMC: _hmc_produce_ensemble
        # TODO:     2- No-U-Turn: _nuts_hmc_produce_ensemble
        # TODO:     3- Riemannian HMC: _rm_hmc_produce_ensemble
        # TODO:     4- My new HMC with automatic adjustment of parameters
        
        # 
        analysis_ensemble, chain_diagnostics = self._hmc_produce_ensemble(initial_state=initial_state, verbose=self._verbose)

        # inflation of the analysis ensemble can be considered with MultiChain MCMC sampler if small steps are taken
        inflation_fac = self.filter_configs['analysis_inflation_factor']
        if inflation_fac > 1.0:
            if self._verbose:
                print('Inflating the analysis ensemble...')
            if self._verbose:
                in_place = False
            else:
                in_place = True
            #
            inflated_ensemble = utility.inflate_ensemble(ensemble=analysis_ensemble, inflation_factor=inflation_fac, in_place=in_place)
            #
            if self._verbose:
                print('inflated? : ', (analysis_ensemble[0][:]!=inflated_ensemble[0][:]).any())
            #
            self.filter_configs['analysis_ensemble'] = inflated_ensemble
        else:
            pass
            
        # Update the analysis ensemble in the filter_configs
        self.filter_configs['analysis_ensemble'] = analysis_ensemble  # list reconstruction processes copying
        
        # update analysis_state
        self.filter_configs['analysis_state'] = utility.ensemble_mean(self.filter_configs['analysis_ensemble'])
        #

        # Check if the gaussian prior needs to be updated
        if self.switch_back_to_GMM:
            self.switched_to_Gaussian_prior = True  # for proper results saving
            print('Switching back to GMM prior...')
            self.switch_back_to_GMM = False
            self.prior_distribution = 'gmm'
            self.filter_configs['prior_distribution'] = 'gmm'
        else:
            self.switched_to_Gaussian_prior = False
        
        # Add chain diagnostics to the output_configs dictionary for proper outputting:
        self.output_configs['filter_statistics'].update({'chain_diagnostics':chain_diagnostics})
        #
        return chain_diagnostics
        # ==========================================(End Analysis step)==========================================
        #

    #
    def _hmc_produce_ensemble(self, initial_state=None, analysis_ensemble_in=None, verbose=False):
        """
        Use HMC sampling scheme to construct the Markov chain and collect ensembles from the stationary distribution.
        This is the function you should call first to generate the ensemble, everything else is called/controlled by
        this method.
        
        Args:
            initial_state: state (position) used to initialize the Markov chain
            analysis_ensemble_in (default None): an analysis ensemble (list of model.state_vector objects) 
                that is updated by the collected ensemble (in place) instead of creating a new list.
            verbose (default False): can be used for extra on-screen printing while debugging, and testing
            
        Returns:
            analysis_ensemble: the samples collected by the HMC sampler from the posterior distribution
            chain_diagnostics: a dictionary containing the diagnostics of the chain such as acceptance rate, and
                effective sample size, etc.
            
        """
        # Collect all required parameters:
        # Check the initial state and initialize a placeholder for the proposed state
        if initial_state is None:
            # get the forecast state as the mean of the forecast ensemble. Will not be used for GMM!
            initialize_chain_strategy = self.filter_configs['gmm_prior_settings']['initialize_chain_strategy']
            if initialize_chain_strategy is None:
                initialize_chain_strategy = 'forecast_mean'
            else:
                assert isinstance(initialize_chain_strategy, str)
                initialize_chain_strategy = initialize_chain_strategy.lower()
                initialize_chain_strategies = ['forecast_mean', 'highest_weight']
                if initialize_chain_strategy not in initialize_chain_strategies:
                    print("Chain initialization policy is not recognized. \n"
                                     "You have to pass on of the values:\n %s" % repr(initialize_chain_strategies))
                    raise ValueError()
            
            if initialize_chain_strategy == 'forecast_mean':
                try:
                    initial_state = self.prior_mean.copy()
                except(ValueError, NameError, AttributeError):
                    initial_state = None
                finally:
                    if initial_state is None:
                        try:
                            initial_state = self.forecast_state.copy()
                        except(ValueError, NameError, AttributeError):
                            initial_state = utility.ensemble_mean(self.filter_configs['forecast_ensemble'])
            elif initialize_chain_strategy == 'highest_weight':
                gmm_prior_weights = self.prior_distribution_statistics['gmm_weights']
                winner_index = np.argmax(gmm_prior_weights)  # for multiple occurrences, the first winner is taken
                initial_state = self.prior_distribution_statistics['gmm_means'][winner_index].copy()
            else:
                print("Chain initialization policy is not recognized. \n"
                                 "You have to pass on of the values:\n %s" % repr(initialize_chain_strategies))
                raise ValueError()
        else:
            pass

        # Check the placeholder of the analysis ensemble to avoid reconstruction
        if analysis_ensemble_in is None:
            # analysis_ensemble = self.filter_configs['analysis_ensemble']
            analysis_ensemble = None
        else:
            analysis_ensemble = analysis_ensemble_in

        if analysis_ensemble is None:
            analysis_ensemble = []
            append_members = True
        else:
            append_members = False

        if len(analysis_ensemble) != 0 and len(analysis_ensemble) != self.sample_size:
            print("Dimension mismatch between size of the analysis ensemble and the sample size!")
            raise ValueError()

        # # Get the name of the symplectic integrator
        # symplectic_integrator = self.symplectic_integrator

        # Construct a placeholder for the momentum vectors to avoid reconstruction and copying
        current_momentum = self._momentum.generate_momentum()
        # proposed_momentum = current_momentum.copy()

        #
        if self.tempering_scheme is not None:
            tempering_scheme = self.tempering_scheme
            tempering_parameters = self.tempering_parameters
            # extract parameters based on supported tempering schemes
            if tempering_scheme not in self._supported_tempering_schemes:
                raise NotImplementedError("The tempering scheme [%s] is not supported yet!" % tempering_scheme)
            else:
                # Construct tempered chains and collect ensembles from the Coolest chain
                pass
            #
            # TODO: Finish the tempered chains implementation after finishing single chain counterpart.
            # TODO: consider separation to separate method
            print("Tempered chains are not ready yet!")
            raise NotImplementedError()
            #
        else:
            # Construct a single chain
            if self.optimize_to_converge:
                # replace the burn-in stage by an optimization step
                print("Optimization step is not implemented yet!")
                raise NotImplementedError()
            else:
                # Burn-in based on a fixed number of steps
                diagnostic_scheme = self.chain_parameters['convergence_diagnostic']  # for early termination
                #
                # prepare to save chain diagnostics (statistics)
                acceptance_flags = []
                acceptance_probabilities = []
                uniform_random_numbers = []
                #
                
                # start constructing the chain(s)
                burn_in_steps = self.chain_parameters['burn_in_steps']
                current_state = initial_state.copy()
                for burn_ind in xrange(burn_in_steps):
                    # carry-out one HMC cycle and get the next state.
                    # (replace the old one... This may be hard if diagnostic tools are used)
                    # generate a fresh synthetic momentum vector. I use the place holder to avoid recreating state vectors
                    current_momentum = self._momentum.generate_momentum(momentum_vec=current_momentum)
                        
                    try:
                        proposed_state, proposed_momentum = self._hmc_propose_state(current_state=current_state,
                                                                                    current_momentum=current_momentum)
                        
                        traj = self.model.integrate_state(initial_state=proposed_state.copy(),
                                                 checkpoints=self.filter_configs['timespan'])
                        # print(">>>>>>>==================>>>>>>>. CURRENT_RMSE", utility.calculate_rmse(current_state, traj[-1]))
                        # print(">>>>>>>==================>>>>>>>. proposal_RMSE", utility.calculate_rmse(proposed_state, traj[-1]))
                                               
                        # add current_state to the ensemble as it is the most updated membe
                        accept_proposal, energy_loss, a_n, u_n = self._hmc_MH_accept_reject(current_state=current_state,
                                                                                            proposed_state=proposed_state,
                                                                                            current_momentum=current_momentum,
                                                                                            proposed_momentum=proposed_momentum,
                                                                                            verbose=verbose
                                                                                            )
                    except(ValueError):
                        accept_proposal, energy_loss, a_n, u_n = False, 0, 0, 1
                    
                    # save probabilities for chain diagnostics evaluation
                    acceptance_probabilities.append(a_n)
                    uniform_random_numbers.append(u_n)  
                    #
                    if accept_proposal:
                        acceptance_flags.append(1)
                        if verbose:
                            print('Burning step [%d]: J=%f' % (burn_ind, self._hmc_total_energy(proposed_state, proposed_momentum)))
                        current_state = proposed_state.copy()  # Do we need to copy?                        
                    else:
                        acceptance_flags.append(0)
                        # do nothing, unless we decide later to keep all or some of the burned states, may be for diagnostics or so!
                        pass
            #
            # Burn-in stage is over. Start collecting samples:
            mixing_steps = self.chain_parameters['mixing_steps']

            # Construct a placeholder for the momentum vectors to avoid reconstruction and copying
            try:
                current_momentum
            except (ValueError, AttributeError, NameError):
                current_momentum = self._momentum.generate_momentum()

            # current_state = initial_state.copy()  # un-necessary copy!
            #
            for ens_ind in xrange(self.sample_size):
                # To reduce correlation (mixing_steps-1) samples are discarded
                for mixing_ind in xrange(1, mixing_steps+1):
                    # propose a state given xn, decide whether to accept or reject xn
                    # generate a fresh synthetic momentum vector. I use the place holder to avoid recreating state vectors
                    current_momentum = self._momentum.generate_momentum(momentum_vec=current_momentum)
                    try:
                        proposed_state, proposed_momentum = self._hmc_propose_state(current_state=current_state.copy(),
                                                                                    current_momentum=current_momentum.copy())
                        # update the ensemble
                        accept_proposal, energy_loss, a_n, u_n = self._hmc_MH_accept_reject(current_state=current_state.copy(),
                                                                                            current_momentum=current_momentum.copy(),
                                                                                            proposed_state=proposed_state.copy(),
                                                                                            proposed_momentum=proposed_momentum.copy(),
                                                                                            verbose=verbose
                                                                                            )
                    except(ValueError):
                        accept_proposal, energy_loss, a_n, u_n = False, 0, 0, 1
                    
                    # save probabilities for chain diagnostics evaluation
                    acceptance_probabilities.append(a_n)
                    uniform_random_numbers.append(u_n)  
                    #
                    if accept_proposal:
                        acceptance_flags.append(1)
                        current_state = proposed_state.copy()
                        #
                        if verbose:
                            print("Ensemble member [%d] Mixing step [%d] : u_n=%4.3f, a_n=%4.3f" % (ens_ind, mixing_ind, u_n, a_n))
                            print('Mixing: J=', self._hmc_total_energy(proposed_state, proposed_momentum))
                    else:
                        acceptance_flags.append(0)
                        # do nothing, unless we decide later to keep all or some of the thinning states, may be for diagnostics or so!
                        pass
                    
                # add current_state to the ensemble as it is the most updated membe
                
                if append_members:
                    analysis_ensemble.append(current_state)  # append already copies the object and add reference to it.
                else:
                    analysis_ensemble[ens_ind][:] = current_state[:].copy()  # same. no need to copy.                
        #
        if self._verbose:
            liner = "--."*30
            print("\n%s\nEnsemble produced, and ready to return...\n%s\n" %(liner, liner))
            print("sample_size:", self.sample_size)
            print("acceptance_flags:", acceptance_flags)
            print("observation", self.filter_configs['observation'])
            print("reference_state:", self.filter_configs['reference_state'][100:200])
            print("forecast_state:", self.filter_configs['forecast_state'][100:200])
            print("analysis_state:", utility.ensemble_mean(analysis_ensemble)[100:200])
            print("Prior Variances", self.prior_variances)
            print("\nNow return...\n%s\n" % liner)
            
        # prepare chain statistics and diagnostic measures to be returned:
        acceptance_rate = float(np.asarray(acceptance_flags).sum()) / len(acceptance_flags) * 100.0
        rejection_rate = (100.0 - acceptance_rate)
        chain_diagnostics = dict(acceptance_rate=acceptance_rate,
                                 rejection_rate=rejection_rate,
                                 acceptance_probabilities=acceptance_probabilities,
                                 acceptance_flags=acceptance_flags,
                                 uniform_random_numbers=uniform_random_numbers
                                 )
        
        # return the collected ensembles, and chain diagnostics.
        # print('first member test', isinstance(analysis_ensemble[0], np.ndarray))
        return analysis_ensemble, chain_diagnostics
        #
    
    #
    def _hmc_MH_accept_reject(self, current_state, current_momentum, proposed_state, proposed_momentum, verbose=False):
        """
        Metropolis-Hastings accept/reject criterion based on loss of energy between current and proposed states
        proposed_state, proposed_momentum are the result of forward propagation of current_state, current_momentum
        using the symplectic integrator
        
        Args:
            current_state: current (position) state of the chain
            current_momentum: current (momentum) state of the chain
            proposed_state: proposed (position) state of the chain
            proposed_momentum: current (momentum) state of the chain
            verbose (default False): can be used for extra on-screen printing while debugging, and testing
            
        Returns:
            accept_state: True/False, whether to accept the proposed state/momentum or reject them
            energy_loss: the difference between total energy of the proposed and the current pair
            a_n: Metropolis-Hastings (MH) acceptance probability
            u_n: the uniform random number compared to a_n to accept/reject the sate (MH)
            
        """
        symplectic_integrator = self.symplectic_integrator.lower()
        #
        if symplectic_integrator in ['verlet', '2-stage', '3-stage', '4-stage']:
            current_energy = self._hmc_total_energy(current_state, current_momentum)
            proposed_energy = self._hmc_total_energy(proposed_state, proposed_momentum)
            energy_loss = proposed_energy - current_energy
            if verbose:
                print('energy_loss', energy_loss)
            if abs(energy_loss) > 500:  # this should avoid overflow errors
                if energy_loss < 0:
                    sign = -1
                else:
                    sign = 1
                energy_loss = sign * 500
        elif symplectic_integrator in ['hilbert', 'infinite']:
            raise NotImplementedError("Infinite dimensional symplectic integrator is being investigated closely!")
        else:
            raise ValueError("Unrecognized symplectic integrator [%s]" % symplectic_integrator)

        # calculate probability of acceptance and decide whether to accept or reject the proposed state
        a_n = min(1.0, np.exp(-energy_loss))
        if verbose:
            print('MH-acceptance_probability', a_n)
        u_n = np.random.rand()
        if a_n > u_n:
            accept_state = True
        else:
            accept_state = False
        #
        return accept_state, energy_loss, a_n, u_n
        #
    
    #
    def _hmc_total_energy(self, current_state, current_momentum):
        """
        Evaluate the total energy function (the Hamiltonian) for a given position/momentum pair
        Total Energy: Kinetic + Potential energy 
        
        Args:
            current_state: current (position) state of the chain
            current_momentum: current (momentum) state of the chain
            
        Returns:
            total_energy: Kinetic + Potential energy of the passed position/momentum pair
            
        """
        potential_energy = self._hmc_potential_energy_value(current_state)
        kinetic_energy = self._momentum.evaluate_kinetic_energy(current_momentum)
        total_energy = potential_energy + kinetic_energy
        #
        return total_energy
        #

    #
    def _hmc_propose_state(self, current_state, current_momentum=None):
        """
        Given the current state, advance the chain and generate a new state.
        This generally means: build a full Hamiltonian Trajectory.
        If proposed_state is not None, it will be updated, otherwise a new object of the same type as current_state
        will be created and returned.
        
        Args:
            current_state: current (position) state of the chain
            current_momentum: current (momentum) state of the chain
            
        Returns:
            proposed_state: position variable at the end of the Hamiltonian trajectory
            proposed_momentum: momentum variable at the end of the Hamiltonian trajectory
            
                proposed_state, proposed_momentum are the result of forward propagation of 
                current_state, current_momentum using the symplectic integrator
        """
        # generate a momentum vector if none is passed
        if current_momentum is None:
            self._momentum.generate_momentum()
        #
        # construct a Hamiltonian trajectory to get a proposed state...
        proposed_state, proposed_momentum = self._prob_Hamiltonian_trajectory(current_state=current_state, current_momentum=current_momentum)
        #
        return proposed_state, proposed_momentum
        #

    #
    def _prob_Hamiltonian_trajectory(self, current_state, current_momentum, perturb_step_size=True):
        """
        Given the current tuple in phase-space (current_state, current_momentum) construct a Hamiltonian trajectory and
        return the last state and momentum as proposed_state and proposed_momentum.
        Make sure the current state is not altered.
        
        Args:
            current_state: current (position) state of the chain
            current_momentum: current (momentum) state of the chain
            perturb_step_size (default True): add random perturbation to the pseudo-step size of the symplectic 
                integrator to reduce independence of the results on the specific choice of the step size.
        
        Returns:
            proposed_state: position variable at the end of the Hamiltonian trajectory
            proposed_momentum: momentum variable at the end of the Hamiltonian trajectory
            
                proposed_state, proposed_momentum are the result of forward propagation of 
                current_state, current_momentum using the symplectic integrator
            
        """
        # 1- Retrieve the symplectic integrator name,
        symplectic_integrator = self.symplectic_integrator
        # 2- Retrieve the symplectic integrator parameters,
        symplectic_integrator_parameters = self.symplectic_integrator_parameters
        m = symplectic_integrator_parameters['num_steps']  # number of steps should be generally large (not really!)
        h0 = symplectic_integrator_parameters['step_size']  # reference step size should be such that rejection rate is 25% to 30%
        if perturb_step_size:
            u = (np.random.rand() - 0.5) * 0.4  # perturb step-size:
            h = (1 + u) * h0
        else:
            h = h0

        # create placeholders for proposed and current states
        proposed_state = current_state.copy()
        proposed_momentum = current_momentum.copy()
        #
        if symplectic_integrator == 'verlet':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            for step_ind in xrange(m):
                temp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.add(temp_momentum.scale(h/2.0))
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)  # this has to be returned on the form of a StateVector as well!
                proposed_momentum = proposed_momentum.axpy(-h, dj)
                #
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.axpy(h/2.0, tmp_momentum)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        #
        elif symplectic_integrator == '2-stage':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            a1 = (3 - np.sqrt(3.0)) / 6.0
            a2 = 1 - 2.0 * a1
            b1 = 0.5
            for step_ind in xrange(m):
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.add(tmp_momentum.scale(a1*h))
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)
                proposed_momentum = proposed_momentum.add(dj.scale(-b1 * h))
                #
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.add(tmp_momentum.scale(a2*h))
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)
                proposed_momentum = proposed_momentum.add(dj.scale(-b1 * h))
                #
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.add(tmp_momentum.scale(a1*h))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        #
        elif symplectic_integrator == '3-stage':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            a1 = 0.11888010966548
            b1 = 0.29619504261126
            a2 = 0.5 - a1
            b2 = 1 - 2*b1
            for step_ind in xrange(m):
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.axpy(a1*h, tmp_momentum)
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)
                proposed_momentum = proposed_momentum.axpy(-b1*h, dj)
                #                
                
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.axpy(a2*h, tmp_momentum)
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)
                proposed_momentum = proposed_momentum.axpy(-b2*h, dj)
                #
                
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.axpy(a2*h, tmp_momentum)
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)
                proposed_momentum = proposed_momentum.axpy(-b1 * h, dj)
                #
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.axpy((a1*h), tmp_momentum)
                
        #
        elif symplectic_integrator == '4-stage':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            a1 = 0.071353913450279725904
            a2 = 0.268548791161230105820
            b1 = 0.191667800000000000000
            b2 = 0.5 - b1
            a3 = 1.0 - 2.0 * (a1 + a2)
            for step_ind in xrange(m):
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.add(tmp_momentum.scale(a1*h))
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)
                proposed_momentum = proposed_momentum.add(dj.scale(-b1 * h))
                #
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.add(tmp_momentum.scale(a2*h))
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)
                proposed_momentum = proposed_momentum.add(dj.scale(-b2 * h))
                #
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.add(tmp_momentum.scale(a3*h))
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)
                proposed_momentum = proposed_momentum.add(dj.scale(-b2 * h))
                #
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.add(tmp_momentum.scale(a2*h))
                #
                dj = self._hmc_potential_energy_gradient(proposed_state)
                proposed_momentum = proposed_momentum.add(dj.scale(-b1 * h))
                #
                tmp_momentum = self._momentum.inv_mass_mat_prod_momentum(proposed_momentum, in_place=False)
                proposed_state = proposed_state.add(tmp_momentum.scale(a1*h))
        #
        elif symplectic_integrator == 'hilbert':
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            print("Hilbert Integrator is not yet implemented. I am checking some important stuff first in the paper!")
            raise NotImplementedError()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        else:
            print("Unrecognized symplectic integrator of the Hamiltonian dynamics")
            raise ValueError()
        #
        return proposed_state, proposed_momentum
        #

    #
    def _hmc_potential_energy(self, current_state):
        """
        Evaluate the potential energy value and gradient
        
        Args:
            current_state: current (position) state of the chain
        
        Returns:
            potential_energy_value: value of the potential energy function at the given position,
                this is the negative-log of the target posterior
            potential_energy_gradient: gradient of the potential energy function
                
        """
        potential_energy_value = self._hmc_potential_energy_value(current_state)
        potential_energy_gradient = self._hmc_potential_energy_gradient(current_state)
        #
        return potential_energy_value, potential_energy_gradient
        #
    
    #
    def _hmc_potential_energy_value(self, current_state):
        """
        Evaluate the potential energy value only
        
        Args:
            current_state: current (position) state of the chain
            
        Returns:
            potential_energy_value: value of the potential energy function at the given position,
                this is the negative-log of the target posterior
            
        """
        # Retrieve the observation vector if obs is None
        observation = self.filter_configs['observation']

        # Retrieve the forecast parameters based on the prior distribution then evaluate the potential energy
        prior_distribution = self.prior_distribution
        #
        if prior_distribution == 'gaussian':
            # Get the forecast state if the prior is Gaussian and forecast_state is None.
            try:
                forecast_state = self.filter_configs['forecast_state'].copy()
            except (ValueError, AttributeError, TypeError):
                forecast_ensemble = self.filter_configs['forecast_ensemble']
                forecast_state = utility.ensemble_mean(forecast_ensemble)
            finally:
                if forecast_state is None:
                    forecast_ensemble = self.filter_configs['forecast_ensemble']
                    forecast_state = utility.ensemble_mean(forecast_ensemble)
                    self.filter_configs['forecast_ensemble'] = forecast_state.copy()
            #
            # Potential energy := 3D-Var cost functional in this case, right?
            #
            # 1- Observation/Innovation term
            model_observation = self.model.evaluate_theoretical_observation(current_state)
            # innovations = model_observation.copy().scale(-1.0).add(observation)  # innovations = y - H(x)
            innovations = observation.axpy(-1.0, model_observation, in_place=False)  # innovations = y - H(x)
            # print("innovations", innovations)
            scaled_innovations = self.model.observation_error_model.error_covariances_inv_prod_vec(innovations, in_place=False)
            observation_term = 0.5 * scaled_innovations.dot(innovations)
            #
            # 2- Background term
            lu, piv = self.prior_distribution_statistics['B_lu'], self.prior_distribution_statistics['B_piv']
            # print(self.prior_distribution_statistics['B'].get_numpy_array())
            #
            deviations = current_state.axpy(-1.0, forecast_state, in_place=False)
            # print("deviations", deviations[10:200])
            scaled_deviations_numpy = lu_solve((lu, piv), deviations.get_numpy_array())
            scaled_deviations = self.model.state_vector()
            scaled_deviations[:] = scaled_deviations_numpy.copy()
            # print("deviations", deviations)
            # print("scaled_deviations", scaled_deviations)
            #
            background_term = 0.5 * scaled_deviations.dot(deviations)
            potential_energy_value = background_term + observation_term
            if self._verbose:
                print("Chain is running; Background_term", background_term, "Observation_term", observation_term)

        elif prior_distribution.lower() in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            # The GMM model should be attached to the filter object right now.
            #
            # 1- Observation term ( same as Gaussian prior) keep here in case we add different prior...
            model_observation = self.model.evaluate_theoretical_observation(current_state)  # H_k(x_k)
            # innovations = model_observation.axpy(-1.0, observation)  # innovations =  y - H(x)
            innovations = observation.axpy(-1.0, model_observation, in_place=False)  # innovations = y - H(x)
            scaled_innovations = self.model.observation_error_model.error_covariances_inv_prod_vec(innovations, in_place=False)
            #
            observation_term = 0.5 * scaled_innovations.dot(innovations)
            #
            # 2- Background term
            background_term = self._hmc_gmm_evaluate_potential_log_terms_val(current_state=current_state)
            #
            # Add up the forecast and observation term
            potential_energy_value = observation_term + background_term
            #
            if self._verbose:
                print("Chain is running; Background_term", background_term, "Observation_term", observation_term)
            #
        else:
            print("Prior distribution [%s] is not supported!" % prior_distribution)
            raise ValueError()
        #
        return potential_energy_value
        #

    #
    def _hmc_gmm_evaluate_potential_log_terms_val(self, current_state):
        """
        Evaluate the forecast terms in the potential energy function formulation in the case of GMM prior.
        These are the terms other than the observation/innovation term.
        Two of them include LOVELY logarithmic terms, this is why I named it like that.
        
        Args:
            current_state: current (position) state of the chain
            
        Returns:
            potential_energy_value: scalar,
                the value of the forecast terms int the posterior negative logarithm (potential energy function).
            
        """
        # Extract GMM components for ease of access
        gmm_optimal_covar_type = self.prior_distribution_statistics['gmm_optimal_covar_type']
        gmm_num_components = self.prior_distribution_statistics['gmm_num_components']
        gmm_weights = self.prior_distribution_statistics['gmm_weights']
        gmm_means = self.prior_distribution_statistics['gmm_means']
        # retrieve the values of the logarithm of determinant of each component's covariance matrix
        gmm_covariances_det_log = np.abs(self.prior_distribution_statistics['gmm_covariances_det_log'])  # we need the log(|det(C)|)
        try:
            gmm_covariances = self.prior_distribution_statistics['gmm_covariances']
        except (ValueError, NameError, AttributeError):
            gmm_covariances = None
        try:
            gmm_precisions = self.prior_distribution_statistics['gmm_precisions']
        except(ValueError, NameError, AttributeError):
            gmm_precisions = None

        # Loop over all components, calculate the weighted deviations from components' means and locate the maximum term
        # Values of the weighted innovations of the current state from each component mean.
        weighted_deviations = np.zeros(gmm_num_components)
        max_term_exponent = - np.infty
        max_term_ind = None  # to check later
        component_exponents = []  # to check later
        for comp_ind in xrange(gmm_num_components):
            component_covariances_det_log = gmm_covariances_det_log[comp_ind]
            component_weight = gmm_weights[comp_ind]
            component_mean = gmm_means[comp_ind]
            # weighted innovation
            # innovation = current_state.copy()
            # innovation = innovation.scale(-1.0).add(component_mean); innovation.scale(-1.0)
            innovation = current_state.axpy(-1.0, component_mean, in_place=False)
            if gmm_precisions is not None:  # use precisions instead of inverting covariances
                if gmm_optimal_covar_type == 'tied':
                    precisions = gmm_precisions
                elif gmm_optimal_covar_type in ['diag', 'spherical', 'full']:
                    precisions = gmm_precisions[comp_ind]
                else:
                    raise ValueError("Unrecognized optimal_covar_type! Found '%s'! " % gmm_optimal_covar_type)
                #
                if gmm_optimal_covar_type in ['diag', 'spherical']:
                    scaled_innovation = innovation.multiply(precisions, in_place=False)
                else:
                    scaled_innovation = precisions.vector_product(innovation, in_place=False)
            #
            elif gmm_covariances is not None:  # Precisions are not available, use inverse of covariances here
                if gmm_optimal_covar_type == 'tied':
                    covariances = gmm_covariances
                elif gmm_optimal_covar_type in ['diag', 'spherical', 'full']:
                    covariances = gmm_covariances[comp_ind]
                else:
                    raise ValueError("Unrecognized optimal_covar_type! Found '%s'! " % gmm_optimal_covar_type)
                #
                if gmm_optimal_covar_type in ['diag', 'spherical']:
                    precisions = covariances.reciprocal(in_place=False)
                    scaled_innovation = innovation.multiply(precisions, in_place=False)
                else:
                    scaled_innovation = covariances.solve(innovation)
            #
            else:
                raise ValueError("Neither Covariances nor precisions of the GM model are available!")
            #
            j_val = 0.5 * scaled_innovation.dot(innovation)
            weighted_deviations[comp_ind] = j_val

            component_exponent = np.log(component_weight) - (0.5*component_covariances_det_log) - j_val
            if np.abs(component_exponent) == np.inf:
                print("component value is infinity")
                print('np.log(component_weight)', np.log(component_weight))
                print('component_covariances_det_log', component_covariances_det_log)
                print('j_val', j_val)
            component_exponents.append(component_exponent)
            # print('In J_val Num of comp: %d. \t Component_exponent %s' % (gmm_num_components, component_exponent))
            if component_exponent > max_term_exponent:
                max_term_exponent = component_exponent
                max_term_ind = comp_ind

        # Check to see if a proper max_term_ind is found or not. Proceed if all are OK
        if max_term_ind is None:
            print("Failed to update 'max_term_ind'! Here are the values: %s" % repr(component_exponents))
            raise ValueError
        else:
            # Now proceed with evaluating the forecast term... Check first if no index is found:
            # retrieve information of the term with maximum value
            covariances_det_log_first = gmm_covariances_det_log[max_term_ind]
            weight_first = gmm_weights[max_term_ind]
            j_val_first = weighted_deviations[max_term_ind]
            # get the indices of the other components
            if max_term_ind == 0:
                rest_indices = np.arange(1, gmm_num_components)
            elif max_term_ind == gmm_num_components-1:
                rest_indices = np.arange(0, gmm_num_components-1)
            elif max_term_ind>0 and max_term_ind<gmm_num_components-1:
                rest_indices = np.zeros(gmm_num_components-1, dtype=np.int)
                rest_indices[0:max_term_ind] = range(0, max_term_ind)
                rest_indices[max_term_ind: ] = range(max_term_ind+1, gmm_num_components)
            else:
                raise ValueError("max_term_ind is=%d. This is unexpected" % str(max_term_ind))

        # Start evaluating the forecast term by summing up all sub-log-terms
        # Initialize first
        if gmm_num_components > 1:
            forecast_term_val = - np.log(weight_first) + 0.5 * np.log(covariances_det_log_first)
            forecast_term_val += weighted_deviations[max_term_ind]
        else:
            forecast_term_val = weighted_deviations[max_term_ind]
        #
        # Loop over the other terms from 2 to number of components less 1
        sum_val = 0.0
        for comp_ind in xrange(gmm_num_components-1):
            other_ind = rest_indices[comp_ind]
            covariances_det_log = gmm_covariances_det_log[other_ind]
            weight = gmm_weights[other_ind]
            j_val = weighted_deviations[other_ind]
            exponent = np.log(weight) - np.log(weight_first) + 0.5*covariances_det_log_first - 0.5*covariances_det_log + j_val_first - j_val
            sum_val += np.exp(exponent)

        # Now evaluate the forecast term and return
        forecast_term_val -= np.log(1 + sum_val)
        # 
        return forecast_term_val
        #

    #
    def _hmc_potential_energy_gradient(self, current_state, fd_validation=False, debug=False):
        """
        Evaluate the potential energy gradient only
        
        Args:
            current_state: current (position) state of the chain
            fd_validation (default False): use finiti difference approximation of the gradient to validate the 
                calculated gradient.
            debug (default False): used to check the entries of the calculated gradient for invalid (inf) values
        
        Returns:
            potential_energy_gradient: the gradient of the potential energy function at the given position,
                this is the derivative of the negative-log of the target posterior.
            
        """
        observation = self.filter_configs['observation']
        #
        if self.prior_distribution == 'gaussian':
            # Get the forecast state if the prior is Gaussian and forecast_state is None.
            try:
                forecast_state = self.filter_configs['forecast_state']
            except (ValueError, AttributeError, TypeError):
                forecast_ensemble = self.filter_configs['forecast_ensemble']
                forecast_state = utility.ensemble_mean(forecast_ensemble)
            else:
                if forecast_state is None:
                    forecast_ensemble = self.filter_configs['forecast_ensemble']
                    forecast_state = utility.ensemble_mean(forecast_ensemble)
                    self.filter_configs['forecast_ensemble'] = forecast_state
            #
            # Potential energy = 3D-Var cost functional in this case, right?
            #
            # 1- Observation/Innovation term
            model_observation = self.model.evaluate_theoretical_observation(current_state)
            innovations = model_observation.axpy(-1.0, observation)
            observation_term = self.model.observation_error_model.error_covariances_inv_prod_vec(innovations)
            # 
            # 2- Background term
            lu, piv = self.prior_distribution_statistics['B_lu'], self.prior_distribution_statistics['B_piv']
            deviations = current_state.axpy(-1.0, forecast_state, in_place=False)
            scaled_deviations_numpy = lu_solve((lu, piv), deviations.get_numpy_array())
            background_term = self.model.state_vector()
            background_term[:] = scaled_deviations_numpy.copy()
            #
            potential_energy_gradient = background_term.add(self.model.observation_operator_Jacobian_T_prod_vec(current_state, observation_term))
        #
        elif self.prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            # 
            # 1- Observation/Innovation term
            model_observation = self.model.evaluate_theoretical_observation(current_state)
            innovations = model_observation.axpy(-1.0, observation)  # innovations = H(x) - y
            observation_term = self.model.observation_error_model.error_covariances_inv_prod_vec(innovations)
            observation_term = self.model.observation_operator_Jacobian_T_prod_vec(current_state, observation_term)
            #
            # 2- Background term
            background_term = self._hmc_gmm_evaluate_potential_log_terms_grad(in_state=current_state)
            #
            # 3- add up the two terms to find the full gradient
            potential_energy_gradient = background_term.add(observation_term)
        #
        else:
            print("Prior distribution [%s] is not supported!" % self.prior_distribution)
            raise ValueError()

        # ---------------------------------------------------------------------------------------- #
         # Finite difference validation of the gradient:
        # ----------------------------------------------- #
        if fd_validation:
            print('current_state', current_state)
            
            eps = 1e-5
            fd_grad = potential_energy_gradient.copy()
            fd_grad[:] = 0
            for st_ind in xrange(self.model.state_size()):
                test_current_state = current_state.copy()
                test_current_state[st_ind] += eps
                f1 = self._hmc_potential_energy_value(test_current_state)
                test_current_state[st_ind] -= 2.0*eps
                f2 = self._hmc_potential_energy_value(test_current_state)
                fd_grad[st_ind] = (f1 - f2) / (2.0*eps)

                rel_err = (fd_grad[st_ind] - potential_energy_gradient[st_ind]) / fd_grad[st_ind]
                print('Gradient Validatiion: i=[%3d], grad[%3d]=%+8.6e \t; FD-grad[%3d]=%+8.6e; \t relerr = %+8.6e'
                      % (st_ind, st_ind, potential_energy_gradient[st_ind], st_ind, fd_grad[st_ind], rel_err))
            # ---------------------------------------------------------------------------------------- #

        if self._verbose:
            if np.isinf(potential_energy_gradient[:]).any():
                print('potential_energy_gradient', potential_energy_gradient)
                print('current_state', current_state)
                print('observation_term', observation_term)
                print('background_term', background_term)
                print('innovations', innovations)
                print('model_observation', model_observation)
        else:
            pass
        #
        return potential_energy_gradient
        #

    #
    def _hmc_gmm_evaluate_potential_log_terms_grad(self, in_state):
        """
        Evaluate the gradient of the forecast terms in the potential energy function formulation in the case of GMM prior.
        :return: a state_vector: derivative of the forecast term with respect to hte passed current_state
        
        Args:
            current_state: current (position) state of the chain
        
        Returns:
            potential_energy_gradient: model.state_vector,
                the derivative of the forecast terms in the posterior negative logarithm (potential energy function).
        """
        #
        current_state = in_state.copy()  # unnecessary remove after debugging
        #
        # Extract GMM components for ease of access
        gmm_optimal_covar_type = self.prior_distribution_statistics['gmm_optimal_covar_type']
        gmm_num_components = self.prior_distribution_statistics['gmm_num_components']
        gmm_weights = self.prior_distribution_statistics['gmm_weights']
        gmm_means = gmm_covariances = self.prior_distribution_statistics['gmm_means']
        # retrieve the values of the logarithm of determinant of each component's covariance matrix
        gmm_covariances_det_log = np.abs(self.prior_distribution_statistics['gmm_covariances_det_log'])
        try:
            gmm_covariances = self.prior_distribution_statistics['gmm_covariances']
        except (ValueError, NameError, AttributeError):
            gmm_covariances = None
        try:
            gmm_precisions = self.prior_distribution_statistics['gmm_precisions']
        except(ValueError, NameError, AttributeError):
            gmm_precisions = None

        # Loop over all components, calculate the weighted deviations from components' means and locate the maximum term
        # Values of the weighted innovations of the current state from each component mean.
        weighted_deviations = np.zeros(gmm_num_components)  # holds the value of J(x_{i,k})
        scaled_deviations_list = []  # a list containing deviation from component mean scaled by inverse covariance matrix
        max_term_exponent = - np.infty
        max_term_ind = None  # to check later
        component_exponents = []  # to check later
        for comp_ind in xrange(gmm_num_components):
            component_covariances_det_log = gmm_covariances_det_log[comp_ind]
            component_weight = gmm_weights[comp_ind]
            component_mean = gmm_means[comp_ind].copy()
            #
            # weighted innovation
            innovation = current_state.axpy(-1.0, component_mean, in_place=False)
            if gmm_precisions is not None:  # use precisions instead of inverting covariances
                if gmm_optimal_covar_type == 'tied':
                    precisions = gmm_precisions
                elif gmm_optimal_covar_type in ['diag', 'spherical', 'full']:
                    precisions = gmm_precisions[comp_ind].copy()
                else:
                    raise ValueError("Unrecognized optimal_covar_type! Found '%s'! " % gmm_optimal_covar_type)
                #
                if gmm_optimal_covar_type in ['diag', 'spherical']:
                    scaled_innovation = innovation.multiply(precisions, in_place=False)
                else:
                    scaled_innovation = precisions.vector_product(innovation, in_place=False)
            #
            elif gmm_covariances is not None:  # Precisions are not available, use inverse of covariances here
                if gmm_optimal_covar_type == 'tied':
                    covariances = gmm_covariances
                elif gmm_optimal_covar_type in ['diag', 'spherical', 'full']:
                    covariances = gmm_covariances[comp_ind].copy()
                else:
                    print("Unrecognized optimal_covar_type! Found '%s'! " % gmm_optimal_covar_type)
                    raise ValueError()
                #
                if gmm_optimal_covar_type in ['diag', 'spherical']:
                    precisions = covariances.reciprocal(in_place=False)
                    scaled_innovation = innovation.multiply(precisions, in_place=False)
                else:
                    scaled_innovation = covariances.solve(innovation)
            #
            else:
                print("Neither Covariances nor precisions of the GM model are available!")
                raise ValueError()
            #
            # Hold the scaled deviation vector for later use
            scaled_deviations_list.append(scaled_innovation)
            # Evaluate and hold the value of the weighted deviations
            j_val = 0.5 * scaled_innovation.dot(innovation)
            weighted_deviations[comp_ind] = j_val.copy()

            component_exponent = np.log(component_weight) - (0.5*component_covariances_det_log) - j_val
            component_exponents.append(component_exponent)
            #
            if component_exponent > max_term_exponent:
                max_term_exponent = component_exponent
                max_term_ind = comp_ind
            #

        # Check to see if a proper max_term_ind is found or not. Proceed if all are OK
        if max_term_ind is None:
            print("Failed to update 'max_term_ind'! Here are the values: %s" % repr(component_exponents))
            raise ValueError
        else:
            # Now proceed with evaluating the forecast term... Check first if no index is found:
            # retrieve information of the term with maximum value
            covariances_det_log_first = gmm_covariances_det_log[max_term_ind]
            weight_first = gmm_weights[max_term_ind]
            j_val_first = weighted_deviations[max_term_ind]
            # get the indices of the other components
            if max_term_ind == 0:
                rest_indices = np.arange(1, gmm_num_components)
            elif max_term_ind == gmm_num_components-1:
                rest_indices = np.arange(0, gmm_num_components-1)
            elif max_term_ind>0 and max_term_ind<gmm_num_components-1:
                rest_indices = np.zeros(gmm_num_components-1, dtype=np.int)
                rest_indices[0: max_term_ind] = range(0, max_term_ind)
                rest_indices[max_term_ind: ] = range(max_term_ind+1, gmm_num_components)
            else:
                print("max_term_ind is=%d. This is unexpected" % str(max_term_ind))
                raise ValueError()

        #
        # Evaluate the same terms in the summation used repeatedly
        # Loop over the other terms from 2 to number of components less 1
        sum_val = 0.0
        sum_terms = np.empty(gmm_num_components)
        sum_terms[max_term_ind] = np.infty
        for comp_ind in xrange(gmm_num_components-1):
            other_ind = rest_indices[comp_ind]
            covariances_det_log = gmm_covariances_det_log[other_ind]
            weight = gmm_weights[other_ind]
            j_val = weighted_deviations[other_ind]
            exponent = np.log(weight) - np.log(weight_first) + 0.5*covariances_det_log_first - 0.5*covariances_det_log + j_val_first - j_val
            sum_term = np.exp(exponent)
            sum_terms[other_ind] = sum_term.copy()
            sum_val += sum_term
        gradient_fac = -1.0 / (1 + sum_val)

        # Start evaluating the forecast gradient
        # Initialize first
        gradient = self.model.state_vector()
        gradient[:] = 0.0
        for comp_ind in xrange(gmm_num_components-1):
            other_ind = rest_indices[comp_ind]
            sum_term_val = sum_terms[other_ind].copy()
            # 
            gradients_deviation = scaled_deviations_list[max_term_ind].axpy(-1.0, scaled_deviations_list[other_ind], in_place=False)
            gradient = gradient.add(gradients_deviation.scale(sum_term_val))

        # Now evaluate the forecast gradient term and return
        forecast_term_grad = scaled_deviations_list[max_term_ind].add(gradient.scale(gradient_fac))
        #
        return forecast_term_grad
        #

    #
    def print_cycle_results(self):
        """
        Print filtering results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        
        Args:
        
        Returns:
            None
            
        """
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().print_cycle_results()
        else:
            # old-stype class
            super(HMCFilter, self).print_cycle_results()
        pass  # Add more...
        #

    #
    def save_cycle_results(self, output_dir=None, cleanup_out_dir=False):
        """
        Save filtering results from the current cycle to file(s).
        Check the output directory first. If the directory does not exist, create it.
        
        Args:
            output_dir: full path of the directory to save results in
            clean_out_dir (default False): remove the contents of the output directory
                  
        Returns:
            None
            
        """
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
        model_conf = self.model.get_model_configs()
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
                self.model.write_state(state=forecast_state, directory=cycle_states_out_dir, file_name='forecast_mean')
                # save analysis mean
                analysis_state = self.filter_configs['analysis_state']
                self.model.write_state(state=analysis_state, directory=cycle_states_out_dir, file_name='analysis_mean')
            else:
                raise ValueError("Unsupported ensemble moment: '%s' !" % (file_output_moment_name))
        else:
            # start outputting the whole ensemble members (both forecast and analysis ensembles of course).
            # check if all ensembles are to be saved or just one of the supported ensemble moments
            for ens_ind in xrange(self.sample_size):
                if file_output_separate_files:
                    # print('saving ensemble member to separate files: %d' % ens_ind)
                    forecast_ensemble_member = self.filter_configs['forecast_ensemble'][ens_ind]
                    self.model.write_state(state=forecast_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='forecast_ensemble_'+str(ens_ind),
                                            append=False
                                            )
                    #
                    analysis_ensemble_member = self.filter_configs['analysis_ensemble'][ens_ind]
                    self.model.write_state(state=analysis_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='analysis_ensemble_'+str(ens_ind),
                                            append=False
                                            )
                else:
                    # print('saving ensemble member to same file with resizing: %d' % ens_ind)
                    # save results to different files. For moments
                    forecast_ensemble_member = self.filter_configs['forecast_ensemble'][ens_ind]
                    self.model.write_state(state=forecast_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='forecast_ensemble',
                                            append=True
                                            )
                    #
                    analysis_ensemble_member = self.filter_configs['analysis_ensemble'][ens_ind]
                    self.model.write_state(state=analysis_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='analysis_ensemble',
                                            append=True
                                            )
        # save reference state TODO: We may need to save time(s) along with state(s)...
        reference_state = self.filter_configs['reference_state']
        self.model.write_state(state=reference_state, directory=cycle_states_out_dir, file_name='reference_state')

        #
        # Save observation to file; use model to write observations to file(s)
        # save analysis mean
        observation = self.filter_configs['observation']
        self.model.write_observation(observation=observation, directory=cycle_observations_out_dir, file_name='observation', append=False)

        # Save filter statistics to file
        # 1- Output filter RMSEs: RMSEs are saved to the same file. It's meaningless to create a new file for each cycle
        rmse_file_name = 'rmse'
        if file_output_file_format in ['text', 'ascii']:
            rmse_file_name += '.dat'
            rmse_file_path = os.path.join(filter_statistics_dir, rmse_file_name)
            if not os.path.isfile(rmse_file_path):
                # rmse file does not exist. create file and add header.
                header = "RMSE Results: Filter: '%s' \n %s \t %s \t %s \t %s \t %s \n" % (self._filter_name,
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
            
            #
            filter_configs = self.filter_configs
            if filter_configs['prior_distribution'] == 'gaussian':
                gmm_prior_settings = {}
                prior_distribution = 'gaussian'
                gmm_conf = {}
                                
            elif filter_configs['prior_distribution'] == 'gmm':
                switched_to_Gaussian_prior = self.switched_to_Gaussian_prior
                
                if switched_to_Gaussian_prior:
                    gmm_prior_settings = {}
                    prior_distribution = 'gaussian'
                    gmm_conf = {}
                    self.switched_to_Gaussian_prior = False  # for the next cycle.
                else:
                    gmm_prior_settings = filter_configs['gmm_prior_settings']
                    prior_distribution = 'gmm'
                    gmm_conf = output_configs['filter_statistics']['gmm_prior_statistics']
                
            
            # TODO: Rethink the outputting strategy of this filter...
            # save filter and model configurations (a copy under observation directory and another under state directory)...
            #
            filter_conf= dict(filter_name=filter_configs['filter_name'],
                              prior_distribution=prior_distribution,
                              gmm_prior_settings=gmm_prior_settings,
                              ensemble_size=filter_configs['ensemble_size'],
                              apply_preprocessing=filter_configs['apply_preprocessing'],
                              apply_postprocessing=filter_configs['apply_postprocessing'],
                              chain_parameters=filter_configs['chain_parameters'],
                              timespan=filter_configs['timespan'],
                              analysis_time=filter_configs['analysis_time'],
                              observation_time=filter_configs['observation_time'],
                              forecast_time=filter_configs['forecast_time'],
                              forecast_first=filter_configs['forecast_first']
                              )
            #
            output_conf = output_configs            
            
            # Save chain diagnostics (statistics)
            try:
                chain_diagnostics = output_configs['filter_statistics']['chain_diagnostics']
                # print('chain_diagnostics', chain_diagnostics)
            except (ValueError, NameError, AttributeError, KeyError):
                print("Couldn't retrieve chain diagnostics for the filter?")
                # print('output_configs', output_configs)
                chain_diagnostics = {}            

            if prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
                utility.write_dicts_to_config_file('setup.dat', cycle_observations_out_dir,
                                                   [filter_conf, output_configs, gmm_conf, chain_diagnostics], ['Filter Configs', 'Output Configs', 'GMM-Prior Configs', 'Chain Diagnostics'])
                utility.write_dicts_to_config_file('gmm_prior_configs.dat', cycle_observations_out_dir, gmm_conf, 'GMM-Prior Configs')
                utility.write_dicts_to_config_file('setup.dat', cycle_states_out_dir,
                                                   [filter_conf, output_configs, gmm_conf, chain_diagnostics], ['Filter Configs', 'Output Configs', 'GMM-Prior Configs', 'Chain Diagnostics'])
                utility.write_dicts_to_config_file('gmm_prior_configs.dat', cycle_observations_out_dir, gmm_conf, 'GMM-Prior Configs')
            elif prior_distribution in ['gaussian', 'normal']:
                utility.write_dicts_to_config_file('setup.dat', cycle_observations_out_dir,
                                                   [filter_conf, output_configs, chain_diagnostics], ['Filter Configs', 'Output Configs', 'Chain Diagnostics'])
                utility.write_dicts_to_config_file('setup.dat', cycle_states_out_dir,
                                                   [filter_conf, output_configs, chain_diagnostics], ['Filter Configs', 'Output Configs', 'Chain Diagnostics'])
            else:
                print("Unsupported Prior distribution [%s]" % prior_distribution)
                raise ValueError()
            #
        else:
            print("Unsupported output format: '%s' !" % file_output_file_format)
            raise ValueError()
            #

#
#
class Momentum:
    """
    The momentum variable P in the HMC context is an essential component of the sampling process.
    Here we model it giving it all the necessary functionality to ease working with it.
    
    Initialize the parameters of the momentum variable:
        - The mass matrix (M)
        - Square root of M for generating scaled momentum
        - Inverse of M for evaluating the kinetic energy
    
    Args:
        mass_matrix: a scalar or a numpy array.
             If it is scalar, it will be replicated based on the mass_matrix_shape.
             If it is a one-D numpy array, it has to be put on the diagonal of the mass matrix.
             If it is a full square numpy array, so be it will be used as a mass matrix.
        mass_matrix_shape (default None): shape of the mass matrix,
            This should really be 'diagonal'
        dimension: space dimension of the momentum variable
        model: model object
        diag_val_thresh: a lower threshold value of the diagonal of the mass matrix to aviod overflow on 
            finding the mass matrix inverse.
    
    Returns:
        None
    
    """

    def __init__(self, mass_matrix, mass_matrix_shape='diagonal', dimension=None, model=None, verbose=False):
        
        self._initialize_momentum_entities(mass_matrix=mass_matrix,
                                           mass_matrix_shape=mass_matrix_shape,
                                           dimension=dimension,
                                           model=model
                                           )
                                           #
        self._verbose = verbose
        self._initialized = True
        #

    #
    def _initialize_momentum_entities(self, mass_matrix, mass_matrix_shape='diagonal', dimension=None, model=None):
        """
        Initialize the parameters of the momentum variable:
            - The mass matrix (M)
            - Square root of M for generating scaled momentum
            - Inverse of M for evaluating the kinetic energy
        
        Args:
            mass_matrix: a scalar or a numpy array.
                 If it is scalar, it will be replicated based on the mass_matrix_shape.
                 If it is a one-D numpy array, it has to be put on the diagonal of the mass matrix.
                 If it is a full square numpy array, so be it will be used as a mass matrix.
            mass_matrix_shape (default None): shape of the mass matrix,
                This should really be 'diagonal'
            dimension: space dimension of the momentum variable
            model: model object
            diag_val_thresh: a lower threshold value of the diagonal of the mass matrix to aviod overflow on 
                finding the mass matrix inverse.
                
        Returns:
            None
                
        """
        self._mass_matrix_shape = mass_matrix_shape.lower()
        if self._mass_matrix_shape not in ['diagonal', 'full']:
            raise ValueError("The mass matrix shape [%s] is not recognized or supported!" %mass_matrix_shape)

        if model is None:
            self.model = None
            # Everything will be done locally using numpy arrays.
            self._use_model_object = False
            if dimension is None:
                raise ValueError("Dimensions of the momentum vector must be passed")
            else:
                assert isinstance(dimension, int)
                self._dimension = dimension
            #
        else:
            # This should be the right decision always... We want hmc to be used in the context of DA, right!
            self._use_model_object = True
            self.model = model
            #
            if dimension is not None:
                state_size = model.state_size()
                if dimension != state_size:
                    raise ValueError("Dimension of momentum is %d, while position state size is %d!" % (dimension, state_size))
                else:
                    pass
            else:
                dimension = model.state_size()
            #
            # we are all good, now start creating the mass matrix object
            self._dimension = dimension
            if isinstance(mass_matrix, int) or isinstance(mass_matrix, float):
                if self._mass_matrix_shape == 'diagonal':
                    mass_matrix_diagonal = model.state_vector()
                    mass_matrix_diagonal[:] = mass_matrix.copy()
                elif self._mass_matrix_shape == 'full':
                    mass_matrix_diagonal = model.StateMatrix()
                    mass_matrix_diagonal[:, :] = mass_matrix.copy()
                else:
                    raise ValueError("Mass matrix shape [%s] is not recognized!" % self._mass_matrix_shape)
                #
                # Attach the final version to the momentum object...
                self._mass_matrix = mass_matrix_diagonal
            #
            elif isinstance(mass_matrix, np.ndarray):
                #
                ndim = mass_matrix.ndim
                if mass_matrix_shape.lower() == 'diagonal' and mass_matrix.size == self._dimension:
                    self._mass_matrix = model.state_vector()
                    self._mass_matrix[:] = mass_matrix.copy()
                elif mass_matrix_shape.lower() == 'full' and mass_matrix.size == pow(self._dimension, 2):
                    self._mass_matrix = model.StateMatrix()
                    self._mass_matrix[:, :] = mass_matrix.copy()
                else:
                    raise ValueError("Dimension mismatch! Couldn't create the mass matrix object!")
            #
            elif isinstance(mass_matrix, StateVector):
                raise NotImplementedError("Not yet. Please pass a scalar or a numpy array!")
            #
            elif isinstance(mass_matrix, model.StateMatrix):
                raise NotImplementedError("Not yet. Please pass a scalar or a numpy array!")
            #
            else:
                raise ValueError("mass_matrix seed passed is not recognized")

            # generate the square root and inverse. inverse should be either waived, avoided, or a system solver is used
            if self._mass_matrix_shape == 'diagonal':
                self._mass_matrix_sqrt = self._mass_matrix.sqrt(in_place=False)
                self._mass_matrix_inv = self._mass_matrix.reciprocal(in_place=False)
            elif self._mass_matrix_shape == 'full':
                self._mass_matrix_sqrt = self._mass_matrix.cholesky(in_place=False)
                self._mass_matrix_inv = self._mass_matrix.inverse(in_place=False)
                # self._mass_matrix_inv = self._mass_matrix.presolve()
                #

    #
    def update_mass_matrix(self, mass_matrix, mass_matrix_shape=None, dimension=None, model=None):
        """
        Update the mass matrix values. This might be needed in the middle of the chain, or if parallel chains
        are run with different/local settings.
            - The mass matrix (M)
            - Square root of M for generating scaled momentum
            - Inverse of M for evaluating the kinetic energy
        
        Args:
            mass_matrix: a scalar or a numpy array.
                 If it is scalar, it will be replicated based on the mass_matrix_shape.
                 If it is a one-D numpy array, it has to be put on the diagonal of the mass matrix.
                 If it is a full square numpy array, so be it will be used as a mass matrix.
            mass_matrix_shape (default None): shape of the mass matrix,
                This should really be 'diagonal'
            dimension: space dimension of the momentum variable
            model: model object
            diag_val_thresh: a lower threshold value of the diagonal of the mass matrix to aviod overflow on 
                finding the mass matrix inverse.
            
        Returns:
            None
            
        """
        #
        if mass_matrix_shape is None:
            mass_matrix_shape = self._mass_matrix_shape
        if model is None:
            model = self.model
        if dimension is None:
            dimension = self._dimension
        #
        if mass_matrix_shape == self._mass_matrix_shape and dimension==self._dimension and model is self.model:
            # Then just update mass matrix, its inverse, and square root in place
            if isinstance(mass_matrix, int) or isinstance(mass_matrix, float):
                if mass_matrix_shape == 'diagonal':
                    self._mass_matrix[:] = mass_matrix.copy()
                elif self._mass_matrix_shape == 'full':
                    self._mass_matrix[:, :] = mass_matrix.copy()
                else:
                    raise ValueError("Mass matrix shape [%s] is not recognized!" % self._mass_matrix_shape)
            #
            elif isinstance(mass_matrix, np.ndarray):
                #
                if mass_matrix_shape.lower() == 'diagonal' and mass_matrix.size == self._dimension:
                    self._mass_matrix[:] = mass_matrix.copy()
                elif mass_matrix_shape.lower() == 'full' and mass_matrix.size == pow(self._dimension, 2):
                    self._mass_matrix[:, :] = mass_matrix.copy()
                else:
                    raise ValueError("Dimension mismatch! Couldn't create the mass matrix object!")
            #
            elif isinstance(mass_matrix, StateVector):
                #
                if mass_matrix_shape.lower() == 'diagonal' and mass_matrix.size == self._dimension:
                    self._mass_matrix[:] = mass_matrix[:].copy()
                else:
                    raise ValueError("Dimension mismatch! Couldn't create the mass matrix object!")
            #
            elif isinstance(mass_matrix, model.StateMatrix):
                raise NotImplementedError("Not yet. Please pass a scalar or a numpy array!")
            #
            else:
                raise ValueError("mass_matrix seed passed is not recognized")

            # generate the square root and inverse. inverse should be either waived, avoided, or a system solver is used
            if mass_matrix_shape == 'diagonal':
                self._mass_matrix_sqrt = self._mass_matrix.sqrt(in_place=False)
                self._mass_matrix_inv = self._mass_matrix.reciprocal(in_place=False)
            elif mass_matrix_shape == 'full':
                self._mass_matrix_sqrt = self._mass_matrix.cholesky(in_place=False)
                self._mass_matrix_inv = self._mass_matrix.inverse(in_place=False)
        else:
            self._initialize_momentum_entities(mass_matrix=mass_matrix,
                                               mass_matrix_shape=mass_matrix_shape,
                                               dimension=dimension,
                                               model=model
                                               )
                                               #

    #
    def mass_mat_prod_momentum(self, momentum_vec, in_place=True):
        """
        Product of the mass matrix by a momentum vector
        """
        if self._use_model_object:
            if self._mass_matrix_shape == 'diagonal':
                momentum_vec = momentum_vec.multiply(self._mass_matrix, in_place=in_place)
            elif self._mass_matrix_shape == 'full':
                momentum_vec = self._mass_matrix.vector_product(momentum_vec, in_place=in_place)
            else:
                raise ValueError("Unrecognized mass matrix shape!!")
        else:
            raise NotImplementedError("Mass matrix as a numpy array is Not yet supported! Use model object instead!")
        #
        return momentum_vec
        #

    #
    def inv_mass_mat_prod_momentum(self, momentum_vec, in_place=True):
        """
        Product of the inverse of the mass matrix by a momentum vector
        """
        if self._use_model_object:
            if self._mass_matrix_shape == 'diagonal':
                out_momentum_vec = momentum_vec.multiply(self._mass_matrix_inv, in_place=in_place)
            elif self._mass_matrix_shape == 'full':
                out_momentum_vec = self._mass_matrix_inv.vector_product(momentum_vec, in_place=in_place)
            else:
                raise ValueError("Unrecognized mass matrix shape!!")
        else:
            raise NotImplementedError("Mass matrix as a numpy array is Not yet supported! Use model object instead!")
        
        #
        return out_momentum_vec
        #

    #
    def generate_momentum(self, momentum_vec=None):
        """
        Generate a random momentum
        If momentum_vec is None a new vector will be created, otherwise it will be updated
        """
        # use numpy.random.randn to generate a multivariate standard normal vector
        randn_vec = np.random.randn(self._dimension)
        # scale the momentum_vec by standard deviations and prepare to return
        if self._use_model_object:
            if momentum_vec is None:
                # create a new momentum_vec
                momentum_vec = self.model.state_vector()
            else:
                pass  # do nothing, just update it
            #
            # Now update the entries of the vector. I am not using _set_vector_ref because we need to discuss about slicing more!
            momentum_vec[:] = randn_vec[:].copy()
            # scale:
            if self._mass_matrix_shape == 'diagonal':
                momentum_vec = momentum_vec.multiply(self._mass_matrix_sqrt)
            elif self._mass_matrix_shape == 'full':
                momentum_vec = self._mass_matrix_sqrt.vector_product(momentum_vec)
            else:
                raise ValueError("Unrecognized mass matrix shape!!")
        else:
            raise NotImplementedError("Mass matrix as a numpy array is Not yet supported! Use model object instead!")
        #
        return momentum_vec
        #

    #
    def evaluate_kinetic_energy(self, momentum_vec):
        """
        Evaluate the value of the Kinetic energy. $0.5 * P^T * M^{-1} * P$
        """
        if self._use_model_object:
            if self._mass_matrix_shape == 'diagonal':
                scaled_momentum = momentum_vec.multiply(self._mass_matrix_inv, in_place=False)
            elif self._mass_matrix_shape == 'full':
                scaled_momentum = self._mass_matrix_inv.vector_product(momentum_vec, in_place=False)
            else:
                raise ValueError("Unrecognized mass matrix shape!!")
            #
            kinetic_energy = 0.5 * scaled_momentum.dot(momentum_vec)
        else:
            raise NotImplementedError("Mass matrix as a numpy array is Not yet supported! Use model object instead!")
        #
        return kinetic_energy
        #
        
        

