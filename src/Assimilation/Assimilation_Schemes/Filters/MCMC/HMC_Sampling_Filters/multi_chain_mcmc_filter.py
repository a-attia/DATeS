
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
    MultiChainMCMC:                                                                       
    ---------------
    A multi-chain MCMC sampler for data assimilation.
    The sampler proceeds by smpling the posterior distribution by creating several Markov chains, 
    each starting at the mean of one of the components of the prior GMM approximation of the prior distribution.
    The Proposal density can be Gaussian, or HMC. If the proposal density is Gaussian, the parameters of the proposal density function, 
    i.e. the covariance is chosen locally from the covariance of the local ensemble points in the corresponding mixture component.                                                                    
    Similarly, for HMC, the mass matrix is set to the precisions/variances of the corresponding component in the prior 
    e.g. inflation can be done for wider range of navigation.
    - Publications:                                                                       
      i) Ahmed Attia, Azam Moosavi, and Adrian Sandu (2015). Cluster Sampling Filters for non-Gaussian Data Assimilation.
         ArXiv Prepring: https://arxiv.org/abs/1607.03592
"""


import dates_utility as utility
import numpy as np

from filters_base import FiltersBase
from hmc_filter import HMCFilter


class MultiChainMCMC(HMCFilter):
    """
    A class implementing the Hamiltonian/Hybrid Monte-Carlo Sampling filtering family developed by Ahmed attia.
    
    MC_MCMC filter constructor:
                                                        
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
                * prior_distribution (default 'gmm'): prior probability distribution;
                    this has to be 'GMM'.
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
                
                * proposal_density (default 'HMC'): The kernel of the proposal density,
                    'Gaussian': a Gaussian jumping distribution is used to propose state for the chain
                    'HMC': Hamiltonian Monte Carlo is used
                *gaussian_proposal_covariance_shape (default 'full') shape of the covariance matrix of the 
                    Gaussian jumping distribution
                    - 'full': full covariance matrix
                    - 'diagonal': diagonal covariance matrix
                * fix_gaussian_proposal_covar_diagonal (default False): use maximum covariance on the diagonal
                * gaussian_proposal_variance (default None):  can be accepted if is scalar.
                * ensemble_size_weighting_strategy (default 'weight_mean_likelihood'):
                    This is used to adjus the ensemble size sampled from each of chains under the individual 
                    components.
                    - 'weight_mean_likelihood': total sample size x prior weight x likelihood value of the prior-mean
                    - 'weight_mean_prior': total sample size x prior weight x prior value of the prior-mean
                    - 'weight_mean_posterior': total sample size x prior weight x posterior value of the prior-mean
                    - 'mean_posterior': total sample size x prior value of the prior-mean
                    - 'exact': total sample size x prior weight /square root of determinant of component covariance matrix            
                * reduce_burn_in_stage (default True): burn_in_stage for each chain = burn_in_stage/num_of_chains
                * forced_num_chains (default None):  # can be either None. 1 is to force single chain
                * reset_random_seed_per_chain (default True): use the same sequence of random number/vectors for all chains
                * random_seed_per_chain (default 1): used if reset_random_seed_per_chain is set to True
                * inter_comp_inflation_factor (default 1.0): inflate componentes away from the joint mean
                * intra_comp_inflation_factor (default 1.0)  # inflate ensemble members within each of the components
                
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
    _filter_name = "MC_MCMC-F"
    #    
    
    #
    def __init__(self, filter_configs=None, output_configs=None):
        
        # MultiChainMCMC inherits _def_local_filter_configs from HMCFilter. We need to update it with
        # the default settings of this filter. Not all entries are udpated.
        MultiChainMCMC._def_local_filter_configs.update(dict(prior_distribution='gmm',
                                                             proposal_density='HMC',
                                                             gaussian_proposal_covariance_shape='full',
                                                             fix_gaussian_proposal_covar_diagonal=False,
                                                             gaussian_proposal_variance = None,
                                                             ensemble_size_weighting_strategy = 'weight_mean_likelihood',
                                                             reduce_burn_in_stage=True,
                                                             forced_num_chains = None,
                                                             reset_random_seed_per_chain=True,
                                                             random_seed_per_chain= 1,
                                                             inter_comp_inflation_factor=1.0,
                                                             intra_comp_inflation_factor=1.0
                                                             )
                                                        )
        
        mcmcmc_filter_configs = utility.aggregate_configurations(filter_configs, MultiChainMCMC._def_local_filter_configs)
        mcmcmc_output_configs = utility.aggregate_configurations(output_configs, MultiChainMCMC._local_def_output_configs)
        HMCFilter.__init__(self, filter_configs=mcmcmc_filter_configs, output_configs=mcmcmc_output_configs)

        # Update filter name (required due to the design of HMCFilter constructor)
        self.filter_configs['filter_name'] = MultiChainMCMC._filter_name
        #
        
        # general settings of the Markov chain
        updt_chain_param = dict(forced_num_chains=None,  # can be either None. 1 is to force single chain                              
                                proposal_covariances_domain='local'  # 'local' or 'global' on the ensemble domain
                                )
        self.filter_configs['chain_parameters'].update(updt_chain_param)
        self.chain_parameters.update(updt_chain_param)

        # proposal-density-dependent parameters
        self.proposal_density = self.filter_configs['proposal_density'].lower()        
        #
        # Note that the momentum object has already be initialized, but has to be updated per components
        #        
        try:
            self._verbose = self.output_configs['verbose']
        except(AttributeError, NameError, KeyError):
            self._verbose = False
        #
        self.__initialized = True
        #
        
    #
    def analysis(self):
        """
        Analysis step, the forecast ensemble is used to build an approximation of the prior distribution using a GMM,
        then the posterior is constructed and sampled using a single or multi-chain approach. 
        The proposal kernel along with it's parameters is defined in the configurations dictionary.
        
        Args:
        
        Returns:
            chain_diagnostics: a dictionary containing the diagnostics of the chain such as acceptance rate, and
                effective sample size, etc.
            
        """
        # TODO: Consider applying inflation (interm, and intra) mixture components. This MAY require building GMM twice!
        #
        
        # 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #                              Get the general chain settings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # num_of_chains: holds the number of Markov Chains to run in parallel (sequentially here)
        
        # =========================================(Begin Analysis step)=========================================
        #
        # check if the number of chains is forced to be 1, otherwise set the number of chains to the number of prior components
        forced_num_chains = self.chain_parameters['forced_num_chains']
        if forced_num_chains is not None:
            if forced_num_chains != 1:
                raise NotImplementedError("Only a single chain can be forced so far, otherwise initialization is not well-defined!")
            else:
                num_of_chains = forced_num_chains
        else:
            # set the number of chains to the number of components in the GMM prior approximation:
            if self.prior_distribution == 'gmm':
                num_of_chains = self.prior_distribution_statistics['gmm_num_components']
            elif self.prior_distribution == 'gaussian':
                num_of_chains = 1
            else:
                raise ValueError("Unknown prior distribution! %s" % self.prior_distribution)
        # -----------
        if self._verbose:
            print('num_of_chains', num_of_chains)
        # -----------
            
        # get a reference to the dynamical model used
        model = self.filter_configs['model']
        
        if not num_of_chains >= 1:
            raise ValueError("Number of chains not greater than or equal to 1? \nHow is this even possible? num_of_chains = %s" % repr(num_of_chains))
            
        elif num_of_chains == 1:
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # A single chain, means that the prior is best approximated by a Gaussian distribution.
            # If the proposal_density is HMC, the originial HMC is called
            # If the proposal_density is Gaussian, a standard MCMC with Gaussian proposal density samples the posterior            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.proposal_density == 'hmc':
                # just call the analysis step in the standard HMC sampling filter. 
                chain_diagnostics = HMCFilter.analysis(self)
                # the return value (chain_diagnostics) is already udpated to self.output_configs['filter_statistics']
                #
                # Now all what's necessary is already done in the parent class's analysis function. 
                return  # We are all good; terminate this function.
                
            elif self.proposal_density == 'gaussian':
                # create a single Markov chain, which proposal density is a Gaussian, with mean centered around 
                # the current state of the chain, and the covariance is set globally to the covariance (maybe diagonal) 
                # of the ensemble
                
                # --------------------------------------------
                # a) retrieve/set the chain configurations:
                # --------------------------------------------
                # i] get the forecast state as the mean of the forecast ensemble. Will not be used for GMM!
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
                        initial_state = self.prior_mean
                    except(ValueError, NameError, AttributeError):
                        initial_state = None
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
                    print("Chain initialization policy is not recognized. \n"
                          "You have to pass on of the values:\n %s" % repr(initialize_chain_strategies))
                    raise ValueError()

                # ii] set the covariance of the proposal density:
                gaussian_proposal_covariance_shape = self.filter_configs['gaussian_proposal_covariance_shape'].lower()
                if gaussian_proposal_covariance_shape == 'diagonal':
                    # The diagonal of the covariance matrix only is needed. It is saved as model.state_vecor
                    try:
                        prior_variances = self.prior_variances
                    except (AttributeError, ValueError, NameError):
                        self.generate_prior_info()
                        prior_variances = self.prior_variances
                    
                    covars_fix_diag = self.filter_configs['fix_gaussian_proposal_covar_diagonal']
                    if covars_fix_diag:
                        # scaling by a scalar would be optimal in this case
                        gaussian_proposal_variance = self.filter_configs['gaussian_proposal_variance']
                        if np.isscalar(gaussian_proposal_variance):
                            gaussian_proposal_covariance = gaussian_proposal_variance                            
                        else:
                            gaussian_proposal_covariance = prior_variances.min()  # it's a numpy array
                        gaussian_proposal_precisions = 1.0 / gaussian_proposal_covariance
                        gaussian_proposal_sqrtm = np.sqrt(gaussian_proposal_covariance)
                    else:
                        # here we need a state vector to store the diagonal of the covariance matrix
                        gaussian_proposal_covariance = model.state_vector()
                        gaussian_proposal_covariance[:] = prior_variances[:].copy()
                        gaussian_proposal_precisions = gaussian_proposal_covariance.reciprocal(in_place=False)  # do we need a copy?                   
                        gaussian_proposal_sqrtm = gaussian_proposal_covariance.sqrt()
                    
                elif gaussian_proposal_covariance_shape == 'full':
                    try:
                        self.prior_distribution_statistics['B']
                    except(ValueError, NameError, AttributeError):
                        
                        try:
                            balance_factor = self.filter_configs['hybrid_background_coeff']
                        except (NameError, ValueError, AttributeError, KeyError):
                            # default value is 0 in case it is missing from the filter configurations
                            balance_factor = self._def_local_filter_configs['hybrid_background_coeff']
                        fac = balance_factor  # this multiplies the modeled Background error covariance matrix
                        if fac == 0:                
                            ensemble_covariances = self.model.ensemble_covariance_matrix(forecast_ensemble, localize=self.filter_configs['localize_covariances'])                
                                
                        elif fac == 1:
                            ensemble_covariances = self.model.background_error_model.B.copy().scale(fac)
                        elif 0 < fac < 1:
                            model_covariances = self.model.background_error_model.B.copy().scale(fac)
                            # print('Original model_covariances:', model_covariances)
                            # print('forecast_ensemble:', forecast_ensemble)
                            ensemble_covariances = self.model.ensemble_covariance_matrix(forecast_ensemble, localize=self.filter_configs['localize_covariances'])
                            # print('ensemble_covariances', ensemble_covariances)
                            # print('ensemble_covariances.scale(1-fac)', ensemble_covariances.scale(1-fac))
                            ensemble_covariances = model_covariances.add(ensemble_covariances.scale(1-fac))
                        else:
                            print("The balance factor %s has to be a scalar in the interval [0, 1]" % repr(fac))
                            raise ValueError()
                       
                        #
                        model_variances = ensemble_covariances.diag().copy()  # this returns a 1D numpy array.
                        ensemble_covariances_sqrtm = ensemble_covariances.cholesky(in_place=True)  # do we need a copy?
                else:
                    print("Unrecognized shape for the covariance matrix of the Gaussian proposal density %s" \
                           % repr(gaussian_proposal_covariance_shape))
                    raise ValueError()
                                
                # iii] set the number of steps in the chain (burn-in, mixing --> total number of chain steps):                
                burn_in_steps = self.chain_parameters['burn_in_steps']
                mixing_steps = self.chain_parameters['mixing_steps']
                                
                # --------------------------------------------
                # b) start constructing the chain:
                # --------------------------------------------
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
                    raise NotImplementedError("Tempered chains are not ready yet!")
                    #
                else:
                    #
                    if self.optimize_to_converge:
                        # replace the burn-in stage by an optimization step
                        print("Optimization step for convergence is not implemented yet!")
                        raise NotImplementedError()
                    else:
                        # 
                        diagnostic_scheme = self.chain_parameters['convergence_diagnostic']  # for early termination of the burn-in stage
                        #
                        # Initialize the chain(s) repositories;
                        # Check the placeholder of the analysis ensemble to avoid reconstruction
                        analysis_ensemble = []
                        acceptance_flags = []
                        acceptance_probabilities = []
                        uniform_random_numbers = []
                        #                        
                        # set the chain state (to be updated at each step in the chain using MH-acceptance/rejection) 
                        current_state = initial_state.copy()
                        #
                        # --------------------------------------
                        # start constructing the chain(s)
                        # --------------------------------------
                        #
                        # Start the BURN-IN stage:
                        #
                        proposed_state = model.state_vector() # a placeholder for a proposal state from a Gaussian distribution
                        model_state_size = model.state_size()
                        #
                        for burn_ind in xrange(burn_in_steps):
                            # use the Gaussian kernel to propose a state, then update the chain, and update the chain statistics
                            proposed_state[:] = np.random.randn(model_state_size)
                            # scale and shift the proposed state:
                            # Scale by standard deviation(s)
                            if gaussian_proposal_covariance_shape == 'diagonal':
                                if covars_fix_diag:
                                    proposed_state = proposed_state.scale(gaussian_proposal_sqrtm)
                                else:
                                    proposed_state = proposed_state.multiply(gaussian_proposal_sqrtm)                                
                            elif gaussian_proposal_covariance_shape == 'full':
                                proposed_state = ensemble_covariances_sqrtm.transpose_vector_product(proposed_state)
                            else:
                                raise ValueError("Unrecognized shape for the covariance matrix of the Gaussian proposal density %s" 
                                                    % repr(gaussian_proposal_covariance_shape))
                            # shift by the current state
                            proposed_state = proposed_state.add(current_state)
                            # proposed state from a Gaussian proposal density is ready...
                            
                            # Evaluate the acceptance probability, and use Metropolis-Hastings to update chain state
                            # 
                            accept_proposal, energy_loss, a_n, u_n = self.Gaussian_accept_reject(current_state, proposed_state, verbose=self._verbose)
                            #                             
                            acceptance_probabilities.append(a_n)
                            uniform_random_numbers.append(u_n)         
                            if accept_proposal:
                                acceptance_flags.append(1)
                                current_state = proposed_state.copy()
                            else:
                                acceptance_flags.append(0)
                                # do nothing, unless we decide later to keep all or some of the burned states, may be for diagnostics or so!
                                pass
                            # ------------
                            if self._verbose:
                                print("Burning[%d]: u_n=%4.3f, a_n=%4.3f" % (burn_ind, u_n, a_n))
                                print("Burning: J=", self._hmc_potential_energy_value(proposed_state))
                            # ------------
                            #
                        # Start the sampling/mixing stage:
                        #
                        for ens_ind in xrange(self.sample_size):
                            for mixing_ind in xrange(mixing_steps):
                                # use the Gaussian kernel to propose a state, then update the chain, and update the chain statistics
                                proposed_state[:] = np.random.randn(model_state_size)
                                # scale and shift the proposed state:
                                # Scale by standard deviation(s)                            
                                if gaussian_proposal_covariance_shape == 'diagonal':
                                    if covars_fix_diag:
                                        proposed_state = proposed_state.scale(gaussian_proposal_sqrtm)
                                    else:
                                        proposed_state = proposed_state.multiply(gaussian_proposal_sqrtm)                                
                                elif gaussian_proposal_covariance_shape == 'full':
                                    proposed_state = ensemble_covariances_sqrtm.transpose_vector_product(proposed_state)
                                else:
                                    raise ValueError("Unrecognized shape for the covariance matrix of the Gaussian proposal density %s" 
                                    % repr(gaussian_proposal_covariance_shape))
                                # shift by the current state
                                proposed_state = proposed_state.add(current_state)
                                # proposed state from a Gaussian proposal density is ready...
                                
                                # Evaluate the acceptance probability, and use Metropolis-Hastings to update chain state
                                accept_proposal, energy_loss, a_n, u_n = self.Gaussian_accept_reject(current_state, proposed_state, verbose=verbose)
                                #  
                                acceptance_probabilities.append(a_n)
                                uniform_random_numbers.append(u_n)                                   
                                if accept_proposal:
                                    acceptance_flags.append(1)                                    
                                    current_state = proposed_state.copy()
                                else:
                                    acceptance_flags.append(0)
                                    # do nothing, unless we decide later to keep all or some of the burned states, may be for diagnostics or so!
                                    pass
                                # ------------
                                if verbose:
                                    print("Ensemble member [%d] Mixing step [%d] : u_n=%4.3f, a_n=%4.3f" % (ens_ind, mixing_ind, u_n, a_n))
                                    print("Mixing: J=", self._hmc_potential_energy_value(proposed_state))
                                # ------------
                            
                            # add current_state to the ensemble as it is the most updated member
                            analysis_ensemble.append(current_state)  # append already copies the object and add reference to it.
                
                # This is very specific to QG-like models where some indexes are forced to some values e.g. dirichlet BC
                # It has to be re-written if it is a good idea...
                try:
                    zero_inds = self.model.model_configs['boundary_indexes']
                    for member in analysis_ensemble:
                        member[zero_inds] = 0.0
                    if self._verbose:
                        print("Successfully zeroed boundaries in ensemble members...")
                except (AttributeError, KeyError):
                    print("Attempt to zero bounds failed. Continuing...")
                #
                
                # We can consider analysis ensemble (for testing) in the Multi-Chain smapler approach if small steps are ataken by the symplectic integrator!
                inflation_fac = self.filter_configs['analysis_inflation_factor']
                if inflation_fac > 1.0:
                    analysis_ensemble = utility.inflate_ensemble(ensemble=analysis_ensemble, inflation_factor=inflation_fac, in_place=True)
            
                # update the configuration dictionaries with the generated ensemble
                self.filter_configs['analysis_ensemble'] = analysis_ensemble
                # update analysis_state
                self.filter_configs['analysis_state'] = utility.ensemble_mean(self.filter_configs['analysis_ensemble'])
                #
                
            else:
                print("Unsupported Proposal kernel %s" % repr(self.proposal_density))
                raise NotImplementedError()
                
        else:  # num_chains > 1
            #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # A MULTI-CHAIH sampler is created to sample the multi-modal mixture posterior
            # Everything in the multi-chain analysis step should go inside this branch
            # This branch is for the multi-chain sampler with (strictly) more than a single chain
            # If the proposal_density is HMC, the originial HMC is called for each component
            # If the proposal_density is Gaussian, a standard MCMC with Gaussian proposal samples each component.
            # The prarameters of the proposal density are either fixed (if the configurations say so), or infered from
            # the prior ensemble. The hyperparameters (parameters of the proposal kernel) can be set locally 
            # (individually for each component) or globally (equal for all components).
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # prposal_covariance_domain :'local' --> the covariance (for the proposal density)
            #  is taken from the corresponding component. 'global': the proposal covariance is fixed for all 
            prposal_covariance_domain = self.filter_configs['proposal_covariances_domain'].lower()
            #
            burn_in_steps = self.chain_parameters['burn_in_steps']
            mixing_steps = self.chain_parameters['mixing_steps']
            #
            # Initialize the analysis ensemble repository ( a list of model.state_vector()s )
            analysis_ensemble = []
            #
            acceptance_flags = []
            acceptance_probabilities = []
            uniform_random_numbers = []
            #
            # retrieve the GMM required information
            gmm_num_components = self.prior_distribution_statistics['gmm_num_components']
            gmm_means = self.prior_distribution_statistics['gmm_means']
            gmm_weights = self.prior_distribution_statistics['gmm_weights']
            gmm_optimal_covar_type = self.prior_distribution_statistics['gmm_optimal_covar_type']
                        
            # decide on the sample size per component:            
            # weighting strategy. check the possible values down in the branching
            weighting_strategy =  self.filter_configs['ensemble_size_weighting_strategy']  # 'weight_mean_likelihood'  this seems to be the best strategy!
            #
            if weighting_strategy in ['mean_likelihood', 'weight_mean_likelihood']:                
                # get likelihoods of the means
                means_likelihoods = np.empty(gmm_num_components)
                for comp_ind in xrange(gmm_num_components):
                    comp_mean = gmm_means[comp_ind].copy()
                    comp_mean_likelihood = self.evaluate_likelihood(comp_mean)  # log-likelihoods are evaluated by default
                    means_likelihoods[comp_ind] = comp_mean_likelihood
                # shift the log-likelihoods to avoid overflow errors
                means_likelihoods = means_likelihoods - np.max(means_likelihoods)
                min_log_likelihood = np.min(means_likelihoods)
                lower_bound = - 40
                if min_log_likelihood < lower_bound:
                    means_likelihoods = means_likelihoods + min_log_likelihood + lower_bound
                if self._verbose:
                    print('means_likelihoods', means_likelihoods)
                    print('gmm_weights', gmm_weights)
                    
                if weighting_strategy == 'mean_likelihood':                    
                    ensemble_weights = means_likelihoods
                else:
                    ensemble_weights = means_likelihoods * np.squeeze(np.asarray(gmm_weights))
                            
            elif  weighting_strategy == 'weight_mean_prior':
                means_prior = np.empty(gmm_num_components)
                for comp_ind in xrange(gmm_num_components):
                    comp_mean = _means[comp_ind]                    
                    means_prior[comp_ind] = self.evaluate_prior_pdf(comp_mean)
                ensemble_weights = means_prior * np.squeeze(np.asarray(gmm_weights))
                
            elif weighting_strategy == 'weight_mean_posterior':
                means_posterior = np.empty(gmm_num_components)
                for comp_ind in xrange(gmm_num_components):
                    comp_mean = gmm_means[comp_ind].copy()
                    current_energy = self._hmc_potential_energy_value(current_state)
                    means_posterior[comp_ind] = np.exp(-current_energy)
                ensemble_weights = means_posterior * np.squeeze(np.asarray(gmm_weights))
                
            elif weighting_strategy == 'mean_posterior':
                means_posterior = np.empty(gmm_num_components)
                for comp_ind in xrange(gmm_num_components):
                    comp_mean = gmm_means[comp_ind].copy()
                    current_energy = self._hmc_potential_energy_value(current_state)
                    means_posterior[comp_ind] = np.exp(-current_energy)
                ensemble_weights = means_posterior
                
            elif weighting_strategy == 'exact':
                gmm_covariances_det_log = np.abs(self.prior_distribution_statistics['gmm_covariances_det_log'])  # we need the log(|det(C)|)
                if self._verbose:
                    print('gmm_weights', gmm_weights)
                    print('gmm_covariances_det_log', gmm_covariances_det_log)
                    print('np.exp(np.squeeze(0.5 * gmm_covariances_det_log))', np.exp(np.squeeze(0.5 * gmm_covariances_det_log)))
                scaled_det_logs = np.exp(np.squeeze(- 0.5 * (gmm_covariances_det_log - np.min(gmm_covariances_det_log))))
                ensemble_weights = np.squeeze(np.asarray(gmm_weights)) * (scaled_det_logs)
                
            else:
                print("Unknown Weighting Strategy for Ensemble Size per Mixture Component %s !" % weighting_strategy)
                raise ValueError
            
            if (ensemble_weights == np.inf).any():
                print("Failed to assign proper local ensemble sizes;\nLikelihood function values include infinities! Aborting!")
                raise ValueError
                                
            # normalized likelihoods (not enough. Corrected by multiplying by prior weights)
            if self._verbose:
                print('Un-normalized ensemble_weights:', ensemble_weights)
            ensemble_weights = ensemble_weights / float(np.sum(ensemble_weights))
            if self._verbose:
                print('Normalized ensemble_weights:', ensemble_weights)
            chains_ensemble_sizes = np.asarray([np.floor(n*self.sample_size) for n in ensemble_weights ], dtype=np.int)
            if self._verbose:
                print('chains_ensemble_sizes', chains_ensemble_sizes)
            excess = self.sample_size - chains_ensemble_sizes.sum()
            loc_ind = 0
            while excess > 0: 
                chains_ensemble_sizes[loc_ind%len(chains_ensemble_sizes)] += 1
                excess -= 1
                loc_ind += 1
            if np.sum(chains_ensemble_sizes) != self.sample_size:
                print("Attention required; The original ensemble size is not the same as the sum of \
                        local ensemble sizes. \n Original ensemble size = %d.\n  \
                        Local ensemble sizes: %s " % (self.sample_size, repr(chains_ensemble_sizes))
                      )
            else:
                if self._verbose:
                    print('updated chains_ensemble_sizes: ', chains_ensemble_sizes)
            
            # Now we have the ensemble sizes per chains already set. 
            # Test for false (zero) chains to be sampled
            num_none_zero = np.size(np.nonzero(chains_ensemble_sizes))
            if num_none_zero == 1:
                if self._verbose:
                    print("OK, some components will not be sampled due to extrememly low likelihoods...")
                tmp_forced_num_chains = self.chain_parameters['forced_num_chains']
                self.chain_parameters['forced_num_chains'] = 1
                chain_diagnostics = self.analysis()
                self.chain_parameters['forced_num_chains'] = tmp_forced_num_chains
                print(chain_diagnostics)
                return chain_diagnostics
                
            # Flag to check if the burn-in steps should be reduced (divide by number of componentes)
            reduce_burn_in_stage = self.filter_configs['reduce_burn_in_stage']
            reset_random_seed_per_chain = self.filter_configs['reset_random_seed_per_chain']
            if reset_random_seed_per_chain:
                random_seed_per_chain = self.filter_configs['random_seed_per_chain']
            #
            # Loop over all chains. All chains will run sequentially here. Multiprocessing will be considered later
            for chain_ind in xrange(num_of_chains):
                #
                # Initialize local repositories:
                local_ensemble_size = chains_ensemble_sizes[chain_ind]
                
                # Let's consider the case where a component should be neglected (very small relative importance ratio)!
                if local_ensemble_size == 0:
                    continue
                    
                if reduce_burn_in_stage:
                    local_burn_in_steps = burn_in_steps / num_of_chains  # 
                else:
                    local_burn_in_steps = burn_in_steps
                    
                local_chain_length = local_burn_in_steps + local_ensemble_size*mixing_steps
                
                # Reset the seed of the random number generators for each of the chains
                if reset_random_seed_per_chain:
                    np.random.seed(random_seed_per_chain)
                
                # All generated sample points will be kept for testing and efficiency analysis
                local_ensemble_repository = []
                local_acceptance_flags = []
                local_acceptance_probabilities = []
                local_uniform_random_numbers = []
                #
                    
                local_chain_state = gmm_means[chain_ind].copy()  # initial state = component mean
                
                model_state_size = model.state_size()
                #
                if self.proposal_density == 'hmc':
                    # 
                    # Update the momentum information based on variances of the local component:
                        
                    if self.mass_matrix_strategy == 'identity':
                        mass_matrix_diagonal = self.mass_matrix_scaling_factor
                        
                    elif self.mass_matrix_strategy in ['prior_variances', 'prior_precisions']:
                        # get the variances of the local component, based on the domain of the proposal covariances
                        if prposal_covariance_domain == 'local':
                            #
                            if gmm_optimal_covar_type == 'tied':
                                component_variances = self.prior_variances_list[0].copy()  # all components have the same variances
                            else:
                                component_variances = self.prior_variances_list[chain_ind].copy()                            
                            #
                        elif prposal_covariance_domain == 'global':
                            #
                            component_variances = self.prior_variances.copy()
                            #
                        else:
                            raise ValueError("the 'prposal_covariance_domain' [%s] is not recognized!" % str(prposal_covariance_domain))
                        if not isinstance(component_variances, np.ndarray):
                            component_variances = component_variances[:].copy()
                        #
                        print("Collecting [%d] ensemble points from this component %d (out of %d components)" % (local_ensemble_size, chain_ind+1, num_of_chains))
                        variances_thresh = 1e-15
                        if (component_variances<variances_thresh).any():
                            print("Caution: Some of the variances of component number [%d] are very small; Thresholding until Tapering is incorporated!" % chain_ind)
                            component_variances[(component_variances<variances_thresh).astype(int)] = variances_thresh
                        
                        if self.mass_matrix_strategy == 'prior_variances':
                            mass_matrix_diagonal = component_variances
                        else :
                            mass_matrix_diagonal = 1.0 / component_variances
                        #
                        # A check for improper values: This may result in slow execution, but it is good for testing the scheme until it is stable
                        if np.isnan(mass_matrix_diagonal).any():
                            print("Found NAN in the proposed mass matrix diagonal! Replacing with 1")
                            mass_matrix_diagonal[np.isnan(mass_matrix_diagonal)] = 1
                        elif np.isinf(mass_matrix_diagonal).any():
                            print("Found INF in the proposed mass matrix diagonal! Replacing with 1")
                            mass_matrix_diagonal[np.isnan(mass_matrix_diagonal)] = 1
                        else:
                            pass
                                                        
                    #
                    else:
                        raise ValueError("Unrecognized mass matrix strategy!")
                        
                    #
                    mass_matrix_shape = self._momentum._mass_matrix_shape                    
                    if mass_matrix_shape == 'diagonal':
                        self.update_momentum(mass_matrix=mass_matrix_diagonal, mass_matrix_shape=mass_matrix_shape, dimension=model_state_size, model=model)
                            #update_momentum(self, mass_matrix=None, mass_matrix_shape=None, dimension=None, model=None, diag_val_thresh=1e-4)
                    else:
                        print("You REALLY <<SHOULD>> avoid non-diagonal mass matrix settings")
                        raise NotImplementedError()
                                        
                    # sample the current chain (using HMC proposal)
                    # --------------------------------------------
                    # b) start constructing the local chain:
                    # --------------------------------------------
                    # i] get the forecast state as the mean of the forecast ensemble. Will not be used for GMM!                    
                    local_chain_initial_state = gmm_means[chain_ind].copy()
                    
                    # current state of the local chain
                    current_state = local_chain_initial_state.copy()
                    #                    
                    # Construct a placeholder for the momentum vectors to avoid reconstruction and copying
                    current_momentum = self._momentum.generate_momentum()
                    
                    diagnostic_scheme = self.chain_parameters['convergence_diagnostic']  # for early termination
                    # Initialize the chain(s) repositories;
                    
                    # Start the burn-in stage:
                    #
                    for burn_ind in xrange(local_burn_in_steps):
                        # use the HMC kernel to propose a state, then update the chain, and update the chain statistics
                        #
                        current_momentum = self._momentum.generate_momentum(momentum_vec=current_momentum)
                        try:
                            proposed_state, proposed_momentum = self._hmc_propose_state(current_state=current_state,
                                                                                        current_momentum=current_momentum)
                        # Consider putting this in a DEGUGGING MODE
                        # if np.isinf(proposed_state[:]).any():
                        #     print("The proposed state contains inifinit values?")
                        #     print(proposed_state)
                        #     raise ValueError
                        
                        # update the ensemble
                            accept_proposal, energy_loss, a_n, u_n \
                                            = self._hmc_MH_accept_reject(current_state=current_state,
                                                                         current_momentum=current_momentum,
                                                                         proposed_state=proposed_state,
                                                                         proposed_momentum=proposed_momentum,
                                                                         verbose=self._verbose
                                                                         )
                        except(ValueError):
                            accept_proposal, energy_loss, a_n, u_n = False, 0, 0, 1
                                                    
                        local_acceptance_probabilities.append(a_n)
                        local_uniform_random_numbers.append(u_n)         
                        if accept_proposal:
                            local_acceptance_flags.append(1)
                            current_state = proposed_state.copy()
                        else:
                            local_acceptance_flags.append(0)
                            # do nothing, unless we decide later to keep all or some of the burned states, may be for diagnostics or so!
                            pass
                        # ------------
                        if self._verbose:
                            print("Burning step [%d] : u_n=%4.3f, a_n=%4.3f" % (burn_ind, u_n, a_n))
                            print('Burning: J=', self._hmc_total_energy(proposed_state, proposed_momentum))
                        # ------------
                        
                    # Start the sampling/mixing stage:
                    #
                    for ens_ind in xrange(local_ensemble_size):
                        for mixing_ind in xrange(mixing_steps):
                            # use the HMC kernel to propose a state, then update the chain, and update the chain statistics
                            #
                            current_momentum = self._momentum.generate_momentum(momentum_vec=current_momentum)
                            try:
                                proposed_state, proposed_momentum = self._hmc_propose_state(current_state=current_state,
                                                                                            current_momentum=current_momentum)
                                # update the ensemble
                                accept_proposal, energy_loss, a_n, u_n \
                                                = self._hmc_MH_accept_reject(current_state=current_state,
                                                                             current_momentum=current_momentum,
                                                                             proposed_state=proposed_state,
                                                                             proposed_momentum=proposed_momentum,
                                                                             verbose=self._verbose
                                                                             )
                            except(ValueError):
                                accept_proposal, energy_loss, a_n, u_n = False, 0, 0, 1
                                                        
                            local_acceptance_probabilities.append(a_n)
                            local_uniform_random_numbers.append(u_n)         
                            if accept_proposal:
                                local_acceptance_flags.append(1)                                    
                                current_state = proposed_state.copy()
                            else:
                                local_acceptance_flags.append(0)
                                # do nothing, unless we decide later to keep all or some of the burned states, may be for diagnostics or so!
                                pass
                            # ------------
                            if self._verbose:
                                print("Chain [%d] ; Ensemble member [%d]; Mixing step [%d] : u_n=%4.3f, a_n=%4.3f" % (chain_ind, ens_ind, mixing_ind, u_n, a_n))
                                print('Mixing: J=', self._hmc_total_energy(proposed_state, proposed_momentum))
                            # ------------
                        
                        # add current_state to the ensemble as it is the most updated member
                        local_ensemble_repository.append(current_state.copy())  # append already copies the object and add reference to it.                        
                        #
                                
                    
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Add the locally collected ensemble to the analysis_ensemble repository. 
                    # This should be augmented with the local updates        
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # This is very specific to QG-like models where some indexes are forced to some values e.g. dirichlet BC
                    # It has to be re-written if it is a good idea...
                    #
                    try:
                        zero_inds = self.model.model_configs['boundary_indexes']
                        for analysis_member in local_ensemble_repository:
                            analysis_member[zero_inds] = 0.0
                            analysis_ensemble.append(analysis_member)
                    except (AttributeError, KeyError):
                        print("Attempt to zero bounds failed. Continuing...")
                        for analysis_member in local_ensemble_repository:
                            analysis_ensemble.append(analysis_member)
                    
                    #
                    for acceptance_flag in local_acceptance_flags:
                        acceptance_flags.append(acceptance_flag)
                    #
                    for acceptance_probability in local_acceptance_probabilities:
                        acceptance_probabilities.append(acceptance_probability)
                    #
                    for uniform_random_number in local_uniform_random_numbers:
                        uniform_random_numbers.append(uniform_random_number)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    #
                    
                elif self.proposal_density == 'gaussian':
                    #
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Create multiple Markov chains, each has a proposal density that is a Gaussian, with mean centered 
                    # around the current state of the local chain, and the covariance is set locally to the covariance 
                    # (maybe diagonal) of the local component.
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # 
                    # a) retrieve/set the chain configurations:
                    # --------------------------------------------
                    # i] get the forecast state as the mean of the forecast ensemble. Will not be used for GMM!                    
                    local_chain_initial_state = gmm_means[chain_ind].copy()
                    
                    # current state of the local chain
                    current_state = local_chain_initial_state.copy()
                    
                    # this flag is to set the diagonal of the proposal covariance matrix to single value
                    covars_fix_diag = self.filter_configs['fix_gaussian_proposal_covar_diagonal']
                    
                    # ii] set the covariance of the proposal density:
                    gaussian_proposal_covariance_shape = self.filter_configs['gaussian_proposal_covariance_shape'].lower()
                    if gaussian_proposal_covariance_shape == 'diagonal':
                        # get the variances of the local component, based on the domain of the proposal covariances
                        if prposal_covariance_domain == 'local':
                            #
                            if gmm_optimal_covar_type == 'tied':
                                component_variances = self.prior_variances_list[0].copy()  # all components have the same variances
                            else:
                                component_variances = self.prior_variances_list[chain_ind].copy()
                            #
                        elif prposal_covariance_domain == 'global':
                            #
                            component_variances = self.prior_variances.copy()
                            #
                        else:
                            print("the 'prposal_covariance_domain' [%s] is not recognized!" % str(prposal_covariance_domain))
                            raise ValueError
                            
                        #
                        if covars_fix_diag:
                            # scaling by a scalar would be optimal in this case
                            gaussian_proposal_variance = self.filter_configs['gaussian_proposal_variance']
                            if np.isscalar(gaussian_proposal_variance) :
                                gaussian_proposal_covariance = gaussian_proposal_variance
                            else:                                
                                gaussian_proposal_covariance = component_variances.min()  # it's a numpy array
                            gaussian_proposal_precisions = 1.0 / gaussian_proposal_covariance
                            gaussian_proposal_sqrtm = np.sqrt(gaussian_proposal_covariance)
                        else:
                            # here we need a state vector to store the diagonal of the covariance matrix
                            gaussian_proposal_covariance = model.state_vector()
                            gaussian_proposal_covariance[:] = self.prior_variances.copy()
                            gaussian_proposal_precisions = gaussian_proposal_covariance.reciprocal(in_place=False)  # do we need a copy?                   
                            gaussian_proposal_sqrtm = gaussian_proposal_covariance.sqrt()
                        
                    elif gaussian_proposal_covariance_shape == 'full':
                        # TODO: Check for hybridization, and localization
                        # set the covariance of the proposal exactly to the covariance of the corresponding component                        
                        if gmm_optimal_covar_type in ['diag', 'spherical']:
                            gaussian_proposal_sqrtm = self.prior_distribution_statistics['gmm_covariances'][chain_ind]  # state vector
                            gaussian_proposal_covariance_shape = 'diagonal'  # force the shape to be diagonal
                            
                        elif gmm_optimal_covar_type == 'tied':
                            # Consider evaluating the cholesky decomposition once instead of re-evaluating it!
                            component_covariances = self.prior_distribution_statistics['gmm_covariances']  # state_matrix
                            gaussian_proposal_sqrtm = component_covariances.cholesky(in_place=True)
                            
                        elif gmm_optimal_covar_type == 'full':
                            component_covariances = self.prior_distribution_statistics['gmm_covariances'][chain_ind]  # state_matrix
                            gaussian_proposal_sqrtm = component_covariances.cholesky(in_place=True)
                            
                        else:
                            raise ValueError("This is unexpected!. optimal_covar_type = '%s' " % gmm_optimal_covar_type)
                                                
                    else:
                        raise ValueError("Unrecognized shape for the covariance matrix of the Gaussian proposal density %s" 
                                            % repr(gaussian_proposal_covariance_shape))
                                    
                    # iii] set the number of steps in the chain (burn-in, mixing --> total number of chain steps):                
                    
                    # --------------------------------------------
                    # b) start constructing the local chain:
                    # --------------------------------------------
                    #
                    diagnostic_scheme = self.chain_parameters['convergence_diagnostic']  # for early termination
                    # Initialize the chain(s) repositories;
                    # Check the placeholder of the analysis ensemble to avoid reconstruction
                    
                    
                    # Start the burn-in stage:
                    proposed_state = model.state_vector() # a placeholder for a proposal state from a Gaussian distribution
                    model_state_size = model.state_size()
                    #
                    for burn_ind in xrange(local_burn_in_steps):
                        # use the Gaussian kernel to propose a state, then update the chain, and update the chain statistics
                        proposed_state[:] = np.random.randn(model_state_size)
                        # scale and shift the proposed state:
                        # Scale by standard deviation(s)                            
                        if gaussian_proposal_covariance_shape == 'diagonal':
                            if covars_fix_diag:
                                proposed_state = proposed_state.scale(gaussian_proposal_sqrtm)
                            else:
                                proposed_state = proposed_state.multiply(gaussian_proposal_sqrtm)                                
                        elif gaussian_proposal_covariance_shape == 'full':
                            proposed_state = gaussian_proposal_sqrtm.transpose_vector_product(proposed_state)
                        else:
                            raise ValueError("Unrecognized shape for the covariance matrix of the Gaussian proposal density %s" 
                                                % repr(gaussian_proposal_covariance_shape))
                        # shift by the current state
                        proposed_state = proposed_state.add(current_state)
                        # proposed state from a Gaussian proposal density is ready...
                        
                        # Evaluate the acceptance probability, and use Metropolis-Hastings to update chain state
                        accept_proposal, energy_loss, a_n, u_n = self.Gaussian_accept_reject(current_state, proposed_state, verbose=self._verbose)
                        #                             
                        local_acceptance_probabilities.append(a_n)
                        local_uniform_random_numbers.append(u_n)         
                        if accept_proposal:
                            local_acceptance_flags.append(1)
                            current_state = proposed_state.copy()
                        else:
                            local_acceptance_flags.append(0)
                            # do nothing, unless we decide later to keep all or some of the burned states, may be for diagnostics or so!
                            pass
                        # ------------
                        if self._verbose:
                            print("Burninig step [%d] : u_n=%4.3f, a_n=%4.3f" % (burn_ind, u_n, a_n))
                            print('Burning: J=', self._hmc_potential_energy_value(proposed_state))
                        # ------------
                        
                    # Start the sampling/mixing stage:
                    #
                    for ens_ind in xrange(local_ensemble_size):
                        for mixing_ind in xrange(mixing_steps):
                            # use the Gaussian kernel to propose a state, then update the chain, and update the chain statistics
                            proposed_state[:] = np.random.randn(model_state_size)
                            # scale and shift the proposed state:
                            # Scale by standard deviation(s)                            
                            if gaussian_proposal_covariance_shape == 'diagonal':
                                if covars_fix_diag:
                                    proposed_state = proposed_state.scale(gaussian_proposal_sqrtm)
                                else:
                                    proposed_state = proposed_state.multiply(gaussian_proposal_sqrtm)                                
                            elif gaussian_proposal_covariance_shape == 'full':
                                proposed_state = gaussian_proposal_sqrtm.transpose_vector_product(proposed_state)
                            else:
                                raise ValueError("Unrecognized shape for the covariance matrix of the Gaussian proposal density %s" 
                                % repr(gaussian_proposal_covariance_shape))
                            # shift by the current state
                            proposed_state = proposed_state.add(current_state)
                            # proposed state from a Gaussian proposal density is ready...
                            
                            # Evaluate the acceptance probability, and use Metropolis-Hastings to update chain state
                            # 
                            accept_proposal, energy_loss, a_n, u_n = self.Gaussian_accept_reject(current_state, proposed_state, verbose=self._verbose)
                            #  
                            local_acceptance_probabilities.append(a_n)
                            local_uniform_random_numbers.append(u_n)                                   
                            if accept_proposal:
                                local_acceptance_flags.append(1)
                                current_state = proposed_state.copy()
                            else:
                                local_acceptance_flags.append(0)
                                # do nothing, unless we decide later to keep all or some of the burned states, may be for diagnostics or so!
                                pass                            
                            # ------------
                            if self._verbose:
                                print("Chain [%d]; Ensemble member [%d];Mixing step [%d] : u_n=%4.3f, a_n=%4.3f" % (chain_ind, ens_ind, mixing_ind, u_n, a_n))
                                print('Mixing: J=', self._hmc_potential_energy_value(proposed_state))
                            # ------------
                        
                        # add current_state to the ensemble as it is the most updated member
                        local_ensemble_repository.append(current_state)  # append already copies the object and add reference to it.
                        #
                                
                    
                    #
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Add the locally collected ensemble to the analysis_ensemble repository. 
                    # This should be augmented with the local updates        
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # This is very specific to QG-like models where some indexes are forced to some values e.g. dirichlet BC
                    # It has to be re-written if it is a good idea...
                    #
                    try:
                        zero_inds = self.model.model_configs['boundary_indexes']
                        for analysis_member in local_ensemble_repository:
                            analysis_member[zero_inds] = 0.0
                            analysis_ensemble.append(analysis_member)
                    except (AttributeError, KeyError):
                        print("Attempt to zero bounds failed. Continuing...")
                        for analysis_member in local_ensemble_repository:
                            analysis_ensemble.append(analysis_member)
                    
                        
                    #
                    for acceptance_flag in local_acceptance_flags:
                        acceptance_flags.append(acceptance_flag)
                    #
                    for acceptance_probability in local_acceptance_probabilities:
                        acceptance_probabilities.append(acceptance_probability)
                    #
                    for uniform_random_number in local_uniform_random_numbers:
                        uniform_random_numbers.append(uniform_random_number)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    #
                    
                else:
                    print("Unsupported Proposal kernel %s" % repr(self.proposal_density))
                    raise NotImplementedError()
                
                # End for loop over the chains assigned the task of sampling for each component of the GMM prior
                pass

            #
            # We can consider analysis ensemble (for testing) in the Multi-Chain smapler approach if small steps are ataken by the symplectic integrator!
            inflation_fac = self.filter_configs['analysis_inflation_factor']
            if inflation_fac > 1.0:
                analysis_ensemble = utility.inflate_ensemble(ensemble=analysis_ensemble, inflation_factor=inflation_fac, in_place=True)
 
            #
            # update the configuration dictionaries with the generated ensemble, and analysis state
            self.filter_configs['analysis_ensemble'] = analysis_ensemble
            analysis_state = utility.ensemble_mean(analysis_ensemble)
            self.filter_configs['analysis_state'] = analysis_state
            #
            # ------------
            #  if self._verbose:
            #      print('sampling from %d chains' % num_of_chains)
            #      # print('analysis_ensemble', analysis_ensemble)            
            #      # print('analysis_state', analysis_state)
            # ------------
            #
            
            # This branch is for the multi-chain sampler with (strictly) more than a single chain
            pass
            
        #
        acceptance_rate = float(np.asarray(acceptance_flags).sum()) / len(acceptance_flags) * 100.0
        rejection_rate = (100.0 - acceptance_rate)
        chain_diagnostics = dict(acceptance_rate=acceptance_rate,
                                 rejection_rate=rejection_rate,
                                 acceptance_probabilities=acceptance_probabilities,
                                 acceptance_flags=acceptance_flags,
                                 uniform_random_numbers=uniform_random_numbers
                                 )
        
        # Add chain diagnostics to the output_configs dictionary for proper outputting:
        self.output_configs['filter_statistics'].update(dict(chain_diagnostics=chain_diagnostics))
        return chain_diagnostics
        # ==========================================(End Analysis step)==========================================
        #
     
    #
    def Gaussian_accept_reject(self, current_state, proposed_state, verbose=False):
        """
        Metropolis-Hastings accept/reject criterion based on loss of postential energy between current and proposed states
        
        Args:
            current_state: current state of the chain
            proposed_state: proposed state of the chain using Gaussian jumping distribution
            verbose (default False): can be used for extra on-screen printing while debugging, and testing
            
            Returns:
            accept_state: True/False, whether to accept the proposed state/momentum or reject them
            energy_loss: the difference between total energy of the proposed and the current pair
            a_n: Metropolis-Hastings (MH) acceptance probability
            u_n: the uniform random number compared to a_n to accept/reject the sate (MH)
        
        Returns:
            None
            
        """
        #
        # Evaluate the acceptance probability, and use Metropolis-Hastings to update chain state
        # 1- value of the negative-log of the posterior (unscaled) at the current state:
        current_energy = self._hmc_potential_energy_value(current_state)
        proposed_energy = self._hmc_potential_energy_value(proposed_state)
        energy_loss = proposed_energy - current_energy
        #
        
        thresh = 500  # a threshold on the energy loss to avoid overflow errors
        if abs(energy_loss) > thresh:  # this should avoid overflow errors
            energy_loss = np.sign(energy_loss) * thresh
        
        # calculate probability of acceptance and decide whether to accept or reject the proposed state
        a_n = min(1.0, np.exp(-energy_loss))
        #
        if verbose:
            print('energy_loss', energy_loss)
            print('MH-acceptance_probability', a_n)
        #
        u_n = np.random.rand()
        if a_n > u_n:
            accept_state = True
        else:
            accept_state = False
        #
        # we will return all information required for chain diagnostics. HMC should be udpted similarly
        return accept_state, energy_loss, a_n, u_n
        #

    #
    def evaluate_prior_pdf(self, state):
        """
        Evaluate the GMM prior at a given state
        
        Args:
            state: model.state_vector
        
        Returns:
            value of the prior (GMM) at the given state
            
        """
        gmm_optimal_model = self.prior_distribution_statistics['gmm_optimal_model']
        log_probability = gmm_optimal_model.score(np.squeeze(state[:]))
        #
        return np.exp(log_probability)
        #
        
    #
    def evaluate_likelihood(self, state, return_log_likelihood=True, apply_scaling=False, obs_err_model=None):
        """
        Evaluate the likelihood function at a given state
         
        Args:
            state: model.state_vector
            return_log_likelihood (default True): if False, the scaling factor of the likelihood function is evaluated
            apply_scaling: Falg to normalize the likelihood. Default is false. This option requires the observation_error_model to be passed
            obs_err_model (default None): model.observation_error_model. If None, the observaiton_error model is 
                requested from self.model or self.filter_configs['model']
        
        Returns:
            value of the likelihood for the given state
        
        """        
        # 1- Observation term ( same as Gaussian prior) keep here in case we add different prior...
        # get a reference to the dynamical model used        
        model = self.filter_configs['model']
        
        # get the measurements vector
        observation = self.filter_configs['observation']
        
        model_observation = model.evaluate_theoretical_observation(state)  # H_k(x_k)        
        innovations = model_observation.axpy(-1.0, observation)  # innovations =  H(x) - y
        if obs_err_model is not None:
            scaled_innovations = obs_err_model.error_covariances_inv_prod_vec(innovations, in_place=False)
        else:
            scaled_innovations = model.observation_error_model.error_covariances_inv_prod_vec(innovations, in_place=False)
        log_likelihood = - 0.5 * scaled_innovations.dot(innovations)
        #
        
        if apply_scaling:
            try:
                detR = obs_err_model.detR
            except(ValueError, AttributeError, NameError):
                print("Couldn't find the determinant of the observation error covariance matrix. \
                        Scaling cannot be done! \nReturning Un-Scaled likelihood")
                detR = None
            #
            # If detR is not calculated during the error_model initialization, it is replaced with None
            if detR is not None:  
                if detR == np.inf:
                    print("The retrieved determinant of the observation error covariance matrix is INFINITY. \
                        Scaling cannot be done! \nReturning Un-Scaled likelihood")
                
                else:
                    # Now scale the likelihood
                    try:
                        obs_vec_size = model.observation_vector_size()
                        pi = np.pi
                        # 
                        shift = obs_vec_size * np.log(2*pi)
                        shift += np.log(detR)
                        shift *= -0.5
                        #
                        log_likelihood +=  shift
                        #
                    except(ValueError, AttributeError, NameError):
                        print("Couldn't retrieve the model, or the observation vector information from the observation error model. \
                               Scaling cannot be done! \nReturning Un-Scaled likelihood")
        #
        if return_log_likelihood:
            return log_likelihood
        else:
            return np.exp(log_likelihood)
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
            super(MultiChainMCMC, self).print_cycle_results()
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
        #
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().save_cycle_results(output_dir=output_dir, cleanup_out_dir=cleanup_out_dir)
        else:
            # old-stype class
            super(MultiChainMCMC, self).save_cycle_results(output_dir=output_dir, cleanup_out_dir=cleanup_out_dir)
        pass  # Add more...
        #
        
        
