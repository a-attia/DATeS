
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
    HMCSmoother:
    ------------
    A class implementing the Hamiltonian/Hybrid Monte-Carlo Sampling family for smoothing developed by Ahmed Attia, Vishwas Rao, Razvan Stefanescu, and Adrian Sandu.
    - Publications:
        i) Ahmed Attia, Vishwas Rao, and Adrian Sandu. "A Hybrid Monte-Carlo sampling smoother for four-dimensional data assimilation." International Journal for Numerical Methods in Fluids 83.1 (20170: 90-112.
       ii) Ahmed Attia, Razvan Stefanescu, and Adrian Sandu. "The reduced-order Hybrid Monte Carlo sampling smoother." International Journal for Numerical Methods in Fluids 83.1 (2017): 28-51.
       iii) Ahmed Attia, Mahesh Narayanamurthi, and Vishwas Rao, "Cluster sampling smoother for 4DDA" Under Development

    This is a class derived from the HMCSmoother class for DA sequential smoothing.
"""


import numpy as np
import os

try:
    import cPickle as pickle
except:
    import pickle

from models_base import ModelsBase


import dates_utility as utility
from smoothers_base import SmootherBase
from state_vector_base import StateVectorBase as StateVector
from state_matrix_base import StateMatrixBase as StateMatrix
from observation_vector_base import ObservationVectorBase as ObservationVector

from scipy.linalg import lu_factor, lu_solve


class HMCSmoother(SmootherBase):
    """
    A class implementing the Hamiltonian/Hybrid Monte-Carlo Sampling family for smoothing developed by Ahmed Attia (2014/2015/2016).

    HMC smoother constructor.

    Args:
        smoother_configs:  dict, a dictionary containing HMC smoother configurations.
            Supported configuarations:
                * model (default None):  model object
                * smoother_name (default None): string containing name of the smoother; used for output.
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
                    This is where the smoother output (analysis state) will be saved and returned.
                * forecast_ensemble (default None): a placeholder of the forecast/background ensemble.
                    All ensembles are represented by list of model.state_vector objects
                * forecast_state (default None): model.state_vector object containing the forecast state.
                * smoother_statistics: dict,
                    A dictionary containing updatable smoother statistics. This will be updated by the smoother.

        output_configs: dict, a dictionary containing screen/file output configurations.
            Supported configuarations:
                * scr_output (default False): Output results to screen on/off switch
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default 'Assimilation_Results'): relative path (to DATeS root directory)
                    of the directory to output results in

                * smoother_statistics_dir (default 'Smoother_Statistics'): directory where smoother statistics (such as RMSE, ESS,...) are saved
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

    """
    #
    _smoother_name = "HMC-S"
    #
    _def_local_smoother_configs = dict(smoother_name=_smoother_name,
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
                                     smoother_statistics=dict(forecast_rmse=None,
                                                            analysis_rmse=None,
                                                            initial_rmse=None,
                                                            rejection_rate=None
                                                            )
                                     )
    _local_def_output_configs = dict(scr_output=True,
                                     file_output=False,
                                     smoother_statistics_dir='Smoother_Statistics',
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
    def __init__(self, smoother_configs=None, output_configs=None):

        smoother_configs = utility.aggregate_configurations(smoother_configs, HMCSmoother._def_local_smoother_configs)
        output_configs = utility.aggregate_configurations(output_configs, HMCSmoother._local_def_output_configs)
        SmootherBase.__init__(self, smoother_configs=smoother_configs, output_configs=output_configs)
        #
        try:
            self.model = self.smoother_configs['model']
        except(KeyError, AttributeError, NameError):
            print("A model object has to be passed in the confighurations dictionary!")
            raise AssertionError
        else:
            if self.model is None or not isinstance(self.model, ModelsBase):
                print("A model object has to be passed in the confighurations dictionary!")
                raise AssertionError
            else:
                # All is good model object assigned...
                pass
        #
        try:
            self._model_step_size = self.model._default_step_size
        except:
            self._model_step_size = FDVAR.__time_eps
        self._time_eps = HMCSmoother.__time_eps
        # print("self.__time_eps", self._time_eps)
        # print("self._model_step_size", self._model_step_size)

        #
        # the following configuration are smoother-specific.
        # validate the ensemble size
        if self.smoother_configs['ensemble_size'] is None:
            try:
                forecast_ensemble_size = len(self.smoother_configs['forecast_ensemble'])
            except (ValueError, AttributeError, TypeError):
                forecast_ensemble_size = 0
            try:
                analysis_ensemble_size = len(self.smoother_configs['analysis_ensemble'])
            except (ValueError, AttributeError, TypeError):
                analysis_ensemble_size = 0

            self.sample_size = max(forecast_ensemble_size, analysis_ensemble_size)
            #
        else:
            self.sample_size = self.smoother_configs['ensemble_size']

        self.symplectic_integrator = self.smoother_configs["chain_parameters"]['Symplectic_integrator'].lower()
        self.symplectic_integrator_parameters = {
            'automatic_tuning_scheme': self.smoother_configs["chain_parameters"]['Automatic_tuning_scheme'],
            'step_size': self.smoother_configs["chain_parameters"]['Hamiltonian_step_size'],
            'num_steps': int(self.smoother_configs["chain_parameters"]['Hamiltonian_num_steps'])
        }

        # supported: [prior_precisions, prior_variances, identity]
        self.mass_matrix_strategy = self.smoother_configs["chain_parameters"]['Mass_matrix_type'].lower()
        self.mass_matrix_scaling_factor = self.smoother_configs["chain_parameters"]['Mass_matrix_scaling_factor']

        self.chain_parameters = {
            'initial_state': self.smoother_configs["chain_parameters"]['Initial_state'],
            'burn_in_steps': int(self.smoother_configs["chain_parameters"]['Burn_in_num_steps']),
            'mixing_steps': int(self.smoother_configs["chain_parameters"]['Mixing_steps']),
            'convergence_diagnostic': self.smoother_configs["chain_parameters"]['Convergence_diagnostic_scheme']
        }

        self.optimize_to_converge = self.smoother_configs["chain_parameters"]['Burn_by_optimization']
        if self.optimize_to_converge:
            try:
                self.optimization_parameters = self.smoother_configs["chain_parameters"]['Optimization_parameters']
            except(AttributeError, NameError, ValueError):
                # set default parameters...
                print("Optimization for convergence is yet to be implemented in full")
                raise NotImplementedError()

        if self.smoother_configs["chain_parameters"]['Tempering_scheme'] is not None:
            self.tempering_scheme = self.smoother_configs["chain_parameters"]['Tempering_scheme'].lower()
            self.tempering_parameters = self.smoother_configs["chain_parameters"]['Tempering_parameters']
        else:
            self.tempering_scheme = None

        self.prior_distribution = self.smoother_configs['prior_distribution'].lower()
        if self.prior_distribution not in self._supported_prior_distribution:
            print("Unrecognized prior distribution [%s]!" % self.prior_distribution)
            raise ValueError()

        if self.prior_distribution == 'gaussian':
            #
            forecast_ensemble = self.smoother_configs['forecast_ensemble']
            if forecast_ensemble is not None:
                self.forecast_state = utility.ensemble_mean(forecast_ensemble)
            else:
                self.forecast_state = None
            # Add smoother statistics to the output configuration dictionary for proper outputting.
            if 'smoother_statistics' not in self.output_configs:
                self.output_configs.update(dict(smoother_statistics={}))

        elif self.prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            # update the smoother name
            self._smoother_name = "ClHMC-F"
            self.smoother_configs['smoother_name'] = self._smoother_name
            # Generate GMM parameters... May be it is better to move this to a method to generate prior info.
            # It might be needed in the forecast step in case FORECAST is carried out first, and forecast ensemble is empty!
            self._gmm_prior_settings = self.smoother_configs['gmm_prior_settings']

            # Add smoother statistics to the output configuration dictionary for proper outputting.
            if 'smoother_statistics' not in self.output_configs:
                self.output_configs.update(dict(smoother_statistics=dict(gmm_prior_statistics=None)))

            # Generate the forecast state only for forecast RMSE evaluation. It won't be needed by the GMM+HMC sampler
            forecast_ensemble = self.smoother_configs['forecast_ensemble']
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
        if self.smoother_configs['forecast_ensemble'] is None:
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

        Returns:
            momentum_obj:

        """
        # create a momentum object with a predefined mass matrix based on smoother parameters.
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
    def smoothing_cycle(self, update_reference=False):
        """
        Apply the smoothing step. Forecast, then Analysis...
        All arguments are accessed and updated in the configurations dictionary.

        Args:
            update_reference (default True): bool,
                A flag to decide whether to update the reference state in the smoother or not.

        """
        # Call basic functionality from the parent class:
        SmootherBase.smoothing_cycle(self, update_reference=update_reference)
        #
        # Add further functionality if you wish...
        #

    #
    def forecast(self, generate_prior_info=False):
        """
        Forecast step of the smoother.
        Use the model object to propagate each ensemble member to the end of the given checkpoints to
        produce and ensemble of forecasts.
        Smoother configurations dictionary is updated with the new results.
        If the prior is assumed to be a Gaussian, we are all set, otherwise we have to estimate it's
        parameters based on the provided forecast ensemble (e.g. in the case of 'prior_distribution' = GMM ).

        """
        # generate the forecast states
        analysis_timespan = np.asarray(self.smoother_configs['analysis_timespan'])
        wb = self.smoother_configs['window_bounds']
        initial_time = self.smoother_configs['analysis_time']
        #

        analysis_ensemble = list(self.smoother_configs['analysis_ensemble'])
        initial_state = self.smoother_configs['analysis_state']

        #
        if (wb[-1]-analysis_timespan[-1]) > self._time_eps:
            np.append(analysis_timespan, wb[-1])

        if abs(initial_time - analysis_timespan[0]) > self._time_eps:
            print("The initial time has to be at the initial time of the assimilation window here!")
            raise ValueError
        #
        elif (analysis_timespan[0] - initial_time)> self._time_eps:
            # propagate the forecast state to the beginning of the assimilation window
            local_ckeckpoints = [initial_time, analysis_timespan[0]]
            tmp_trajectory = self.model.integrate_state(initial_state, local_ckeckpoints)
            if isinstance(tmp_trajectory, list):
                initial_state = tmp_trajectory[-1].copy()
            else:
                initial_state = tmp_trajectory.copy()

        else:
            # We are good to go; the initial time matches the beginning of the assimilation timespan
            pass

        #
        forecast_ensemble = utility.propagate_ensemble(ensemble=analysis_ensemble,
                                                       model=self.smoother_configs['model'],
                                                       checkpoints=analysis_timespan,
                                                       in_place=False)
        forecast_state = utility.ensemble_mean(forecast_ensemble)

        self.smoother_configs.update({'forecast_ensemble': forecast_ensemble,
                                      'forecast_state': forecast_state,
                                      'forecast_time': analysis_timespan[-1]
                                      })

        #
        analysis_trajectory = self.model.integrate_state(initial_state, analysis_timespan)
        self.smoother_configs.update({'analysis_trajectory': analysis_trajectory,
                                      'analysis_timespan': analysis_timespan
                                      })
        #
        if generate_prior_info:
            self.generate_prior_info()
        #

    #
    def generate_prior_info(self):
        """
        Generate the statistics of the approximation of the prior distribution.
            - if the prior is Gaussian, the covariance (ensemble/hybrid) matrix is generated and
              optionally localized.
            - if the prior is GMM, the parameters of the GMM (weights, means, covariances) are generated
              based on the gmm_prior_settings.

        """
        # Read the forecast ensemble...
        forecast_ensemble = self.smoother_configs['forecast_ensemble']
        #
        # Inflate the ensemble if required.
        # This should not be required in the context of HMC, but we added it for testing...
        inflation_fac=self.smoother_configs['forecast_inflation_factor']
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
            self.smoother_configs['forecast_ensemble'] = inflated_ensemble
        else:
            pass

        #
        prior_variances_threshold=self.smoother_configs['prior_variances_threshold']
        #
        # Now update/calculate the prior covariances and formulate a presolver from the prior covariances.
        # Prior covariances here are found as a linear combination of both static and flow-dependent versions of the B.
        if self.prior_distribution in ['gaussian', 'normal']:
            try:
                balance_factor = self.smoother_configs['hybrid_background_coeff']
            except (NameError, ValueError, AttributeError, KeyError):
                # default value is 0 in case it is missing from the smoother configurations
                balance_factor = self._def_local_smoother_configs['hybrid_background_coeff']
            #
            fac = balance_factor  # this multiplies the modeled Background error covariance matrix
            if 0.0 < fac < 1.0:
                if self._verbose:
                    print("Hyberidizing the background error covariance matrix")
                model_covariances = self.model.background_error_model.B.copy().scale(fac)
                ensemble_covariances = self.model.ensemble_covariance_matrix(forecast_ensemble, localize=self.smoother_configs['localize_covariances'])
                model_covariances = model_covariances.add(ensemble_covariances.scale(1-fac))
            elif fac == 0.0:
                model_covariances = self.model.ensemble_covariance_matrix(forecast_ensemble, localize=self.smoother_configs['localize_covariances'])
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
            if self.smoother_configs['localize_covariances']:
                if self._verbose:
                    print('Localizing the background error covariance matrix...')
                loc_func = self.smoother_configs['localization_function']
                loc_radius = self.smoother_configs['localization_radius']
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
            gmm_prior_settings = self.smoother_configs['gmm_prior_settings']
            #
            ensemble_size = self.sample_size
            state_size = self.model.state_size()
            forecast_ensemble = self.smoother_configs['forecast_ensemble']
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
                # Fall to Gaussian prior, recalculate the prior distribution info, then apply traditional HMC smoother
                self.switch_back_to_GMM = True # set the flag first
                self.prior_distribution = 'gaussian'  # now update the prior distribution in the configurations dict.
                self.smoother_configs['prior_distribution'] = 'gaussian'
                #
                try:  # cleanup the GMM statistics attached to the output config if they are still there
                    output_configs['smoother_statistics']['gmm_prior_statistics'] = {}
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
                    if self.smoother_configs['gmm_prior_settings']['use_oringinal_hmc_for_one_comp']:
                        print('Switching to Gaussian prior[Single chain]. One component detected!')
                        # Fall to Gaussian prior, recalculate the prior distribution info, then apply traditional HMC smoother
                        self.switch_back_to_GMM = True # set the flag first
                        self.prior_distribution = 'gaussian'  # now update the prior distribution in the configurations dict.
                        self.smoother_configs['prior_distribution'] = 'gaussian'
                        self.generate_prior_info()  # proceed with prior info generation
                        return
                    else:
                        pass
                #
                else:
                    print("How is the number of mixture components negative???")
                    raise ValueError()
                #
                # GMM successfully generated. attach proper information to the smoother object
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
                prior_variances_threshold = self.smoother_configs['prior_variances_threshold']
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

            # Add GMM statistics to smoother statistics
            gmm_statistics = dict(gmm_num_components=self.prior_distribution_statistics['gmm_num_components'],
                                  gmm_optimal_covar_type=self.prior_distribution_statistics['gmm_optimal_covar_type'],
                                  gmm_weights=self.prior_distribution_statistics['gmm_weights'],
                                  gmm_lables=self.prior_distribution_statistics['gmm_lables'],
                                  gmm_covariances_det_log=self.prior_distribution_statistics['gmm_covariances_det_log'],
                                  gmm_inf_criteria=self.prior_distribution_statistics['gmm_inf_criteria']
                                  )
            self.output_configs['smoother_statistics']['gmm_prior_statistics'] = gmm_statistics
        #
        else:
            print("Prior distribution [%s] is not yet supported!" % self.smoother_configs['prior_distribution'])
            raise ValueError()


    #
    def analysis(self):
        """
        Analysis step.

        Returns:
            chain_diagnostics:

        """
        #
        # Obtain information about the prior distribution from the forecast ensemble
        # Covariance localization and hybridization are carried out if requested
        if self.smoother_configs['forecast_ensemble'] is None:
            print("To carry out the analysis step, a forecast ensemble must be provided; None found in the configurationd dictionary!")
            raise ValueError
        #
        self.generate_prior_info()
        #

        # If the momentum parameters are not yet initialized, then do so.
        # This can be the case if the momentum is not initialized in the smoother constructor.
        if self._momentum is None:
            self._momentum = self.initialize_momentum()
        elif isinstance(self._momentum, Momentum):
            self.update_momentum()
        else:
            print("The current momentum is not an instance of the right object!")
            raise ValueError()
            #

        #
        # get the forecast state as the mean of the forecast ensemble. Will not be used for GMM!
        initialize_chain_strategy = self.smoother_configs['gmm_prior_settings']['initialize_chain_strategy']
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
                initial_state = utility.ensemble_mean(self.smoother_configs['forecast_ensemble'])

        elif initialize_chain_strategy == 'highest_weight':
            gmm_prior_weights = self.prior_distribution_statistics['gmm_weights']
            winner_index = np.argmax(gmm_prior_weights)  # for multiple occurrences, the first winner is taken
            initial_state = self.prior_distribution_statistics['gmm_means'][winner_index].copy()
        else:
            raise ValueError("Chain initialization policy is not recognized. \n"
                             "You have to pass on of the values:\n %s" % repr(initialize_chain_strategies))

        # print('forecast_state', forecast_state)
        # print('obs', observation)
        # print('ref_s', self.smoother_configs['reference_state'])
        #
        # TODO: After finishing Traditional HMC implementation, update this branching to call:
        # TODO:     1- Traditional HMC: _hmc_produce_ensemble
        # TODO:     2- No-U-Turn: _nuts_hmc_produce_ensemble
        # TODO:     3- Riemannian HMC: _rm_hmc_produce_ensemble
        # TODO:     4- My new HMC with automatic adjustment of parameters

        #
        analysis_ensemble, chain_diagnostics = self._hmc_produce_ensemble(initial_state=initial_state, verbose=self._verbose)

        # inflation of the analysis ensemble can be considered with MultiChain MCMC sampler if small steps are taken
        inflation_fac = self.smoother_configs['analysis_inflation_factor']
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
            self.smoother_configs['analysis_ensemble'] = inflated_ensemble
        else:
            pass

        # Update the analysis ensemble in the smoother_configs
        self.smoother_configs['analysis_ensemble'] = analysis_ensemble  # list reconstruction processes copying

        # update analysis_state
        self.smoother_configs['analysis_state'] = utility.ensemble_mean(self.smoother_configs['analysis_ensemble'])
        #

        # Check if the gaussian prior needs to be updated
        if self.switch_back_to_GMM:
            self.switched_to_Gaussian_prior = True  # for proper results saving
            print('Switching back to GMM prior...')
            self.switch_back_to_GMM = False
            self.prior_distribution = 'gmm'
            self.smoother_configs['prior_distribution'] = 'gmm'
        else:
            self.switched_to_Gaussian_prior = False

        # Add chain diagnostics to the output_configs dictionary for proper outputting:
        self.output_configs['smoother_statistics'].update({'chain_diagnostics':chain_diagnostics})
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
            initialize_chain_strategy = self.smoother_configs['gmm_prior_settings']['initialize_chain_strategy']
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
                            initial_state = utility.ensemble_mean(self.smoother_configs['forecast_ensemble'])
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
            # analysis_ensemble = self.smoother_configs['analysis_ensemble']
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
                #
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

                    if True:

                        proposed_state, proposed_momentum = self._hmc_propose_state(current_state=current_state,
                                                                                    current_momentum=current_momentum)

                        if verbose:
                            print(">>>>>>>==================>>>>>>>. RMSE current/proposed", utility.calculate_rmse(current_state, proposed_state))

                        # add current_state to the ensemble as it is the most updated membe
                        accept_proposal, energy_loss, a_n, u_n = self._hmc_MH_accept_reject(current_state=current_state,
                                                                                            proposed_state=proposed_state,
                                                                                            current_momentum=current_momentum,
                                                                                            proposed_momentum=proposed_momentum,
                                                                                            verbose=verbose
                                                                                            )

                    else:
                        try:
                            proposed_state, proposed_momentum = self._hmc_propose_state(current_state=current_state,
                                                                                        current_momentum=current_momentum)

                            if verbose:
                                print(">>>>>>>==================>>>>>>>. RMSE current/proposed", utility.calculate_rmse(current_state, proposed_state))

                            # add current_state to the ensemble as it is the most updated membe
                            accept_proposal, energy_loss, a_n, u_n = self._hmc_MH_accept_reject(current_state=current_state,
                                                                                                proposed_state=proposed_state,
                                                                                                current_momentum=current_momentum,
                                                                                                proposed_momentum=proposed_momentum,
                                                                                                verbose=verbose
                                                                                                )
                        except(ValueError):
                            accept_proposal, energy_loss, a_n, u_n = False, 0, 0, 1

                    print("HMC SMOOTHER >> BURN-IN STEP: Step [%3d]; accept_proposal %5s, energy_loss=%+12.8e, a_n=%3.2f " % (burn_ind, accept_proposal, energy_loss, a_n))

                    # save probabilities for chain diagnostics evaluation
                    acceptance_probabilities.append(a_n)
                    uniform_random_numbers.append(u_n)
                    #
                    if accept_proposal:
                        acceptance_flags.append(1)
                        if verbose:
                            print("Burning step [%d]: J=%f" % (burn_ind, self._hmc_total_energy(proposed_state, proposed_momentum)))
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

                    print("HMC SMOOTHER: MIXING STEP >> Ensemble member [%3d] Mixing step [%3d] : accept_proposal %5s; energy_loss=%+12.8e, a_n=%4.3f" % (ens_ind, mixing_ind, accept_proposal, energy_loss, a_n))


                    if accept_proposal:
                        acceptance_flags.append(1)
                        current_state = proposed_state.copy()
                        #
                        if verbose:
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
            print("observation", self.smoother_configs['observation'])
            print("reference_state:", self.smoother_configs['reference_state'][100:200])
            print("forecast_state:", self.smoother_configs['forecast_state'][100:200])
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
        Evaluate the potential energy value only. Potential energy here, is the 4D-Var cost functional.

        Args:
            current_state: current (position) state of the chain

        Returns:
            potential_energy_value: value of the potential energy function at the given position,
                this is the negative-log of the target posterior

        """
        #
        if isinstance(current_state, np.ndarray):
            in_state = self.model.state_vector()
            in_state[:] = current_state.copy()
        else:
            in_state = current_state.copy()

        # This will keep track of propagated state over the assimilation window
        local_state = in_state.copy()

        #
        model = self.model
        window_bounds = self.smoother_configs['window_bounds']

        #
        # Retrieve the observations list, and time settings
        observations_list = self.smoother_configs['observations_list']
        obs_checkpoints = np.asarray(self.smoother_configs['obs_checkpoints'])
        analysis_time = self.smoother_configs['analysis_time']

        #
        # Retrieve the forecast parameters based on the prior distribution then evaluate the potential energy
        prior_distribution = self.prior_distribution
        forecast_time = self.smoother_configs['forecast_time']

        # Check time settings:
        if (analysis_time - obs_checkpoints[0] ) >= self._model_step_size:
            print("Observations have to follow the assimilation times in this implementation!")
            raise ValueError
        else:
            pass

        if abs(forecast_time-analysis_time)>self._time_eps or abs(window_bounds[0]-forecast_time)>self._time_eps:
            print("This implementation requires the forecast and analysis time to coincide at the beginning of the assimilation window!")
            print("Forecast time: ", forecast_time)
            print("Forecast time: ", analysis_time)
            print("window_bounds: ", str(window_bounds))
            raise ValueError
        else:
            pass

        #
        if prior_distribution == 'gaussian':
            #
            # Get the forecast state if the prior is Gaussian and forecast_state is None.
            try:
                forecast_state = self.smoother_configs['forecast_state'].copy()
            except (ValueError, AttributeError, TypeError):
                forecast_ensemble = self.smoother_configs['forecast_ensemble']
                forecast_state = utility.ensemble_mean(forecast_ensemble)
            finally:
                if forecast_state is None:
                    forecast_ensemble = self.smoother_configs['forecast_ensemble']
                    forecast_state = utility.ensemble_mean(forecast_ensemble)
                    self.smoother_configs['forecast_state'] = forecast_state.copy()
                    #

            #
            if self._verbose:
                print("In objective_function_value:")
                print("in-state", current_state)
                print("forecast_state", forecast_state)
                print("obs_checkpoints", obs_checkpoints)
                # print("observations_list", observations_list)
                raw_input("\npress Enter to continue")
                #

            #
            # 1- Observation/Innovation term
            #
            if (forecast_time - obs_checkpoints[0]) >= self._model_step_size:
                print("forecast time can't be after the first observation time instance!")
                print("forecast_time", forecast_time)
                print("obs_checkpoints[0]", obs_checkpoints[0])
                raise AssertionError
                #
            elif abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
                # an observation exists at the analysis time
                Hx = model.evaluate_theoretical_observation(local_state)
                obs_innov = observations_list[0].copy().scale(-1.0)
                obs_innov = obs_innov.add(Hx)
                # obs_innov = Hx.axpy(-1.0, observations_list[0], in_place=False)
                scaled_obs_innov = obs_innov.copy()
                scaled_obs_innov = model.observation_error_model.invR.vector_product(scaled_obs_innov)
                # scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
                observation_term = scaled_obs_innov.dot(obs_innov)
                #
            else:
                obs_checkpoints = np.insert(obs_checkpoints, 0, forecast_time)
                observation_term = 0.0

            #
            num_obs_points = len(observations_list)
            # Forward to observation time instances and update observation term
            for iter_ind, t0, t1 in zip(xrange(num_obs_points), obs_checkpoints[: -1], obs_checkpoints[1: ]):
                local_ckeckpoints = np.array([t0, t1])
                tmp_trajectory = model.integrate_state(initial_state=local_state, checkpoints=local_ckeckpoints)
                if isinstance(tmp_trajectory, list):
                    local_state = tmp_trajectory[-1].copy()
                else:
                    local_state = tmp_trajectory.copy()
                #
                Hx = model.evaluate_theoretical_observation(local_state)

                # obs_innov = Hx.axpy(-1.0, observations_list[iter_ind], in_place=False)
                obs_innov = observations_list[iter_ind].copy().scale(-1.0)
                obs_innov = obs_innov.add(Hx, in_place=False)
                scaled_obs_innov = obs_innov.copy()
                scaled_obs_innov = model.observation_error_model.invR.vector_product(scaled_obs_innov)

                observation_term += scaled_obs_innov.dot(obs_innov)
                #

                #
                if self._verbose:
                    print("subinterval:" + repr(local_ckeckpoints))
                    print("local_state (at end of subinterval above): ", local_state)
                    print("H(x) (at end of subinterval above): ", Hx)
                    print("observation:", observations_list[iter_ind])
                    print("obs_innov", obs_innov)
                    print("scaled_obs_innov", scaled_obs_innov)
                    print("observation_term", observation_term)
                    raw_input("\npress Enter to continue")
                    #


            #
            # 2- Background term
            lu, piv = self.prior_distribution_statistics['B_lu'], self.prior_distribution_statistics['B_piv']
            # print(self.prior_distribution_statistics['B'].get_numpy_array())
            #
            state_dev = forecast_state.copy().scale(-1.0)  # <- state_dev = - forecast_state
            state_dev = state_dev.add(in_state, in_place=False)  # state_dev = x -
            scaled_state_dev = model.state_vector()
            scaled_state_dev[:] = lu_solve((lu, piv), state_dev.get_numpy_array().copy())
            background_term = scaled_state_dev.dot(state_dev)

            # scaled_state_dev = model.background_error_model.invB.vector_product(scaled_state_dev)
            # background_term = scaled_state_dev.dot(state_dev)

            # deviations = local_state.axpy(-1.0, forecast_state, in_place=False)
            # print("deviations", deviations)
            # scaled_deviations_numpy = lu_solve((lu, piv), deviations.get_numpy_array())
            # scaled_deviations = self.model.state_vector()
            # scaled_deviations[:] = scaled_deviations_numpy.copy()
            # print("deviations", deviations)
            # print("scaled_deviations", scaled_deviations)
            # #
            # background_term = scaled_deviations.dot(deviations)

            #
            potential_energy_value = 0.5 * (background_term + observation_term)
            if self._verbose:
                print("forecast_state", forecast_state)
                print("state_dev", state_dev)
                print("scaled_deviations", scaled_state_dev)
                print("background_term", background_term)
                #
                print("Smoothing Chain is running; Background_term", background_term, "Observation_term", observation_term, "potential_energy_value", potential_energy_value)
                #

        elif prior_distribution.lower() in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            # The GMM model should be attached to the smoother object right now.
            #
            #
            if self._verbose:
                print("In objective_function_value:")
                print("in-state", current_state)
                print("obs_checkpoints", obs_checkpoints)
                print("observations_list", observations_list)
                raw_input("\npress Enter to continue")
                #

            #
            # 1- Observation/Innovation term
            #
            if (forecast_time - obs_checkpoints[0]) >= self._model_step_size:
                print("forecast time can't be after the first observation time instance!")
                print("forecast_time", forecast_time)
                print("obs_checkpoints[0]", obs_checkpoints[0])
                raise AssertionError
                #
            elif abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
                # an observation exists at the analysis time
                Hx = model.evaluate_theoretical_observation(local_state)
                obs_innov = observations_list[0].copy().scale(-1.0)
                obs_innov = obs_innov.add(Hx)
                # obs_innov = Hx.axpy(-1.0, observations_list[0], in_place=False)
                scaled_obs_innov = obs_innov.copy()
                scaled_obs_innov = model.observation_error_model.invR.vector_product(scaled_obs_innov)
                # scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
                observation_term = 0.5 * (scaled_obs_innov.dot(obs_innov))
                #
            else:
                obs_checkpoints = np.insert(obs_checkpoints, 0, forecast_time)
                observation_term = 0.0

            #
            num_obs_points = len(observations_list)
            # Forward to observation time instances and update observation term
            for iter_ind, t0, t1 in zip(xrange(num_obs_points), obs_checkpoints[: -1], obs_checkpoints[1: ]):
                local_ckeckpoints = np.array([t0, t1])
                tmp_trajectory = model.integrate_state(initial_state=local_state, checkpoints=local_ckeckpoints)
                if isinstance(tmp_trajectory, list):
                    local_state = tmp_trajectory[-1].copy()
                else:
                    local_state = tmp_trajectory.copy()
                #
                Hx = model.evaluate_theoretical_observation(local_state)

                # obs_innov = Hx.axpy(-1.0, observations_list[iter_ind], in_place=False)
                obs_innov = observations_list[iter_ind].copy().scale(-1.0)
                obs_innov = obs_innov.add(Hx, in_place=False)
                scaled_obs_innov = obs_innov.copy()
                scaled_obs_innov = model.observation_error_model.invR.vector_product(scaled_obs_innov)

                observation_term += 0.5 * (scaled_obs_innov.dot(obs_innov))
                #

                #
                if self._verbose:
                    print("subinterval:" + repr(local_ckeckpoints))
                    print("local_state (at end of subinterval above): ", local_state)
                    print("H(x) (at end of subinterval above): ", Hx)
                    print("observation:", observations_list[iter_ind])
                    print("obs_innov", obs_innov)
                    print("scaled_obs_innov", scaled_obs_innov)
                    print("observation_term", observation_term)
                    raw_input("\npress Enter to continue")
                    #


            #
            # 2- Background term
            background_term = self._hmc_gmm_evaluate_potential_log_terms_val(local_state)
            #
            # Add up the forecast and observation term
            potential_energy_value = observation_term + background_term
            #
            if self._verbose:
                print("Smoothing Chain is running; Background_term", background_term, "Observation_term", observation_term, "potential_energy_value", potential_energy_value)
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
    def _hmc_potential_energy_gradient(self, current_state, fd_validation=False, FD_eps=1e-7, FD_central=True, debug=True):
        """
        Evaluate the potential energy gradient only.  Potential energy here, is the 4D-Var cost functional.

        Args:
            current_state: current (position) state of the chain
            fd_validation (default False): use finiti difference approximation of the gradient to validate the
                calculated gradient.
            debug (default False): used to check the entries of the calculated gradient for invalid (inf) values

        Returns:
            potential_energy_gradient: the gradient of the potential energy function at the given position,
                this is the derivative of the negative-log of the target posterior.

        """
        #
        if isinstance(current_state, np.ndarray):
            local_state = self.model.state_vector()
            local_state[:] = current_state.copy()
        else:
            local_state = current_state.copy()

        #
        model = self.model
        window_bounds = self.smoother_configs['window_bounds']

        #
        # Retrieve the observations list, and time settings
        observations_list = self.smoother_configs['observations_list']
        analysis_time = self.smoother_configs['analysis_time']
        if self.smoother_configs['obs_checkpoints'] is None:
            print("Couldn't find observation checkpoints in self.smoother_configs['obs_checkpoints']; None found!")
            raise ValueError
        else:
            obs_checkpoints = np.asarray(self.smoother_configs['obs_checkpoints'])

        #
        # Retrieve the forecast parameters based on the prior distribution then evaluate the potential energy
        prior_distribution = self.prior_distribution
        forecast_time = self.smoother_configs['forecast_time']

        # Check time settings:
        if (analysis_time - obs_checkpoints[0] ) >= self._model_step_size:
            print("Observations have to follow the assimilation times in this implementation!")
            raise ValueError
        else:
            pass

        if abs(forecast_time-analysis_time)>self._time_eps or (window_bounds[0]-forecast_time)>self._time_eps:
            print("This implementation requires the forecast and analysis time to coincide at the beginning of the assimilation window!")
            print("Forecast time: ", forecast_time)
            print("Forecast time: ", analysis_time)
            print("window_bounds: ", str(window_bounds))
            raise ValueError
        else:
            pass


        #
        if prior_distribution == 'gaussian':
            # Get the forecast state if the prior is Gaussian and forecast_state is None.
            try:
                forecast_state = self.smoother_configs['forecast_state']
            except (ValueError, AttributeError, TypeError):
                forecast_ensemble = self.smoother_configs['forecast_ensemble']
                forecast_state = utility.ensemble_mean(forecast_ensemble)
            else:
                if forecast_state is None:
                    forecast_ensemble = self.smoother_configs['forecast_ensemble']
                    forecast_state = utility.ensemble_mean(forecast_ensemble)
                    self.smoother_configs['forecast_ensemble'] = forecast_state

            #
            # 1- Evaluate the observation terms:
            analysis_time = self.smoother_configs['analysis_time']

            if (analysis_time - obs_checkpoints[0] ) >= self._model_step_size:
                print("Observations have to follow the assimilation times in this implementation!")
                print("analysis_time", analysis_time)
                print("obs_checkpoints[0]", obs_checkpoints[0])
                raise ValueError
            else:
                pass

            #
            if self._verbose:
                print("In objective_function_gradient:")
                print("in-state", current_state)
                print("forecast_state", forecast_state)
                print("obs_checkpoints", obs_checkpoints)
                print("observations_list", observations_list)
                raw_input("\npress Enter to continue")

            #
            # 1- forward checkpointing:
            checkpointed_state = local_state.copy()
            checkpointed_states = []  # this holds states only at observation time instances
            checkpointed_times = []

            if abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
                checkpointed_states.append(local_state)
                checkpointed_times.append(forecast_time)
                #
            elif (obs_checkpoints[0] - forecast_time) >= self._model_step_size:
                local_ckeckpoints = [forecast_time, obs_checkpoints[0]]

                tmp_trajectory = model.integrate_state(initial_state=checkpointed_state.copy(), checkpoints=local_ckeckpoints)
                #
                if isinstance(tmp_trajectory, list):
                    checkpointed_state = tmp_trajectory[-1].copy()
                else:
                    checkpointed_state = tmp_trajectory.copy()
                checkpointed_states.append(checkpointed_state)
                checkpointed_times.append(obs_checkpoints[0])
            else:
                print("forecast time can't be after the first observation time!")
                print("obs_checkpoints[0]", obs_checkpoints[0])
                print("forecast_time", forecast_time)
                raise ValueError

            #
            for t0, t1 in zip(obs_checkpoints[:-1], obs_checkpoints[1:]):
                local_checkpoints = np.array([t0, t1])
                local_trajectory = model.integrate_state(initial_state=checkpointed_state, checkpoints=local_checkpoints)
                if isinstance(local_trajectory, list):
                    checkpointed_state = local_trajectory[-1].copy()
                else:
                    checkpointed_state = local_trajectory.copy()
                checkpointed_states.append(checkpointed_state.copy())
                checkpointed_times.append(t1)
                #

            if self._verbose:
                print("checkpointed_times", checkpointed_times)
                print("checkpointed_states", checkpointed_states)

            #
            # 2- backward propagation, and sensitivity update:
            last_obs_ind = len(observations_list) - 1
            #
            if len(checkpointed_states) != len(observations_list):
                print("Checkpointed states don't match observation time indexes!")
                raise ValueError


            # Initialize the sensitivity matrix:
            Hx = model.evaluate_theoretical_observation(checkpointed_states[-1])
            obs = observations_list[-1]
            obs_innov = Hx.axpy(-1.0, obs, in_place=False)  # innov = H(x) - y
            scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
            lambda_ = model.observation_operator_Jacobian_T_prod_vec(checkpointed_states[-1], scaled_obs_innov)

            #
            if self._verbose:
                print("\nIn objective_function_gradient:")
                print("checkpointed_states[-1]", checkpointed_states[-1])
                print("final obs", obs)
                print("obs_innov", obs_innov)
                print("scaled_obs_innov", scaled_obs_innov)
                print("Initial lambda_", lambda_)
                raw_input("\npress Enter to continue")


            # backward propagation, and update sensitivity matrix (lambda)
            adjoint_integrator = model._adjoint_integrator
            for t0_ind, t1_ind in zip(xrange(last_obs_ind-1, -1, -1), xrange(last_obs_ind, 0, -1)):
                t0, t1 = obs_checkpoints[t0_ind], obs_checkpoints[t1_ind]
                #
                lambda_k = adjoint_integrator.integrate_adj(y=checkpointed_states[t0_ind],
                                                            lambda_=lambda_,
                                                            tin=t0,
                                                            tout=t1
                                                            )
                #
                if isinstance(lambda_k, np.ndarray):
                    try:
                        lambda_k = model.state_vector(lambda_k)
                    except():
                        tmp_lamda = model.state_vector()
                        tmp_lamda[:] = lambda_k.copy()
                        lambda_k = tmp_lamda
                elif isinstance(lambda_k, (StateVector, StateMatrix)):
                    pass
                else:
                    print("Returned Sensitivity matrix is of unrecognized Type: %s" % repr(type(lambda_k)) )
                    raise TypeError

                Hx = model.evaluate_theoretical_observation(checkpointed_states[t0_ind])
                obs = observations_list[t0_ind]
                obs_innov = Hx.axpy(-1.0, obs, in_place=False)  # innov = H(x) - y
                scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
                obs_correction_term = model.observation_operator_Jacobian_T_prod_vec(checkpointed_states[t0_ind], scaled_obs_innov)
                #
                lambda_ = lambda_k.add(obs_correction_term, in_place=False)
                #
                #
                if self._verbose:
                    print("\nIn objective_function_gradient: Iteration %s" % repr([t0_ind, t1_ind]))
                    print("t0 = %f;\t t1=%f" % (t0, t1))
                    print("local_state", local_state)
                    print("checkpointed_states[%d]:" %t0_ind, checkpointed_states[t0_ind])
                    print("observations_list[%d]:" %t0_ind, observations_list[t0_ind])
                    print("obs_innov", obs_innov)
                    print("obs_correction_term", obs_correction_term)
                    print("lambda_k", lambda_k)
                    print("lambda_", lambda_)
                    raw_input("\npress Enter to continue")

            #
            if (obs_checkpoints[0] - forecast_time) >= self._model_step_size:
                lambda_ = adjoint_integrator.integrate_adj(y=local_state,
                                                           lambda_=lambda_,
                                                           tin=forecast_time,
                                                           tout=obs_checkpoints[0]
                                                           )
                if self._verbose:
                    print("(obs_checkpoints[0] - forecast_time) >= self._model_step_size")
                    print("Propagating back from time %f to time %f" % (obs_checkpoints[0], forecast_time))
                if isinstance(lambda_, np.ndarray):
                    try:
                        observation_term = model.state_vector()
                        observation_term[:] = lambda_[:]
                    except:
                        observation_term = model.state_vector()
                        observation_term[:] = lambda_.copy()
                elif isinstance(lambda_, (StateVector, StateMatrix)):
                    observation_term = lambda_
                else:
                    print("Returned Sensitivity matrix is of unrecognized Type: %s" % repr(type(lambda_)) )
                    raise TypeError
                #
            elif abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
                if self._verbose:
                    print("abs(forecast_time - obs_checkpoints[0]) <= self._time_eps")
                observation_term = lambda_
            #
            else:
                print("Forecast time point can't be after first observation time!")
                raise ValueError
            #
            if self._verbose:
                print("|||||||| FINAL LAMBDA (observation term) >>>>> ", observation_term)
                raw_input("\npress Enter to continue")


            #
            # 2- Background term
            lu, piv = self.prior_distribution_statistics['B_lu'], self.prior_distribution_statistics['B_piv']
            deviations = local_state.axpy(-1.0, forecast_state, in_place=False)
            scaled_deviations_numpy = lu_solve((lu, piv), deviations.get_numpy_array())
            background_term = self.model.state_vector()
            background_term[:] = scaled_deviations_numpy[:]
            #
            # potential_energy_gradient = background_term.add(self.model.observation_operator_Jacobian_T_prod_vec(local_state, observation_term))  # why!?
            potential_energy_gradient = background_term.add(observation_term)
        #
        elif prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            #
            #
            # 1- Evaluate the observation terms:
            analysis_time = self.smoother_configs['analysis_time']

            if (analysis_time - obs_checkpoints[0] ) >= self._model_step_size:
                print("Observations have to follow the assimilation times in this implementation!")
                print("analysis_time", analysis_time)
                print("obs_checkpoints[0]", obs_checkpoints[0])
                raise ValueError
            else:
                pass

            #
            if self._verbose:
                print("In objective_function_gradient:")
                print("in-state", state)
                print("forecast_state", forecast_state)
                print("obs_checkpoints", obs_checkpoints)
                print("observations_list", observations_list)
                raw_input("\npress Enter to continue")

            #
            # a) forward checkpointing:
            checkpointed_state = local_state.copy()
            checkpointed_states = []  # this holds states only at observation time instances
            checkpointed_times = []

            if abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
                checkpointed_states.append(local_state)
                checkpointed_times.append(forecast_time)
                #
            elif (obs_checkpoints[0] - forecast_time) >= self._model_step_size:
                local_ckeckpoints = [forecast_time, obs_checkpoints[0]]

                tmp_trajectory = model.integrate_state(initial_state=checkpointed_state.copy(), checkpoints=local_ckeckpoints)
                #
                if isinstance(tmp_trajectory, list):
                    checkpointed_state = tmp_trajectory[-1].copy()
                else:
                    checkpointed_state = tmp_trajectory.copy()
                checkpointed_states.append(checkpointed_state)
                checkpointed_times.append(obs_checkpoints[0])
            else:
                print("forecast time can't be after the first observation time!")
                print("obs_checkpoints[0]", obs_checkpoints[0])
                print("forecast_time", forecast_time)
                raise ValueError

            #
            for t0, t1 in zip(obs_checkpoints[:-1], obs_checkpoints[1:]):
                local_checkpoints = np.array([t0, t1])
                local_trajectory = model.integrate_state(initial_state=checkpointed_state, checkpoints=local_checkpoints)
                if isinstance(local_trajectory, list):
                    checkpointed_state = local_trajectory[-1].copy()
                else:
                    checkpointed_state = local_trajectory.copy()
                checkpointed_states.append(checkpointed_state.copy())
                checkpointed_times.append(t1)
                #

            if self._verbose:
                print("checkpointed_times", checkpointed_times)
                print("checkpointed_states", checkpointed_states)

            #
            # b) backward propagation, and sensitivity update:
            last_obs_ind = len(observations_list) - 1
            #
            if len(checkpointed_states) != len(observations_list):
                print("Checkpointed states don't match observation time indexes!")
                raise ValueError


            # Initialize the sensitivity matrix:
            Hx = model.evaluate_theoretical_observation(checkpointed_states[-1])
            obs = observations_list[-1]
            obs_innov = Hx.axpy(-1.0, obs, in_place=False)  # innov = H(x) - y
            scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
            lambda_ = model.observation_operator_Jacobian_T_prod_vec(checkpointed_states[-1], scaled_obs_innov)

            #
            if self._verbose:
                print("\nIn objective_function_gradient:")
                print("checkpointed_states[-1]", checkpointed_states[-1])
                print("final obs", obs)
                print("obs_innov", obs_innov)
                print("scaled_obs_innov", scaled_obs_innov)
                print("Initial lambda_", lambda_)
                raw_input("\npress Enter to continue")


            # backward propagation, and update sensitivity matrix (lambda)
            adjoint_integrator = model._adjoint_integrator
            for t0_ind, t1_ind in zip(xrange(last_obs_ind-1, -1, -1), xrange(last_obs_ind, 0, -1)):
                t0, t1 = obs_checkpoints[t0_ind], obs_checkpoints[t1_ind]
                #
                lambda_k = adjoint_integrator.integrate_adj(y=checkpointed_states[t0_ind],
                                                            lambda_=lambda_,
                                                            tin=t0,
                                                            tout=t1
                                                            )
                #
                if isinstance(lambda_k, np.ndarray):
                    try:
                        lambda_k = model.state_vector(lambda_k)
                    except():
                        tmp_lamda = model.state_vector()
                        tmp_lamda[:] = lambda_k.copy()
                        lambda_k = tmp_lamda
                elif isinstance(lambda_k, (StateVector, StateMatrix)):
                    pass
                else:
                    print("Returned Sensitivity matrix is of unrecognized Type: %s" % repr(type(lambda_k)) )
                    raise TypeError

                Hx = model.evaluate_theoretical_observation(checkpointed_states[t0_ind])
                obs = observations_list[t0_ind]
                obs_innov = Hx.axpy(-1.0, obs, in_place=False)  # innov = H(x) - y
                scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
                obs_correction_term = model.observation_operator_Jacobian_T_prod_vec(checkpointed_states[t0_ind], scaled_obs_innov)
                #
                lambda_ = lambda_k.add(obs_correction_term, in_place=False)
                #
                #
                if self._verbose:
                    print("\nIn objective_function_gradient: Iteration %s" % repr([t0_ind, t1_ind]))
                    print("t0 = %f;\t t1=%f" % (t0, t1))
                    print("local_state", local_state)
                    print("checkpointed_states[%d]:" %t0_ind, checkpointed_states[t0_ind])
                    print("observations_list[%d]:" %t0_ind, observations_list[t0_ind])
                    print("obs_innov", obs_innov)
                    print("obs_correction_term", obs_correction_term)
                    print("lambda_k", lambda_k)
                    print("lambda_", lambda_)
                    raw_input("\npress Enter to continue")

            #
            if (obs_checkpoints[0] - forecast_time) >= self._model_step_size:
                lambda_ = adjoint_integrator.integrate_adj(y=local_state,
                                                           lambda_=lambda_,
                                                           tin=forecast_time,
                                                           tout=obs_checkpoints[0]
                                                           )
                if self._verbose:
                    print("(obs_checkpoints[0] - forecast_time) >= self._model_step_size")
                    print("Propagating back from time %f to time %f" % (obs_checkpoints[0], forecast_time))
                if isinstance(lambda_, np.ndarray):
                    try:
                        observation_term = model.state_vector()
                        observation_term[:] = lambda_[:]
                    except:
                        observation_term = model.state_vector()
                        observation_term[:] = lambda_.copy()
                elif isinstance(lambda_, (StateVector, StateMatrix)):
                    observation_term = lambda_
                else:
                    print("Returned Sensitivity matrix is of unrecognized Type: %s" % repr(type(lambda_)) )
                    raise TypeError
                #
            elif abs(forecast_time - obs_checkpoints[0]) <= self._time_eps:
                if self._verbose:
                    print("abs(forecast_time - obs_checkpoints[0]) <= self._time_eps")
                observation_term = lambda_
            #
            else:
                print("Forecast time point can't be after first observation time!")
                raise ValueError
            #
            if self._verbose:
                print("|||||||| FINAL LAMBDA (observation term) >>>>> ", observation_term)
                raw_input("\npress Enter to continue")



            #
            # 2- Background term
            background_term = self._hmc_gmm_evaluate_potential_log_terms_grad(in_state=local_state)
            #
            # 3- add up the two terms to find the full gradient
            potential_energy_gradient = background_term.add(observation_term)
        #
        else:
            print("Prior distribution [%s] is not supported!" % self.prior_distribution)
            raise ValueError()

        #
        if fd_validation:
            self.__validate_gradient(local_state, potential_energy_gradient, FD_eps=FD_eps, FD_central=FD_central)

            print('local_state', local_state)

        #
        if self._verbose:
            print("^"*100+"\n"+"^"*100)
            print("\nIn _hmc_potential_energy_gradient: ")
            print("forecast_state", forecast_state)
            print("local_state", local_state)
            print("background_term: ", background_term)
            print("observation_term", observation_term)
            print("lambda_", lambda_)

            print("Gradient:", potential_energy_gradient)
            print("v"*100+"\n"+"v"*100)

        if debug:
            if np.isinf(potential_energy_gradient[:]).any():
                print('potential_energy_gradient', potential_energy_gradient)
                print('local_state', local_state)
                print('observation_term', observation_term)
                print('background_term', background_term)
                print('innovations', innovations)
                print('model_observation', model_observation)

        #
        if isinstance(current_state, np.ndarray):
            potential_energy_gradient = potential_energy_gradient.get_numpy_array()

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
    #
    def __validate_gradient(self, state, gradient, FD_eps=1e-5, FD_central=False):
        """
        Use Finite Difference to validate the gradient

        Args:
            state:
            gradient:
            FD_eps:
            FD_central:

        """
        #
        # Use finite differences to validate the Gradient (if needed):
        state_size = self.model.state_size()
        # Finite Difference Validation:
        eps = FD_eps
        grad_array = gradient.get_numpy_array()
        if isinstance(state, np.ndarray):
            state_array = state.copy()
        elif isinstance(state, (StateVector, StateMatrix)):
            state_array = state.get_numpy_array()
        else:
            print("Passed state is of unrecognized Type: %s" % repr(type(state)) )
            raise TypeError

        sep = "\n"+"~"*80+"\n"
        # print some statistics for monitoring:
        print(sep + "FD Validation of the Gradient" + sep)
        print("  + Maximum gradient entry :", grad_array.max())
        print("  + Minimum gradient entry :", grad_array.min())

        #
        perturbed_state = self.model.state_vector()
        state_perturb = np.zeros_like(grad_array)
        fd_grad = np.zeros_like(grad_array)
        #
        if not FD_central:
            f0 = self._hmc_potential_energy_value(state)
        #

        for i in xrange(state_size):
            state_perturb[:] = 0.0
            state_perturb[i] = eps
            #
            if FD_central:
                perturbed_state[:] = state_array - state_perturb
                f0 = self._hmc_potential_energy_value(perturbed_state)
            #
            perturbed_state[:] = state_array + state_perturb
            f1 = self._hmc_potential_energy_value(perturbed_state)

            if FD_central:
                fd_grad[i] = (f1-f0)/(2.0*eps)
            else:
                fd_grad[i] = (f1-f0)/(eps)

            err = (grad_array[i] - fd_grad[i]) / fd_grad[i]
            print(">>>>Gradient/FD>>>> %4d| Grad = %+8.6e\t FD-Grad = %+8.6e\t Rel-Err = %+8.6e <<<<" % (i, grad_array[i], fd_grad[i], err))



    #
    def print_cycle_results(self):
        """
        Print smoothing results from the current cycle to the main terminal
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
            super(HMCSmoother, self).print_cycle_results()
        pass  # Add more...
        #

    #
    def save_cycle_results(self, output_dir=None, cleanup_out_dir=False):
        """
        Save smoothing results from the current cycle to file(s).
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
        # clean-up output directory; this is set to true only if the smoother is called once, otherwise smoothing_process should handle it.
        if cleanup_out_dir:
            parent_path, out_dir = os.path.split(file_output_directory)
            utility.cleanup_directory(directory_name=out_dir, parent_path=parent_path)
        # check the output sub-directories...
        smoother_statistics_dir = os.path.join(file_output_directory, output_configs['smoother_statistics_dir'])
        model_states_dir = os.path.join(file_output_directory, output_configs['model_states_dir'])
        observations_dir = os.path.join(file_output_directory, output_configs['observations_dir'])
        file_output_variables = output_configs['file_output_variables']  # I think it's better to remove it from the smoother base...

        if not os.path.isdir(smoother_statistics_dir):
            os.makedirs(smoother_statistics_dir)
        if not os.path.isdir(model_states_dir):
            os.makedirs(model_states_dir)
        if not os.path.isdir(observations_dir):
            os.makedirs(observations_dir)

        # check if results are to be saved to separate files or appended on existing files.
        # This may be overridden if not adequate for some output (such as model states), we will see!
        file_output_separate_files = output_configs['file_output_separate_files']
        # This is useful for saving smoother statistics but not model states or observations as models should handle both
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
                                          smoother_statistics_dir=smoother_statistics_dir,
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
                forecast_state = self.smoother_configs['forecast_state']
                self.model.write_state(state=forecast_state, directory=cycle_states_out_dir, file_name='forecast_mean')
                # save analysis mean
                analysis_state = self.smoother_configs['analysis_state']
                self.model.write_state(state=analysis_state, directory=cycle_states_out_dir, file_name='analysis_mean')
            else:
                raise ValueError("Unsupported ensemble moment: '%s' !" % (file_output_moment_name))
        else:
            # start outputting the whole ensemble members (both forecast and analysis ensembles of course).
            # check if all ensembles are to be saved or just one of the supported ensemble moments
            for ens_ind in xrange(self.sample_size):
                if file_output_separate_files:
                    # print('saving ensemble member to separate files: %d' % ens_ind)
                    forecast_ensemble_member = self.smoother_configs['forecast_ensemble'][ens_ind]
                    self.model.write_state(state=forecast_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='forecast_ensemble_'+str(ens_ind),
                                            append=False
                                            )
                    #
                    analysis_ensemble_member = self.smoother_configs['analysis_ensemble'][ens_ind]
                    self.model.write_state(state=analysis_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='analysis_ensemble_'+str(ens_ind),
                                            append=False
                                            )
                else:
                    # print('saving ensemble member to same file with resizing: %d' % ens_ind)
                    # save results to different files. For moments
                    forecast_ensemble_member = self.smoother_configs['forecast_ensemble'][ens_ind]
                    self.model.write_state(state=forecast_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='forecast_ensemble',
                                            append=True
                                            )
                    #
                    analysis_ensemble_member = self.smoother_configs['analysis_ensemble'][ens_ind]
                    self.model.write_state(state=analysis_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='analysis_ensemble',
                                            append=True
                                            )
        # save reference state TODO: We may need to save time(s) along with state(s)...
        reference_state = self.smoother_configs['reference_state']
        self.model.write_state(state=reference_state, directory=cycle_states_out_dir, file_name='reference_state')

        #
        # Save observations to file; use model to write observations to file(s)
        for observation in self.smoother_configs['observations_list']:
            file_name = utility.try_file_name(directory=cycle_observations_out_dir, file_prefix='observation')
            self.model.write_observation(observation=observation, directory=cycle_observations_out_dir, file_name=file_name, append=False)

        # Save smoother statistics to file
        # Output the RMSE results; it's meaningless to create a new file for each cycle:
        rmse_file_name = 'rmse.txt'  # RMSE are always saved in text files
        rmse_file_path = os.path.join(smoother_statistics_dir, rmse_file_name)
        # Create a header for the file if it is newely created
        if not os.path.isfile(rmse_file_path):
            # rmse file does not exist. create file and add header.
            header = "RMSE Results: Smoother: '%s' \n %s \t %s \t %s \t %s \n" % (self._smoother_name,
                                                                                  'Forecast-Time'.rjust(20),
                                                                                  'Analysis-Time'.rjust(20),
                                                                                  'Forecast-RMSE'.rjust(20),
                                                                                  'Analysis-RMSE'.rjust(20),
                                                                                  )
            if False:
                # get the initial RMSE and add it if forecast is done first...
                initial_time = self.smoother_configs['forecast_time']
                initial_rmse = self.output_configs['smoother_statistics']['initial_rmse']
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
            # rmse file does exist. Header should be already there!
            pass

        # Now rmse results file exists --> Append rmse results to the file:
        forecast_time = self.smoother_configs['forecast_time']
        analysis_time = self.smoother_configs['analysis_time']
        #
        forecast_rmse = self.output_configs['smoother_statistics']['forecast_rmse'][0]
        analysis_rmse = self.output_configs['smoother_statistics']['analysis_rmse'][0]
        if self._verbose:
            print("forecast_time, forecast_rmse, analysis_time, analysis_rmse")
            print(forecast_time, forecast_rmse, analysis_time, analysis_rmse)
        output_line = u" {0:20.14e} \t {1:20.14e} \t {2:20.14e} \t {3:20.14e} \n".format(forecast_time,
                                                                                         analysis_time,
                                                                                         forecast_rmse,
                                                                                         analysis_rmse
                                                                                         )
        # now write the rmse results to file
        with open(rmse_file_path, mode='a') as file_handler:
            file_handler.write(output_line)
        #

        # Write Smoother configurations
        if file_output_file_format in ['text', 'ascii']:
            #
            smoother_configs = self.smoother_configs
            if smoother_configs['prior_distribution'] == 'gaussian':
                gmm_prior_settings = {}
                prior_distribution = 'gaussian'
                gmm_conf = {}

            elif smoother_configs['prior_distribution'] == 'gmm':
                switched_to_Gaussian_prior = self.switched_to_Gaussian_prior

                if switched_to_Gaussian_prior:
                    gmm_prior_settings = {}
                    prior_distribution = 'gaussian'
                    gmm_conf = {}
                    self.switched_to_Gaussian_prior = False  # for the next cycle.
                else:
                    gmm_prior_settings = smoother_configs['gmm_prior_settings']
                    prior_distribution = 'gmm'
                    gmm_conf = output_configs['smoother_statistics']['gmm_prior_statistics']


            # TODO: Rethink the outputting strategy of this smoother...
            # save smoother and model configurations (a copy under observation directory and another under state directory)...
            #
            smoother_conf= dict(smoother_name=smoother_configs['smoother_name'],
                              prior_distribution=prior_distribution,
                              gmm_prior_settings=gmm_prior_settings,
                              ensemble_size=smoother_configs['ensemble_size'],
                              apply_preprocessing=smoother_configs['apply_preprocessing'],
                              apply_postprocessing=smoother_configs['apply_postprocessing'],
                              chain_parameters=smoother_configs['chain_parameters'],
                              window_bounds=smoother_configs['window_bounds'],
                              analysis_time=smoother_configs['analysis_time'],
                              obs_checkpoints=smoother_configs['obs_checkpoints'],
                              forecast_time=smoother_configs['forecast_time']
                              )
            #
            output_conf = output_configs

            # Save chain diagnostics (statistics)
            try:
                chain_diagnostics = output_configs['smoother_statistics']['chain_diagnostics']
                # print('chain_diagnostics', chain_diagnostics)
            except (ValueError, NameError, AttributeError, KeyError):
                print("Couldn't retrieve chain diagnostics for the smoother?")
                # print('output_configs', output_configs)
                chain_diagnostics = {}

            if prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
                utility.write_dicts_to_config_file('setup.dat', cycle_observations_out_dir,
                                                   [smoother_conf, output_configs, gmm_conf, chain_diagnostics], ['Smoother Configs', 'Output Configs', 'GMM-Prior Configs', 'Chain Diagnostics'])
                utility.write_dicts_to_config_file('gmm_prior_configs.dat', cycle_observations_out_dir, gmm_conf, 'GMM-Prior Configs')
                utility.write_dicts_to_config_file('setup.dat', cycle_states_out_dir,
                                                   [smoother_conf, output_configs, gmm_conf, chain_diagnostics], ['Smoother Configs', 'Output Configs', 'GMM-Prior Configs', 'Chain Diagnostics'])
                utility.write_dicts_to_config_file('gmm_prior_configs.dat', cycle_observations_out_dir, gmm_conf, 'GMM-Prior Configs')
            elif prior_distribution in ['gaussian', 'normal']:
                utility.write_dicts_to_config_file('setup.dat', cycle_observations_out_dir,
                                                   [smoother_conf, output_configs, chain_diagnostics], ['Smoother Configs', 'Output Configs', 'Chain Diagnostics'])
                utility.write_dicts_to_config_file('setup.dat', cycle_states_out_dir,
                                                   [smoother_conf, output_configs, chain_diagnostics], ['Smoother Configs', 'Output Configs', 'Chain Diagnostics'])
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
