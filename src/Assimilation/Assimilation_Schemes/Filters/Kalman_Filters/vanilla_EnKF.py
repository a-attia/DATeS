
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
    This module contains an implementation of the Vanilla Stochastic EnKF.

    EnKF:
    -----
        A class implementing the stochastic ensemble Kalman Filter [Evensen 1994, Burgers et al. 1998].


"""


import numpy as np
import scipy
import scipy.io as sio
from scipy.linalg import lu_factor, lu_solve
import os
import shutil
import re

import dates_utility as utility
from filters_base import FiltersBase
from state_vector_base import StateVectorBase as state_vector
from observation_vector_base import ObservationVectorBase as observation_vector


class EnKF(FiltersBase):
    """
    Stochastic (perturbed observations) Ensemble-Kalman filtering with randomized observations. [Evensen 1994, Burgers et al. 1998]

    Args:
        filter_configs:  dict,
            A dictionary containing EnKF filter configurations.
            Supported configuarations:
            --------------------------
                * model (default None):  model object
                * filter_name (default None): string containing name of the filter; used for output.
                * hybrid_background_coeff (default 0.0): used when hybrid background errors are used,
                    this multiplies the modeled Background error covariance matrix.
                    Will be effective only if the covariance localization is done in full space, otherwise it is meaningless.
                * inflation_factor (default 1.09): covariance inflation factor
                * obs_covariance_scaling_factor (default 1): observation covariance scaling factor (rfactor2 in Sakov's Code)
                * obs_adaptive_prescreening_factor (default None): Added adaptive observation prescreening (kfactor in Sakov's Code)
                * localize_covariances (default True): bool,
                    apply covariance localization to ensemble covariances.
                    This is done by default using Shur product, and is requested from the model.
                    This is likely to be updated in future releases to be carried out here with more options.
                * localization_method: method used to carry out filter localization to remove sporious correlations.
                  Three localization methods are supported:
                     - 'covariance_filtering': involves modification of the update equations by replacing
                        the state error covariance by its element-wise product with some distance-dependent
                        correlation matrix. This is done by localizing covariances projected in the observation space.
                     -  'local_analysis': uses a local approximation of the forecast covariance for updating
                        a state vector element, calculated by building a local window around this element.
                * localization_radius (default np.infty): radius of influence of covariance decorrelation
                * localization_function ('gaspari-cohn'): the covariance localization function
                    'gaspari-cohn', 'gauss', etc.
                    These functions has to be supported by the model to be used here.
                * prior_distribution (default 'gaussian'): prior probability distribution;
                    this shoule be either 'gaussian' or 'GMM'.
                    - 'Gaussian': the prior distribution is approximated based on the forecast ensemble,
                    - 'GMM': the prior distribution is approximated by a GMM constructed using EM algorithm
                             using the forecast ensemble.
                * gmm_prior_settings: dict,
                    This is a configurations dictionary of the GMM approximation to the prior.
                    This will be used only if the prior is assumed to be non-Gaussian, and better estimate is needed,
                    i.e. it is used only if 'prior_distribution' is set to 'GMM'.
                    The implementation in this case follows the cluster EnKF described by [cite].
                    The configurations supported are:
                       - clustering_model:
                       - cov_type:
                       - localize_covariances:
                       - localization_radius:
                       - localization_function:
                       - inf_criteria:
                       - number_of_components:
                       - min_number_of_components:
                       - max_number_of_components:
                       - min_number_of_points_per_component:
                       - invert_uncertainty_param:
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
            --------------------------
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
    _filter_name = "EnKF"
    #
    _def_local_filter_configs = dict(model=None,
                                     filter_name=_filter_name,
                                     hybrid_background_coeff=0.0,
                                     forecast_inflation_factor=1.0,  # applied to forecast ensemble
                                     inflation_factor=1.09,  # applied to analysis ensemble
                                     obs_covariance_scaling_factor=1.0,
                                     obs_adaptive_prescreening_factor=None,
                                     localize_covariances=True,
                                     localization_method='covariance_filtering',
                                     localization_radius=np.infty,
                                     localization_function='gaspari-cohn',
                                     prior_distribution='gaussian',
                                     gmm_prior_settings=dict(clustering_model='gmm',
                                                             cov_type='diag',
                                                             localize_covariances=False,
                                                             localization_radius=np.infty,
                                                             localization_function='gaspari-cohn',
                                                             inf_criteria='aic',
                                                             number_of_components=None,
                                                             min_number_of_components=None,
                                                             max_number_of_components=None,
                                                             min_number_of_points_per_component=1,
                                                             invert_uncertainty_param=True
                                                             ),
                                     ensemble_size=None,
                                     analysis_ensemble=None,
                                     analysis_state=None,
                                     forecast_ensemble=None,
                                     forecast_state=None,
                                     filter_statistics=dict(forecast_rmse=None,
                                                            analysis_rmse=None,
                                                            initial_rmse=None
                                                            )
                                     )
    #
    _local_def_output_configs = dict(scr_output=True,
                                     file_output=False,
                                     filter_statistics_dir='Filter_Statistics',
                                     model_states_dir='Model_States_Repository',
                                     observations_dir='Observations_Rpository',
                                     file_output_moment_only=True,
                                     file_output_moment_name='mean',
                                     file_output_file_name_prefix='EnKF_results',
                                     file_output_file_format='txt',
                                     file_output_separate_files=False
                                     )
                                     #
    _supported_prior_distribution = ['gaussian', 'normal', 'gmm', 'gaussian-mixture', 'gaussian_mixture']
    #

    #
    def __init__(self, filter_configs=None, output_configs=None):

        # aggregate configurations, and attach filter_configs, output_configs to the filter object.
        filter_configs = utility.aggregate_configurations(filter_configs, EnKF._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, EnKF._local_def_output_configs)
        #
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(filter_configs=filter_configs, output_configs=output_configs)
        else:
            # old-stype class
            super(EnKF, self).__init__(filter_configs=filter_configs, output_configs=output_configs)

        #
        self.model = self.filter_configs['model']
        #
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

        # retrieve the observation vector size from model:
        self.observation_size = self.model.observation_vector_size()

        if self.filter_configs['localize_covariances']:
            loc_radius = self.filter_configs['localization_radius']
            try:
                if np.isinf(loc_radius):
                    self.filter_configs['localize_covariances'] = False
            except ValueError:
                pass
            except TypeError:
                self.filter_configs['localize_covariances'] = False
            finally:
                if loc_radius is None:
                    self.filter_configs['localize_covariances'] = False

        #
        # Generate prior-distribution information:
        self.prior_distribution = self.filter_configs['prior_distribution'].lower()
        #
        if self.prior_distribution in ['gaussian', 'normal']:
            #
            forecast_ensemble = self.filter_configs['forecast_ensemble']
            if forecast_ensemble is not None:
                self.forecast_state = utility.ensemble_mean(forecast_ensemble)
            else:
                self.forecast_state = None

        elif self.prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            # Generate GMM parameters... May be it is better to move this to a method to generate prior info.
            # It might be needed in the forecast step in case FORECAST is carried out first, and forecast ensemble is empty!
            self._gmm_prior_settings = self.filter_configs['gmm_prior_settings']

            if 'filter_statistics' not in self.output_configs:
                # Add filter statistics to the output configuration dictionary for proper outputting.
                self.output_configs.update(dict(filter_statistics=dict(gmm_prior_statistics=None)))

            # Generate the forecast state only for forecast RMSE evaluation. It won't be needed by the GMM+HMC sampler
            forecast_ensemble = self.filter_configs['forecast_ensemble']
            if forecast_ensemble is not None:
                self.forecast_state = utility.ensemble_mean(forecast_ensemble)
            else:
                self.forecast_state = None
        #
        else:
            print("Unrecognized prior distribution [%s]!" % self.prior_distribution)
            raise ValueError()
        #
        try:
            self._verbose = self.output_configs['verbose']
        except(AttributeError, NameError):
            self._verbose = False
        #
        self.__initialized = True
        #

    #
    def filtering_cycle(self, update_reference=False):
        """
        Carry out a single filtering cycle. Forecast followed by analysis or the other way around.
        All required variables are obtained from 'filter_configs' dictionary.

        Args:
            update_reference (default True): bool,
                A flag to decide whether to update the reference state in the filter or not.

        Returns:
            None

        """
        # Call basic functionality from the parent class:
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().filtering_cycle(update_reference=update_reference)
        else:
            # old-stype class
            super(EnKF, self).filtering_cycle(update_reference=update_reference)
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
        analysis_ensemble = self.filter_configs['analysis_ensemble']
        timespan = self.filter_configs['timespan']
        forecast_ensemble = utility.propagate_ensemble(ensemble=analysis_ensemble,
                                                       model=self.filter_configs['model'],
                                                       checkpoints=timespan,
                                                       in_place=False)
        self.filter_configs['forecast_ensemble'] = forecast_ensemble
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        self.filter_configs['forecast_state'] = forecast_state.copy()

        # Add more functionality after building the forecast ensemble if desired!
        # Obtain information about the prior distribution from the forecast ensemble
        # Covariance localization and hybridization are carried out if requested
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

        Args:
            None

        Returns:
            None

        """
        #
        # Read the forecast ensemble...
        forecast_ensemble = self.filter_configs['forecast_ensemble']
        #
        if self.filter_configs['prior_distribution'].lower() in ['gaussian', 'normal']:
            #
            # Localization is now moved to the analysis step

            # Evaluate the ensemble variances of the ensemble
            self.prior_variances = utility.ensemble_variances(self.filter_configs['forecast_ensemble'])

            #
            try:
                self.prior_distribution_statistics
            except (NameError, AttributeError):
                self.prior_distribution_statistics = dict()
            #

        # Construct a Gaussian Mixture Model (GMM) representation/approximation of the prior distribution:
        elif self.filter_configs['prior_distribution'].lower() in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
            #
            gmm_prior_settings = self.filter_configs['gmm_prior_settings']
            #
            ensemble_size = self.sample_size
            state_size = self.model.state_size()
            forecast_ensemble = self.filter_configs['forecast_ensemble']
            forecast_ensemble_numpy = np.empty((ensemble_size, state_size))  # This formulation is for GMM inputs
            # Create a numpy.ndarray containing the forecast ensemble to be used by the GMM generator
            for ens_ind in xrange(ensemble_size):
                # GMM builder requires the forecast ensemble to be Nens x Nvar
                forecast_ensemble_numpy[ens_ind, :] = forecast_ensemble[ens_ind].get_numpy_array()
            # Now generate the GMM model
            gmm_converged, gmm_lables, gmm_weights, gmm_means, gmm_covariances, gmm_precisions, gmm_optimal_covar_type \
                = utility.generate_gmm_model_info(ensemble=forecast_ensemble_numpy,
                                                  clustering_model=gmm_prior_settings['clustering_model'],
                                                  cov_type=gmm_prior_settings['cov_type'],
                                                  inf_criteria=gmm_prior_settings['inf_criteria'],
                                                  number_of_components=gmm_prior_settings['number_of_components'],
                                                  min_number_of_components=gmm_prior_settings['min_number_of_components'],
                                                  max_number_of_components=gmm_prior_settings['max_number_of_components'],
                                                  min_number_of_points_per_component=gmm_prior_settings['min_number_of_points_per_component'],
                                                  invert_uncertainty_param=gmm_prior_settings['invert_uncertainty_param']
                                                  )
            if not gmm_converged:
                print("The GMM model construction process failed. EM algorithm did NOT converge!")
                raise ValueError()
            else:
                try:
                    self.prior_distribution_statistics
                except (NameError, AttributeError):
                    self.prior_distribution_statistics = dict()
                #
                gmm_num_components = int(np.max(gmm_lables)) + 1
                if gmm_num_components > 1:
                    gmm_weights = np.asarray(gmm_weights)
                elif gmm_num_components == 1:
                    gmm_weights = np.asarray([gmm_weights])
                else:
                    print("How is the number of mixture components negative???")
                    raise ValueError()
                #
                # GMM successfully generated. attach proper information to the filter object
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
                    mean_vector[:] = gmm_means[comp_ind, :]
                    means_list.append(mean_vector)
                self.prior_distribution_statistics['gmm_means'] = means_list
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
                        covariances_det_log_list = []
                        prior_variances = self.model.state_vector(); prior_variances[:] = 0.0
                        joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                        mean_vector = self.model.state_vector()  # temporary state vector overwritten each iteration
                        for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                            # retrieve and store the covariance matrix for each component.
                            variances_vector = self.model.state_vector()
                            gmm_variances = gmm_covariances[comp_ind, :]
                            variances_vector[:] = gmm_variances
                            covariances_list.append(variances_vector)
                            # Evaluate the determinant logarithm of the diagonal covariance matrix
                            covariances_det_log = np.sum(np.log(gmm_variances))
                            covariances_det_log_list.append(covariances_det_log)
                            #
                            # find the joint variance and the joint mean for momentum construction
                            prior_variances = prior_variances.add(variances_vector.scale(gmm_weights[comp_ind]))
                            mean_vector[:] = gmm_means[comp_ind, :]
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
                                mean_vector[:] = gmm_means[comp_ind, :]
                                # deviation = mean_vector.scale(-1.0).add(joint_mean); deviation.scale(-1.0)
                                deviation = mean_vector.axpy(-1.0, joint_mean, in_place=False)
                                # prior_variances = prior_variances.add(deviation.multiply(deviation, in_place=True).scale(gmm_weights[comp_ind]))
                                prior_variances = prior_variances.add(deviation.square().scale(gmm_weights[comp_ind]))
                            self.prior_variances = prior_variances.get_numpy_array()
                            self.prior_mean = joint_mean
                            if self._verbose:
                                print('prior variances:', self.prior_variances)
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
                            joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                        else:
                            calculate_det_log = False
                        gmm_precisions_list = []
                        prior_variances = self.model.state_vector(); prior_variances[:] = 0.0
                        for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                            precisions_vector = self.model.state_vector()
                            precisions_vector[:] = gmm_precisions[comp_ind, :]
                            gmm_precisions_list.append(precisions_vector)
                            #
                            if calculate_det_log:
                                # Evaluate the determinant logarithm of the diagonal covariance matrix from precision matrix
                                covariances_det_log = - np.sum(np.log(gmm_precisions[comp_ind, :]))
                                covariances_det_log_list.append(covariances_det_log)
                            #
                            if prior_variances_from_precisions:
                                # find the joint variance and the joint mean for momentum construction
                                prior_variances = prior_variances.add(precisions_vector.reciprocal().scale(gmm_weights[comp_ind]))
                                mean_vector = self.model.state_vector()
                                mean_vector[:] = gmm_means[comp_ind, :]
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
                                    mean_vector[:] = gmm_means[comp_ind, :]
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
                elif gmm_optimal_covar_type == 'tied':
                    if gmm_covariances is None:
                        self.prior_distribution_statistics['gmm_covariances'] = None
                        self.prior_distribution_statistics['gmm_covariances_det_log'] = None
                        self.prior_variances = None
                    else:
                        covariances_matrix = self.model.state_matrix()
                        covariances_matrix[:, :] = gmm_covariances[:, :]
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
                            #
                        else:
                            singular_vals = np.linalg.svd(gmm_covariances.get_numpy_array(), compute_uv=False)
                        self.prior_distribution_statistics['gmm_covariances'] = covariances_matrix
                        covariances_det_log = np.sum(np.log(singular_vals))
                        # Evaluate and replicate the logarithm of the covariances matrix's determinant for all components.
                        self.prior_distribution_statistics['gmm_covariances_det_log'] \
                            = np.asarray([covariances_det_log for i in xrange(self.prior_distribution_statistics['gmm_num_components'])])

                        # find the joint variance and the joint mean for momentum construction
                        prior_variances = self.model.state_vector()
                        prior_variances[:] = covariances_matrix.diag()
                        joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                        for comp_ind in xrange(gmm_num_components):
                            mean_vector = self.model.state_vector()
                            mean_vector[:] = gmm_means[comp_ind, :]
                            joint_mean = joint_mean.add(mean_vector.scale(gmm_weights[comp_ind]))
                        #
                        # proceed with evaluating joint variances and the joint mean
                        if self._gmm_prior_settings['approximate_covariances_from_comp']:
                            # use means and covariances of the components to calculate/approximate mean and covariance matrix of the mixture.
                            mean_vector = self.model.state_vector()
                            for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                                mean_vector[:] = gmm_means[comp_ind, :]
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
                        precisions_matrix[:, :] = gmm_precisions[:, :]

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
                            prior_variances[:] = precisions_matrix.diag()
                            prior_variances = prior_variances.reciprocal()
                            joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                            for comp_ind in xrange(gmm_num_components):
                                mean_vector = self.model.state_vector()
                                mean_vector[:] = gmm_means[comp_ind, :]
                                joint_mean = joint_mean.add(mean_vector.scale(gmm_weights[comp_ind]))
                            #
                            # proceed with evaluating joint variances and the joint mean
                            if self._gmm_prior_settings['approximate_covariances_from_comp']:
                                # use means and covariances of the components to calculate/approximate mean and covariance matrix of the mixture.
                                mean_vector = self.model.state_vector()
                                for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                                    mean_vector[:] = gmm_means[comp_ind, :]
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
                elif gmm_optimal_covar_type == 'full':
                    if gmm_covariances is None:
                        self.prior_distribution_statistics['gmm_covariances'] = None
                        self.prior_distribution_statistics['gmm_covariances_det_log'] = None
                        self.prior_variances = None
                    else:
                        covariances_list = []
                        covariances_det_log_list = []
                        prior_variances = self.model.state_vector(); prior_variances[:] = 0.0
                        joint_mean = self.model.state_vector(); joint_mean[:] = 0.0
                        for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                            covariances_matrix = self.model.state_matrix()
                            covariances_matrix[:, :] = gmm_covariances[comp_ind, :, :]
                            covariances_list.append(covariances_matrix)
                            # retrieve and store the covariance matrix for each component.
                            variances_vector = self.model.state_vector()
                            variances_vector[:] = covariances_matrix.diag()
                            covariances_list.append(variances_vector)
                            # Evaluate and replicate the logarithm of the covariances matrix's determinant for all components.
                            singular_vals = np.linalg.svd(gmm_covariances[comp_ind, :, :], compute_uv=False)
                            covariances_det_log = np.sum(np.log(singular_vals))
                            covariances_det_log_list.append(covariances_det_log)
                            #
                            # find the joint variance and the joint mean for momentum construction
                            prior_variances = prior_variances.add(variances_vector.scale(gmm_weights[comp_ind]))
                            mean_vector = self.model.state_vector()
                            mean_vector[:] = gmm_means[comp_ind, :]
                            joint_mean = joint_mean.add(mean_vector.scale(gmm_weights[comp_ind]))

                        self.prior_distribution_statistics['gmm_covariances'] = covariances_list
                        self.prior_distribution_statistics['gmm_covariances_det_log'] = np.asarray(covariances_det_log_list)
                        #
                        # proceed with evaluating joint variances and the joint mean
                        if self._gmm_prior_settings['approximate_covariances_from_comp']:
                            # use means and covariances of the components to calculate/approximate mean and covariance matrix of the mixture.
                            mean_vector = self.model.state_vector()
                            for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                                mean_vector[:] = gmm_means[comp_ind, :]
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
                        prior_variances = self.model.state_vector(); prior_variances[:] = 0.0
                        for comp_ind in xrange(self.prior_distribution_statistics['gmm_num_components']):
                            precisions_matrix = self.model.state_matrix()
                            precisions_matrix[:, :] = gmm_precisions[comp_ind, :, :]
                            gmm_precisions_list.append(precisions_matrix)
                            precisions_vector = self.model.state_vector()
                            precisions_vector[:] = precisions_matrix.diag()
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
                                mean_vector[:] = gmm_means[comp_ind, :]
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
                                    mean_vector[:] = gmm_means[comp_ind, :]
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

    #
    def analysis(self, all_to_numpy=True):
        """
        Analysis step:

        Args:
            all_to_numpy (default False): bool,
                convert all data structures to Numpy and re-place results into target structures only in the end.

        Returns:
            None. Only self.filter_configs is updated.

        """
        # Check if the forecast ensemble should be inflated;
        f = self.filter_configs['forecast_inflation_factor']
        utility.inflate_ensemble(self.filter_configs['forecast_ensemble'], f)

        #
        if all_to_numpy:
            # get the forecast state as the mean of the forecast ensemble. Will not be used for GMM!
            state_size = self.model.state_size()
            try:
                observation_size = self.observation_size
            except(NameError, AttributeError):
                observation_size = self.model.observation_vector_size()
                #
            forecast_ensemble = self.filter_configs['forecast_ensemble']
            forecast_state = utility.ensemble_mean(forecast_ensemble)
            ensemble_size = self.sample_size

            # observation error covariance matrix
            R = self.model.observation_error_model.R.get_numpy_array()
            try:
                sqrtR = self.model.observation_error_model.sqrtR.get_numpy_array()
            except (AttributeError, NameError, ValueError):
                sqrtR = np.linalg.cholesky(R)
            #
            forecast_ensemble_np = np.empty((state_size, ensemble_size))
            for ens_ind in xrange(ensemble_size):
                # forecast_ensemble[:, ens_ind] = forecast_ensemble[ens_ind][:]
                forecast_ensemble_np[:, ens_ind] = forecast_ensemble[ens_ind].get_numpy_array()
            forecast_state_np = np.mean(forecast_ensemble_np, 1)

            # get the measurements vector
            observation = self.filter_configs['observation'].get_numpy_array()

            #
            # PREPARE for ASSIMILATION:
            # ---------------------------

            # Model-observation of the forecast ensemble members:
            HE = np.empty((observation_size, ensemble_size))
            for ens_ind in xrange(ensemble_size):
                HE[:, ens_ind] = self.model.evaluate_theoretical_observation(forecast_ensemble[ens_ind]).get_numpy_array()
            # Mean of forecasted observations:
            Hx = np.mean(HE, 1)

            # Observation innovations:
            # obs_innovations = observation - Hx
            obs_innovations = observation - self.model.evaluate_theoretical_observation(forecast_state).get_numpy_array()

            # Forecast Ensemble Anomalies matrix [forecast ensemble members - ensemble mean]
            A = forecast_ensemble_np
            for ens_ind in xrange(ensemble_size):
                A[:, ens_ind] -= forecast_state_np

            # Forecast Ensemble Observations Anomalies matrix [HE(e) - Hx], e = 1,2,..., ensemble_size
            HA = HE  # reuse --> in-place
            for ens_ind in xrange(ensemble_size):
                HA[:, ens_ind] -= Hx

            #
            # START ASSIMILATION:
            # ---------------------------

            # standardised innovation and ensemble anomalies
            # sqrtR_lu, sqrtR_piv = lu_factor( sqrtR )
            # s = obs_innovations / np.sqrt(ensemble_size-1)
            # s = lu_solve((sqrtR_lu, sqrtR_piv) , s)
            # S = np.empty_like(HA)
            # for ens_ind in xrange(ensemble_size):
            #     S[:, ens_ind] = (lu_solve((sqrtR_lu, sqrtR_piv) , HA[:, ens_ind])) / np.sqrt(ensemble_size-1)

            sqrtR_inv = np.linalg.inv( sqrtR )
            s = sqrtR_inv.dot(obs_innovations) / np.sqrt(ensemble_size-1)
            S = sqrtR_inv.dot(HA) / np.sqrt(ensemble_size-1)

            # get the current state of the random number generator
            current_random_state = np.random.get_state()

            # observation covariance scaling factor
            rfactor = float(self.filter_configs['obs_covariance_scaling_factor'])

            # Generate the random observations (global perturbations of observations for the EnKF)
            D = np.random.randn(observation_size, ensemble_size)
            D *= 1.0/(rfactor * np.sqrt(ensemble_size-1.0))
            d = np.mean(D, 1)
            for ens_ind in xrange(ensemble_size):
                D[:, ens_ind] -= d
            D *= np.sqrt(ensemble_size / (ensemble_size-1.0))
            # Restore the state of the gaussian random number generator:
            np.random.set_state(current_random_state)

            # Analysis is carried out based on the tpe of localization in what follows:
            #
            localize_covariances = self.filter_configs['localize_covariances']
            if not localize_covariances:
                # Global analysis (No Localization):
                if ensemble_size <= observation_size:
                    # Calculte G = (I + (S^T * S))^{-1} * S^T
                    G = np.dot(np.transpose(S), S)
                    G[np.diag_indices_from(G)] += 1.0
                    G = np.dot(np.linalg.inv(G), np.transpose(S))
                else:
                    # Calculte G = S^T * (I + S * S^T)^{-1}
                    G = np.dot(S, np.transpose(S))
                    G[np.diag_indices_from(G)] += 1.0
                    G = np.dot(S.T, np.linalg.inv(G))

                # Evaluate the Ensemble-Mean update:
                ens_mean_update = np.dot(np.dot(A, G), s)  # dx

                # Evaluate Ensemble-Anomalies update:
                if rfactor != 1.0:
                    # rescale S, and G
                    S *= 1.0 / np.sqrt(rfactor)
                    if ensemble_size <= observation_size:
                        # RE-Calculte G = (I + (S^T * S))^{-1} * S^T
                        G = np.dot(np.transpose(S), S)
                        G[np.diag_indices_from(G)] += 1.0
                        G = np.dot(np.linalg.inv(G), np.transpose(S))
                    else:
                        # RE-Calculte G = S^T * (I + S * S^T)^{-1}
                        G = np.dot(S, np.transpose(S))
                        G[np.diag_indices_from(G)] += 1.0
                        G = np.dot(S.T, np.linalg.inv(G))
                else:
                    pass
                # Now Evaluate A = A * (I + G * (D - S)):
                D -= S
                G = G.dot(D)
                G[np.diag_indices_from(G)] += 1.0
                A = A.dot(G)

            else:
                # Apply Localization based on the localization function:
                localization_method = self.filter_configs['localization_method']
                localization_function = self.filter_configs['localization_function']
                localization_radius = self.filter_configs['localization_radius']
                #
                if re.match(r'\Acovariance(-|_)*filtering\Z', localization_method, re.IGNORECASE):
                    # Evaluate the Kalman gain matrix (with HPH^T, and PH^T localized based on the filter settings)
                    K = self._calc_Kalman_gain(A, HA)

                    # Evaluate the Ensemble-Mean update (dx):
                    ens_mean_update = np.dot(K, obs_innovations)

                    # Recalculate the Kalman gain with observation variances/covariances multiplied by rfactor
                    if rfactor != 1:
                        K = self._calc_Kalman_gain(A, HA, rfactor)

                    # Update Ensemble-Anomalies matrix:
                    # get the current state of the random number generator
                    current_random_state = np.random.get_state()
                    # Generate the random observations (global perturbations of observations for the EnKF)
                    D = np.random.randn(observation_size, ensemble_size)
                    D *= np.sqrt(rfactor)
                    D = np.dot(sqrtR, D)
                    d = np.mean(D, 1)
                    for ens_ind in xrange(ensemble_size):
                        D[:, ens_ind] -= d
                    D *= np.sqrt(ensemble_size / (ensemble_size-1.0))
                    # Restore the state of the gaussian random number generator:
                    np.random.set_state(current_random_state)

                    # Now Evaluate A = A + K * (D - HA):
                    D -= HA
                    D = np.dot(K, D)
                    A += D


                elif re.match(r'\Alocal(-|_)*analysis\Z', localization_method, re.IGNORECASE):
                    raise NotImplementedError("TO BE UPDATED...")
                    # ens_mean_update np.empty(state_size)
                    # for state_ind in xrange(state_size):
                    #     # find local observation:

                    pass
                    # # ------------------^^^^ TRANSLATE ^^^^-------------------
                    # Check 'assimilate.m' line 188
                    # for i = 1 : n
                    #     [localobs, coeffs] = find_localobs(prm, i, pos);
                    #     ploc = length(localobs);
                    #     if ploc == 0
                    #         continue
                    #     end
                    #
                    #     Sloc = S(localobs, :) .* repmat(coeffs, 1, m);
                    #
                    #     if m <= ploc
                    #         Gloc = inv(speye(m) + Sloc' * Sloc) * Sloc';
                    #     else
                    #         Gloc = Sloc' * inv(speye(ploc) + Sloc * Sloc');
                    #     end
                    #
                    #     dx(i) = A(i, :) * Gloc * (s(localobs) .* coeffs);
                    #
                    #     if rfactor ~= 1
                    #         Sloc = Sloc / sqrt(rfactor);
                    #         if m <= ploc
                    #             Gloc = inv(speye(m) + Sloc' * Sloc) * Sloc';
                    #         else
                    #             Gloc = Sloc' * inv(speye(ploc) + Sloc * Sloc');
                    #         end
                    #     end
                    #
                    #     Dloc = D(localobs, :) .* repmat(coeffs, 1, m);
                    #     A(i, :) = A(i, :) + A(i, :) * Gloc * (Dloc - Sloc);
                    # end
                    # -------------------------------------

                else:
                    print("Localization method '%s' is not Supported/Recognized!" % localization_method)
                    raise ValueError

            # Inflate the ensemble if required; this is done by magnifying the matrix of ensemble-anomalies:
            inflation_fac=self.filter_configs['inflation_factor']
            if inflation_fac > 1.0:
                if self._verbose:
                    print('Inflating the forecast ensemble...')
                #
                A *= inflation_fac
                #
                if self._verbose:
                    print('inflated? : ', (analysis_ensemble[0][:]!=inflated_ensemble[0][:]).any())

            # Now we are good to go; update the ensemble mean, and ensemble-anomalies using ens_mean_update, and A
            ens_mean_update_vec = self.model.state_vector()
            ens_mean_update_vec[:] = np.squeeze(ens_mean_update)
            analysis_state = forecast_state.copy()
            analysis_state = analysis_state.add(ens_mean_update_vec)
            analysis_mean_np = np.squeeze(analysis_state.get_numpy_array())
            try:
                analysis_ensemble = self.filter_configs['analysis_ensemble']
            except(ValueError, KeyError, NameError, AttributeError):
                analysis_ensemble = []
            finally:
                if analysis_ensemble is None:
                    analysis_ensemble = []
                elif isinstance(analysis_ensemble, list):
                    if len(analysis_ensemble)!=0 and len(analysis_ensemble)!=ensemble_size:
                        analysis_ensemble = []
                else:
                    print("analysis_ensemble type is not recognized!")
                    raise TypeError

            if len(analysis_ensemble) == 0:
                # Append analysis states
                for ens_ind in xrange(ensemble_size):
                    analysis_ens_member = self.model.state_vector()
                    analysis_ens_member[:] = np.squeeze(A[:, ens_ind]) + analysis_mean_np
                    analysis_ensemble.append(analysis_ens_member)
            else:
                # update analysis ensemble in place
                for ens_ind in xrange(ensemble_size):
                    analysis_ensemble[ens_ind][:] = np.squeeze(A[:, ens_ind]) + analysis_mean_np

            # Ensemble Rotation:


            # Update analysis ensemble and analysis state (average of the analysis_ensemble)
            self.filter_configs['analysis_ensemble'] = analysis_ensemble

            # Update the analysis_state in the filter_configs dictionary.
            self.filter_configs['analysis_state'] = analysis_state
            #
        #
        else:
            raise NotImplementedError("To be implemented!")

    #
    def _calc_Kalman_gain(self, A, HA, rfactor=1.0):
        """
        Calculate and return Kalman gain.
        All matrices passed and returned from this function are Numpy-based

        Args:
            A: Forecast Ensemble Anomalies matrix [forecast ensemble members - ensemble mean]
            HA: Forecast Ensemble Observations Anomalies matrix [HE(e) - Hx], e = 1,2,..., ensemble_size

        Returns:
            K: Kalman gain matrix

        """
        # model, and dimensionalities:
        model = self.model
        state_size = model.state_size()
        ensemble_size = self.sample_size
        observation_size = model.observation_vector_size()

        # Calculate Kalman Gain, and carry out localization if needed:
        HPHT = (1.0/(ensemble_size - 1)) * np.dot(HA, HA.T)
        PHT = (1.0/(ensemble_size - 1)) * np.dot(A, HA.T)

        # localization info:
        localize_covariances = self.filter_configs['localize_covariances']
        #
        # Apply covariance localization if requested:
        if localize_covariances:

            # get dimensionalities:
            # number of model-grid dimensions
            try:
                num_dimensions = model.model_configs['num_dimensions']
            except(KeyError):
                num_dimensions = None

            # get the observational grid
            observations_positions = model.get_observations_positions()
            num_obs_dims = np.size(observations_positions, 1)
            #
            if num_dimensions is None:
                num_dimensions = num_obs_dims
            elif (max(num_dimensions, num_obs_dims) == 1):
                num_dimensions = num_obs_dims = 1
            else:
                pass

            if num_obs_dims != num_dimensions:
                print("Observational grid dimension mismatches the model grid dimension!. \n  \
                       Observation dimensions = %d, Model state dimensions = %d" % (num_obs_dims, num_dimensions))
                raise ValueError

            try:
                dx = model.model_configs['dx']
            except(KeyError, NameError, AttributeError):
                dx = 1

            if num_dimensions <=1:
                dy = dx
            elif num_dimensions ==2:
                try:
                    dy = model.model_configs['dy']
                except(KeyError, NameError, AttributeError):
                    dy = dx
            else:
                print("Sorry: only up to 2D models are handled Here!")
                raise NotImplementedError

            #
            if observation_size>1:
                # Masking HPHT :if more than one observatin is made:

                localization_function = self.filter_configs['localization_function']
                #
                localization_radii = self.filter_configs['localization_radius']
                equal_radii = None
                if np.isscalar(localization_radii):
                    localization_radius = localization_radii  # just for convenience
                    equal_radii = True
                else:

                    localization_radius = np.asarray(localization_radii[:])
                    equal_radii = False
                    #
                    if localization_radius.size == state_size:
                        tmp_state = model.state_vector()
                        # tmp_state[:] = localization_radius
                        # localization_radius = model.evaluate_theoretical_observation(tmp_state).get_numpy_array()  (Wrong!)
                        pass
                        #
                        # localization_radius are designed for state vectors; we need those for observation gridpoints:
                        state_localization_radius = localization_radius
                        localization_radius = []
                        # obs_grid = model.get_observational_grid()
                        for i in xrange(observation_size):
                            # get localization radius from closest model grid point:
                            nb_inds = []
                            # dx2 = min(dx, dy)
                            dx2 = max(dx, dy)
                            while len(nb_inds) == 0 and dx2 <= dx*state_size:
                                nb_inds = model.get_neighbors_indexes(index=i,
                                                                      radius=dx2,
                                                                      source_index='obs',
                                                                      target_index='state')
                                #
                                if len(nb_inds) > 0:
                                    # found neighboring gridpoints;
                                    # empirically choose the localization radius of the first one (we can follow other strategies!)
                                    localization_radius.append(state_localization_radius[nb_inds[0]])
                                    break

                                # sanity check:
                                if dx2 >= dx*state_size or dx2 >= dy*state_size:  # this might be too much!
                                    if len(nb_inds) == 0:
                                        print("This is an impossible situation: No state grid points found near observation index %d" %i)
                                        print("Terminating!")
                                        raise ValueError
                                    else:
                                        pass

                                # double dx to find nearest gridpoint
                                dx2 *= 2
                        #
                        localization_radius = np.asarray(localization_radius)
                        #

                        # Another sanity check!
                        if localization_radius.size != observation_size:
                            print("Projected localization_radius vector is supposed to be of length equal to the state vector!")
                            print("localization_radius is of length %d" % len(localization_radius))
                            print("observation vector size is %d" % observation_size)
                            raise ValueError

                        if self._verbose:
                            print("Observation localization radii:")
                            print(localization_radius)
                            #
                    #
                    elif localization_radius.size == 1:
                        localization_radius = localization_radius[0]
                        equal_radii = True
                    #
                    else:
                        print("localization_radius.size", localization_radius.size)
                        print("state_size", state_size)
                        print("observation_size", observation_size)
                        print("ensemble_size", ensemble_size)

                        print("The localization radius has to be either a scalar or an iterable of dimension equal to the model state space size.")
                        raise ValueError

                #
                # print "++++++++++++++++ >>", type(localization_radius), localization_radius

                # TODO: Now, test if the localization_radius contains a scalr or an iterable (e.g. a list of radii)
                #       Based on the size of the iterable (if not scalar) decide whether the localization radii are given
                #       in the observation space or the state space, and save the inputs/outputs accordingly in the configuration dictionary.
                # TODO: If there are more than one radius, then we need to pick these radii one by one and apply them to the state entries,
                #       however, we will actually need to project these radii to the observation space. Isn't it obvious?
                #       Just project it by left multiplication with H, i.e. the observation operator.
                # TODO: Inside the loop (in the observation space) use each of the projected radii, and
                #       after you calcluate the coefficients, you are done for the training.

                #
                obs_loc_operator = np.zeros((observation_size, observation_size))
                #
                # Model-grid is of zero, or 1 dimensions:
                if num_dimensions <= 1:

                    try:
                        periodic_bc = model.model_configs['periodic']
                    except(KeyError, NameError, AttributeError):
                        periodic_bc = False

                    # Localization in the observation space:
                    for obs_ind in xrange(observation_size):
                        ref_obs_coord = np.squeeze(observations_positions[obs_ind])
                        distances = np.abs(observations_positions[obs_ind:] - ref_obs_coord).squeeze()
                        if distances.ndim == 0:
                            distances = np.array([distances.item()])

                        #
                        # update distances for periodic models/boundary-conditions (e.g. Lorenz 96)
                        if periodic_bc:
                            rem_distances = (observation_size*dx) - distances
                            up_distances = distances > rem_distances
                            distances[up_distances] = rem_distances[up_distances]
                        #
                        if equal_radii:
                            obs_loc_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                        else:
                            obs_loc_coeffs = utility.calculate_localization_coefficients(localization_radius[obs_ind], distances, localization_function)
                            #
                        obs_loc_operator[obs_ind, obs_ind:] = obs_loc_coeffs
                        obs_loc_operator[obs_ind, :obs_ind] = obs_loc_operator[:obs_ind, obs_ind]
                        #
                    HPHT *= obs_loc_operator

                elif num_dimensions==2:
                    #
                    loc_observations_positions = observations_positions.copy()
                    for obs_ind in xrange(observation_size):
                        ref_obs_coord = np.squeeze(loc_observations_positions[obs_ind, :])
                        #
                        loc_distances = loc_observations_positions[obs_ind:, :]
                        loc_obs_cnt = np.size(loc_distances, 0)
                        for dim_ind in xrange(num_obs_dims):
                            loc_distances[:, dim_ind] -= ref_obs_coord[dim_ind]
                        #
                        distances = np.empty(loc_obs_cnt)
                        for loc_obs_ind in xrange(loc_obs_cnt):
                            distances[loc_obs_ind] = np.linalg.norm(loc_distances[loc_obs_ind, :])
                        #
                        if equal_radii:
                            obs_loc_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                        else:
                            obs_loc_coeffs = utility.calculate_localization_coefficients(localization_radius[obs_ind], distances, localization_function)

                        # obs_loc_coeffs is a row in the matrix H Decorr HT (which is also a column due to symmetry for the cartesian-gridded models.
                        # Just make sure to throw an exception (e.g NotImplementedError) if self.model.configs['grid_type'] is not 'cartes|zian'
                        # Now, we need to project each of these columns/rows to the full space by left, and right multiplication (properly) by the observation operator.
                        # once we have HT Decorr H, (i.e. the rows), we need to return them in the configuration dictionary for Azam to use in the ML stuff.
                        # the question now is:
                        # which parts of the localization radii, we want

                        obs_loc_operator[obs_ind, obs_ind:] = obs_loc_coeffs
                        obs_loc_operator[obs_ind, :obs_ind] = obs_loc_operator[:obs_ind, obs_ind]
                        #

                    HPHT *= obs_loc_operator
                    #
                else:
                    print("Only up to 2D models are handled Here!")
                    raise NotImplementedError
            else:
                # It's a single observation, no masking here is needed
                pass

            # Mask PHT:
            num_dimensions = model.model_configs['num_dimensions']
            #
            if num_dimensions <= 1:
                try:
                    periodic = model.model_configs['periodic']
                except(KeyError, NameError, AttributeError):
                    periodic = False

                distances = observations_positions * dx
                #
                # update distances for periodic models/boundary-conditions (e.g. Lorenz 96)
                if periodic_bc:
                    rem_distances = (observation_size*dx) - distances
                    up_distances = distances > rem_distances
                    distances[up_distances] = rem_distances[up_distances]

                if distances.ndim == 0:
                    distances = np.array([distances.item()])

                #
                if equal_radii:
                    loc_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                else:
                    loc_coeffs = []
                    for rad, dist in zip(localization_radius, distances):
                        coeff = utility.calculate_localization_coefficients(rad, dist, localization_function)
                        loc_coeffs.append(coeff)
                    loc_coeffs = np.asarray(loc_coeffs)

                try:
                    mult = state_size / model.model_configs['num_prognostic_variables']
                except (NameError, AttributeError, KeyError):
                    mult = 1
                nv = state_size / mult

                v = np.empty(state_size)
                for obs_ind in xrange(observation_size):
                    ref_obs_ind = int(np.squeeze(observations_positions[obs_ind]))
                    offset = 0
                    for prog_ind in xrange(mult):
                        v[obs_ind+offset: nv+offset+1] = loc_coeffs[: nv-obs_ind]
                        v[offset: offset+obs_ind] = loc_coeffs[nv-obs_ind: nv]
                        #
                        # v[ref_obs_ind+offset: nv+offset+1] = loc_coeffs[: nv-ref_obs_ind]
                        # v[offset: offset+ref_obs_ind] = loc_coeffs[nv-ref_obs_ind: nv]
                        #
                        offset += nv

                    PHT[:, obs_ind] *= v
                #
            elif num_dimensions==2:
                nx = model.model_configs['nx']
                ny = model.model_configs['ny']
                if nx != ny:
                    print("Only square domains are handled for 2D models!")
                    raise NotImplementedError
                else:
                    # Eshta: Now localize HPT for 2D models....
                    pass
                    model_grid = model.get_model_grid()
                    loc_coeff_coll = np.empty_like(PHT)
                    for obs_ind in xrange(observation_size):
                        ref_obs_ind = np.squeeze(observations_positions[obs_ind, :])
                        #
                        # Distance from this observation to all model grid points, and localization coefficients:
                        loc_distances = model_grid.copy()

                        for dim_ind in xrange(num_dimensions):
                            loc_distances[:, dim_ind] -= ref_obs_ind[dim_ind]
                        st_range = np.size(loc_distances, 0)
                        distances = np.empty(st_range)
                        for st_ind in xrange(st_range):
                            distances[st_ind] = np.linalg.norm(loc_distances[st_ind, :])
                        #
                        if np.isscalar(localization_radius):
                            loc_rad = localization_radius
                            loc_coeffs = utility.calculate_localization_coefficients(loc_rad, distances, localization_function)
                        else:
                            if localization_radius.size == distances.size:
                                loc_rad = localization_radius
                            elif localization_radius.size == observation_size and distances.size == state_size:
                                # print "localization_radius.size == observation_size and distances.size == state_size"
                                if re.match(r'\Alinear\Z', model._observation_operator_type, re.IGNORECASE):

                                    tmp_dist = model.state_vector()
                                    tmp_dist[:] = distances[:]
                                    distances = model.evaluate_theoretical_observation(tmp_dist).get_numpy_array()
                                    loc_rad = localization_radius

                                    loc_coeffs = utility.calculate_localization_coefficients(loc_rad, distances, localization_function)
                                    coeff_vec = model.observation_vector()
                                    coeff_vec[:] = loc_coeffs[:]
                                    loc_coeffs = model.observation_operator_Jacobian_T_prod_vec(None, loc_coeffs).get_numpy_array()

                                    # obs_loc_rad = model.observation_vector()
                                    # obs_loc_rad[:] = localization_radius[:]
                                    # loc_rad = model.observation_operator_Jacobian_T_prod_vec(None, obs_loc_rad).get_numpy_array()

                                else:
                                    print "Multi-Localization Radii for 2D models is only supported if you provide localization vector in the state space, or the obsevation operator is linear!"
                                    raise ValueError
                                #
                            elif localization_radius.size == state_size and distances.size == observation_size:
                                print ">> The situation 'localization_radius.size == state_size and distances.size == observation_size' is not Implemented yet!"
                                raise NotImplementedError
                            else:
                                print "This situation does not make any sense!"
                                raise ValueError

                        # loc_coeffs = utility.calculate_localization_coefficients(loc_rad, distances, localization_function)
                        #
                        PHT[:, obs_ind] *= loc_coeffs;
                        #

                        loc_coeff_coll[:, obs_ind] = loc_coeffs
                    #
                #
            else:
                print("Only up to 2D models are handled Here!")
                raise NotImplementedError
            #
        else:
            # No localization will take place. HPHT, and PHT won't be changed.
            pass

        # Now formulate K (localized or not):
        # K = PHT (HPHT + R)^{-1}
        R = model.observation_error_model.R.get_numpy_array()
        if rfactor != 1 :
            R *= rfactor
        K = PHT.dot(np.linalg.inv(HPHT+R))
        # Now return the Kalman gain matrix
        return K

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
            super(EnKF, self).print_cycle_results()
        pass  # Add more...
        #

    #
    def save_cycle_results(self, output_dir=None, cleanup_out_dir=False):
        """
        Save filtering results from the current cycle to file(s).
        Check the output directory first. If the directory does not exist, create it.

        Args:
            output_dir: full path of the directory to save results in

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
            # print('writing output directory structure to config file \n \t%s \n' % output_dir_structure_file)
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
                    self.model.write_state(state=forecast_ensemble_member.copy(),
                                            directory=cycle_states_out_dir,
                                            file_name='forecast_ensemble',
                                            append=True
                                            )
                    #
                    analysis_ensemble_member = self.filter_configs['analysis_ensemble'][ens_ind]
                    self.model.write_state(state=analysis_ensemble_member.copy(),
                                            directory=cycle_states_out_dir,
                                            file_name='analysis_ensemble',
                                            append=True
                                            )
        # save reference state
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
        if file_output_file_format in ['txt', 'ascii']:
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

            # save filter and model configurations (a copy under observation directory and another under state directory)...
            filter_configs = self.filter_configs
            filter_conf= dict(filter_name=filter_configs['filter_name'],
                              prior_distribution=filter_configs['prior_distribution'],
                              gmm_prior_settings=filter_configs['gmm_prior_settings'],
                              ensemble_size=filter_configs['ensemble_size'],
                              apply_preprocessing=filter_configs['apply_preprocessing'],
                              apply_postprocessing=filter_configs['apply_postprocessing'],
                              timespan=filter_configs['timespan'],
                              analysis_time=filter_configs['analysis_time'],
                              observation_time=filter_configs['observation_time'],
                              forecast_time=filter_configs['forecast_time'],
                              forecast_first=filter_configs['forecast_first']
                              )
            io_conf = output_configs
            #
            if self.prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
                gmm_conf = self.output_configs['filter_statistics']['gmm_prior_statistics']
                utility.write_dicts_to_config_file('setup.dat', cycle_observations_out_dir,
                                                   [filter_conf, io_conf, gmm_conf], ['Filter Configs', 'Output Configs', 'GMM-Prior Configs'])
                utility.write_dicts_to_config_file('setup.dat', cycle_states_out_dir,
                                                   [filter_conf, io_conf, gmm_conf], ['Filter Configs', 'Output Configs', 'GMM-Prior Configs'])
            else:
                utility.write_dicts_to_config_file('setup.dat', cycle_observations_out_dir,
                                                   [filter_conf, io_conf], ['Filter Configs', 'Output Configs'])
                utility.write_dicts_to_config_file('setup.dat', cycle_states_out_dir,
                                                   [filter_conf, io_conf], ['Filter Configs', 'Output Configs'])
        else:
            print("Unsupported output format: '%s' !" % file_output_file_format)
            raise ValueError()
            #

    #
    def read_cycle_results(self, output_dir, read_err_covars=False):
        """
        Read filtering results from file(s).
        Check the output directory first. If the directory does not exist, raise an IO error.
        If the directory, and files exist, Start retrieving the results properly

        Args:
            output_dir: directory where KF results are saved.
                We assume the structure is the same as the structure created by the KF implemented here
                in 'save_cycle_results'.

        Returns:
            reference_state:
            forecast_state:
            analysis_state:
            observation:
            forecast_err_covar:
            analysis_err_covar:

        """
        # TODO: To be written!
        raise NotImplementedError
        #
