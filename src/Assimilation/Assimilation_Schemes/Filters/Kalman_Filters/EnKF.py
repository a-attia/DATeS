
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
    This module contains several flavors of the Ensemble Kalman Filtering scheme.

    EnKF:
    -----
        A class implementing the stochastic ensemble Kalman Filter [Evensen 1994, Burgers et al. 1998].

    DEnKF:
    ------
        A class implementing the deterministic ensemble Kalman Filter [Sakov, P. and Bertino, L. 2010].

    LLSEnKF:
    --------
    A class implementing the local least squares ensemble Kalman Filter [Anderson 2013]


    Remarks:
    --------
        - The Prior:
            The prior distribution is assumed by default to be Gaussian which statistics are approximated based on the forecast ensemble.
            Here we add the option to use other models (such as Gaussian Mixture Models) to approximate the prior distribution.
            This can be used for example to GMM + EnKF schemes such as Cluster EnKF.

        - Localization:
            Two localization methods are implemented following the enkf-matlab implementations.
            The localization method is switched/controlled via the option "" in the filter configurations dictionary.
            The two localization methods are:
                + 'covariance_localization': involves modification of the update equations by replacing the state error covariance
                  by its element-wise product with some distance-dependent correlation matrix (Houtekamer and Mitchell, 2001; Hamill and Whitaker, 2001).
                + 'local_analysis': uses a local approximation of the forecast covariance for updating a state vector element,
                  calculated by building a local window around this element (Evensen, 2003; Anderson, 2003; Ott et al., 2004).

        - Some EnKF schemes are not supposed to be used with particular localization methods.
            For example, 'covariance_localization' can not be used with the ETKF.

    References
    ------------
    - Anderson, J. L. 2001. A local least squares framework for ensemble filtering. Mon. Wea. Rev. 131, 634-642.
    - Evensen, G. 2003. The ensemble Kalman filter: theoretical formulation and practical implementation. Ocean Dynamics 53, 343-367.
    - Hamill, T. M. and Whitaker, J. S. 2001. Distance-dependent filtering of background error covariance estimates in an ensemble Kalman filter. Mon. Wea. Rev. 129, 2776-2790.
    - Houtekamer, P. L. and Mitchell, H. L. 2001. A Sequential Ensemble Kalman Filter for Atmospheric Data Assimilation. Mon. Wea. Rev. 129, 123-137.
    - Hunt, B. R., Kostelich, E. J. and Szunyogh, I. 2007. Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. Physica D 230, 112-126.
    - Sakov, P. and Bertino, L. 2010. Relation between two common localisation methods for the EnKF. Comput. Geosci (in press).

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
    _def_local_filter_configs = dict(model=None,
                                     filter_name="EnKF",
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
        # reference_state = self.filter_configs['reference_state']
        # xf = self.filter_configs['forecast_state']
        # f_rmse = utility.calculate_rmse(reference_state, xf)
        # print("XXX, forecast state: ", f_rmse)
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
        #
        model = self.model
        state_size = model.state_size()
        observation_size = model.observation_vector_size()
        #
        # get the forecast state as the mean of the forecast ensemble. Will not be used for GMM!

        # Check if the forecast ensemble should be inflated;
        f = self.filter_configs['forecast_inflation_factor']
        forecast_ensemble = utility.inflate_ensemble(self.filter_configs['forecast_ensemble'], f, in_place=False)
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        ensemble_size = len(forecast_ensemble)

        # get the measurements vector
        observation = self.filter_configs['observation']

        #
        if all_to_numpy:

            #
            # Xf = np.empty((state_size, ensemble_size))  # forecast ensemble
            # for ens_ind in xrange(ensemble_size):
            #     Xf[:, ens_ind] = forecast_ensemble[ens_ind].get_numpy_array()
            Xf = utility.ensemble_to_np_array(forecast_ensemble, state_as_col=True, model=model)
            # del forecast_ensemble

            # observation error covariance matrix
            R = model.observation_error_model.R.get_numpy_array()
            try:
                sqrtR = model.observation_error_model.sqrtR.get_numpy_array()
            except (AttributeError, NameError, ValueError):
                sqrtR = np.linalg.cholesky(R)
            # del R

            # # get the current state of the random number generator
            # current_random_state = np.random.get_state()

            # Random observation noise:
            U = np.random.randn(observation_size, ensemble_size)

            # # Restore the state of the gaussian random number generator:
            # np.random.set_state(current_random_state)

            y = observation.get_numpy_array()
            Y = np.empty((observation_size, ensemble_size))
            # model observation:
            for ens_ind in xrange(ensemble_size):
                Y[:, ens_ind] = y + U[:, ens_ind]

            # model observations:
            Yf = np.empty((observation_size, ensemble_size))
            for ens_ind in xrange(ensemble_size):
                Yf[:, ens_ind] = model.evaluate_theoretical_observation(forecast_ensemble[ens_ind]).get_numpy_array()
            Hx = Yf.copy()

            # get means:
            xf_b = np.mean(Xf, 1)  # forecast ensemble mean
            u_b  = np.mean(U,  1)  # observation noise mean
            yf_b = np.mean(Yf, 1)  # model observation mean

            # normalized anomalies:
            for ens_ind in xrange(ensemble_size):
                Xf[:, ens_ind] -= xf_b
                Yf[:, ens_ind] -= (yf_b + U[:, ens_ind] - u_b)
            Xf *= 1.0/np.sqrt(ensemble_size-1)
            Yf *= 1.0/np.sqrt(ensemble_size-1)
            del U, u_b

            # Compute and (optionally) localize the gain matrix
            Pb = np.dot(Xf, Xf.T)

            #
            localize_covariances = self.filter_configs['localize_covariances']
            if localize_covariances:
                radius = self.filter_configs['localization_radius']
                loc_func = self.filter_configs['localization_function']
                Pb = self._cov_mat_loc(Pb, radius, loc_func)

            HP = np.empty((observation_size, state_size))
            HPHT = np.empty((observation_size, observation_size))
            tmp_state = model.state_vector()
            for st_ind in xrange(state_size):
                tmp_state[:] = Pb[:, st_ind]
                HP[:, st_ind] = self.model.observation_operator_Jacobian_prod_vec(forecast_state, tmp_state).get_numpy_array()
            for obs_ind in xrange(observation_size):
                tmp_state[:] = HP[obs_ind, :]
                HPHT[obs_ind, :] = self.model.observation_operator_Jacobian_prod_vec(forecast_state, tmp_state).get_numpy_array()
            del tmp_state

            use_lu = True

            #
            analysis_ensemble = []
            xa = self.model.state_vector()
            #
            if use_lu:
                K_lu, K_piv = lu_factor( HPHT + R )

                for ens_ind in xrange(ensemble_size):
                    s = lu_solve((K_lu, K_piv) , (Y[:, ens_ind] - Hx[:, ens_ind]))
                    s = HP.T.dot(s)
                    xa[:] = s[:] + xf_b[:]
                    analysis_ensemble.append(xa.copy())
            else:
                K = np.dot(HP.T, np.linalg.inv(HPHT + R))
                for ens_ind in xrange(ensemble_size):
                    s = K.dot(Y[:, ens_ind] - Hx[:, ens_ind])
                    xa[:] = s[:] + xf_b[:]
                    analysis_ensemble.append(xa.copy())


            analysis_state = utility.ensemble_mean(analysis_ensemble)

            # Update analysis ensemble and analysis state (average of the analysis_ensemble)
            self.filter_configs['analysis_ensemble'] = analysis_ensemble

            # Update the analysis_state in the filter_configs dictionary.
            self.filter_configs['analysis_state'] = analysis_state
            #

            #
            #
            # #
            # # PREPARE for ASSIMILATION:
            # # ---------------------------
            #
            #
            #
            #
            #
            # # Obserbation Innovation matrix:
            # D = np.random.randn(observation_size, ensemble_size)
            # d_mean = np.mean(D, 1)
            # for ens_ind in xrange(ensemble_size):
            #     D[:, ens_ind] = sqrtR.dot(D[:, ens_ind] - d_mean)
            #
            # for ens_ind in xrange(ensemble_size):
            #     D[:, ens_ind] += observation - self.model.evaluate_theoretical_observation(forecast_ensemble[ens_ind]).get_numpy_array()
            #
            # # Restore the state of the gaussian random number generator:
            # np.random.set_state(current_random_state)
            #
            #
            #
            # # Model-observation of the forecast ensemble members:
            # tmp_state = self.model.state_vector()
            # HP = np.empty((observation_size, state_size))
            # HPHT = np.empty((observation_size, observation_size))
            # for st_ind in xrange(state_size):
            #     tmp_state[:] = Pb[:, st_ind]
            #     HP[:, st_ind] = self.model.observation_operator_Jacobian_prod_vec(forecast_state, tmp_state).get_numpy_array()
            # for obs_ind in xrange(observation_size):
            #     tmp_state[:] = HP[obs_ind, :]
            #     HPHT[obs_ind, :] = self.model.observation_operator_Jacobian_prod_vec(forecast_state, tmp_state).get_numpy_array()
            #
            # K_lu, K_piv = lu_factor( HPHT + R )
            #
            # analysis_ensemble = []
            # xf_update = self.model.state_vector()
            #
            # for e_ind in xrange(ensemble_size):
            #     s = lu_solve((K_lu, K_piv) , D[:, e_ind])
            #     s = HP.T.dot(s)
            #     xf_update[:] = s[:]
            #     analysis_ensemble.append(forecast_state.add(xf_update, in_place=False))

            #
        #
        else:
            raise NotImplementedError("To be implemented!")

        f = self.filter_configs['inflation_factor']
        self.filter_configs['analysis_ensemble'] = utility.inflate_ensemble(self.filter_configs['analysis_ensemble'], f, in_place=True)


    def _cov_mat_loc(self, Pb, loc_radius, loc_func, cor_loc_dir=3):
        """
        Localize the covariance matrix via pointwise product
        """
        model = self.model
        state_size = model.state_size()

        Pb_copy = Pb.copy()


        # Naive Localization:

        if cor_loc_dir is None:
            loc_direct_approach = self.filter_configs['loc_direct_approach']
        else:
            loc_direct_approach = cor_loc_dir

        if loc_direct_approach<1 or loc_direct_approach>6:
            print("loc_direct_approach MUST be an integer from 1-6, see Attia's Notes on OED_Localization!")
            raise ValueError

        if loc_func is None:
            localization_function = self.filter_configs['localization_function']
        else:
            localization_function = loc_func
        localization_radius = loc_radius

        # number of model-grid dimensions
        try:
            num_dimensions = self.model.model_configs['num_dimensions']
        except(KeyError):
            num_dimensions = None
        # get the observational grid
        observations_positions = self.model.get_observations_positions()
        num_obs_dims = np.size(observations_positions, 1)
        #
        if num_dimensions is None:
            num_dimensions = num_obs_dims
        elif (max(num_dimensions, num_obs_dims) == 1):
            num_dimensions = num_obs_dims = 1
        else:
            pass
        #
        try:
            periodic_bc = self.model.model_configs['periodic']
        except(KeyError, NameError, AttributeError):
            periodic_bc = False

        if num_dimensions <= 1:
            try:
                model_grid = self.model.get_model_grid()
                model_grid = np.asarray(model_grid).squeeze()
            except:
                model_grid = np.arange(state_size)

            # Localization in the observation space:
            for st_ind in xrange(state_size):
                ref_st_coord = model_grid[st_ind]
                # distances = np.abs(model_grid[st_ind: ] - ref_st_coord).squeeze()
                distances = np.abs(model_grid[:] - ref_st_coord).squeeze()
                if distances.ndim == 0:
                    distances = np.array([distances.item()])

                #
                # update distances for periodic models/boundary-conditions (e.g. Lorenz 96)
                if periodic_bc:
                    rem_distances = (model_grid[-1]-model_grid[0]) +(model_grid[1]-model_grid[0])  - distances
                    up_distances = distances > rem_distances
                    distances[up_distances] = rem_distances[up_distances]
                #
                # print("distances", distances)
                if np.isscalar(localization_radius):
                    loc_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                else:
                    # loc_coeffs = utility.calculate_localization_coefficients(localization_radius[st_ind: ], distances, localization_function)
                    loc_coeffs = utility.calculate_localization_coefficients(localization_radius[st_ind], distances, localization_function)
                #

                if loc_direct_approach == 1:
                    # radius is fixed over rows of the covariance matrix:
                    Pb[st_ind, : ] *= loc_coeffs

                elif loc_direct_approach == 2:
                    # radius is fixed over rows of the covariance matrix:
                    Pb[:, st_ind ] *= loc_coeffs

                elif loc_direct_approach == 3 or loc_direct_approach == 6:
                    # print("\n\n\n\n\n >>>>>>> COOL.... <<<<<<<<\n\n\n\n\n")
                    # radius is fixed over rows of the covariance matrix:
                    #
                    if np.isscalar(localization_radius):
                        Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                        Pb[st_ind:, st_ind ] *= loc_coeffs[st_ind: ]

                    else:
                        vert_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                        Pb[st_ind:, st_ind ] *= (0.5 * (loc_coeffs[st_ind: ] + vert_coeffs[st_ind: ]))
                        Pb[st_ind, st_ind: ] *= (0.5 * (loc_coeffs[st_ind: ] + vert_coeffs[st_ind: ]))

                elif loc_direct_approach == 4:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                    Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                    Pb[st_ind: , st_ind] *= loc_coeffs[st_ind: ]

                elif loc_direct_approach == 5:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied upwards:
                    Pb[st_ind, : st_ind+1 ] *= loc_coeffs[st_ind: ]
                    Pb[ : st_ind+1, st_ind] *= loc_coeffs[st_ind: ]

                else:
                    print("loc_direct_approach MUST be an integer from 1-6, see Attia's Notes on OED_Localization!")
                    raise ValueError
                # Pb[st_ind, st_ind] = 0.9 * Pb[st_ind, st_ind] + 0.1


        elif num_dimensions==2:
            #
            # WARNING: This should not be called online for big models. Consider calculating offline, and writing to file!
            try:
                model_grid = self.model.get_model_grid()
            except:
                try:
                    nx = self.model.model_configs['nx']
                    ny = self.model.model_configs['ny']
                    dx = self.model_configs['dx']
                    dy = self.model_configs['dy']
                except:
                    nx = np.int(np.floor(np.sqrt(state_size)))
                    ny = state_size / nx
                    dx = 1.0
                    dy = 1.0
                    model_grid = np.empty((state_size, 2))

                    x_indexes = np.arange(nx) * dx
                    y_indexes = np.arange(ny) * dy
                    model_grid[:, 0] = list(x_indexes) * ny  # test if reshaping is faster!
                    model_grid[:, 1] = np.repeat(y_indexes, nx)
                    #

            for st_ind, ref_coord in enumerate(model_grid):
                # coords = model_grid[st_ind: , :]
                coords = model_grid
                distance = np.sqrt( (coords[:, 0]-ref_coord[0])**2 + (coords[:, 1]-ref_coord[1])**2 )

                if np.isscalar(localization_radius):
                    loc_coeffs = utility.calculate_localization_coefficients(radius=localization_radius,
                                                                             distances=distance,
                                                                             method=localization_function)
                else:
                    loc_coeffs = utility.calculate_localization_coefficients(radius=localization_radius[st_ind],
                                                                             distances=distance,
                                                                             method=localization_function)

                # Pb[st_ind, st_ind:] *= loc_coeffs
                # Pb[st_ind:, st_ind] *= loc_coeffs
                if loc_direct_approach == 1:
                    # radius is fixed over rows of the covariance matrix:
                    Pb[st_ind, : ] *= loc_coeffs

                elif loc_direct_approach == 2:
                    # radius is fixed over rows of the covariance matrix:
                    Pb[:, st_ind ] *= loc_coeffs

                elif loc_direct_approach == 3 or loc_direct_approach == 6:
                    # radius is fixed over rows of the covariance matrix:
                    #
                    if np.isscalar(localization_radius):
                        Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                        Pb[st_ind:, st_ind ] *= loc_coeffs[st_ind: ]

                    else:
                        vert_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                        Pb[st_ind:, st_ind ] *= (0.5 * (loc_coeffs[st_ind: ] + vert_coeffs[st_ind: ]))
                        Pb[st_ind, st_ind: ] *= (0.5 * (loc_coeffs[st_ind: ] + vert_coeffs[st_ind: ]))
                    #

                elif loc_direct_approach == 4:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                    Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                    Pb[st_ind: , st_ind] *= loc_coeffs[st_ind: ]

                elif loc_direct_approach == 5:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied upwards:
                    Pb[st_ind, : st_ind+1 ] *= loc_coeffs[st_ind: ]
                    Pb[ : st_ind+1, st_ind] *= loc_coeffs[st_ind: ]

                else:
                    print("loc_direct_approach MUST be an integer from 1-6, see Attia's Notes on OED_Localization!")
                    raise ValueError
                pass
                #
        #
        else:
            print("Only up to 2D models are handled Here!")
            raise NotImplementedError

        #
        return Pb



    #
    def _calc_Kalman_gain(self, A, HA, rfactor=1.0, cor_loc_dir=3):
        """
        Calculate and return Kalman gain.
        All matrices passed and returned from this function are Numpy-based

        Args:
            A: Forecast Ensemble Anomalies matrix [forecast ensemble members - ensemble mean]
            HA: Forecast Ensemble Observations Anomalies matrix [HE(e) - Hx], e = 1,2,..., ensemble_size
            cor_loc_dir: determins the approach taken to localize HPHT for different localization radii [1:6] methods are currently supported

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

            self._validate_localization_radii()
            localization_function = self.filter_configs['localization_function']
            orig_loc_radius = self.filter_configs['localization_radius']

            if self._verbose:
                print("Started Localization step in the observation space; ")
                print("Localization radius/radii: %s" % repr(localization_radius))

            # model_grid, and observation_grid, and dimensions; for calculating distances:
            model_grid = model.get_model_grid()
            observation_grid = model.get_observation_grid()
            num_dimensions = np.size(model_grid, 1)
            num_obs_dims = np.size(observation_grid, 1)
            #
            if num_obs_dims != num_dimensions:
                print("Observational grid dimension mismatches the model grid dimension!. \n  \
                       Observation dimensions = %d, Model state dimensions = %d" % (num_obs_dims, num_dimensions))
                raise ValueError
            else:
                num_dims = num_obs_dims

            try:
                periodic_bc = model.model_configs['periodic']
            except(KeyError, NameError, AttributeError):
                periodic_bc = False

            if periodic_bc:
                try:
                    dx = model.model_configs['dx']
                except:
                    dx = 1

            # Masking HPHT < if more than one observatin is made >
            if observation_size>1:
                if np.isscalar(orig_loc_radius):
                    localization_radius = orig_loc_radius
                    equal_radii = True
                else:
                    # make sure th localization radii are in the observations space; nearest is taken if it is in the state space
                    equal_radii = False
                    #
                    if orig_loc_radius.size == observation_size:
                        # loclaization radii are of the right size
                        localization_radius = orig_loc_radius
                        #
                    elif orig_loc_radius.size == state_size:
                        # find closes to each observation points
                        localization_radius = np.empty(observation_size)

                        for obs_ind in xrange(observation_size):
                            obs_grid_point = observation_grid[obs_ind, :]
                            dists = utility.euclidean_distance(model_grid, obs_grid_point)
                            min_dist = dists.min()
                            min_loc = np.where(dists == min_dist)[0][0]
                            rad = localization_radius[min_loc]
                            if periodic_bc:
                                # TODO: this is incorrect for more than one dimension... Raise an error, or upgrade!
                                dists = utility.euclidean_distance(model_grid-((state_size-1)*dx), obs_grid_point)
                                min_dist_b = dists.min()
                                if min_dist_b < min_dist:
                                    min_loc = np.where(dists == min_dist_b)[0][0]
                                    rad = localization_radius[min_loc]
                            localization_radius[obs_ind] = rad

                    else:
                        print("localization_radius.size", localization_radius.size)
                        print("state_size", state_size)
                        print("observation_size", observation_size)
                        print("ensemble_size", ensemble_size)

                        print("The localization radius has to be either a scalar or an iterable of dimension equal to the model state space sizeor observation size!")
                        raise ValueError
                #
                if self._verbose:
                    print("\n\n Localization Radius in Observation space:\n %s \n\n" % repr(localization_radius))

                #
                # Start Localization in the observation space:
                #
                obs_loc_operator = np.zeros((observation_size, observation_size))
                #
                if periodic_bc:
                    cir_grid = observation_grid-((state_size-1)*dx)

                # Localization in the observation space:
                for obs_ind in xrange(observation_size):
                    ref_obs_coord = observation_grid[obs_ind, :]
                    distances = utility.euclidean_distance(observation_grid, ref_obs_coord)

                    #
                    # update distances for periodic models/boundary-conditions (e.g. Lorenz 96)
                    if periodic_bc:
                        rem_distances = utility.euclidean_distance(cir_grid, ref_obs_coord)
                        for i in xrange(distances.size):
                            distances[i] = min(distances[i], rem_distances[i])

                    #
                    if equal_radii:
                        obs_loc_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                    else:
                        obs_loc_coeffs = utility.calculate_localization_coefficients(localization_radius[obs_ind], distances, localization_function)
                        #

                    if cor_loc_dir == 1:
                        # radius is fixed over rows of the covariance matrix:
                        HPHT[obs_ind, : ] *= obs_loc_coeffs

                    elif cor_loc_dir == 2:
                        # radius is fixed over rows of the covariance matrix:
                        HPHT[:, obs_ind] *= obs_loc_coeffs

                    elif cor_loc_dir == 3 or cor_loc_dir == 6:
                        # print("\n\n\n\n\n >>>>>>> COOL.... <<<<<<<<\n\n\n\n\n")
                        # radius is fixed over rows of the covariance matrix:
                        #
                        if equal_radii:
                            HPHT[obs_ind, obs_ind: ] *= obs_loc_coeffs[obs_ind: ]
                            HPHT[obs_ind:, obs_ind ] *= obs_loc_coeffs[obs_ind: ]
                        else:
                            vert_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                            fac = 0.5 * (obs_loc_coeffs[obs_ind: ] + vert_coeffs[obs_ind: ])
                            HPHT[obs_ind:, obs_ind] *= fac
                            HPHT[obs_ind, obs_ind: ] *= fac

                    elif cor_loc_dir == 4:
                        # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                        HPHT[obs_ind, obs_ind: ] *= obs_loc_coeffs[obs_ind: ]
                        HPHT[obs_ind: , obs_ind] *= obs_loc_coeffs[obs_ind: ]

                    elif cor_loc_dir == 5:
                        # radius is fixed over rows, and columnsn of the covariance matrix, and are varied upwards:
                        HPHT[obs_ind, : obs_ind+1 ] *= obs_loc_coeffs[obs_ind: ]
                        HPHT[ : obs_ind+1, obs_ind] *= obs_loc_coeffs[obs_ind: ]

                    else:
                        print("cor_loc_dir MUST be an integer from 1-6, see Attia's Notes on OED_Localization!")
                        raise ValueError

            #
            # Mask PHT:
            #
            if periodic_bc:
                cir_grid = model_grid - ((state_size-1)*dx)

            # loclaization of PHT by columns
            for obs_ind in xrange(observation_size):
                #
                ref_obs_coord = observation_grid[obs_ind, :]
                distances = utility.euclidean_distance(model_grid, ref_obs_coord)
                #
                # update distances for periodic models/boundary-conditions (e.g. Lorenz 96)
                if periodic_bc:
                    rem_distances = utility.euclidean_distance(cir_grid, ref_obs_coord)
                    for i in xrange(distances.size):
                        distances[i] = min(distances[i], rem_distances[i])

                #
                if equal_radii:
                    loc_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                else:
                    loc_coeffs = utility.calculate_localization_coefficients(localization_radius[obs_ind], distances, localization_function)

                PHT[:, obs_ind] *= loc_coeffs
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

    def _validate_localization_radii(self):
        """
        """
        model = self.filter_configs['model']
        state_size = model.state_size()
        observation_size = model.observation_size()

        loc_radius = self.filter_configs['localization_radius']
        if utility.isscalar(loc_radius):
            pass
        elif utility.isiterable(loc_radius):
            loc_radius = np.asarray([l for l in loc_radius]).squeeze().flatten()
            if loc_radius.size == 1:
                loc_radius = loc_radius[0]
            elif loc_radius.size == state_size:
                pass
            elif loc_radius.size == observation_size:
                pass
            else:
                print("Localization radius is an iterable of the wrong size!")
                print("Localizaiton radius size: %d" % loc_radius.size)
                print("State dimension: %d" % state_size)
                print("Observation dimension: %d" % observation_size)
                raise ValueError
                #
        self.filter_configs.update({'localization_radius': loc_radius})  # is it necessary!
        #  Done... Return None...

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
                filter_name = self.filter_configs['filter_name']
                header = "RMSE Results: Filter: '%s' \n %s \t %s \t %s \t %s \t %s \n" % (filter_name,
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

#
#
class DEnKF(EnKF):
    """
    Deterministic Ensemble-Kalman filtering with. This mainly differs from the class 'EnKF' in the analysis step; everything else is identical


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
    _def_local_filter_configs = EnKF._def_local_filter_configs.copy()
    _local_def_output_configs = EnKF._local_def_output_configs.copy()
    _supported_prior_distribution = EnKF._supported_prior_distribution
    #

    def __init__(self, filter_configs=None, output_configs=None):

        DEnKF._def_local_filter_configs.update(dict(filter_name="DEnKF"))
        DEnKF._local_def_output_configs.update(dict(file_output_file_name_prefix='DEnKF_results'))

        # aggregate configurations, and attach filter_configs, output_configs to the filter object.
        filter_configs = utility.aggregate_configurations(filter_configs, DEnKF._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, DEnKF._local_def_output_configs)

        #
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(filter_configs=filter_configs, output_configs=output_configs)
        else:
            # old-stype class
            super(DEnKF, self).__init__(filter_configs=filter_configs, output_configs=output_configs)

        # EnKF.__init__(self, filter_configs, output_configs)
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
        forecast_ensemble = self.filter_configs['forecast_ensemble']
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        forecast_state_np = forecast_state.get_numpy_array()

        # Check if the forecast ensemble should be inflated;
        f = self.filter_configs['forecast_inflation_factor']

        # print("Inflating with: ", f)
        # print("Beofre Inflation, X0: ", forecast_ensemble[0])

        utility.inflate_ensemble(forecast_ensemble, f, in_place=True)
        # print("After Inflation, X0: ", self.filter_configs['forecast_ensemble'][0])


        #
        if all_to_numpy:
            # get the forecast state as the mean of the forecast ensemble. Will not be used for GMM!
            state_size = self.model.state_size()
            try:
                observation_size = self.observation_size
            except(NameError, AttributeError):
                observation_size = self.model.observation_vector_size()
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
                forecast_ensemble_np[:, ens_ind] = np.squeeze(forecast_ensemble[ens_ind].get_numpy_array())

            # get the measurements vector
            observation = self.filter_configs['observation'].get_numpy_array()

            #
            # PREPARE for ASSIMILATION:
            # ---------------------------
            # Model-observation of the forecast ensemble members:
            HE = np.empty((observation_size, ensemble_size))
            for ens_ind in xrange(ensemble_size):
                obs_vec = self.model.evaluate_theoretical_observation(forecast_ensemble[ens_ind])
                HE[:, ens_ind] = obs_vec.get_numpy_array()

            # Mean of forecasted observations:
            Hx = np.squeeze(np.mean(HE, 1))

            # Observation innovations:
            # obs_innovations = observation - Hx
            obs_innovations = observation - self.model.evaluate_theoretical_observation(forecast_state).get_numpy_array()

            if self._verbose:
                print('Maximum observation-innovation magnitude = %f' % np.abs(obs_innovations).max())

            # observation covariance scaling factor
            rfactor = float(self.filter_configs['obs_covariance_scaling_factor'])

            # Forecast Ensemble Anomalies matrix [forecast ensemble members - ensemble mean]
            A = forecast_ensemble_np.copy()
            for ens_ind in xrange(ensemble_size):
                A[:, ens_ind] -= forecast_state_np

            # Forecast Ensemble Observations Anomalies matrix [HE(e) - Hx], e = 1,2,..., ensemble_size
            HA = HE.copy()  # should reuse --> in-place
            for ens_ind in xrange(ensemble_size):
                HA[:, ens_ind] -= Hx

            #
            # START ASSIMILATION:
            # ---------------------------

            # standardised innovation and ensemble anomalies
            # sqrtR_lu, sqrtR_piv = lu_factor( sqrtR )
            # s = obs_innovations / np.sqrt(ensemble_size-1.0)
            # s = lu_solve((sqrtR_lu, sqrtR_piv) , s)
            # S = np.empty_like(HA)
            # for ens_ind in xrange(ensemble_size):
            #     S[:, ens_ind] = (lu_solve((sqrtR_lu, sqrtR_piv) , HA[:, ens_ind])) / np.sqrt(ensemble_size-1.0)

            sqrtR_inv = np.linalg.inv(sqrtR)
            s = sqrtR_inv.dot(obs_innovations) / np.sqrt(ensemble_size-1.0)
            S = sqrtR_inv.dot(HA) / np.sqrt(ensemble_size-1.0)

            # Analysis is carried out based on the tpe of localization in what follows:
            #
            localize_covariances = self.filter_configs['localize_covariances']
            if not localize_covariances:
                # Global analysis (No Localization):
                if ensemble_size <= observation_size:
                    # Calculte G = (I + (S^T * S))^{-1} * S^T
                    G = np.dot(S.T, S)
                    G[np.diag_indices_from(G)] += 1.0
                    G = np.dot(np.linalg.inv(G), S.T)
                else:
                    # Calculte G = S^T * (I + S * S^T)^{-1}
                    G = np.dot(S, S.T)
                    G[np.diag_indices_from(G)] += 1.0
                    G = np.dot(S.T, np.linalg.inv(G))

                # Evaluate the Ensemble-Mean update:
                ens_mean_update = np.dot(A.dot(G), s)  # dx

                # Evaluate Ensemble-Anomalies update:
                if rfactor != 1.0:
                    # rescale S, and G
                    S *= 1.0 / np.sqrt(rfactor)
                    if ensemble_size <= observation_size:
                        # RE-Calculte G = (I + (S^T * S))^{-1} * S^T
                        G = np.dot(S.T, S)
                        G[np.diag_indices_from(G)] += 1.0
                        G = np.dot(np.linalg.inv(G), S.T)
                    else:
                        # RE-Calculte G = S^T * (I + S * S^T)^{-1}
                        G = np.dot(S, S.T)
                        G[np.diag_indices_from(G)] += 1.0
                        G = np.dot(S.T, np.linalg.inv(G))
                else:
                    pass
                # Now Evaluate A = A * (I - 0.5 * G * S):
                T_R = -0.5 * G.dot(S)
                T_R[np.diag_indices_from(T_R)] += 1.0
                A = A.dot(T_R)

            else:
                # Apply Localization based on the localization function:
                localization_method = self.filter_configs['localization_method']
                # localization_function = self.filter_configs['localization_function']
                # localization_radius = self.filter_configs['localization_radius']
                #
                if re.match(r'\Acovariance(-|_)*filtering\Z', localization_method, re.IGNORECASE):
                    # Evaluate the Kalman gain matrix (with HPH^T, and PH^T localized based on the filter settings)
                    K = self._calc_Kalman_gain(A, HA)

                    # Evaluate the Ensemble-Mean update (dx):
                    ens_mean_update = K.dot(obs_innovations)

                    # Recalculate the Kalman gain with observation variances/covariances multiplied by rfactor
                    if rfactor != 1:
                        K = self._calc_Kalman_gain(A, HA, rfactor)

                    # Now Evaluate A = A + K * (D - HA):
                    A -= (0.5 * K.dot(HA))

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
                    #     A(i, :) = A(i, :) - A(i, :) * 0.5 * Gloc * Sloc;
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

            #
            # Now we are good to go; update the ensemble mean, and ensemble-anomalies using ens_mean_update, and A
            analysis_mean_np = forecast_state_np + ens_mean_update
            analysis_state = self.model.state_vector()
            analysis_state[:] = analysis_mean_np
            #
            analysis_ensemble = []
            for ens_ind in xrange(ensemble_size):
                analysis_ens_member = self.model.state_vector()
                analysis_ens_member[:] = np.squeeze(A[:, ens_ind]) + analysis_mean_np
                analysis_ensemble.append(analysis_ens_member)

            # Ensemble Rotation:

            # Update analysis ensemble and analysis state (average of the analysis_ensemble)
            self.filter_configs['analysis_ensemble'] = analysis_ensemble

            # Update the analysis_state in the filter_configs dictionary.
            self.filter_configs['analysis_state'] = analysis_state.copy()
            tes = self.filter_configs['analysis_state'].get_numpy_array()

            #
        #
        else:
            raise NotImplementedError("To be implemented!")

#
#
class ETKF(EnKF):
    """
    A class implementing the ensemble transform Kalman Filter. This mainly differs from the class 'EnKF' in the analysis step; everything else is identical

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
    _def_local_filter_configs = EnKF._def_local_filter_configs.copy()
    _local_def_output_configs = EnKF._local_def_output_configs.copy()
    #
    _supported_prior_distribution = EnKF._supported_prior_distribution
    #

    def __init__(self, filter_configs=None, output_configs=None):

        ETKF._def_local_filter_configs.update(dict(filter_name="ETKF"))
        ETKF._local_def_output_configs.update(dict(file_output_file_name_prefix='ETKF_results'))
        #
        # aggregate configurations, and attach filter_configs, output_configs to the filter object.
        filter_configs = utility.aggregate_configurations(filter_configs, ETKF._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, ETKF._local_def_output_configs)

        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(filter_configs=filter_configs, output_configs=output_configs)
        else:
            # old-stype class
            super(ETKF, self).__init__(filter_configs=filter_configs, output_configs=output_configs)

        # EnKF.__init__(self, filter_configs, output_configs)
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

            # observation covariance scaling factor
            rfactor = float(self.filter_configs['obs_covariance_scaling_factor'])

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
            #
            sqrtR_inv = np.linalg.inv( sqrtR )
            s = sqrtR_inv.dot(obs_innovations) / np.sqrt(ensemble_size-1)
            S = sqrtR_inv.dot(HA) / np.sqrt(ensemble_size-1)

            # Analysis is carried out based on the tpe of localization in what follows:
            #
            localize_covariances = self.filter_configs['localize_covariances']
            if not localize_covariances:
                # Global analysis (No Localization):
                M = np.dot(S.T, S)
                M[np.diag_indices_from(M)] += 1.0
                G = np.dot(np.linalg.inv(M), S.T)

                # Evaluate the Ensemble-Mean update:
                ens_mean_update = np.dot(np.dot(A, G), s)  # dx

                # Evaluate Ensemble-Anomalies update:
                if rfactor != 1.0:
                    # rescale S, and G
                    S *= 1.0 / np.sqrt(rfactor)
                    M = np.dot(S.T, S)
                    M[np.diag_indices_from(M)] += 1.0
                    G = np.dot(np.linalg.inv(M), S.T)

                else:
                    pass
                # Now Evaluate A = A * (I - 0.5 * G * S):
                A = np.dot(A, scipy.linalg.sqrtm(M))

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

                    # Now Evaluate A = A + K * (D - HA):
                    A -= 0.5 * K.dot(HA)

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
                    #     M = inv(speye(m) + Sloc' * Sloc);
                    #     Gloc = M * Sloc';
                    #
                    #     dx(i) = A(i, :) * Gloc * (s(localobs) .* coeffs);
                    #
                    #     if rfactor ~= 1
                    #         Sloc = Sloc / sqrt(rfactor);
                    #         M = inv(speye(m) + Sloc' * Sloc);
                    #         Dloc = D(localobs, :) .* repmat(coeffs, 1, m);
                    #         A(i, :) = A(i, :) + A(i, :) * Gloc * (Dloc - Sloc);
                    #     end
                    #
                    #     A(i, :) = A(i, :) - A(i, :) * 0.5 * Gloc * Sloc;
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
            self.filter_configs['analysis_state'] = analysis_state.copy()
            #
        #
        else:
            raise NotImplementedError("To be implemented!")

#
#
class LLSEnKF(FiltersBase):
    """
    A class implementing the local least squares ensemble Kalman filter (Anderson 2013).
    Both Stochastic (Purturbed-Observation) and deterministic (Ensemble Transform / Square Root) versions are possible.
    In this implementation, During the first phase of the implementation, we will ignore the pre and post processing,
    and assume the observational error matrix is diagonal.
    Local least squares implementations of Kalman filter.
    Reference:  @article{anderson2003local,
                         title={A local least squares framework for ensemble filtering},
                         author={Anderson, Jeffrey L},
                         journal={Monthly Weather Review},
                         volume={131},
                         number={4},
                         pages={634--642},
                         year={2003}
                         }

    Args:
        model_object: a reference to the model object. Error statistics are loaded from the model object.
                      Forecast is carried out by calling time integrator attached to the model object.
        filter_type: this has to be either 'stochastic' or 'deterministic'
                        - 'stochastic' (or 'perturbed_observations') will lead to the stochastic
                                       (perturbed observation) version of the filter implementation.
                        - 'deterministic' (or 'square_root' or 'sqrt') will lead to the deterministic
                                       (square-root) version of the filter implementation.
                        THIS IS MOVED TO THE CONFIGURATIONS DICTIONARY...
        filter_configs: a dictionary containing filter configurations.

    """
    _def_local_filter_configs = dict(filter_name="LLS-EnKF",
                                     ensemble_size=None,
                                     analysis_ensemble=None,
                                     analysis_state=None,
                                     forecast_ensemble=None,
                                     forecast_state=None,
                                     filter_type='stochastic',
                                     regression_model_type='global',
                                     regression_model_radius=np.infty,
                                     rank_observation_with_state=True,
                                     refernce_state=None,
                                     filter_statistics=dict(forecast_rmse=None,
                                                            analysis_rmse=None,
                                                            rejection_rate=None
                                                            )
                                     )
    _local_def_output_configs = dict(scr_output=True,
                                     file_output=False,
                                     file_output_moment_only=True,
                                     file_output_moment_name='mean',
                                     file_output_directory='Assimilation_Results',
                                     files_output_file_name_prefix='KF_results',
                                     files_output_file_format='text',
                                     files_output_separate_files=True
                                     )

    def __init__(self, filter_configs=None, output_configs=None):

        # aggregate configurations, and attach filter_configs, output_configs to the filter object.
        filter_configs = utility.aggregate_configurations(filter_configs, LLSEnKF._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, LLSEnKF._local_def_output_configs)

        FiltersBase.__init__(self, filter_configs=filter_configs, output_configs=output_configs)
    #
        # the following configuration are filter-specific.
        self._filter_type = self.filter_configs['filter_type']
        self._regression_model_type = self.filter_configs['regression_model_type']
        self._regression_model_radius = self.filter_configs['regression_model_radius']
        self._rank_observation_with_state = self.filter_configs['rank_observation_with_state']

        #
        if self._filter_type.lower() in ['deterministic', 'square_root', 'square-root', 'sqrt', 'eakf']:
            self._filter_type = 'deterministic'
        elif self._filter_type.lower() in ['stochastic', 'perturbed-obs', 'perturbed-observations',
                                           'perturbed_obs', 'perturbed_observations', 'enkf']:
            self._filter_type = 'stochastic'

        if self._regression_model_type.lower() != 'global' and self._regression_model_radius == np.infty:
            self._regression_model_type = 'global'
        elif self._regression_model_type.lower() != 'global' and self._regression_model_radius < np.infty:
            self._regression_model_type = 'local'

    def filtering_cycle(self, checkpoints=None, obs=None):
        """
        Apply the filtering step. Forecast, then Analysis...
        All arguments are accessed and updated in the configurations dictionary.
        """
        FiltersBase.filtering_cycle(self)

    def forecast(self):
        """
        Forecast step: propagate each ensemble member to the end of the given checkpoints to produce and ensemble of
                       forecasts. Filter configurations dictionary is updated with the new results.
        """
        # generate the forecast states
        analysis_ensemble = self.filter_configs['analysis_ensemble']
        timespan = self.filter_configs['timespan']
        forecast_ensemble = utility.propagate_ensemble(ensemble=analysis_ensemble,
                                                       model=self.filter_configs['model'],
                                                       checkpoints=timespan,
                                                       in_place=False)
        self.filter_configs['forecast_ensemble'] = forecast_ensemble
        forecast_state = utility.ensemble_mean(self.filter_configs['forecast_ensemble'])
        self.filter_configs['forecast_state'] = forecast_state
        # Add more functionality after building the forecast ensemble if desired!
        #
        # # generate the forecast state
        # analysis_ensemble = self.filter_configs['analysis_ensemble']
        # if analysis_ensemble is None or len(analysis_ensemble)<1:
        #     raise ValueError("Either no analysis ensemble is initialized or it is an empty list!")
        # else:
        #     time_span = self._checkpoints
        #     # print('time_span', time_span)
        #     self.filter_configs['forecast_ensemble'] = []
        #     for ens_member in analysis_ensemble:
        #         # print('ens_member', ens_member)
        #         trajectory = model.integrate_state(initial_state=ens_member, checkpoints=time_span)
        #         # print('forecast_state', trajectory[-1])
        #         self.filter_configs['forecast_ensemble'].append(trajectory[-1].copy())
        #
        # # print('analysis_ensemble', self.filter_configs['analysis_ensemble'])
        # # print('forecast_ensemble', self.filter_configs['forecast_ensemble'])

    def analysis(self):
        """
        Analysis step:
        """
        #
        # Check if the forecast ensemble should be inflated;
        f = self.filter_configs['forecast_inflation_factor']
        utility.inflate_ensemble(self.filter_configs['forecast_ensemble'], f)

        #
        # Check the filter type.
        filter_type = self._filter_type
        model = self.filter_configs['model']
        state_size = model._state_size
        obs_vec_size = model._observation_vector_size

        observation_vector = self.filter_configs['observation']  # get the measurements vector (y^o)
        forecast_ensemble = self.filter_configs['forecast_ensemble']
        #
        # this is how list of objects should be copied...
        analysis_ensemble = [member.copy() for member in forecast_ensemble]

        ensemble_size = self.filter_configs['ensemble_size']
        forecast_state = utility.ensemble_mean(forecast_ensemble)

        # Check the regression model type.
        regression_model_type = self._regression_model_type
        #
        # evaluate the ensemble of background observations and use it calculate prior observation variances (sigma_k,k)
        background_observation_ensemble = np.empty((obs_vec_size, ensemble_size))
        for ens_ind in xrange(ensemble_size):
            ens_member = forecast_ensemble[ens_ind]
            background_observation_ensemble[:, ens_ind] = (model.evaluate_theoretical_observation(
                ens_member)).get_raw_vector_ref()
        # Evaluate observations background/prior variances:
        obs_cov_b = np.var(background_observation_ensemble, axis=1, dtype=np.float64)

        if filter_type == 'stochastic':
            #
            # if the regression model is global, build it using the whole information coming from the joint prior:
            if regression_model_type.lower() == 'global':
                influence_radius = model._state_size
            elif regression_model_type.lower() == 'local':
                influence_radius = self._regression_model_radius
            else:
                raise ValueError("Unrecognized regression model!")

            # get the diagonal of R (this should be moved to pre-processing step later
            R_diag = model.observation_error_model.R.diag()

            # loop over each entry of the observation vector
            for j in xrange(obs_vec_size):
                y_o = observation_vector[j]  # scalar observation in turn
                r = R_diag[j]  # observation error variance corresponding to current scalar observation
                sig_yy_b = obs_cov_b[j]  # background observation variance
                # updated observation covariance:
                sig_yy_u = (r * sig_yy_b) / (r + sig_yy_b)

                # generate perturbed-observations from y_o from N(y_o, r)
                randn = utility.mvn_rand_vec(ensemble_size) * np.sqrt(r)
                randn = randn - np.mean(randn)
                perturbed_observations = randn + y_o

                # rank observations to avoid large updates (see page 639 in Anderson's paper)
                # The branching will be updated after debugging...
                if self._rank_observation_with_state:
                    # use each observation ensemble member to update the associated state vector
                    # get a list of indexes of state entries to update based on the radius of influence:
                    neighbors_indexes_list = model.get_neighbors_indexes(index=j,
                                                                               radius=influence_radius,
                                                                               source_index='observation_vector')
                    # Rank observation increments
                    perturbed_observations.sort()
                    background_observation_ensemble.sort()
                    for e in xrange(ensemble_size):
                        y_o_e = perturbed_observations[e]  # measured observation
                        y_b_e = background_observation_ensemble[j, e]  # background observation
                        # evaluate updated observation
                        y_u_e = sig_yy_u * ((1/sig_yy_b)*y_b_e + (1/r)*y_o_e)

                        del_y = y_u_e - y_b_e  # observation increment
                        # print('del_y=', del_y)
                        # propagate observation increment to state components
                        for state_ind in neighbors_indexes_list:
                            # calculate the prior covariances (get state_ind entry from all forecast ensemble members)
                            x_list = []
                            for i in xrange(ensemble_size):
                                x_list.append(forecast_ensemble[e][i])
                            # print('x_list', x_list)
                            # print('background_observation_ensemble[state_ind, :]', background_observation_ensemble[state_ind, :])
                            cov = self.two_var_covariance(x_list, background_observation_ensemble[state_ind, :])
                            # print('cov', cov, 'sig_yy_b', sig_yy_b)
                            fac = cov/sig_yy_b  # ratio of covariances coming from the regression model...
                            del_x = fac * del_y
                            # print('del_y', del_y, 'del_x', del_x)
                            # print('analysis_ensemble[%3d][%3d]: del_x =%+5.3e --> x_u=%+9.7e'
                            #       % (e, state_ind, del_x, analysis_ensemble[e][state_ind]))
                            analysis_ensemble[e][state_ind] += del_x  # check the corresponding decoration method
                            # print('analysis_ensemble[%3d][%3d]: del_x =%+5.3e --> x_u=%+9.7e'
                            #       % (e, state_ind, del_x, analysis_ensemble[e][state_ind]))

                else:
                    # use each observation ensemble member to update the associated state vector
                    # get a list of indexes of state entries to update based on the radius of influence:
                    neighbors_indexes_list = model.get_neighbors_indexes(index=j,
                                                                               radius=influence_radius,
                                                                               source_index='observation_vector')
                    for e in xrange(ensemble_size):
                        y_o_e = perturbed_observations[e]  # measured observation
                        y_b_e = background_observation_ensemble[j, e]  # background observation
                        # evaluate updated observation
                        y_u_e = sig_yy_u * ((1/sig_yy_b)*y_b_e + (1/r)*y_o_e)

                        del_y = y_u_e - y_b_e  # observation increment
                        # print('del_y=', del_y)
                        # propagate observation increment to state components
                        for state_ind in neighbors_indexes_list:
                            # calculate the prior covariances (get state_ind entry from all forecast ensemble members)
                            x_list = []
                            for i in xrange(ensemble_size):
                                x_list.append(forecast_ensemble[e][i])
                            # print('x_list', x_list)
                            # print('background_observation_ensemble[state_ind, :]', background_observation_ensemble[state_ind, :])
                            cov = self.two_var_covariance(x_list, background_observation_ensemble[state_ind, :])
                            # print('cov', cov, 'sig_yy_b', sig_yy_b)
                            fac = cov/sig_yy_b  # ratio of covariances coming from the regression model...
                            del_x = fac * del_y
                            # print('del_y', del_y, 'del_x', del_x)
                            # print('analysis_ensemble[%3d][%3d]: del_x =%+5.3e --> x_u=%+9.7e'
                            #       % (e, state_ind, del_x, analysis_ensemble[e][state_ind]))
                            analysis_ensemble[e][state_ind] += del_x  # check the corresponding decoration method
                            # print('analysis_ensemble[%3d][%3d]: del_x =%+5.3e --> x_u=%+9.7e'
                            #       % (e, state_ind, del_x, analysis_ensemble[e][state_ind]))

            self.filter_configs['analysis_ensemble'] = analysis_ensemble
            self.filter_configs['analysis_state'] = utility.ensemble_mean(self.filter_configs['analysis_ensemble'])
            # print('forecast_ensemble', self.filter_configs['forecast_ensemble'])
            # print('analysis_ensemble', self.filter_configs['analysis_ensemble'])
            # Check the radius of influence.
            #
        elif filter_type == 'deterministic':
            pass
        else:
            raise ValueError("Unrecognized filter type!")


    def cycle_preprocessing(self):
        """
        PreProcessing on the passed data before applying the data assimilation filter cycle.
        Applied if needed based on the passed options in the configurations dictionary...
        """
        # Create singular vectors matrix (obs_err_cov_singular_vectors) of the observation error covariance matrix if
        # not available.
        # Project ensemble members to the space spanned by the columns of obs_err_cov_singular_vectors.
        model = self.filter_configs['model']
        try:
            U = self.obs_err_cov_singular_vectors
        except:
            self.obs_err_cov_singular_vectors, s, _ = model.observation_error_model.R.svd()
            if s.size != model._observation_vector_size:
                raise ValueError("Observation error covariance matrix is not full-rank!")
            else:
                U = self.obs_err_cov_singular_vectors
        # project the prior (joint-state) ensemble onto the singular vector (columns of U).
        raise NotImplementedError()

    def cycle_postprocessing(self):
        """
        PostProcessing on the passed data before applying the data assimilation filter cycle.
        Applied if needed based on the passed options in the configurations dictionary...
        """
        # Project updated ensemble members back to the full space using obs_err_cov_singular_vectors.
        raise NotImplementedError()

    def save_cycle_results(self, full_out_dir=None, relative_out_dir=None, separate_files=None):
        """
        Save filtering results from the current cycle to file(s).
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        Input:
            out_dir: directory to put results in. The directory
        """
        raise NotImplementedError

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

    def print_cycle_results(self):
        """
        Print filtering results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        """
        FiltersBase.print_cycle_results(self)


    @staticmethod
    def two_var_covariance(list1, list2):
        """
        Given two iterables, calculate the covariance of the two variables which samples are stored in these iterables.
        """
        ens1 = np.asarray(list1)
        avg1 = np.mean(ens1)
        ens2 = np.asarray(list2)
        avg2 = np.mean(ens2)

        n = ens1.size
        cov = (1.0/(n-1)) * np.sum((ens1-avg1)*(ens2-avg2))
        return cov
