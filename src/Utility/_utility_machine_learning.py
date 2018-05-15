
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
    A module providing utility functions required to handle machine learning related operations.
"""


import sys
import numpy as np

from sklearn.mixture import GMM as GMM_Model
from sklearn.mixture import VBGMM as VBGMM_Model

# from _utility_file_IO import *

# All utility functions using Machine Learning algorithms should be listed here.
#
_supported_clustering_models = ['gmm', 'vbgmm', 'vb_gmm', 'vb-gmm']


#
def generate_gmm_model_info(ensemble,
                            clustering_model='gmm',
                            cov_type=None,
                            inf_criteria='aic',
                            number_of_components=None,
                            min_number_of_components=None,
                            max_number_of_components=None,
                            min_number_of_points_per_component=1,
                            invert_uncertainty_param=False,
                            verbose=False
                            ):
    """
    Build the best Gaussian mixture model fitting to an ensemble of states.
    
    Args:
        ensemble: A two dimensional numpy array with each row representing an ensemble member
        clustering_model:
        cov_type:
        inf_criteria:
        number_of_components:
        min_number_of_components:
        max_number_of_components:
        min_number_of_points_per_component:
        invert_uncertainty_param:
    
    Returns:
        optimal_model:
        converged:
        lables: 
        weights: 
        means: 
        covariances: 
        precisions: 
        optimal_covar_type:
        
    """
    # Make sure ensemble is an a two dimensional numpy array with each row representing an ensemble member
    assert isinstance(ensemble, np.ndarray)
    ensemble_size = int(np.size(ensemble, 0))  # just in case!
    #
    if isinstance(cov_type, str):
        if cov_type.lower() == 'best':
            cov_type = None
    # default maximum number of mixture components (half the ensemble size) if it is not passed as argument
    if max_number_of_components is None or max_number_of_components >= ensemble_size:
        max_number_of_components = ensemble_size

    if min_number_of_components is None:
        min_number_of_components = 1

    #
    clustering_model = clustering_model.lower()
    if clustering_model not in _supported_clustering_models:
        raise ValueError("The clustering model passed is not supported."
                         "Passed [%s]."
                         "Supported[%s]" % (clustering_model, str(_supported_clustering_models)))

    # print(" Clustering Process of %4d ensemble members started..." %ensemble_size ) ,
    # sys.stdout.flush()
    # print('\r'+' '.rjust(50)+'\r'),

    #
    if number_of_components is None:
        #
        #  information criteria (aic,bic). value: {the lower the better}
        optimal_inf_criterion = np.infty

        # Loop over all number of components to find the one with optimal information criterion value
        for tmp_num_comp in range(min_number_of_components, max_number_of_components+1):
            #
            # cluster and return the cluster model and the number of components.
            # Either save all or just add one more clustering step for optimal number of components (storage vs speed).
            if clustering_model == 'gmm':
                temp_model, temp_criterion_value, temp_covar_type = GMM_clustering(ensemble,
                                                                                   tmp_num_comp,
                                                                                   cov_type,
                                                                                   inf_criteria
                                                                                   )
            elif clustering_model in ['vbgmm', 'vb_gmm', 'vb-gmm']:
                temp_model, temp_criterion_value, temp_covar_type = VBGMM_clustering(ensemble,
                                                                                     tmp_num_comp,
                                                                                     cov_type,
                                                                                     inf_criteria
                                                                                     )
            else:
                raise ValueError("\n\tclustering algorithm ["+clustering_model+"] is not implemented!")

            # compare criterion to previously evaluated
            if temp_criterion_value < optimal_inf_criterion:
                optimal_inf_criterion = temp_criterion_value
                optimal_model = temp_model
                optimal_covar_type = temp_covar_type

        # Prediction to find the lables and hyper-parameters of the GM model
        converged = optimal_model.converged_
        lables = optimal_model.predict(ensemble)  # labels start at  'zero'
        weights = np.round(optimal_model.weights_, 3)
        means = optimal_model.means_

        if clustering_model == 'gmm':
            covariances = optimal_model.covars_
            precisions = None
        elif clustering_model in ['vbgmm', 'vb_gmm', 'vb-gmm']:
            covariances = None
            precisions = optimal_model.precs_
        else:
            # shouldn't be reached at all
            raise ValueError("\n\tclustering algorithm ["+clustering_model+"] is not implemented!")

    else:
        # The number of components is already passed. No loop for finding information criteria...
        #
        if clustering_model == 'gmm':
            optimal_model , optimal_inf_criterion , temp_covar_type = GMM_clustering(ensemble,
                                                                                     number_of_components,
                                                                                     cov_type,
                                                                                     inf_criteria
                                                                                     )
        elif clustering_model in ['vbgmm', 'vb_gmm', 'vb-gmm']:
            optimal_model , optimal_inf_criterion , temp_covar_type = VBGMM_clustering(ensemble, number_of_components, cov_type, inf_criteria)
        else:
            raise ValueError("\n\tclustering algorithm ["+clustering_model+"] is not implemented!")

        #
        # Prediction to find the lables and hyper-parameters of the GM model
        converged = optimal_model.converged_
        lables = optimal_model.predict(ensemble)  # labels start at  'zero'
        weights = np.round(optimal_model.weights_, 5)
        means = optimal_model.means_
        optimal_covar_type = temp_covar_type

        if clustering_model == 'gmm':
            covariances = optimal_model.covars_
            precisions = None
            #
        elif clustering_model in ['vbgmm','vb_gmm']:
            precisions = optimal_model.precs_
            covariances = None
            #
        else:
            # shouldn't be reached at all
            raise ValueError("\n\tclustering algorithm ["+clustering_model+"] is not implemented!")
            
    # ---------------------------- ================================================== ----------------------------
    # Now all the GMM parameters are found appropriately except for the inverse of the uncertainty parameter.
    # ---------------------------- ================================================== ----------------------------
    # Now we need to check the number of points (lables) in each mixture component, 
    # and make sure the number of elements per component is at least equal to min_number_of_points_per_component
    # If any of the components covers fewer elements, the number of components is decreased by one, and recurrence
    # is initiated. Since lables are non zero contiguous numbers, we can use the following counter efficiently:
    # ------------------------------------------------------------------------------------------------------------
    gmm_number_of_components = len(weights)
    #
    if verbose and  (gmm_number_of_components != np.size(means, 0)):
        print("This is odd! Number of component's means=%d, while length of weights=%d. They should match!" % (gmm_number_of_components, np.size(means, 0)))
    if verbose and  (gmm_number_of_components != optimal_model.n_components):
        print("Caution: Number of component's from model object=%d, while length of weights=%d. They should match!" % (optimal_model.n_components, gmm_number_of_components))
    #
    lables_counts = np.bincount(lables, minlength=gmm_number_of_components)  # minlenght guarantee that the bincounter returns array of length gmm_number_of_components
    num_zero_assignments = gmm_number_of_components - np.count_nonzero(lables_counts)  # number of components where None of the data points (hard) predicted into
    if num_zero_assignments > 0:
        min_elements_count = 0
        num_comp_to_reduce = num_zero_assignments
    else:
        min_elements_count = min(lables_counts)
        num_comp_to_reduce = 1  # just reduce the number of components by one (iteratively)
    
    # Check the number of components against the bounds on number of components:
    if gmm_number_of_components > max_number_of_components:
        # Case 2: gmm_number of components is greater than the maximum number of components allowed; decrease:
        new_number_of_components = gmm_number_of_components - num_comp_to_reduce
        if verbose:
            print("Number of GMM components is %d. Number of components is MORE than the maximum allowed of %d" \
                            % (gmm_number_of_components, max_number_of_components))
            print("Reducing number of GMM components to %d, and refitting the GMM..." % new_number_of_components)
        #
        optimal_model, converged, lables, weights, means, covariances, precisions, optimal_covar_type =  \
            generate_gmm_model_info(ensemble,
                                    clustering_model=clustering_model,
                                    cov_type=cov_type,
                                    inf_criteria=inf_criteria,
                                    number_of_components=new_number_of_components,
                                    min_number_of_components=min_number_of_components,
                                    max_number_of_components=max_number_of_components,
                                    min_number_of_points_per_component=min_number_of_points_per_component,
                                    invert_uncertainty_param=invert_uncertainty_param,
                                    verbose=verbose
                                    )
        #
    elif gmm_number_of_components < min_number_of_components:
        # Case 3: gmm_number of components is less than the minimum number of components allowed; increase:
        new_number_of_components = gmm_number_of_components + 1
        if verbose:
            print("Number of components %d is LESS than the minimum allowed of %d" \
                            % (gmm_number_of_components, min_number_component))
            print("Increasing number of GMM components to %d, and refitting the GMM..." % new_number_of_components)
        #
        optimal_model, converged, lables, weights, means, covariances, precisions, optimal_covar_type =  \
            generate_gmm_model_info(ensemble,
                                    clustering_model=clustering_model,
                                    cov_type=cov_type,
                                    inf_criteria=inf_criteria,
                                    number_of_components=new_number_of_components,
                                    min_number_of_components=min_number_of_components,
                                    max_number_of_components=max_number_of_components,
                                    min_number_of_points_per_component=min_number_of_points_per_component,
                                    invert_uncertainty_param=invert_uncertainty_param,
                                    verbose=verbose
                                    )
        # pass:
    else:
        # Every thing is fine: max_number_of_components >= gmm_number_of_components >= min_number_of_components;
        pass
        
    # Now check the number of labels assigned to each GMM component:
    if min_elements_count < min_number_of_points_per_component:
        new_number_of_components = gmm_number_of_components - 1
        if verbose:
            print("Adjusting number of GMM components to %d; With restriction: %d <= number of GMM components <= %d" \
                        % (new_number_of_components, min_number_of_components, max_number_of_components))
        if (new_number_of_components > max_number_of_components) or (new_number_of_components < min_number_of_components):
            # Clash, failed to converge under passed settings; destroy the results, and return
            converged = False
            optimal_model = None
            lables = None
            weights = None
            means = None
            covariances = None
            precisions = None
            optimal_covar_type = None
        else:
            # Reduced the number of components by one, and recurse:
            if verbose:
                print("Number of GMM components is %d. Some components are assigned lables %d < %d = min_number_of_points_per_component" \
                                % (gmm_number_of_components, min_elements_count, min_number_of_points_per_component))
                print("Reducing number of GMM components to %d, and refitting the GMM..." % new_number_of_components)
            #
            optimal_model, converged, lables, weights, means, covariances, precisions, optimal_covar_type =  \
                generate_gmm_model_info(ensemble,
                                        clustering_model=clustering_model,
                                        cov_type=cov_type,
                                        inf_criteria=inf_criteria,
                                        number_of_components=new_number_of_components,
                                        min_number_of_components=min_number_of_components,
                                        max_number_of_components=max_number_of_components,
                                        min_number_of_points_per_component=min_number_of_points_per_component,
                                        invert_uncertainty_param=invert_uncertainty_param,
                                        verbose=verbose
                                        )
    #
    else:
        # Everything is fine now to return
        pass
    
    # ---------------------------- ================================================== ----------------------------
    # Finally: (optionally) invert the Covariance/Precision array(s):
    # ---------------------------- ================================================== ----------------------------
    if invert_uncertainty_param:
        #
        if precisions is None and covariances is not None:
            # find precisions matrix given the covariances matrix(es)
            if optimal_covar_type in ['diag', 'spherical']:
                precisions = 1/covariances
            elif optimal_covar_type == 'tied':
                if np.ndim(covariances) != 2:
                    raise ValueError(" Dimension of the covariances matrix is not correct! It should be 2"
                                     " -> np.ndim(covariances)=%d" % np.ndim(covariances)
                                     )
                else:
                    precisions = np.linalg.inv(covariances)
            elif optimal_covar_type == 'full':
                if np.ndim(covariances) != 3:
                    raise ValueError(" Dimension of the covariances matrix is not correct! It should be 3"
                                     " -> np.ndim(covariances)=%d" % np.ndim(covariances)
                                     )
                else:
                    precisions = np.empty_like(covariances)
                    for comp_ind in xrange(np.size(covariances, 0)):
                        precisions[comp_ind, :, :] = np.linalg.inv(covariances[comp_ind, :, :])
            else:
                # shouldn't be reached at all
                raise ValueError("\n\tcovariance type is not recognized. How did this happened!"
                                 "optimal_covar_type received: "+optimal_covar_type)
            #
        elif covariances is None and precisions is not None:
            #
            if optimal_covar_type in ['diag', 'spherical']:
                covariances = 1/precisions
            elif optimal_covar_type == 'tied':
                if np.ndim(precisions) != 2:
                    raise ValueError(" Dimension of the precisions matrix is not correct! It should be 2"
                                     " -> np.ndim(precisions)=%d" % np.ndim(precisions)
                                     )
                else:
                    covariances = np.linalg.inv(precisions)
            elif optimal_covar_type == 'full':
                if np.ndim(precisions) != 3:
                    raise ValueError(" Dimension of the precisions matrix is not correct! It should be 3"
                                     " -> np.ndim(precisions)=%d" % np.ndim(precisions)
                                     )
                else:
                    covariances = np.empty_like(precisions)
                    for comp_ind in xrange(np.size(precisions, 0)):
                        covariances[comp_ind, :, :] = np.linalg.inv(precisions[comp_ind, :, :])
            else:
                # shouldn't be reached at all
                # shouldn't be reached at all
                raise ValueError("\n\tcovariance type is not recognized. How did this happened!"
                                 "optimal_covar_type received: "+optimal_covar_type)
            #
        else:
            raise ValueError("Unexpected Error caused by Precisions and Covariances. "
                             "Seems both are None or both are known!")

    if verbose and converged:
        print("Clustering Process of [%d] ensemble members Finished."
              "Clusters found: [%d]" % (ensemble_size , optimal_model.n_components))
    elif not converged:
        print("Unfortunately: Convergence of GMM under passed settings has failed!!!")
    else:
        pass
    
    # ---------------------------- ================================================== ----------------------------
    # Done; Now return...
    return optimal_model, converged, lables, weights, means, covariances, precisions, optimal_covar_type
    # ---------------------------- ================================================== ----------------------------
    #


#
# -------------------------------------------------------------------------------------------------------------------- #
#
def GMM_clustering(Ensemble,
                   num_comp,
                   covariance_type,
                   inf_criteria
                   ):
    """
    Standard Gaussian Mixture model with EM
    
    Args:
        Ensemble: 
        num_comp: 
        covariance_type: 
        inf_criteria: 
    
    Returns:
         gmm_model: 
         opt_inf_criterion: 
         optimal_covar_type: 
         
    """
    #
    inf_criteria = inf_criteria.lower()

    if covariance_type is not None:
        optimal_covar_type = covariance_type
        #
        # one model
        gmm_model = GMM_Model(n_components=num_comp , covariance_type=covariance_type)
        gmm_model.fit(Ensemble)
        #
        if inf_criteria =='aic':
            opt_inf_criterion = gmm_model.aic(Ensemble)
        elif inf_criteria == 'bic':
            opt_inf_criterion=gmm_model.bic(Ensemble)
        else:
            raise ValueError("\n\t Information Criterion passed is not recognized! "
                             "AIC or BIC only are accepted, received ["+inf_criteria+"]!")
    else:
        covariance_types = ['diag', 'spherical', 'tied', 'full']
        opt_inf_criterion = np.infty
        gmm_model = None
        optimal_covar_type = None
        #
        # best model by comparing several covariance types:
        for cov_type in covariance_types:
            #
            temp_gmm_model = GMM_Model(n_components=num_comp, covariance_type=cov_type)
            temp_gmm_model.fit(Ensemble)
            #
            if inf_criteria=='aic':
                temp_inf_criterion = temp_gmm_model.aic(Ensemble)
            elif inf_criteria.lower()=='bic':
                temp_inf_criterion=temp_gmm_model.bic(Ensemble)
            else:
                raise ValueError("\n\t Information Criterion passed is not recognized! AIC or BIC only are accepted, received ["+inf_criteria+"]!")

            if temp_inf_criterion < opt_inf_criterion:
                opt_inf_criterion  = temp_inf_criterion
                gmm_model      = temp_gmm_model
                optimal_covar_type = cov_type

    return gmm_model, opt_inf_criterion, optimal_covar_type
    #


#
#
def VBGMM_clustering(Ensemble,
                     num_comp,
                     covariance_type,
                     inf_criteria,
                     alpha=1.0,
                     random_state=None,
                     thresh=None,
                     tol=0.001,
                     verbose=False,
                     min_covar=None,
                     n_iter=10,
                     params='wmc',
                     init_params='wmc'
                     ):
    """
    Variational Gaussian Mixture model with EM
    
    Args:
        Ensemble:
        num_comp:
        covariance_type:
        inf_criteria:
        alpha=1.0:
        random_state=None:
        thresh=None:
        tol=0.001:
        verbose=False:
        min_covar=None:
        n_iter=10:
        params='wmc':
        init_params='wmc'
        
    Returns:
        gmm_model: 
        opt_inf_criterion: 
        optimal_covar_type:
        
    """
    #
    inf_criteria = inf_criteria.lower()

    if covariance_type is not None:
        optimal_covar_type = covariance_type
        #
        # one model
        gmm_model = VBGMM_Model(n_components=num_comp,
                                covariance_type=covariance_type,
                                alpha=alpha,
                                random_state=random_state,
                                thresh=thresh,
                                verbose=verbose,
                                min_covar=min_covar,
                                n_iter=n_iter,
                                params=params,
                                init_params=init_params
                                )
        gmm_model.fit(Ensemble)
        #
        if inf_criteria =='aic':
            opt_inf_criterion = gmm_model.aic(Ensemble)
        elif inf_criteria == 'bic':
            opt_inf_criterion = gmm_model.bic(Ensemble)
        else:
            raise ValueError("\n\t Information Criterion passed is not recognized! "
                             "AIC or BIC only are accepted, received ["+inf_criteria+"]!")

    else:
        covariance_types = ['diag', 'spherical', 'tied', 'full']
        opt_inf_criterion = np.infty
        gmm_model = None
        optimal_covar_type = None

        # best model by comparing several covariance types:
        for cov_type in covariance_types:
            #
            temp_gmm_model = VBGMM_Model(n_components=num_comp,
                                         covariance_type=cov_type,
                                         alpha=alpha,
                                         random_state=random_state,
                                         thresh=thresh,
                                         tol=tol,
                                         verbose=verbose,
                                         min_covar=min_covar,
                                         n_iter=n_iter,
                                         params=params,
                                         init_params=init_params
                                         )
            temp_gmm_model.fit(Ensemble)
            #
            if inf_criteria=='aic':
                temp_inf_criterion = temp_gmm_model.aic(Ensemble)
            elif inf_criteria.lower()=='bic':
                temp_inf_criterion = temp_gmm_model.bic(Ensemble)
            else:
                raise ValueError("\n\t Information Criterion passed is not recognized! AIC or BIC only are accepted, recieved ["+inf_criteria+"]!")

            if temp_inf_criterion < opt_inf_criterion:
                opt_inf_criterion  = temp_inf_criterion
                gmm_model      = temp_gmm_model
                optimal_covar_type = cov_type

    return gmm_model, opt_inf_criterion, optimal_covar_type
    #
    
