
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
    A module providing functions that handle statistics-related operations; this includes generating random vectors, etc.
"""


import numpy as np
from _utility_misc import ensemble_to_np_array

from state_vector_numpy import StateVectorNumpy as StateVector
from state_matrix_numpy import StateMatrixNumpy as StateMatrix

# All utility functions required to conduct statistical operations should be listed here.


def mvn_rand_vec(vec_size):
    """
    Function generating a standard normal random vector with values truncated at -/+3

    Args:
        vec_size:

    Returns:
        randn_vec:

    """
    randn_vec = np.random.randn(vec_size)
    flg = randn_vec > 3
    randn_vec[flg] = 3
    flg = randn_vec < -3
    randn_vec[flg] = -3

    return randn_vec


def ensemble_mean(vectors_list):
    """
    Given a list of State or Observation vectors, the mean is evaluated and returned

    Args:
        vectors_list: a list of state or observation vectors

    Returns:
        ens_average: a vecotr of the same type as the entries of vecotrs_list, containing the mean of objects of vectors_list.
    """
    # assert isinstance(vectors_list, list)

    ensemble_size = len(vectors_list)
    ens_average = vectors_list[0].copy()
    for vec_ind in xrange(1, ensemble_size):
        ens_average = ens_average.add(vectors_list[vec_ind])

    ens_average = ens_average.scale(1.0/ensemble_size)
    return ens_average


def add_ensemble(first_ensemble, second_ensemble, in_place=True):
    """
    Add two ensembles. if in_place, the first is overwritten

    Args:
        first_ensemble:
        second_ensemble:
        in_place:

    Returns:
        result_vectors_list

    """
    if not in_place:
        result_vectors_list = first_ensemble.copy()
    else:
        result_vectors_list = first_ensemble

    for result_vec in result_vectors_list:
        result_vec.add(second_ensemble, in_place=True)

    return result_vectors_list


def ensemble_variances(ensemble, sample_based=True, ddof=1.0, return_state_vector=False):
    """
    Calculate the ensemble-based statevector variances.

    Args:
        ensemble:
        sample_based:
        return_state_vector:

    Returns:
        _raw_vector_ref

    """
    mean_vec = ensemble_mean(ensemble)
    ensemble_size = len(ensemble)
    var_vec = mean_vec.copy()   # we need an alternative
    var_vec[:] = 0.0
    for ens_ind in xrange(ensemble_size):
        innovation_vec = mean_vec.copy()
        # innovation_vec.axpy(-1.0, ensemble[ens_ind])
        innovation_vec.scale(-1.0).add(ensemble[ens_ind])
        var_vec.add(innovation_vec.square())

    if sample_based:
        scale_factor = 1.0/float(ensemble_size-ddof)
    else:
        scale_factor = 1.0/ensemble_size

    var_vec.scale(scale_factor)

    if return_state_vector:
        return var_vec
    else:
        return var_vec.get_numpy_array()


def ensemble_covariance_matrix(ensemble, model=None, corr=False, zero_nans=False, ddof=1.0):
    """
    Construct and ensemble-based covariance matrix given an ensemble of states

    Args:
        ensemble: list of StateVector objects, or a numpy array;
            if it is a numpy array, each column is a state vector
        model: model object
        corr: if True, the correlation matrix is returned (divide the covariance matrix rows by the diagonal)
        zero_nans: This is used if corr is True only;
            If any of the variances is ZERO, the correlation will contain nans. If zero_nans is True, these correlations are replaced with zeros

    Returns:
        covariance_mat: StateMatrix if model is not None, and ensemble is a list of model.StateVector objects,
            otherwise it is a numpy array

    Remark:
        Try model.ensemble_covariance_matrix() if you want a sparse/localized matrix,

    """
    # Here I will just construct the covariance matrix to enable full decorrelation.
    if isinstance(ensemble, np.ndarray):
        if ensemble.ndim != 2:
            print("Tha passed 'ensemble' object has to be either a numpy 2D array, or a list of model state vectors...")
            raise AssertionError
        else:
            state_size, ensemble_size = np.shape(ensemble)
            perturbations = ensemble.copy()
            ens_mean = np.mean(ensemble, 1).copy()

            #
            for ens_ind in xrange(ensemble_size):
                perturbations[:, ens_ind] -= ens_mean

            perturbations /= np.sqrt(ensemble_size - ddof)
            covariance_mat = perturbations.dot(perturbations.T)

            if corr:
                variances = covariance_mat.diagonal().copy()
                stdevs = np.sqrt(variances)
                if zero_nans:
                    stdevs[np.where(stdevs==0)[0]] = np.inf
                for ens_ind, stdev in zip(xrange(stdevs.size), stdevs):
                    covariance_mat[ens_ind, ens_ind:] /= (stdevs[ens_ind:] * stdev)
                    covariance_mat[ens_ind:, ens_ind] = covariance_mat[ens_ind, ens_ind:]


    elif isinstance(ensemble, list):

        ensemble_size = len(ensemble)
        if model is not None:
            state_size = model.state_size()
        else:
            try:
                state_size = ensemble[0].size()
            except:
                try:
                    state_size = ensemble[0].size
                except:
                    state_size = len(ensemble[0])

        ens_anomalies = np.empty((state_size, ensemble_size))

        # construct ensemble of anomalies matrix:
        # ens_mean = ensemble_mean(ensemble).scale(-1)  # of course it is negated to be added later than subtracted
        ens_mean = ensemble_mean(ensemble)

        scl = 1.0 / np.sqrt(float(ensemble_size - ddof))
        for ens_ind in xrange(ensemble_size):
            # state = ensemble[ens_ind].add(ens_mean, in_place=False)
            state = ensemble[ens_ind].axpy(-1, ens_mean, in_place=False)
            # state.scale(scl, in_place=True)
            ens_anomalies[:, ens_ind] = state.get_numpy_array()
        ens_anomalies *= scl

        if model is not None:
            covariances = model.state_matrix()
            covariances[:, :] = np.dot(ens_anomalies, ens_anomalies.T)
        else:
            covariances = np.dot(ens_anomalies, ens_anomalies.T)

        #
        if corr:
            variances = covariances.diagonal().copy()
            stdevs = np.sqrt(variances)
            if zero_nans:
                stdevs[np.where(stdevs==0)[0]] = np.inf
            for ens_ind, stdev in zip(xrange(stdevs.size), stdevs):
                covariances[ens_ind, ens_ind:] /= (stdevs[ens_ind:] * stdev)
                covariances[ens_ind:, ens_ind] = covariances[ens_ind, ens_ind:]


    else:
        print("Tha passed 'ensemble' object has to be either a numpy 2D array, or a list of model state vectors...")
        raise AssertionError

    #
    return covariances
    #


def ensemble_precisions(ensemble, sample_based=True, return_state_vector=False):
    """
    Calculate the ensemble-based statevector precisions (variance reciprocals).

    Args:
        ensemble:
        sample_based:
        return_state_vector:

    """
    # calculate variances:
    var_vec = ensemble_variances(ensemble, sample_based=sample_based, return_state_vector=True)

    #  get the reciprocal and return
    if return_state_vector:
        return var_vec.reciprocal()
    else:
        sample_variances = var_vec.get_numpy_array()
        return 0.1 / sample_variances


def generate_ensemble(ensemble_size, ensemble_average, noise_model):
    """
    Create a list/ensemble of states of size ensemble_size with noise vectors created using
        noise_model centered around ensemble_mean.

    Args:
        ensemble_size:
        ensemble_average:
        noise_model:

    Returns:
        perturbed_ensemble
    """
    perturbed_ensemble = []
    for ens_ind in xrange(ensemble_size):
        noise_vec = noise_model.generate_noise_vec()
        perturbed_ensemble.append(noise_vec.add(ensemble_average))

    return perturbed_ensemble
