
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
    A module providing functions that handle DA functionalities; such as ensemble propagation, and inflation, etc.
"""


import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Failed to Import matplotlib. Plotting won't work in this session... Proceeding ...")

import scipy
from scipy import stats as scipy_stats
import re

from _utility_stat import ensemble_variances
from _utility_stat import ensemble_mean

from _utility_misc import isiterable, isscalar

import copy

#
def propagate_ensemble(ensemble, model, checkpoints, in_place=True):
    """
    Given a list (ensemble) of model states, use the model object to propagate each state forward in time
    and create/modify the ensemble based on the flag in in_place.

    Args:
        ensemble: list of model state vectors. Exceptions should be handled more carefully

    """
    if ensemble is None or len(ensemble) < 1:
            raise ValueError("Ensemble is None or it is an empty list!")
    else:
        ens_size = len(ensemble)
        if in_place:
            out_ensemble = ensemble
        else:
            out_ensemble = list(ensemble)  # reconstruction processes copying

        for ens_ind in xrange(ens_size):
            tmp_traject = model.integrate_state(initial_state=ensemble[ens_ind], checkpoints=checkpoints)
            if not isinstance(tmp_traject, list):
                tmp_traject = [tmp_traject]
            out_ensemble[ens_ind] = tmp_traject[-1].copy()
            # if in_place:
            #     out_ensemble[ens_ind] = tmp_traject[-1].copy()
            # else:
            #     out_ensemble.append(tmp_traject[-1].copy())

    return out_ensemble


#
def create_synthetic_observations(model, observation_checkpoints, reference_checkpoints, reference_trajectory):
    """
    create synthetic observations given reference trajectory
    """
    # Create synthetic observations
    observations_list = []
    if reference_checkpoints[0] == observation_checkpoints[0]:
        observation_vec = model.evaluate_theoretical_observation(reference_trajectory[0])
        observations_list.append(observation_vec.add(
            model.observation_error_model.generate_noise_vec(noise_type='gaussian'), in_place=False))
    elif observation_checkpoints[0] >= reference_checkpoints[0]:
        observation_checkpoints.insert(0, reference_checkpoints[0])
    else:
        raise ValueError("take a look at checkpoints and observation_checkpoints!")

    for time_ind in xrange(1, np.size(observation_checkpoints)):
        try:
            time_point = observation_checkpoints[time_ind]
            obs_ind = np.squeeze(np.where(reference_checkpoints == time_point))
            observation_vec = model.evaluate_theoretical_observation(reference_trajectory[obs_ind])
            observations_list.append(observation_vec.add(
                model.observation_error_model.generate_noise_vec(noise_type='gaussian'), in_place=False))
        except:
            ref_pre_ind = np.max(np.squeeze(np.where(reference_checkpoints < observation_checkpoints[time_ind])))
            local_checkpoints = [reference_checkpoints[ref_pre_ind], observation_checkpoints[time_ind]]
            local_trajectory = model.integrate_state(initial_state=model._reference_initial_condition,
                                                     checkpoints=local_checkpoints)
            observation_vec = model.evaluate_theoretical_observation(local_trajectory[-1])
            observations_list.append(observation_vec.add(
                model.observation_error_model.generate_noise_vec(noise_type='gaussian'), in_place=False))

    return observations_list
#
# ==================================================================================================================== #


# ==================================================================================================================== #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
#
def calculate_rmse(first, second, vec_size=None):
    """
    Calculate the root mean squared error between two vectors of the same type.
    No exceptions are handled explicitly here

    Args:
        first:
        second:
        vec_size: length of each of the two vectors

    Returns:
        rmse

    """
    if not isinstance(first, type(second)):
        print('first', first)
        print('second', second)
        raise TypeError(" The two vectors must be of the same type!"
                        "First: %s\n"
                        "Second: %s" % (repr(type(first)), repr(type(first))))
    if vec_size is None:
        vec_size = first.size
    #
    scld_second = second.copy()
    scld_second = scld_second.scale(-1)
    scld_second = scld_second.add(first)
    rmse = scld_second.norm2()
    rmse = rmse/np.sqrt(vec_size)

    return rmse
#


#
#
def calculate_localization_coefficients(radius, distances, method='Gauss'):
    """
    Evaluate the spatial decorrelation coefficients based on the passed vector of distances and the method.

    Args:
        radius: decorrelation radius
        distances: vector containing distances based on which decorrelation coefficients are calculated.
        method: Localization mehtod. Methods supported:
            1- Gaussian 'Gauss'
            2- Gaspari_Cohn
            3- Cosine
            4- Cosine_squared
            5- Exp3
            6- Cubic
            7- Quadro
            8- Step
            9- None

    Returns:
        coefficients: a vector containing decorrelation coefficients.
            If the passed radius, and distances are both scalars, coefficients is scalar too

    """
    _methods_supported = ['gauss', 'cosine', 'cosine_squared', 'gaspari_cohn', 'exp3', 'cubic', 'quadro', 'step']
    #
    if isscalar(distances):
        distances = np.array([distances], dtype=np.float)  # this is local now
    elif isiterable(distances):
        distances = np.asarray(distances).flatten().astype(np.float)
    else:
        print("distances must be either a scalar, or an iterable! Unknown Type!" % type(distances))
        raise TypeError
        #

    #
    if method is None:
        if distances.size == 1:
            return 1
        else:
            return np.ones_like(distances)

    if isinstance(distances, np.ndarray):
        if distances.ndim == 0:
            distances = np.array([distances.item()])

    if distances.size == 1:
        distances = distances[0]
    else:
        pass
        # good to go!

    #
    # print("radius", radius)
    # print("distances", distances)
    # coefficients = np.zeros_like(distances)
    #
    if np.size(distances)==1:

        if isscalar(radius):
            pass
        else:
            if len(radius) == 1:
                radius = radius[0]
            else:
                print("Radius is expected to be either a scalar, or an iterable of lenght 1!, NOT %s " % type(radius))
                raise TypeError

        # a scalar is passed in this case...
        if re.match(r'\Agauss\Z', method, re.IGNORECASE):
            coefficients = np.exp(-0.5*((distances/radius)**2))
        elif re.match(r'\Aexp3\Z', method, re.IGNORECASE):
            coefficients = np.exp(-0.5 * (distances/radius) ** 3)
        elif re.match(r'\Acosine\Z', method, re.IGNORECASE):
            thresh = radius * 2.3167
            if distances <= thresh:
                red = distances/thresh
                coefficients = (1 + np.cos(red*np.pi)) / 2.0
            else:
                coefficients = 0.0
        elif re.match(r'\Acosine(_|-)*squared\Z', method, re.IGNORECASE):
            thresh = radius * 3.2080
            if distances <= thresh:
                red = distances/thresh
                coefficients = ((1 + np.cos(red*np.pi)) / 2.0) ** 2
            else:
                coefficients = 0.0
        elif re.match(r'\Agaspari(_|-)*cohn\Z', method, re.IGNORECASE):
            thresh = radius * 1.7386
            if distances <= thresh:
                red = distances/thresh
                r2 = red ** 2
                r3 = red ** 3
                coefficients = 1.0 + r2 * (-r3/4.0 + r2/2.0) + r3 * (5.0/8.0) - r2 * (5.0/3.0)
            elif distances <= thresh*2:
                red = distances/thresh
                r1 = red
                r2 = red ** 2
                r3 = red ** 3
                coefficients = r2 * (r3/12.0 - r2/2.0) + r3 * (5.0/8.0) + r2 * (5.0/3.0) - r1 * 5.0 + 4.0 - (2.0/3.0) / r1
            else:
                coefficients = 0.0
        elif re.match(r'\Acubic\Z', method, re.IGNORECASE):
            thresh = radius * 1.8676
            if distances <= thresh:
                red = distances/thresh
                coefficients = (1.0 - (red) ** 3) ** 3
            else:
                coefficients = 0.0
        elif re.match(r'\Aquadro\Z', method, re.IGNORECASE):
            thresh = radius * 1.7080
            if distances <= thresh:
                red = distances/thresh
                coefficients = (1.0 - (red) ** 4) ** 4
            else:
                coefficients = 0.0
        elif re.match(r'\Astep\Z', method, re.IGNORECASE):
            if distances < radius:
                coefficients = 1.0
            else:
                coefficients = 0.0
        else:
            # Shouldn't be reached if we keep the first test.
            raise ValueError("The Localization method '%s' is not supported."
                             "Supported methods are: %s" % (method, repr(_methods_supported)))
    #
    else:
        #
        if np.isscalar(radius):
            pass
        else:
            # multiple radii are given. Each will be taken to the corresponding distance point.
            # distances, and radius have to be of equal sizes
            radius = np.asarray(radius).squeeze().astype(np.float)
            if radius.size != distances.size:
                print(radius.size, distances.size)
                print("distances, and radius have to be of equal sizes!")
                raise AssertionError

        #  distances is of dimension greater than one. vector is assumed
        if re.match(r'\Agauss\Z', method, re.IGNORECASE):
            # print("distances", distances)
            # print("radius", radius)
            coefficients = np.exp(-0.5*((distances/radius)**2))

        elif re.match(r'\Aexp3\Z', method, re.IGNORECASE):
            coefficients = np.exp(-0.5 * (distances/radius) ** 3);

        elif re.match(r'\Acosine\Z', method, re.IGNORECASE):
            coefficients = np.zeros_like(distances)
            thresh = radius * 2.3167
            indexes = (distances <= thresh)
            if np.isscalar(thresh):
                red = distances[indexes] / thresh
            else:
                red = distances[indexes] / thresh[indexes]
            coefficients[indexes] = (1 + np.cos(red*np.pi)) / 2.0

        elif re.match(r'\Acosine(_|-)*squared\Z', method, re.IGNORECASE):
            coefficients = np.zeros_like(distances)
            thresh = radius * 3.2080
            indexes = (distances <= thresh)
            if np.isscalar(thresh):
                red = distances[indexes] / thresh
            else:
                red = distances[indexes] / thresh[indexes]
            coefficients[indexes] = ((1 + np.cos(red*np.pi)) / 2.0) ** 2

        elif re.match(r'\Agaspari(_|-)*cohn\Z', method, re.IGNORECASE):
            coefficients = np.zeros_like(distances)
            thresh = radius * 1.7386
            indexes = (distances <= thresh)
            # print('indexes', indexes)
            if np.isscalar(thresh):
                r2 = (distances[indexes] / thresh) ** 2
                r3 = (distances[indexes] / thresh) ** 3
            else:
                r2 = (distances[indexes] / thresh[indexes]) ** 2
                r3 = (distances[indexes] / thresh[indexes]) ** 3

            coefficients[indexes] = 1.0 + r2 * (-r3/4.0 + r2/2.0) + r3 * (5.0/8.0) - r2 * (5.0/3.0)

            indexes_1 = (distances > thresh)
            indexes_2 = (distances <= thresh*2)
            indexes = np.asarray( [(indexes_1[i] and indexes_2[i]) for i in xrange(np.size(indexes_1))] )

            if np.isscalar(thresh):
                r1 = (distances[indexes] / thresh)
                r2 = (distances[indexes] / thresh) ** 2
                r3 = (distances[indexes] / thresh) ** 3
            else:
                r1 = (distances[indexes] / thresh[indexes])
                r2 = (distances[indexes] / thresh[indexes]) ** 2
                r3 = (distances[indexes] / thresh[indexes]) ** 3

            coefficients[indexes] = r2 * (r3/12.0 - r2/2.0) + r3 * (5.0/8.0) + r2 * (5.0/3.0) - r1 * 5.0 + 4.0 - (2.0/3.0) / r1

        elif re.match(r'\Acubic\Z', method, re.IGNORECASE):
            coefficients = np.zeros_like(distances)
            thresh = radius * 1.8676
            indexes = (distances < thresh)
            if np.isscalar(thresh):
                coefficients[indexes] = (1.0 - (distances[indexes] / thresh) ** 3) ** 3
            else:
                coefficients[indexes] = (1.0 - (distances[indexes] / thresh[indexes]) ** 3) ** 3

        elif re.match(r'\Aquadro\Z', method, re.IGNORECASE):
            coefficients = np.zeros_like(distances)
            thresh = radius * 1.7080
            indexes = (distances < thresh)
            if np.isscalar(thresh):
                coefficients[indexes] = (1.0 - (distances[indexes] / thresh) ** 4) ** 4
            else:
                coefficients[indexes] = (1.0 - (distances[indexes] / thresh[indexes]) ** 4) ** 4

        elif re.match(r'\Astep\Z', method, re.IGNORECASE):
            coefficients = np.zeros_like(distances)
            indexes = (distances < radius)
            coefficients[indexes] = 1.0

        else:
            # Shouldn't be reached if we keep the first test.
            raise ValueError("The Localization method '%s' is not supported."
                             "Supported methods are: %s" % (method, repr(_methods_supported)))

    # print("Getting out of localization coefficients calculator...")
    # print("distances: ", distances)
    # print("radius: ", radius)
    # print("coefficients: ", coefficients)
    return coefficients
#

def calculate_mixed_localization_coefficients(radii, distances, method='Gauss'):
    """
    Evaluate the spatial decorrelation coefficients based on the passed vector of distances and the method.
    This is different from calculate_localization_coefficients, in the sense that radii is an iterable of two entries,
        the localization radius is the product of each two corresponding radii

    Args:
        radius: decorrelation radius
        distances: vector containing distances based on which decorrelation coefficients are calculated.
        method: Localization mehtod. Methods supported:
            1- Gaussian 'Gauss'
            2- Gaspari_Cohn

    Returns:
        coefficients: a vector containing decorrelation coefficients.
            If the passed radius, and distances are both scalars, coefficients is scalar too

    """
    if not isiterable(radii):
        try:
            return calculate_localization_coefficients(radius=radii, distances=distances, method=method)
        except:
            print("Failed to calculate the localization coefficients!")
            raise

    if len(radii) == 2:
        if isscalar(radii[0]) and isscalar(radii[1]):
            li = radii[0]
            lj = radii[1]
        elif isiterable(radii[0]) and isiterable(radii[1]):
            if len(radii[0]) != len(radii[1]):
                print("radii elements must be of equal dimension!")
                raise AssertionError
            else:
                li = np.asarray(radii[0]).flatten()
                lj = np.asarray(radii[1]).flatten()
        elif isscalar(radii[0]):
            lj = np.asarray(radii[1]).flatten()
            li = np.ones_like(lj) * radii[0]
        elif isscalar(radii[1]):
            li = np.asarray(radii[0]).flatten()
            lj = np.ones_like(li) * radii[1]
        else:
            print("Both radii entries must be either scalars, or iterables of equal dimensions!")
            raise ValueError
    else:
        print("Unexpected dimension %d " % len(radii))
        print("Failed to calculate the localization coefficients!")
        raise TypeError

    _methods_supported = ['gauss', 'gaspari_cohn']
    #
    if isscalar(distances):
        distances = np.array([distances], dtype=np.float)  # this is local now
    elif isiterable(distances):
        distances = np.asarray(distances).flatten().astype(np.float)
    else:
        print("distances must be either a scalar, or an iterable! Unknown Type!" % type(distances))
        raise TypeError
        #

    #
    if method is None:
        if distances.size == 1:
            return 1
        else:
            return np.ones_like(distances)

    if isinstance(distances, np.ndarray):
        if distances.ndim == 0:
            distances = np.array([distances.item()])

    if distances.size == 1:
        distances = distances[0]
    else:
        pass
        # good to go!

    #
    # print("radius", radius)
    # print("distances", distances)
    # coefficients = np.zeros_like(distances)
    #
    if np.size(distances)==1:
        if isscalar(li) and isscalar(lj):
            pass
        else:
            if len(li) == 1 and len(lj) == 1:
                li, lj = li[0], lj[0]
            else:
                print("Radii are expected to be either scalars, or iterables of lenght 1!, NOT %s and %s " % (type(li), type(lj)))
                raise TypeError

        # a scalar is passed in this case...
        if re.match(r'\Agauss\Z', method, re.IGNORECASE):
            coefficients = np.exp(-0.5*((distances)**2)/(li*lj))

        elif re.match(r'\Agaspari(_|-)*cohn\Z', method, re.IGNORECASE):
            thresh = (li*lj) * 1.7386
            if distances <= thresh:
                red = distances/thresh
                r2 = red ** 2
                r3 = red ** 3
                coefficients = 1.0 + r2 * (-r3/4.0 + r2/2.0) + r3 * (5.0/8.0) - r2 * (5.0/3.0)
            elif distances <= thresh*2:
                red = distances/thresh
                r1 = red
                r2 = red ** 2
                r3 = red ** 3
                coefficients = r2 * (r3/12.0 - r2/2.0) + r3 * (5.0/8.0) + r2 * (5.0/3.0) - r1 * 5.0 + 4.0 - (2.0/3.0) / r1
            else:
                coefficients = 0.0
        else:
            # Shouldn't be reached if we keep the first test.
            raise ValueError("The Localization method '%s' is not supported."
                             "Supported methods are: %s" % (method, repr(_methods_supported)))
    #
    else:
        #
        if isscalar(li) and isscalar(lj):
            pass
        else:
            if len(li) != len(lj):
                print("Radii are expected to be either scalars, or iterables of the same length")
                raise TypeError

        #  distances is of dimension greater than one. vector is assumed
        if re.match(r'\Agauss\Z', method, re.IGNORECASE):
            # print("distances", distances)
            # print("radii", radii)
            coefficients = np.exp(-0.5*((distances)**2)/(li*lj))

        elif re.match(r'\Agaspari(_|-)*cohn\Z', method, re.IGNORECASE):
            coefficients = np.zeros_like(distances)
            thresh = (li*lj) * 1.7386
            indexes = (distances <= thresh)
            # print('indexes', indexes)
            if np.isscalar(thresh):
                r2 = (distances[indexes] / thresh) ** 2
                r3 = (distances[indexes] / thresh) ** 3
            else:
                r2 = (distances[indexes] / thresh[indexes]) ** 2
                r3 = (distances[indexes] / thresh[indexes]) ** 3

            coefficients[indexes] = 1.0 + r2 * (-r3/4.0 + r2/2.0) + r3 * (5.0/8.0) - r2 * (5.0/3.0)

            indexes_1 = (distances > thresh)
            indexes_2 = (distances <= thresh*2)
            indexes = np.asarray( [(indexes_1[i] and indexes_2[i]) for i in xrange(np.size(indexes_1))] )

            if np.isscalar(thresh):
                r1 = (distances[indexes] / thresh)
                r2 = (distances[indexes] / thresh) ** 2
                r3 = (distances[indexes] / thresh) ** 3
            else:
                r1 = (distances[indexes] / thresh[indexes])
                r2 = (distances[indexes] / thresh[indexes]) ** 2
                r3 = (distances[indexes] / thresh[indexes]) ** 3

            coefficients[indexes] = r2 * (r3/12.0 - r2/2.0) + r3 * (5.0/8.0) + r2 * (5.0/3.0) - r1 * 5.0 + 4.0 - (2.0/3.0) / r1

        else:
            # Shouldn't be reached if we keep the first test.
            raise ValueError("The Localization method '%s' is not supported."
                             "Supported methods are: %s" % (method, repr(_methods_supported)))

    # print("Getting out of localization coefficients calculator...")
    # print("distances: ", distances)
    # print("radii: ", radii)
    # print("coefficients: ", coefficients)
    return coefficients
#


def inflate_ensemble(ensemble, inflation_factor, in_place=True):
    """
    Apply inflation on an ensemble of states

    Args:
        ensemble: list of model states
        inflation_factor: scalr or iterable of length equal to model states
        in_place: overwrite passed ensemble
    Returns:
        inflated_ensemble: inflated ensemble of states

    """
    # print("Inflation with:", inflation_factor)
    assert isinstance(ensemble, list), "The passed ensemble must be a list of model states!"
    if len(ensemble) == 0:
        print( "*"*50 + "\nThe passed ensemble is empty!\n" + "*"*50)
        return

    if np.isscalar(inflation_factor):
        _scalar = True
        _inflation_factor = float(inflation_factor)
    elif isiterable(inflation_factor):
        if len(inflation_factor) == 1:  # Extract inflation factor if it's a single value wrapped in an iterable
            _inflation_factor = np.array(inflation_factor)
            for i in xrange(_inflation_factor.ndim):
                _inflation_factor = _inflation_factor[i]
            _scalar = True
        else:
            _scalar = False

        # check type and size:
        if type(ensemble[0]) is type(inflation_factor):
            if ensemble[0].size == inflation_factor.size:
                _inflation_factor = inflation_factor
            else:
                print("Size Mismatch;\n\tInflation Factor can be an iterable BUT IT MUST be of the same size as a state vector")
                raise AssertionError
        else:
            _inflation_factor = ensemble[0].copy()
            _inflation_factor[:] = inflation_factor[:]

    else:
        print("inflation factor has to be a scalar or an iterable of length equal to model.state_size")
        raise AssertionError


    # Check if inflation is actually required:
    # i.e. if scalar --> >1
    #      if iterable --> at least one entry > 1.0
    if _scalar:
        if _inflation_factor == 1.0:
            return ensemble
        elif _inflation_factor < 0.0:
            print("inflation factor has to be a POSITIVE scalar")
            raise ValueError
        else:
            # Inflation is to be carried-out...
            pass
    else:
        _inf = False
        for i in _inflation_factor:
            if i > 1.0:
                _inf = True
                break
        if not _inf:
            return ensemble
        else:
            # Inflation is to be carried-out
            pass
    #
    # If this point is reached, an ensemble inflation is carried out...
    # print('\ninflating now....\n')
    ensemble_size = len(ensemble)
    mean_vec = ensemble_mean(ensemble)
    if in_place:
        inflated_ensemble = ensemble
        #
        for ens_ind in xrange(ensemble_size):
            state = inflated_ensemble[ens_ind]
            if _scalar:
                state = (state.axpy(-1.0, mean_vec)).scale(_inflation_factor)
            else:
                state = (state.axpy(-1.0, mean_vec)).multiply(_inflation_factor)

            inflated_ensemble[ens_ind] = state.add(mean_vec)
    else:
        inflated_ensemble = []
        mean_vec = ensemble_mean(ensemble)
        #
        for ens_ind in xrange(ensemble_size):
            # print('ens_ind', ens_ind)
            # innovation_vec = (mean_vec.scale(-1.0, in_place=False).add(ensemble[ens_ind])).scale(_inflation_factor)
            state = ensemble[ens_ind].copy()
            if _scalar:
                state = (state.axpy(-1.0, mean_vec)).scale(_inflation_factor)
            else:
                state = (state.axpy(-1.0, mean_vec)).multiply(_inflation_factor)

            inflated_ensemble.append(state.add(mean_vec))
        #
    return inflated_ensemble



# def inflate_ensemble(ensemble, inflation_factor, in_place=True, return_anomalies=False, scale_anomalies=True):
#     """
#     Apply inflation on an ensemble of states
#
#     Args:
#         ensemble: list of model states
#         inflation_factor: scalr or iterable of length equal to model states
#         in_place: overwrite passed ensemble
#         return_anomalies: return inflated anomalies instead with of the inflated ensemble
#         scale_anomalies: if True, ensemble of anomalies is multiplied by 1/sqrt(Nens-1)
#
#     Returns:
#         inflated_ensemble: either inflated ensemble of states or inflated ensemble of state
#             anomalies based on 'return_anomalies'
#
#     """
#     # print("Inflation with:", inflation_factor)
#     assert isinstance(ensemble, list), "The passed ensemble must be a list of model states!"
#     if len(ensemble) == 0:
#         print( "*"*50 + "\nThe passed ensemble is empty!\n" + "*"*50)
#         return
#
#     if np.isscalar(inflation_factor):
#         _scalar = True
#         _inflation_factor = float(inflation_factor)
#     elif isiterable(inflation_factor):
#         if len(inflation_factor) == 1:  # Extract inflation factor if it's a single value wrapped in an iterable
#             _inflation_factor = np.array(inflation_factor)
#             for i in xrange(_inflation_factor.ndim):
#                 _inflation_factor = _inflation_factor[i]
#             _scalar = True
#         else:
#             _scalar = False
#
#         # check type and size:
#         if type(ensemble[0]) is type(inflation_factor):
#             if ensemble[0].size == inflation_factor.size:
#                 _inflation_factor = inflation_factor
#             else:
#                 print("Size Mismatch;\n\tInflation Factor can be an iterable BUT IT MUST be of the same size as a state vector")
#                 raise AssertionError
#         else:
#             _inflation_factor = ensemble[0].copy()
#             _inflation_factor[:] = inflation_factor[:]
#
#     else:
#         print("inflation factor has to be a scalar or an iterable of length equal to model.state_size")
#         raise AssertionError
#
#
#     # Check if inflation is actually required:
#     # i.e. if scalar --> >1
#     #      if iterable --> at least one entry > 1.0
#     if _scalar:
#         if _inflation_factor == 1.0:
#             return ensemble
#         elif _inflation_factor < 0.0:
#             print("inflation factor has to be a POSITIVE scalar")
#             raise ValueError
#         else:
#             # Inflation is to be carried-out...
#             pass
#     else:
#         _inf = False
#         for i in _inflation_factor:
#             if i > 1.0:
#                 _inf = True
#                 break
#         if not _inf:
#             return ensemble
#         else:
#             # Inflation is to be carried-out
#             pass
#     #
#     # If this point is reached, an ensemble inflation is carried out...
#     # print('\ninflating now....\n')
#     ensemble_size = len(ensemble)
#     mean_vec = ensemble_mean(ensemble)
#     if in_place:
#         inflated_ensemble = ensemble
#         #
#         for ens_ind in xrange(ensemble_size):
#             state = inflated_ensemble[ens_ind]
#             if _scalar:
#                 state = (state.axpy(-1.0, mean_vec)).scale(_inflation_factor)
#             else:
#                 state = (state.axpy(-1.0, mean_vec)).multiply(_inflation_factor)
#
#             if return_anomalies:
#                 inflated_ensemble[ens_ind] = state.add(mean_vec)
#                 if scale_anomalies:
#                     inflated_ensemble[ens_ind].scale(1.0/np.sqrt(ensemble_size-1))
#             else:
#                 inflated_ensemble[ens_ind] = state.add(mean_vec)
#     else:
#         inflated_ensemble = []
#         mean_vec = ensemble_mean(ensemble)
#         #
#         for ens_ind in xrange(ensemble_size):
#             # print('ens_ind', ens_ind)
#             # innovation_vec = (mean_vec.scale(-1.0, in_place=False).add(ensemble[ens_ind])).scale(_inflation_factor)
#             state = ensemble[ens_ind].copy()
#             if _scalar:
#                 state = (state.axpy(-1.0, mean_vec)).scale(_inflation_factor)
#             else:
#                 state = (state.axpy(-1.0, mean_vec)).multiply(_inflation_factor)
#
#             if return_anomalies:
#                 inflated_ensemble.append(state)
#                 if scale_anomalies:
#                     inflated_ensemble[ens_ind].scale(1.0/np.sqrt(ensemble_size-1))
#             else:
#                 inflated_ensemble.append(state.add(mean_vec))
#         #
#     return inflated_ensemble
#
#


def rank_hist(ensembles_repo, reference_repo, first_var=0,
                                              last_var=None,
                                              var_skp=1,
                                              draw_hist=False,
                                              target_fig=None,
                                              target_ax=None,
                                              hist_type='relfreq',
                                              first_time_ind=0,
                                              last_time_ind=None,
                                              time_ind_skp=1,
                                              hist_title=None,
                                              hist_max_height=None,
                                              font_size=None,
                                              ignore_indexes=None,
                                              add_fitted_beta=False,
                                              add_uniform=False,
                                              zorder=0):
    """
    Calculate the rank statistics of the true solution/observations w.r.t
    an ensemble of states/observations

    Args:
        ensembles_repo: an ensemble of model states (or model observations).
                        A numpy array of size (state/observation size, ensemble size, time instances)
        reference_repo: a list of reference states (or model observations)
                        A numpy array of size (state/observation size, time instances)
        first_var: initial index in the reference states to evaluate ranks at
        last_var: last index in the reference states to evaluate ranks at
        var_skp: number of skipped variables to reduce correlation effect
        draw_hist: If True, a rank histogram is plotted, and a figure handle is returned,
                   None is returned otherwise
        target_fig, target axes are used if draw_hist is True. If target_ax is not None, histogram is added to it,
        hist_type: 'freq' vs 'relfreq': Frequency vs Relative frequencies for plotting.
                   Used only when 'draw_hist' is True.
        first_time_ind: initial index in the time dimension to evaluate ranks at
        last_time_ind: last index in the time dimension to evaluate ranks at
        time_ind_skp: number of skipped time instances to reduce correlation effect
        hist_title: histogram plot title (if given), and 'draw_hist' is True.
        hist_max_height: ,
        font_size: ,
        ignore_indexes: 1d iterable stating indexes of the state vector to ignore while calculating frequencies/relative frequencies
        add_fitted_beta: fit a beta disgtribution, and add to plot (only if draw_hist is True)
        add_uniform: add a perfect uniform distribution, and add to plot (only if draw_hist is True)
        zorder: order of the bars on the figure

    Returns:
        ranks_freq: frequencies of the rank of truth among ensemble members
        ranks_rel_freq: relative frequencies of the rank of truth among ensemble members
        bins_bounds: bounds of the bar plot
        fig_hist: a matlab.pyplot figure handle of the rank histogram plot

    """
    # Assertions:
    assert isinstance(ensembles_repo, np.ndarray), "First argument 'ensembles_repo' must be a Numpy array!"
    assert isinstance(reference_repo, np.ndarray), "Second argument 'reference_repo' must be a Numpy array!"
    #
    assert isinstance(first_var, int), "'first_var' has to be an integer!"
    if last_var is not None:
        assert isinstance(last_var, int), "'last_var' has to be either None, or an integer!"
    if var_skp is not None:
        assert isinstance(var_skp, int), "'var_skp' has to be either None, or an integer!"

    assert isinstance(first_time_ind, int), "'first_time_ind' has to be an integer!"
    if last_time_ind is not None:
        assert isinstance(last_time_ind, int), "'last_time_ind' has to be either None, or an integer!"
    if time_ind_skp is not None:
        assert isinstance(time_ind_skp, int), "'time_ind_skp' has to be either None, or an integer!"
    #

    if ignore_indexes is not None:
        local_ignore_inds = np.asarray(ignore_indexes).squeeze()
    else:
        local_ignore_inds = None

    # Check dimensionality of inputs:
    # 1- get a squeezed view of 'ensembles_repo'
    loc_ensembles_repo = ensembles_repo.squeeze()
    ens_ndim = loc_ensembles_repo.ndim
    if not (2 <= ens_ndim <= 3):
        print("Ensemble repository has to be either 2, or 3, dimensional Numpy array!")
        print("loc_ensembles_repo is of dimension: %s" %str(ens_ndim))
        raise AssertionError
        #
    elif ens_ndim == 2:
        loc_ensembles_repo_shape = loc_ensembles_repo.shape
        time_inds = xrange(1)
        #
    elif ens_ndim == 3:
        loc_ensembles_repo_shape = loc_ensembles_repo.shape
        time_inds = xrange(loc_ensembles_repo.shape[2])
        #
    else:
        print("This is an impossible situation! Check the code!?")
        raise ValueError
        #
    state_size = loc_ensembles_repo_shape[0]
    ensemble_size = loc_ensembles_repo_shape[1]
    nobs_times = len(time_inds)
    #

    # 2- get a squeezed view of 'reference_repo'
    loc_reference_repo = reference_repo.squeeze()
    ref_dim = loc_reference_repo.ndim
    if ref_dim == 1:
        ref_state_size = loc_reference_repo.size
    elif ref_dim == 2:
        ref_state_size = loc_reference_repo.shape[0]
        if loc_reference_repo.shape[1] != nobs_times:
            print("Mismatch in dimensionality in the time dimension!")
            raise AssertionError
    else:
        print("reference state/observaiton dimension has to be state size by 1 or two !")
        raise AssertionError

    if state_size != ref_state_size:
        print("Mismatch in state/observation size!")
        raise AssertionError

    #
    if not (0 <= first_var <= state_size-1):
        first_var = 0
    #
    if last_var is None:
        last_var = state_size-1
    else:
        if last_var > state_size-1:
            last_var = state_size-1
        elif last_var < 0:
            last_var = 1
    #
    if not (1 <= var_skp):
        first_var = 1

    #
    if not (0 <= first_time_ind <= nobs_times-1):
        first_time_ind = 0
    #
    if last_time_ind is None:
        last_time_ind = nobs_times-1
    else:
        if last_time_ind > nobs_times-1:
            last_time_ind = nobs_times-1
        elif last_time_ind < 0:
            last_time_ind = 1
    #
    if not (1 <= time_ind_skp):
        time_ind_skp = 1

    if not isinstance(hist_type, str):
        hist_type = 'freq'
    #
    # Done with assertion, and validation...
    #

    #
    # Initialize results placeholders:
    ranksmat_length = ensemble_size + 1
    ranks_freq = np.zeros(ranksmat_length, dtype=int)

    #
    # Start calculating ranks (of truth) w.r.t ensembles:
    if nobs_times < 1:
        print("How is that possible? 'nobs_times is [%d] < 1 ?!'" % nobs_times)
        raise ValueError
        #
    elif nobs_times == 1:
        for var_ind in xrange (first_var, last_var+1, var_skp):
            #
            if local_ignore_inds is not None:
                if var_ind in local_ignore_inds:
                    # print("Skipping Index [%d] of the state vector, from rank evaluation..." % var_ind)
                    continue
            else:
                pass
            augmented_vec = loc_ensembles_repo[var_ind, :]
            ref_sol = loc_reference_repo[var_ind]
            augmented_vec = np.append(augmented_vec, ref_sol).squeeze()

            rnk = np.where(np.argsort(augmented_vec) == augmented_vec.size-1)[0][0]
            # rnk = np.argsort(augmented_vec)[-1]  # get rank of true/ref state/observation

            ranks_freq[rnk] += 1
            #
        #
    else:
        for time_ind in xrange (first_time_ind, last_time_ind+1, time_ind_skp):
            for var_ind in xrange (first_var, last_var+1, var_skp):
                #
                if local_ignore_inds is not None:
                    if var_ind in local_ignore_inds:
                        # print("Skipping Index [%d] of the state vector, from rank evaluation...")
                        continue
                else:
                    pass
                augmented_vec = loc_ensembles_repo[var_ind, :, time_ind]
                ref_sol = loc_reference_repo[var_ind, time_ind]
                augmented_vec = np.append(augmented_vec, ref_sol).squeeze()

                rnk = np.where(np.argsort(augmented_vec) == augmented_vec.size-1)[0][0]
                # rnk = np.argsort(augmented_vec)[-1]  # get rank of true/ref state/observation

                ranks_freq[rnk] += 1
                #

    # calculate ranks relative frequences
    ranks_rel_freq = ranks_freq / float(ranks_freq.sum())

    # bounds of rank histogram plot:
    bins_bounds = np.arange(ensemble_size+1)

    if draw_hist:
        beta_label = None
        u_label = None

        # Based on hist_type decide on the height of histogram bars
        if hist_type.lower() == 'freq':
            bins_heights = ranks_freq
            ylabel = 'Frequency'
        elif hist_type.lower() == 'relfreq':
            bins_heights = ranks_rel_freq
            ylabel = 'Relative Frequency'
        else:
            print("Unrecognized histogram plot type %s" % hist_type)
            raise ValueError

        # Start plotting:
        if target_fig is None and target_ax is None:
            fig_hist, ax = plt.subplots(facecolor='white')
        elif target_fig is None:
            ax = target_ax
            fig_hist = ax.get_figure()
        elif target_ax is None:
            fig_hist = target_fig
            ax = fig_hist.gca()
        ax.bar(bins_bounds , bins_heights, width=1, color='green', edgecolor='black', zorder=zorder)

        # Adjust limits of the plot as necessary:
        ax.set_xlim(-0.5, ensemble_size+0.5)
        if hist_max_height is not None and np.isscalar(hist_max_height):
            ax.set_ylim(0, max(hist_max_height, bins_heights.max()+1e-2))
        #
        if hist_title is not None and isinstance(hist_title, str):
            if font_size is not None:
                fig_hist.suptitle(hist_title, fontsize=font_size)
                for tickx, ticky in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
                    tickx.label.set_fontsize(font_size)
                    ticky.label.set_fontsize(font_size)
            else:
                fig_hist.suptitle(hist_title)

        if font_size is not None:
            ax.set_xlabel("Rank", fontsize=font_size)
            ax.set_ylabel(ylabel, fontsize=font_size)
        else:
            ax.set_xlabel("Rank")
            ax.set_ylabel(ylabel)
        #
        if add_fitted_beta:
            # Fit a beta distribution:
            # create data from frequencies:
            data = []
            for fr, bn in zip(ranks_freq, bins_bounds):
                data += [float(bn)] * fr
            data = np.asarray(data)
            # fit beta dist to generated data:
            dist = scipy_stats.beta
            params = dist.fit(data)  # beta distribution parameters
            # print("Beta fitted params: ", params)

            # generate a pdf curve for fitted beta:
            pdf_x = np.linspace(dist.ppf(0.01, params[0], params[1]), dist.ppf(0.99, params[0], params[1]), 100)
            pdf_y = dist.pdf(pdf_x, params[0], params[1])

            # avoid very large values
            pdf_y[np.where(pdf_y>1e+3)[0]] = np.nan


            # shift X values to 0 to ensemble_size
            a, b = np.nanmin(pdf_x), np.nanmax(pdf_x)
            c, d = 0, ensemble_size
            pdf_x = c + ((d-c)/(b-a)) * (pdf_x-a)

            # scale Y's
            if True:
                _ylims = ax.get_ylim()
                y_scale = _ylims[1] - _ylims[0]
                ul = y_scale*0.95 + _ylims[0]
                fac = ul / np.nanmax(pdf_y)
                pdf_y = pdf_y*fac + _ylims[0]
            else:
                a, b = np.nanmin(pdf_y), np.nanmax(pdf_y)
                _ylims = ax.get_ylim()
                y_scale = _ylims[1] - _ylims[0]
                c = y_scale*0.05 + _ylims[0]
                d = y_scale*0.95 + _ylims[0]
                pdf_y = c + ((d-c)/(b-a)) * (pdf_y-a)

            # Adjust pdf_y to the bins:
            if hist_type.lower() == 'freq':
                if np.nanmin(pdf_y) < ranks_freq.min():
                    pdf_y += abs(np.nanmin(pdf_y)-ranks_freq.min())
                else:
                    pdf_y -= abs(np.nanmin(pdf_y)-ranks_freq.min())
            elif hist_type.lower() == 'relfreq':
                if np.nanmin(pdf_y) < ranks_rel_freq.min():
                    pdf_y += abs(np.nanmin(pdf_y)-ranks_rel_freq.min())
                else:
                    pdf_y -= abs(np.nanmin(pdf_y)-ranks_rel_freq.min())
            else:
                print("Unrecognized histogram plot type %s" % hist_type)
                raise ValueError

            zorder += 1
            try:
                if add_uniform:
                    beta_label = r'$\beta$(%3.2f, %3.2f)'%(params[0], params[1])
                else:
                    beta_label = None
                ax.plot(pdf_x, pdf_y, 'r-', linewidth=3, label=beta_label, zorder=zorder)
            except(RuntimeError):
                if add_fitted_uniform:
                    beta_label = 'Beta(%3.2f, %3.2f)'%(params[0], params[1])
                else:
                    beta_label = None
                ax.plot(pdf_x, pdf_y, 'r-', linewidth=3, label=beta_label, zorder=zorder)
            # Update y limits; just in-case!
            ylim = ax.get_ylim()
            if hist_type.lower() == 'freq':
                ax.set_ylim([ylim[0], max(ylim[-1], np.nanmax(pdf_y), ranks_freq.max())])
            elif hist_type.lower() == 'relfreq':
                ax.set_ylim([ylim[0], max(ylim[-1], np.nanmax(pdf_y), ranks_rel_freq.max())])
            else:
                print("Unrecognized histogram plot type %s" % hist_type)
                raise ValueError

        # Add perfect uniform distribution
        if add_uniform:
            # get average height:
            if hist_type.lower() == 'freq':
                avg_height = np.mean(ranks_freq)
            elif hist_type.lower() == 'relfreq':
                avg_height = np.mean(ranks_rel_freq)
            else:
                print("Unrecognized histogram plot type %s" % hist_type)
                raise ValueError
            xlim = ax.get_xlim()
            zorder += 1

            try:
                if add_fitted_beta:
                    u_label = r'$\mathcal{U}$'
                else:
                    u_label = None
                ax.plot(xlim, [avg_height, avg_height], 'b--', linewidth=3, label=u_label, zorder=zorder)
            except(RuntimeError):
                if add_fitted_beta:
                    u_label = 'Uniform'
                else:
                    u_label = None
                ax.plot(xlim, [avg_height, avg_height], 'b--', linewidth=3, label=u_label, zorder=zorder)

        # Add legend
        if add_fitted_beta or add_uniform:
            if beta_label is u_label is None:
                pass
            else:
                ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.075), fancybox=True, shadow=True)

        # Draw everthing
        plt.draw()
        #
    else:
        fig_hist = None
    #
    return ranks_freq, ranks_rel_freq, bins_bounds , fig_hist
    #


#
def random_orthonormal_matrix(dimension):
    """
    Generates a random orthonormal matrix O such that:
    Q * Q^T = I, where I is the identity matrix of size dimension x dimension

    Args:
        dimension: size of the random orthonormal matrix to be generated

    Returns:
        Q: the random orthonormal matrix, Q is of size (dimension x dimension)

    """
    # preserve the state of the random-number generator
    # get the current state of the random number generator
    current_random_state = np.random.get_state()

    Q, R = np.linalg.qr(np.random.randn(dimension, dimension))
    for dim_ind in xrange(dimension):
        if R[dim_ind, dim_ind] < 0:
            Q[:, dim_ind] *= -1.0

    # Restore the state of the gaussian random number generator:
    np.random.set_state(current_random_state)

    return Q
    #

def random_orthonormal_mean_preserving_matrix(dimension):
    """
    Generates a random orthonormal mean-preserving matrix Q, such that:
        Q * Q^T = I, where I is the identity matrix of size dimension x dimension, and
        Q * II = 0, where II is a column vector of ones, and 0 is a column vector of zeros.

    Args:
        dimension: size of the random orthonormal matrix to be generated

    Returns:
        Q: the random orthonormal matrix, Q is of size (dimension x dimension)

    """
    II = np.ones((dimension, 1))
    U, _, _ = np.linalg.svd(II)
    orth_rand_mat = random_orthonormal_matrix(dimension-1)
    #
    Q = scipy.linalg.block_diag(1.0, orth_rand_mat)
    Q = np.dot(Q, U.T)
    Q = np.dot(U, Q)
    #
    return Q



def covariance_trace(ensemble, model=None, row_var=False, ddof=1):
    """
    Evaluate the trace of the covariance matrix given an ensemble of states.

    Args:
        ensemble: a list of model states, or a Numpy array.
                  If it is a numpy array, each column is taken as state (row_vars=False).
        model: model object. Needed of the passed ensemble is a list of model states.
        row_var: active only if ensemble is a Numpy-nd array. Each row is a state variable.
                  Set to True if each column is a state variable.
        ddof: degree of freedom; the variance is corrected by dividing by sample_size - ddof

    Returns:
        trace: the trace of the covariance matrix of the ensemble.
               This is the sum of the ensemble-based variances of the state variables.
    """
    if isinstance(ensemble, np.ndarray):
        if row_var:
            axis = 0
        else:
            axis = 1
        variances = np.var(ensemble, axis=axis, ddof=ddof)
        trace = variances.sum()

    elif isinstance(ensemble, list):
        #
        trace = ensemble_variances(ensemble=ensemble, ddof=ddof).sum()

        # TODO: Consider calculating the variances from the ensemble directly by iterating over entries

    else:
        print("The ensemble has to be either a numpy array or a list of model-based ensemble states.")
        raise TypeError

    return trace

def ensemble_anomalies(ensemble, scale=False, ddof=1.0, in_place=False, inflation_factor=None, model=None):
    """
    Return an ensemble of anomalies given an ensemble;
    This shifts the ensemble by subtracting the ensemble mean.
    If scale is True, the ensemble deviations are scaled by the reciprocal of square root of ensemble size minus ddof

    Args:
        ensemble: list of model states
        scale: Flag to scale ensemble deviations/anomalies
        ddof: only 0 or 1 are accepted; subtracted from ensemble size to avoid biasedness
        in_place: overwrite entries of the passed ensemble/list
        model: an instance of a dynamical model

    Returns:
        ens_anomalies: ensemble of state anomalies

    """
    assert isinstance(ensemble, list), "passed ensemble must be a list of model states!"
    assert ddof in [0, 1], "ddof must be either 0, or 1; Not: %s" %str(ddof)

    if inflation_factor is not None:
        if np.isscalar(inflation_factor):
            _inf = _scalar = True
            _inflation_factor = float(inflation_factor)
        elif isiterable(inflation_factor):
            _inf = True
            if len(inflation_factor) == 1:  # Extract inflation factor if it's a single value wrapped in an iterable
                _inflation_factor = np.array(inflation_factor)
                for i in xrange(_inflation_factor.ndim):
                    _inflation_factor = _inflation_factor[i]
                _scalar = True
            else:
                _scalar = False

            # check type and size:
            if type(ensemble[0]) is type(inflation_factor):
                if ensemble[0].size == inflation_factor.size:
                    _inflation_factor = inflation_factor
                else:
                    print("Size Mismatch;\n\tInflation Factor can be an iterable BUT IT MUST be of the same size as a state vector")
                    raise AssertionError
            else:
                _inflation_factor = ensemble[0].copy()
                _inflation_factor[:] = inflation_factor[:]
        else:
            print("inflation factor has to be a scalar or an iterable of length equal to model.state_size")
            raise AssertionError

        # Check if inflation is actually required:
        # i.e. if scalar --> >1
        #      if iterable --> at least one entry > 1.0
        if _scalar:
            if _inflation_factor == 1.0:
                _inf = False
            elif _inflation_factor < 0.0:
                print("inflation factor has to be a POSITIVE scalar")
                raise ValueError
            else:
                # Inflation is to be carried-out...
                _inf = True  # Just in-case!
                pass
        else:
            _inf = False
            for i in _inflation_factor:
                if i > 1.0:
                    _inf = True
                    break
    else:
        _inf = False


    ens_size = len(ensemble)
    if ens_size <= 1 :
        print("Ensemble must contain at least two state vectors; ensemble passed is of length %d?!" % ens_size)
        raise ValueError

    if scale:
        s_fac = 1.0 / np.sqrt(ens_size - ddof)

    ens_mean = ensemble_mean(ensemble)
    ens_mean = ens_mean.scale(-1.0, in_place=True)  # of course it is negated to be added later than subtracted
    if in_place:
        ens_anomalies = ensemble
    else:
        ens_anomalies = []

    for ens_ind in xrange(ens_size):
        state = ensemble[ens_ind]
        if in_place:
            state.add(ens_mean, in_place=True)
        else:
            innov = state.add(ens_mean, in_place=False)
            ens_anomalies.append(innov)
            state = ens_anomalies[-1]

        # scale by 1/sqrt(ensemble_size = ddof)
        if scale:
            state.scale(s_fac, in_place=True)

        # inflate state anomaly if required
        if _inf:
            if _scalar:
                state.scale(_inflation_factor, in_place=True)
            else:
                state.multiply(_inflation_factor, in_place=True)

    #
    return ens_anomalies

def ensemble_T_dot_state(ensemble, in_state=None, cardinal=None, return_np_array=False):
    """
    Given an ensemble (list of model states), and a state vector,
    return the dot-product of each ensemble member by the passed state
    The result is a list (or a numpy array if return_np_array is True) of the same length/size as the passed ensemble

    Args:
        ensemble: a list of model states.
        in_state: model state
        cardinal: either None, or an integer indicating ith cardinal vector in R^n
        return_np_array: True/False

    Returns:
        res: the dot-product of each ensemble member by the passed state

    """
    # TODO: proceed; this needs to be smart!!!
    assert isinstance(ensemble, list), "passed ensemble must be a list of model states!"

    if in_state is None and cardinal is None:
        print("Either pass a state or an integer indicating a cardinality vector!")
        raise ValueError
    elif in_state is not None and cardinal is not None:
        cardinal = None
    else:
        pass

    ens_size = len(ensemble)
    if ens_size == 0:
        print("Ensemble passed is empty?!")
        raise ValueError

    try:
        state_size = ensemble[0].size()
    except:
        state_size = len(ensemble[0])

    if cardinal is not None:
        assert isinstance(cardinal, int), "cardinal must be cardinal if to be used!; found %s !" % type(cardinal)
        if not (0 <= cardinal < state_size):
            print("cardinal must satisfy 0 <= cardinal < state_size !")
            raise ValueError

    res = [None]* ens_size  # does pre-allocation here help in reducing the cpu-time?!

    if cardinal is not None:
        for ens_ind in xrange(ens_size):
            res[ens_ind] = ensemble[ens_ind][cardinal]

    elif in_state is not None:
        for ens_ind in xrange(ens_size):
            res[ens_ind] = ensemble[ens_ind].dot(in_state)

    else:
        print("There is a BUG, please debug!!! \n This point can't be reached!")
        raise ValueError

    if return_np_array:
        res = np.asarray(res)
    #
    return res


def ensemble_dot_vec(ensemble, in_vec, out_state=None):
    """
    Given an ensemble (list of model states), and an iterable,
    return the dot-product of ensemble-matrix tranposed dot the passed iterable.
    Args:
        ensemble: a list of model states.
        in_vec: an iterable of length equal to the ensemble size

    Returns:
        out_state: the state returned

    """
    # TODO: proceed; this needs to be smart!!!
    assert isinstance(ensemble, list), "passed ensemble must be a list of model states!"

    ens_size = len(ensemble)
    if ens_size == 0:
        print("Ensemble passed is empty?!")
        raise ValueError

    if len(in_vec) != ens_size:
        print("The passed vector must be of size equal to the ensemble size!")
        raise AssertionError

    if isinstance(in_vec, np.ndarray):
        vec = in_vec
    else:
        vec = np.asarray(in_vec)

    if out_state is None:
        out_state = ensemble[0].scale(vec[0], in_place=False)
    else:
        out_state[:] = ensemble[0].scale(vec[0], in_place=False)[:]

    try:
        state_size = out_state.size()
    except:
        state_size = len(out_state)

    #
    for i in xrange(1, ens_size):
        out_state = out_state.axpy(vec[i], ensemble[i], in_place=True)

    return out_state


def ensemble_covariance_dot_state(ensemble, in_state=None, cardinal=None, ddof=1, model=None, use_anomalies=True, out_state=None):
    """
    Given an ensemble of states (list of state vectors), evaluate the effect of the ensemble-based
    covariance matrix on the passed state vector.

    Args:
        ensemble: a list of model states.
                  If it is a numpy array, each column is taken as state (row_vars=False).
        in_state: state vector to multiply ensemble covariance with
        cardinal: if an integer,
        ddof: degree of freedom; the variance is corrected by dividing by sample_size - ddof
        model:
        use_anomalies:

    Returns:
        out_state: The result of multiplying the ensemble-based covariance matrix by the given state.

    """
    assert isinstance(ensemble, list), "passed ensemble must be a list of model states!"

    if in_state is None and cardinal is None:
        print("Either pass a state or an integer indicating a cardinality vector!")
        raise ValueError
    elif in_state is not None and cardinal is not None:
        cardinal = None
    else:
        pass

    assert ddof in [0, 1], "ddof must be either 0, or 1; Not: %s" %str(ddof)

    ens_size = len(ensemble)

    if ens_size == 0:
        print("Ensemble passed is empty?!")
        raise ValueError

    if model is not None:
        state_size = model.state_size()
    else:
        try:
            state_size = ensemble[0].size()
        except:
            state_size = len(ensemble[0])

    #
    if use_anomalies:
        ens_anomalies = ensemble_anomalies(ensemble, scale=True, ddof=ddof, in_place=False)
        vec = ensemble_T_dot_state(ens_anomalies, in_state=in_state, cardinal=cardinal, return_np_array=True)
        out_state = ensemble_dot_vec(ens_anomalies, vec, out_state=out_state)

    else:
        if out_state is None:
            if model is not None:
                out_state = model.state_vector()
            else:
                out_state = ensemble[0].copy()

        if cardinal is not None:
            if model is not None:
                in_state = model.state_vector()
            else:
                in_state = out_state.copy()
            in_state[:] = 0.0
            in_state[cardinal] = 1.0

        ens_mean = ensemble_mean(ensemble).scale(-1)
        devs = np.empty((state_size, ens_size))
        #
        for ens_ind in xrange(ens_size):
            ens_member = ensemble[ens_ind]
            devs[:, ens_ind] = ens_member.add(ens_mean, in_place=False).get_numpy_array()

        result_state_np = devs.T.dot(in_state.get_numpy_array())
        result_state_np = devs.dot(result_state_np)
        result_state_np /= float(ens_size - ddof)
        out_state[:] = result_state_np[:]

    #
    return out_state
