
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
import matplotlib.pyplot as plt

import scipy
import re
from _utility_stat import ensemble_mean

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
    if np.isscalar(distances):
        distances = np.array([distances])  # this is local now
        
    try:
        distances = np.squeeze(np.asarray(distances[:]))  # this is local now
    except:
        print("Failed to read the passed distances!")
        raise ValueError
    
    #
    if method is None:
        if np.size(distances) == 1:
            return 1
        else:
            return np.ones_like(distances)
    
    if isinstance(distances, np.ndarray):
        if distances.ndim == 0:
            distances = np.array([distances.item()])
            
    if np.size(distances) == 1:
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
        # a scalar is passed in this case...
        if re.match(r'\Agauss\Z', method, re.IGNORECASE):
            coefficients = np.exp(-0.5*((distances/float(radius))**2))
        elif re.match(r'\Aexp3\Z', method, re.IGNORECASE):
            coefficients = np.exp(-0.5 * (distances / float(radius)) ** 3)
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
            radius = np.asarray(radius).squeeze()
            if radius.size != distances.size:
                print "distances, and radius have to be of equal sizes!"
                raise AssertionError
            
        #  distances is of dimension greater than one. vector is assumed
        if re.match(r'\Agauss\Z', method, re.IGNORECASE):
            coefficients = np.exp(-0.5*((distances/float(radius))**2))

        elif re.match(r'\Aexp3\Z', method, re.IGNORECASE):
            coefficients = np.exp(-0.5 * (distances / float(radius)) ** 3);

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

    return coefficients
#


def inflate_ensemble(ensemble, inflation_factor, in_place=True):
    """
    Apply inflation on an ensemble of states
    """
    assert isinstance(ensemble, list)
    if isinstance(inflation_factor, int) or isinstance(inflation_factor, float):
        inflation_factor = float(inflation_factor)
    else:
        raise AssertionError("inflation factor has to be a scalar")

        
    if inflation_factor == 1.0:
        pass
    elif inflation_factor < 0:
        raise ValueError("inflation factor has to be a POSITIVE scalar")
    else:
        # print('\ninflating now....\n')
        ensemble_size = len(ensemble)
        mean_vec = ensemble_mean(ensemble)
        if in_place:
            inflated_ensemble = ensemble
            #
            for ens_ind in xrange(ensemble_size):
                state = ensemble[ens_ind]
                state = (state.axpy(-1.0, mean_vec)).scale(inflation_factor)
                # innovation_vec = (mean_vec.scale(-1.0, in_place=False).add(state)).scale(inflation_factor)
                inflated_ensemble[ens_ind] = state.add(mean_vec)
            ensemble = inflated_ensemble
        else:
            inflated_ensemble = []
            mean_vec = ensemble_mean(ensemble)
            #
            for ens_ind in xrange(ensemble_size):
                # print('ens_ind', ens_ind)
                # innovation_vec = (mean_vec.scale(-1.0, in_place=False).add(ensemble[ens_ind])).scale(inflation_factor)
                state = ensemble[ens_ind]
                state = (state.axpy(-1.0, mean_vec)).scale(inflation_factor)
                inflated_ensemble.append(state.add(mean_vec))
                # innovation_vec = ensemble[ens_ind].axpy(-1.0, mean_vec, in_place=False)
                # innovation_vec = innovation_vec.scale(inflation_factor)
                # inflated_ensemble.append(innovation_vec.add(mean_vec).copy())
            ensemble = inflated_ensemble
    #
    return ensemble


def rank_hist(ensembles_repo, reference_repo, first_var=0, 
                                              last_var=None, 
                                              var_skp=1, 
                                              draw_hist=False, 
                                              hist_type='relfreq', 
                                              first_time_ind=0, 
                                              last_time_ind=None,
                                              time_ind_skp=1, 
                                              hist_title=None,
                                              hist_max_height=None,
                                              font_size=None,
                                              ignore_indexes=None,
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
        hist_type: 'freq' vs 'relfreq': Frequency vs Relative frequencies for plotting. 
                   Used only when 'draw_hist' is True.
        first_time_ind: initial index in the time dimension to evaluate ranks at
        last_time_ind: last index in the time dimension to evaluate ranks at
        time_ind_skp: number of skipped time instances to reduce correlation effect
        hist_title: histogram plot title (if given), and 'draw_hist' is True.
        hist_max_height: ,
        font_size: ,
        ignore_indexes: 1d iterable stating indexes of the state vector to ignore while calculating frequencies/relative frequencies
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
        fig_hist = plt.figure(facecolor='white')
        ax = plt.subplot(111)
        ax.bar(bins_bounds , bins_heights, width=1, color='green', edgecolor='black', zorder=zorder)
        
        # Adjust limits of the plot as necessary:
        ax.set_xlim(-0.5, ensemble_size+0.5)
        if hist_max_height is not None and np.isscalar(hist_max_height):
            ax.set_ylim(0, max(hist_max_height, bins_heights.max()+1e-5))
        #
        if hist_title is not None and isinstance(hist_title, str):
            if font_size is not None:
                fig_hist.suptitle(hist_title, fontsize=FS)
                for tickx, ticky in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
                    tickx.label.set_fontsize(FS)
                    ticky.label.set_fontsize(FS)
            else:
                fig_hist.suptitle(hist_title)
        
        if font_size is not None:
            ax.set_xlabel("Rank", fontsize=FS)
            ax.set_ylabel(ylabel, fontsize=FS)
        else:
            ax.set_xlabel("Rank")
            ax.set_ylabel(ylabel)
        #
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
        print "axis : ", axis
        variances = np.var(ensemble, axis=axis, ddof=ddof)
        trace = variances.sum()
        
    elif isinstance(ensemble, list):
        #
        if model is None:
            print("An instance of a model object has to be passed when the ensemble is a list of model states")
            raise ValueError
            
        model = model
        ens_size = len(ensemble)
        state_size = model.state_size()
        ensemble_np = np.empty((state_size, ens_size))
        for ens_ind in xrange(ens_size):
            ensemble_np[:, ens_ind] = ensemble[ens_ind].get_numpy_array()
        
        trace = covariance_trace(ensemble_np, row_var=False)
        
    else:
        print("The ensemble has to be either a numpy array or a list of model-based ensemble states.")
        raise TypeError
    
    return trace

def ensemble_covariance_dot_state(ensemble, in_state, model=None):
    """
    Given an ensemble of states (list of state vectors), evaluate the effect of the ensemble-based
    covariance matrix on the passed state vector.
    
    Args:
        ensemble: a list of model states.
                  If it is a numpy array, each column is taken as state (row_vars=False).
        in_state: state vector
        
    Returns:
        out_state: The result of multiplying the ensemble-based covariance matrix by the given state.
        
    """
    assert isinstance(ensemble, list)
    
    if model is None:
        print("An instance of a model object has to be passed when the ensemble is a list of model states")
        raise ValueError
    
    ens_mean = ensemble_mean(ensemble).scale(-1)
    ens_size = len(ensemble)
    state_size = model.state_size()
    devs = np.empty((state_size, ens_size))
    #
    for ens_ind in xrange(ens_size):
        ens_member = ensemble[ens_ind]
        devs[:, ens_ind] = ens_member.add(ens_mean, in_place=False).get_numpy_array()
    
    result_state_np = devs.T.dot(in_state.get_numpy_array())
    result_state_np = devs.dot(result_state_np) / (ens_size - 1.0)
    
    out_state = model.state_vector()
    out_state[:] = result_state_np[:]
    #
    return out_state    
    
    
    
    
    
