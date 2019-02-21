
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
#   Set of miscelaneous functions to manipulate  plots/figures                                #
# ########################################################################################## #
#

import numpy as np
# MAKE SURE this module does not import any of the other utility modules, and make sure you afoic cyclic import

"""
    A module providing miscelaneous functions
"""

def isiterable(a):
    """
    A simple function to check if
    """
    check = False
    try:
        a.__iter__
        check = True
    except(AttributeError):
        check = False
    return check


def isscalar(a):
    """
    wrapper to numpy.isscalar
    """
    return np.isscalar(a)


def ensemble_to_np_array(ensemble, state_as_col=True, model=None):
    """
    Return a numpy array representation of the passed ensemble. Orientation is based on the second argument 'state_as_col'

    Args:
        ensemble: a list of model states
        state_as_col: if True, each state vector is saved as a column in the numpy array, otherwise, each state is saved as a row
        model: an instance of a model object used to create entries of the passed ensemble

    Returns:
        ens_np: a numpy array with each column/row equal to a model state
    """
    assert isinstance(ensemble, list), "passed ensemble must be a list of model states!"

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

    if state_as_col:
        ens_np = np.empty((state_size, ens_size))
    else:
        ens_np = np.empty((ens_size, state_size))

    for i in xrange(ens_size):
        state = ensemble[i]
        if state_as_col:
            ens_np[:, i] = state.get_numpy_array()
        else:
            ens_np[i, :] = state.get_numpy_array()

    return ens_np


# Calculate Euclidean distance
def euclidean_distance(in_p0, in_p1):
    """
    calculate the Euclidean distances;

    Args:
        in_p0, in_p1 are two numpy arrays where each row is a point coordinates;
        either in_p0, and in_p1 are of equal size (length), or one of them is of size=1, and taken as reference points, otherwise AssertionError is raised

        if both in_p0, and p1 are scalars, the return value is also a scalar

    Returns:
        distances: a Numpy of one dimension containining Euclidean distances between in_p0, in_p1

    """
    if isscalar(in_p0):
        p0 = np.asarray([[in_p0]])
    else:
        p0 = in_p0
    if isscalar(in_p1):
        p1 = np.asarray([[in_p1]])
    else:
        p1 = in_p1

    if isinstance(in_p0, np.ndarray) and isinstance(in_p1, np.ndarray):
        p0, p1 = in_p0, in_p1
    # elif isiterable(in_p1) and isiterable(in_p1):
    #     # give it a try
    #
    #     pass
    else:
        print("in_p0, and in_p1 must either be scalars, or two numpy-arrays with equal dimensions!")
        raise AssertionError

    if p0.ndim == 1:
        p0 = p0.reshape(p0.size, 1)

    if p1.ndim == 1:
        p1 = p1.reshape(p1.size, 1)

    if p0.ndim != p1.ndim:
        print("p0, and p1 are not of equal dimensions! %d! = %d" %(p0.ndim, p1.ndim), p0, p1)
        raise ValueError
    else:
        ndims = p0.ndim


    s0 = np.size(p0, 0)
    s1 = np.size(p1, 0)
    
    if not (min(s0, s1)==1 or s0==s1):
        # distances in this case is an array; we can implement it later
        print("Distances in this case is an array; we can implement it later")
        raise NotImplementedError

    if np.min(p1.shape) == min(s0, s1):
        ref_p, var_p = p1, p0
    else:
        ref_p, var_p = p0, p1

    distances = []
    for i in xrange(min(s0, s1)):
        x0 = ref_p[i, :]
        for j in xrange(max(s0, s1)):
            x1 = var_p[j, :]

            dis = np.sqrt(np.sum((x1 - x0)**2))
            distances.append(dis)

    if len(distances) == 1:
        distances = distances[0]
    else:
        distances = np.asarray(distances)
    return distances


def moving_average(x, radius, periodic=False):
    """
    Return a smoothed version of x, where each entry is replaced with the average of itself and values within radius of r grid points
    if periodic is true, the state is seen as positioned on  a circle
    """
    assert radius >=0, "The radius most be nonegative!"
    if radius == 0:
        return x

    state = np.asarray(x, dtype=np.float).flatten()
    state_size = state.size
    if radius > state_size/2:
        print("For a state size [%d], the moving average radius must be within[%d, %d]" % (state_size, 0, state_size/2))
        raise ValueError

    r = radius
    if periodic:
        state = np.concatenate([state[-r:], state, state[: r]], axis=None)
        smoothed_state = state.copy()
        for i in xrange(r, state_size+r):
            smoothed_state[i] = np.mean(state[i-r: i+r+1])
    else:
        smoothed_state = state.copy()
        for i in xrange(r, state_size-r):
            smoothed_state[i] = np.mean(state[i-r: i+r+1])

    if periodic:
        smoothed_state = smoothed_state[r: state_size+r]

    return smoothed_state


def unique_colors(size, np_seed=2456, gradual=False):
    """
    get uniqe random colors on the format (R, G, B) to be used for plotting with many lines
    """
    # Get the current state of Numpy.Random, and set the seed
    if gradual:
        G_vals = np.linspace(0.0, 0.5, size, endpoint=False)
        R_vals = np.linspace(0.0, 0.3, size, endpoint=False)
        B_vals = np.linspace(0.5, 1, size, endpoint=False)
    else:
        np_rnd_state = np.random.get_state()
        np.random.seed(np_seed)

        G_vals = np.linspace(0.0, 1.0, size, endpoint=False)
        R_vals = np.random.choice(xrange(size), size=size, replace=False) / float(size)
        B_vals = np.random.choice(xrange(size), size=size, replace=False) / float(size)

        # Resend Numpy random state
        np.random.set_state(np_rnd_state)

    return [(r, g, b) for r, g, b in zip(R_vals, G_vals, B_vals)]

def str2bool(v):
    if v.lower().strip() in ('yes', 'true', 't', 'y', '1'):
        val = True
    elif v.lower().strip() in ('no', 'false', 'f', 'n', '0'):
        val = False
    else:
        print("Boolean Value is Expected Yes/No, Y/N, True/False, T/F, 1/0")
        raise ValueError
    return val
str_to_bool = str2bool


def add_subplot_axes(ax, rect=None):
    """  TOBE added to the utility module
    Add a subplot with a given axis
    Args:
        ax: matplotlib axis
        rect: [x-position, y-position, relative width, relative-height]
    Returns:
        suax
    """
    fig = ax.get_figure()
    box = ax.get_position()
    width = box.width
    height = box.height
    if rect is None:
        rect = [0.25, 0.4, 0.4, 0.4] 

    inax_position  = ax.transAxes.transform(rect[0: 2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height])
    # Adjust lable sizes, and tick fontsizes
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax



