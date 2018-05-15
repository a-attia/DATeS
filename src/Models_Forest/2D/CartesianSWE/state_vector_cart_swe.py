
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


import numpy as np
from state_vector_base import StateVectorBase
import cart_swe

class StateVectorCartSWE(StateVectorBase):
    """
    Abstract class defining the model state vector.
    """

    def __init__(self):
        """
        State vector constructor.

        Input:
            model_vector_ref: a reference to the model's vector object.
        """
        self._raw_vector = cart_swe.vec_init()
        assert(self._raw_vector != None)
        self._temp_scalar = cart_swe.new_doublep()
        self._temp_integer = cart_swe.new_intp()

    def __del__(self):
        """
        State vector destructor.
        """
        err = cart_swe.vec_del(self._raw_vector)
        self._raw_vector = None
        cart_swe.delete_doublep(self._temp_scalar)
        cart_swe.delete_intp(self._temp_integer)

    def set_raw_vector_ref(self, model_vector_ref):
        """
        Sets the model vector reference to a new object.

        Input:
            model_vector_ref: a reference to the model's vector object.
        """
        self._raw_vector = model_vector_ref

    def get_raw_vector_ref(self):
        """
        Returns a reference to the enclosed vector object constructed by the model.

        Output:
            : a reference to the model's vector object.
        """
        return self._raw_vector

    def get_numpy_array(self):
        """
        """
        err = cart_swe.vec_get_size(self._temp_integer)
        v_length = cart_swe.intp_value(self._temp_integer)
        array = cart_swe.new_darray(v_length)
        err = cart_swe.vec_copy(self._raw_vector, array)
        l = [cart_swe.darray_getitem(array, i) for i in xrange(v_length)]
        cart_swe.delete_darray(array)
        return np.array(l)

    def get_state_size(self):
        err = cart_swe.vec_get_size(self._temp_integer)
        return cart_swe.intp_value(self._temp_integer)

    def scale(self, alpha):
        """
        BLAS-like methods.

        Input:
            alpha: Scale the vector by the constant alpha.
        """
        err = cart_swe.vec_scale(alpha, self._raw_vector)
        return self

    def copy(self):
        """
        Return a copy of the vector.

        Output:
            :
        """
        vector = StateVectorCartSWE()
        err = cart_swe.vec_copy(self._raw_vector, vector._raw_vector)
        return vector

    def dot(self, other):
        """
        Perform a dot product with other.

        Input:
            other:

        Output:
            :
        """
        err = cart_swe.vec_dot(self._raw_vector, other._raw_vector, self._temp_scalar)
        return cart_swe.doublep_value(self._temp_scalar)

    def axpy(self, alpha, other):
        """
        Add the vector with a scaled vector.

        Input:
            alpha:
            other:
            self :

        Output:
            None
        """
        err = cart_swe.vec_axpy(alpha, other._raw_vector, self._raw_vector)
        return self

    def add(self, other):
        """
        Add other to the vector.

        Input:
            other:

        Output:
            :
        """
        err = cart_swe.vec_add(other._raw_vector, self._raw_vector)
        return self

    def norm2(self):
        """
        Return the 2-norm of the vector.

        Output:
            :
        """
        err = cart_swe.vec_norm2(self._raw_vector, self._temp_scalar)
        return cart_swe.doublep_value(self._temp_scalar)

    #
    #
    # ================================================================================================================ #
    #                                          Emulate Python Descriptors                                              #
    # ================================================================================================================ #
    #
    def __repr__(self):
        raise NotImplementedError('Descriptor is not Implemented')

    def __str__(self):
        raise NotImplementedError('Descriptor is not Implemented')

    def __iter__(self):
        raise NotImplementedError('Descriptor is not Implemented')

    def __contains__(self, item):
        raise NotImplementedError('Descriptor is not Implemented')

    def __getitem__(self, item):
        raise NotImplementedError('Descriptor is not Implemented')

    def __setitem__(self, key, value):
        raise NotImplementedError('Descriptor is not Implemented')

    def __delitem__(self, key):
        raise NotImplementedError('Descriptor is not Implemented')

    def __getslice__(self, i, j):
        raise NotImplementedError('Descriptor is not Implemented')

    def __setslice__(self, i, j, sequence):
        raise NotImplementedError('Descriptor is not Implemented')

    def __delslice__(self, i, j, sequence):
        raise NotImplementedError('Descriptor is not Implemented')

    def __len__(self):
        raise NotImplementedError('Descriptor is not Implemented')

    def __add__(self, other):
        raise NotImplementedError('Descriptor is not Implemented')

    def __sub__(self, other):
        raise NotImplementedError('Descriptor is not Implemented')

    def __mul__(self, other):
        raise NotImplementedError('Descriptor is not Implemented')
    #
    # ================================================================================================================ #
