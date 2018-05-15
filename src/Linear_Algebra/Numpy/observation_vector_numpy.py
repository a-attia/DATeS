
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


import copy
import numpy as np

from observation_vector_base import ObservationVectorBase


class ObservationVectorNumpy(ObservationVectorBase):
    """
    Numpy implementation of observation vector operations.
    
    A model object can choose this implementation to present it's ObservationVector, and the associated operations/functionalities.

    This should be very similar to StateVectorNumpy.
    
    Args:
        model_obs_vector_ref: a reference (pointer) to the model's observation vector object.
            This could be a reference to a numpy array for example.
            It could even be a reference to the initial index of some bizzare model-based structure.
            All is needed is that the implemented methods be aware of this structure and handle it properly.
        time: time instance at which the state is generated; useful for time-dependent models
        
    """

    def __init__(self, model_obs_vector_ref, time=None):
        
        assert isinstance(model_obs_vector_ref, np.ndarray)
        if model_obs_vector_ref.ndim != 1:
            print("model vector passed is not one dimensional NumpyArray!")
            raise AssertionError

        self._raw_vector = model_obs_vector_ref
        #
        self.__size = self._raw_vector.size
        self.__shape = self._raw_vector.shape
        self.__time = time
        #
        
    def set_local(obs, time=None):
        """
        Set the entries of this observation vector object to the entries of obs.
        
        Args:
            obs: 1D iterable, of size equal to the length of this observation vector object.
            time: time instance at which the observation is generated; useful for time-dependent models
            
        Returns:
            None
            
        """
        if len(obs) != self.__size:
            print("the passed observation (obs) has to be 1d iterable of length/size %d" % self.__size)
            raise ValueError
        #
        for i in xrange(self.__size):
            self._raw_vector[i] = obs[i]
        self.__time = time
        #
    
    def set_raw_vector_ref(self, model_obs_vector_ref, time=None):
        """
        Sets the model vector reference to a new object.
        This should do what the constructor is doing. This however, can be called if the reference of the 
        underlying data structure is to be updated after initialization.
        
        Args:
            model_obs_vector_ref: a reference (pointer) to the model's observation vector object.
                This could be a reference to a numpy array for example.
                It could even be a reference to the initial index of some bizzare model-based structure.
                All is needed is that the implemented methods be aware of this structure and handle it properly.
            time: time instance at which the observation is generated; useful for time-dependent models
            
        """
        self._raw_vector = model_obs_vector_ref
        #
        self.__size = self._raw_vector.size
        self.__shape = self._raw_vector.shape
        self.__time = time
        #

    def get_raw_vector_ref(self):
        """
        Returns a reference to the enclosed vector object constructed by the model.
        
        Returns:
            a reference to the model's vector object (ObservationVectorNumpy._raw_vector).
        
        """
        return self._raw_vector
        #

    def get_numpy_array(self):
        """
        Return a copy of the internal vector (underlying reference data structure) converted to Numpy ndarray.
        Since the underlying vector is already a numpy.ndarray, just return a copy.
        
        Returns:
            a Numpy representation of the nuderlying data observation vector structure.
        
        """
        return copy.deepcopy(self._raw_vector)
        #

    def copy(self):
        """
        Return a copy of the vector.
        """
        return copy.deepcopy(self)
        #

    def scale(self, alpha, in_place=True):
        """
        BLAS-like method; scale the underlying ObservationVectorNumpy by the constant (scalar) alpha.
        
        Args:
            alpha: scalar
            in_place: If true scaling is applied (in-place) to self.
                If False, a new copy will be returned.
        
        Returns:
            The ObservationVectorNumpy object scaled by the constant (scalar) alpha.
        
        """
        if not in_place:
            result_vec = self.copy()
        else:
            result_vec = self
        result_vec._raw_vector = result_vec._raw_vector * alpha
        return result_vec
        #

    def dot(self, other, in_place=True):
        """
        Perform a dot product with other.
        
        Args:
            other: another ObservationVectorNumpy object.

        Returns:
            The (scalar) dot-product of self with other
        
        """
        return self._raw_vector.dot(other.get_raw_vector_ref())
        #

    def add(self, other, in_place=True):
        """
        Add other to the vector.
        in_place: If true addition is applied (in-place) to self.
            If False, a new copy will be returned.
            
        Args:
            other: another ObservationVectorNumpy object
        
        Returns:
            ObservationVector; self + other is returned
        
        """
        if not in_place:
            result_vec = self.copy()
        else:
            result_vec = self
        result_vec._raw_vector = result_vec._raw_vector + other._raw_vector
        return result_vec
        #

    def axpy(self, alpha, other, in_place=True):
        """
        Add the vector with a scaled vector.
        
        Args:
            alpha: scalar
            other: another ObservationVectorNumpy object
            in_place: If true scaled addition is applied (in-place) to self.
                If False, a new copy will be returned.
            
        Returns:
            ObservationVector; self + alpha * other is returned
        
        """
        if not in_place:
            result_vec = self.copy()
        else:
            result_vec = self
        result_vec._raw_vector = result_vec._raw_vector + alpha * other._raw_vector
        return result_vec
        #

    def multiply(self, other, in_place=True):
        """
        Return point-wise multiplication with other
        
        Args:
            other: another ObservationVectorNumpy object
            in_place: If true multiplication is applied (in-place) to self.
                If False, a new copy will be returned.
            
        Returns:
            point-wise multiplication of self with other
        
        """
        if not in_place:
            result_vec = self.copy()
        else:
            result_vec = self
        result_vec.set_raw_vector_ref(result_vec.get_raw_vector_ref() * other.get_raw_vector_ref())  # Numpy will take care of
        return result_vec
        #
        
    def norm2(self):
        """
        Return the 2-norm of the ObservationVectorNumpy (self).
        
        Args:
            alpha: scalar
            other: another ObservationVectorNumpy object
            
        Returns:
            2-norm of self
    
        """
        return np.linalg.norm(self._raw_vector)
        #

    def sum(self):
        """
        Return sum of entries of the vector
        """
        return self._raw_vector.sum()
        #

    def prod(self):
        """
        Return product of entries of the vector
        """
        return self._raw_vector.prod()
        #

    def max(self):
        """
        Return maximum of entries of the vector
        """
        return self._raw_vector.max()
        #
        
    def min(self):
        """
        Return minimum of entries of the vector
        """
        return self._raw_vector.min()
        #
        
    def reciprocal(self, in_place=True):
        """
        Return the reciprocal of all entries.
        
        Args:
            in_place: If true reciprocals are evaluated (in-place) to self.
                If False, a new copy will be returned.
            
        Returns:
            the reciprocal (of all entries) of the ObservationVectorNumpy (self)
        
        """
        if not in_place:
            result_vec = self.copy()
        else:
            result_vec = self
        result_vec._raw_vector = 1.0 / result_vec._raw_vector  # Numpy will take care of the inversion here.
        return result_vec

    def square(self, in_place=True):
        """
        Evaluate square of all entries.
        
        Args:
            in_place: If true the squares are evaluated (in-place) to self.
                If False, a new copy will be returned.
                
        Returns:
            the square (of all entries) of the ObservationVectorNumpy (self)
        
        """
        if not in_place:
            result_vec = self.copy()
        else:
            result_vec = self
        result_vec._raw_vector = np.square(result_vec._raw_vector)  # Numpy will take care of the inversion here.
        return result_vec
        #
        
    def sqrt(self, in_place=True):
        """
        Evaluate square root of all entries.
        
        Args:
            in_place: If true the square roots are evaluated (in-place) to self.
                If False, a new copy will be returned.
            
        Returns:
            the square root (of all entries) of the ObservationVectorNumpy (self)
        
        """
        if not in_place:
            result_vec = self.copy()
        else:
            result_vec = self
        result_vec._raw_vector = np.sqrt(result_vec._raw_vector)  # Numpy will take care of the inversion here.
        return result_vec

    def abs(self, in_place=True):
        """
        Evaluate absolute value of all entries.
        
        Args:
            in_place: If true the absolute values are evaluated (in-place) to self.
                If False, a new copy will be returned.
            
        Returns:
            the absolute value (of all entries) of the ObservationVectorNumpy (self)
        
        """
        if not in_place:
            result_vec = self.copy()
        else:
            result_vec = self
        result_vec._raw_vector = np.absolute(result_vec._raw_vector)  # Numpy will take care of the inversion here.
        return result_vec
        #
    
        
    
    #
    # Properties' Setters, and Getters:
    # ------------------------------------
    #
    @property
    def size(self):
        """
        Get the size of the observation vector.
        """
        return self.__size
        #
    @size.setter
    def size(self, value):
        """
        set the size of observation vector. This should not be allowed to be updated by users@!
        """
        self.__size = value
        #
    
    #
    @property
    def shape(self):
        """
        Get the shape of the observation vector.
        """
        return self.__shape
        #
    @shape.setter
    def shape(self, value):
        """
        set the shape vector. This should not be allowed to be updated by users@!
        """
        self.__shape = value
        #
    
    #
    @property
    def time(self):
        """
        Get the time at which this observation is generated (for time-dependent models).
        """
        return self.__time
        #
    @time.setter
    def time(self, value):
        """
        Set the time at which this observation is generated (for time-dependent models)
        """
        self.__time = value
        #
    #
    
    

    
    #
    # Emulate Python Descriptors/Decorators:
    # --------------------------------------
    #
    def __repr__(self):
        return repr(self._raw_vector)

    def __str__(self):
        return str(self._raw_vector)

    def __iter__(self):
        return self._raw_vector.__iter__()

    def __contains__(self, item):
        return self._raw_vector.__contains__(item)

    def __getitem__(self, item):
        if not isinstance(item, int) and np.isscalar(item):
            item = int(item)
        return self._raw_vector.__getitem__(item)

    def __setitem__(self, key, value):
        if not isinstance(key, int) and np.isscalar(key):
            key = int(key)
        return self._raw_vector.__setitem__(key, value)

    def __delitem__(self, key):
        if not isinstance(key, int) and np.isscalar(key):
            key = int(key)
        return self._raw_vector.__delitem__(key)

    def __getslice__(self, i, j):
        return self._raw_vector.__getslice__(i, j)

    def __setslice__(self, i, j, sequence):
        return self._raw_vector.__setslice__(i, j, sequence)

    def __delslice__(self, i, j, sequence):
        return self._raw_vector.__delslice__(i, j, sequence)

    def __len__(self):
        return self._raw_vector.__len__()

    def __add__(self, other):
        return self._raw_vector.__add__(self, other)

    def __sub__(self, other):
        return self._raw_vector.__sub__(self, other)

    def __mul__(self, other):
        return self._raw_vector.__mul__(self, other)
    #
    #
    # ----------------------------------------------------------------------------------------------------- #
    #
    
