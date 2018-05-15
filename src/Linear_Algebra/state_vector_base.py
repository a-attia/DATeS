
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


class StateVectorBase(object):
    """
    Abstract class defining the model state vector.
    
    Base class for model state vector objects.
    A model state vector should provide all basic functionalities such as vector addition, scaling, norm, vector-vector dot-product, etc.
    
    Args:
        model_vector_ref: a reference (pointer) to the model's state vector object.
            This could be a reference to a numpy array for example.
            It could even be a reference to the initial index of some bizzare model-based structure.
            All is needed is that the implemented methods be aware of this structure and handle it properly.
        time: 
    
    """

    def __init__(self, model_vector_ref, time=None):
        
        self._raw_vector = model_vector_ref
        #
        # Update properties of the StateVector object.
        # this should be udpated based on the implementation of the underlying data structure
        self.__size = self._raw_vector.size
        self.__shape = self._raw_vector.shape
        self.__time = time
        #
        raise NotImplementedError
        #
        
    def set_raw_vector_ref(self, model_vector_ref, time=None):
        """
        Sets the model vector reference to a new object. 
        This should do what the constructor is doing. 
        This however, can be called if the reference of the underlying data structure is to be updated after initialization.
        
        Args:
            model_vector_ref: a reference (pointer) to the model's state vector object.
                This could be a reference to a numpy array for example.
                It could even be a reference to the initial index of some bizzare model-based structure.
                All is needed is that the implemented methods be aware of this structure and handle it properly.
                This can be very useful for example when a time integrator is to update a state vector in-place.
            time: time instance at which the state is generated; useful for time-dependent models
        
        """
        self._raw_vector = model_vector_ref
        #
        # Update properties of the StateVector object.
        # this should be udpated based on the implementation of the underlying data structure
        self.__size = self._raw_vector.size
        self.__shape = self._raw_vector.shape
        self.__time = time
        #
        raise NotImplementedError
        #
        
    def set_local(state, time=None):
        """
        Set the entries of this state vector object to the entries of state.
        
        Args:
            state: 1D iterable, of size equal to the length of this state vector object.
            time: time instance at which the state is generated; useful for time-dependent models
            
        Returns:
            None
            
        """
        raise NotImplementedError

    def get_raw_vector_ref(self):
        """
        Return a reference to the wrapped data structure constructed by the model.
        
        Returns:
            a reference to the model's vector object (StateVector._raw_vector).
        
        """
        return self._raw_vector
        #
    
    def get_numpy_array(self):
        """
        Return a copy of the internal vector (underlying reference data structure) converted to Numpy ndarray.
        
        Returns:
            a Numpy representation of the nuderlying data state vector structure.
        
        """
        raise NotImplementedError
        #

    def copy(self):
        """
        Return a (deep) copy of the StateVector object.
        """
        raise NotImplementedError
        #

    def scale(self, alpha, in_place=True):
        """
        BLAS-like method; scale the underlying StateVector by the constant (scalar) alpha.
        
        Args:
            alpha: scalar
            in_place: If True, scaling is applied (in-place) to the passed StateVector (self)
                If False, a new copy will be returned.
            
        Returns:
            The StateVector object scaled by the constant (scalar) alpha.
        
        """
        raise NotImplementedError
        #

    def dot(self, other):
        """
        Perform a dot product with other.
        
        Args:
            other: another StateVector object.

        Returns:
            The (scalar) dot-product of self with other
        
        """
        raise NotImplementedError
        #
    
    def add(self, other, in_place=True):
        """
        Add other to the vector.
        in_place: If True addition is applied (in-place) to the passed vector.
            If False, a new copy will be returned.
            
        Args:
            other: another StateVector object
        
        Returns:
            StateVector; self + other is returned
        
        """
        raise NotImplementedError
        #

    def axpy(self, alpha, other, in_place=True):
        """
        Add the vector with a scaled vector.
        
        Args:
            alpha: scalar
            other: another StateVector object
            in_place: If True scaled addition is applied (in-place) to the passed vector.
                If False, a new copy will be returned.
            
        Returns:
            StateVector; self + alpha * other is returned
        
        """
        raise NotImplementedError
        #

    def multiply(self, other, in_place=True):
        """
        Return point-wise multiplication with other
        
        Args:
            other: another StateVector object
            in_place: If True multiplication is applied (in-place) to the passed vector.
                If False, a new copy will be returned.
            
        Returns:
            point-wise multiplication of self with other
        
        """
        raise NotImplementedError
        #

    def norm2(self):
        """
        Return the 2-norm of the StateVector (self).
        
        Args:
            alpha: scalar
            other: another StateVector object
            
        Returns:
            2-norm of self
        
        """
        raise NotImplementedError
        #
        
    def sum(self):
        """
        Return sum of entries of the StateVector
        """
        raise NotImplementedError
        #
        
    def prod(self):
        """
        Return product of entries of the StateVector
        """
        raise NotImplementedError
        #

    def max(self):
        """
        Return maximum of entries of the StateVector
        """
        raise NotImplementedError
        #
        
    def min(self):
        """
        Return minimum of entries of the StateVector
        """
        raise NotImplementedError
        #

    def reciprocal(self, in_place=True):
        """
        Return the reciprocal of all entries.
        
        Args:
            in_place: If True reciprocals are evaluated (in-place) to the passed vector.
                If False, a new copy will be returned.
            
        Returns:
            the reciprocal (of all entries) of the StateVector (self)
        
        """
        raise NotImplementedError
        #

    def square(self, in_place=True):
        """
        Evaluate square of all entries.
        
        Args:
            in_place: If True the squares are evaluated (in-place) to the passed vector.
                If False, a new copy will be returned.
            
        Returns:
            the square (of all entries) of the StateVector (self)
        
        """
        raise NotImplementedError
        #

    def sqrt(self, in_place=True):
        """
        Evaluate square root of all entries.
        
        Args:
            in_place: If True the square roots are evaluated (in-place) to the passed vector.
                If False, a new copy will be returned.
            
        Returns:
            the square root (of all entries) of the StateVector (self)
        
        """
        raise NotImplementedError
        #

    def abs(self, in_place=True):
        """
        Evaluate absolute value of all entries.
        
        Args:
            in_place: If True the absolute values are evaluated (in-place) to the passed vector.
                If False, a new copy will be returned.
            
        Returns:
            the absolute value (of all entries) of the StateVector (self)
        
        """
        raise NotImplementedError
        #

    def cross(self, other):
        """
        Perform a cross product producing a StateMatrix.
        
        Args:
            other: another StateVector object
        
        Returns:
            StateMatrix object containing the cross product of self with other.
        
        """
        raise NotImplementedError
        #
    
        
    
    #
    # Properties' Setters, and Getters:
    # ------------------------------------
    #
    @property
    def size(self):
        """
        Get the size of the state vector.
        """
        return self.__size
        #
    @size.setter
    def size(self, value):
        """
        set the size of state vector. This should not be allowed to be updated by users@!
        """
        self.__size = value
        #
    
    #
    @property
    def shape(self):
        """
        Get the shape of the state vector.
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
        Get the time at which this state is generated (for time-dependent models).
        """
        return self.__time
        #
    @time.setter
    def time(self, value):
        """
        Set the time at which this state is generated (for time-dependent models)
        """
        self.__time = value
        #
    #
    
    
    
    # Emulate Python Descriptors/Decorators:
    # --------------------------------------
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
    #
    # ----------------------------------------------------------------------------------------------------- #
    #
    
