
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



class ObservationVectorBase(object):
    """
    Base class for model observation vector objects.
    a model observation vector should provide all basic functionalities such as vector addition,
    scaling, norm, vector-vector dot-product, etc.
    This should be very similar to state_vector_base.
   
    Args:
        model_vector_ref: a reference (pointer) to the model's observation vector object.
        This could be a reference to a numpy array for example.
        It could even be a reference to the initial index of some bizzare model-based structure.
        All is needed is that the implemented methods be aware of this structure and handle it properly.
    time: time instance at which the state is generated; useful for time-dependent models
        
    """

    def __init__(self, model_vector_ref, time=None):
        
        self._raw_vector = model_vector_ref
        #
        # Update properties of the ObservationVector object.
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
        This should do what the constructor is doing. This however, can be called if the reference of the 
        underlying data structure is to be updated after initialization.
        
        Args:
            model_vector_ref: a reference (pointer) to the model's observation vector object.
                This could be a reference to a numpy array for example.
                It could even be a reference to the initial index of some bizzare model-based structure.
                All is needed is that the implemented methods be aware of this structure and handle it properly.
            time: time instance at which the state is generated; useful for time-dependent models
            
        Returns:
            None
            
        """
        self._raw_vector = model_vector_ref
        #
        # Update properties of the ObservationVector object.
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
        Returns a reference to the enclosed vector object constructed by the model.
        
        Args:
            None
        
        Returns:
            a reference to the model's vector object (ObservationVectorNumpy._raw_vector).
        
        """
        return self._raw_vector
        #
    
    def get_numpy_array(self):
        """
        Return a copy of the internal vector (underlying reference data structure) converted to Numpy ndarray.
        
        Args:
            None
        
        Returns:
            a Numpy representation of the nuderlying data observation vector structure.
        
        """
        raise NotImplementedError
        #

    def copy(self):
        """
        Return a (deep) copy of the ObservationVector object.
        """
        raise NotImplementedError
        #

    def scale(self, alpha, in_place=True):
        """
        BLAS-like method; scale the underlying ObservationVector by the constant (scalar) alpha.
        
        Args:
            alpha: scalar
            in_place: If True scaling is applied (in-place) to the passed vector.
                If False, a new copy will be returned.
        
        Returns:
            The ObservationVector object scaled by the constant (scalar) alpha.
        
        """
        raise NotImplementedError
        #

    def dot(self, other):
        """
        Perform a dot product with other.
        
        Args:
            other: another ObservationVector object.

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
            other: another ObservationVector object
        
        Returns:
            ObservationVector; self + other is returned
        
        """
        raise NotImplementedError
        #

    def axpy(self, alpha, other, in_place=True):
        """
        Add the vector with a scaled vector.
        
        Args:
            alpha: scalar
            other: another ObservationVector object
            in_place: If True scaled addition is applied (in-place) to the passed vector.
                If False, a new copy will be returned.
            
        Returns:
            ObservationVector; self + alpha * other is returned
        
        """
        raise NotImplementedError
        #

    def multiply(self, other, in_place=True):
        """
        Return point-wise multiplication with other
        
        Args:
            other: another ObservationVector object
            in_place: If True multiplication is applied (in-place) to the passed vector.
                If False, a new copy will be returned.
                
        Returns:
            point-wise multiplication of self with other
        
        """
        raise NotImplementedError
        #

    def norm2(self):
        """
        Return the 2-norm of the ObservationVector (self).
        
        Args:
            alpha: scalar
            other: another ObservationVector object
        
        Returns:
            2-norm of self
        
        """
        raise NotImplementedError
        #
        
    def sum(self):
        """
        Return sum of entries of the ObservationVector
        """
        raise NotImplementedError
        #
        
    def prod(self):
        """
        Return product of entries of the ObservationVector
        """
        raise NotImplementedError
        #

    def max(self):
        """
        Return maximum of entries of the ObservationVector
        """
        raise NotImplementedError
        #
        
    def min(self):
        """
        Return minimum of entries of the ObservationVector
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
            the reciprocal (of all entries) of the ObservationVector (self)
        
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
            the square (of all entries) of the ObservationVector (self)
        
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
            the square root (of all entries) of the ObservationVector (self)
        
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
            the absolute value (of all entries) of the ObservationVector (self)
        
        """
        raise NotImplementedError
        #

    def cross(self, other):
        """
        Perform a cross product producing a ObservationMatrix.
        
        Args:
            other: another ObservationVector object
        
        Returns:
            ObservationMatrix object containing the cross product of self with other.
        
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
        Get the size of the observation vector.
        """
        return self.__shape
        #
    @shape.setter
    def shape(self, value):
        """
        set the size of shape vector. This should not be allowed to be updated by users@!
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
    # ----------------------------------------------------------------------------------------------------- #
    #
    
    
    
    #
    # ----------------------------------------------------------------------------------------------------- #
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
    
