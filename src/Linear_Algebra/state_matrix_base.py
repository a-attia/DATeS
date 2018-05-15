
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



class StateMatrixBase(object):
    """
    Base class for model state matrix objects. A model state matrix should provide all basic functionalities such as matrix addition, matrix-vector product, matrix-inv prod vector, etc.
   
    Abstract class defining DATeS interface to model matrices (such as the model Jacobian, or  background error covariance matrix).
    
    Args:
        model_matrix_ref: a reference to the model's matrix object to be wrapped.
            This could be a reference to a numpy 2d array for example.
            It could even be a reference to the initial index of some bizzare model-based structure.
            All is needed is that the implemented methods be aware of this structure and handle it properly.
    
    """

    def __init__(self, model_matrix_ref):
        
        self._raw_matrix = model_matrix_ref
        #
        # Update properties of the StateVector object.
        self._update_attributes()
        #

    def _update_attributes(self):
        """
        Update the attributes of the reference matrix
        """
        # set basic attributes of the reference matrix
        self.__str_type = 'numpy.ndarray'
        self.__shape = self._raw_matrix.shape
        self.__size = self._raw_matrix.size
        #

    def set_raw_matrix_ref(self, model_matrix_ref):
        """
        Set the model matrix reference to a new object. 
            This should do what the constructor is doing. This however, can be called if 
            the reference of the underlying data structure is to be updated after initialization.
        
        Args:
            model_matrix_ref: a reference (pointer) to the model's state matrix object.
                This could be a reference to a numpy array for example.
                It could even be a reference to the initial index of some bizzare model-based structure.
                All is needed is that the implemented methods be aware of this structure and handle it properly.
        
        Returns:
            None
            
        """
        self._raw_matrix = model_matrix_ref
        #
        # Update properties of the StateMatrix object.
        self._update_attributes()
        #

    def get_raw_matrix_ref(self):
        """
        Returns a reference to the enclosed matrix object constructed by the model.

        Returns:
            a reference to the model's matrix object.
        
        """
        return self._raw_matrix
        #

    def get_numpy_array(self):
        """
        Returns the state matrix (a copy) as a 2D NxN Numpy array.
        
        Returns:
            a Numpy representation (2D NxN Numpy array) of the nuderlying data state matrix structure.
        
        """
        raise NotImplementedError

    def copy(self):
        """
        Return a (deep) copy of the matrix.
        """
        raise NotImplementedError
        #

    def scale(self, alpha, in_place=True):
        """
        BLAS-like method; scale the underlying StateMatrix by the constant (scalar) alpha.
        
        Args:
            alpha: scalar
            in_place: If true scaling is applied (in-place) to (self).
                If False, a new copy will be returned.
        
        Returns:
            The StateMatrix object scaled by the constant (scalar) alpha.
        
        """
        raise NotImplementedError
        #

    def vector_product(self, vector, in_place=True):
        """
        Return the vector resulting from the right multiplication of the matrix and vector.
        
        Args:
            vector: StateVector object
            in_place: If True, matrix-vector product is applied (in-place) to the passed StateVector
                If False, a new copy will be returned.
        
        Returns:
            The product of self by the passed vector.
        
        """
        raise NotImplementedError
        #

    def transpose_vector_product(self, vector, in_place=True):
        """
        Right multiplication of the matrix-transpose.
        
        Args:
            vector: StateVector object
            in_place: If True, matrix-transpose-vector product is applied (in-place) to the passed vector
                If False, a new copy will be returned.
        
        Returns:
            The product of self-transposed by the passed vector.
        
        """
        raise NotImplementedError
        #

    def matrix_product(self, other, in_place=True):
        """
        Right multiplication by a matrix.
        
        Args:
            other: another StateMatrix object of the same type as this object
            in_place: If True, matrix-matrix product is applied (in-place) to  (self)
                If False, a new copy will be returned.
        
        Returns:
            The product of self by the passed StateMatrix (other).
        
        """
        raise NotImplementedError
        #

    def transpose_matrix_product(self, other, in_place=True):
        """
        Right multiplication of the matrix-transpose by a matrix.
        
        Args:
            other: another StateMatrix object of the same type as this object
            in_place: If True, matrix-transpose-matrix product is applied (in-place) to  (self)
                If False, a new copy will be returned.
            
        Returns:
            The product of self-transposed by the passed StateMatrix (other).
        
        """
        raise NotImplementedError
        #

    def transpose(self, in_place=True):
        """
        Return slef transposed
        
        Args:
            in_place: If True, matrix transpose is applied (in-place) to  (self)
                If False, a new copy will be returned.
        
        Returns:
            self (or a copy of it) transposed
        
        """
        raise NotImplementedError
        #

    def matrix_product_matrix_transpose(self, other, in_place=True):
        """
        Right multiplication of the by a matrix-transpose.
        
        Args:
            other: another StateMatrix object of the same type as this object
        
        Returns:
            The product of self by the passed StateMatrix-transposed (other).
        
        """
        raise NotImplementedError
        #
        
    def add(self, other, in_place=True):
        """
        Add a matrix.
        
        Args:
            other: another StateMatrix object of the same type as this object
            in_place: If True, matrix addition is applied (in-place) to (self)
                If False, a new copy will be returned.
        
        Returns:
            The sum of self with the passed StateMatrix (other).
        
        """
        raise NotImplementedError
        #

    def addAlphaI(self, alpha, in_place=True):
        """
        Add a scaled identity matrix.
        
        Args:
            alpha: scalar
            in_place: If True, scaled diagonal is added (in-place) to (self)
                If False, a new copy will be returned.
            
        Returns:
            self + alpha * Identity matrix
        
        """
        raise NotImplementedError
        #

    def presolve(self):
        """
        Prepare for performing a linear solve (by, for example, performing an LU factorization).
        
        """
        raise NotImplementedError
        #

    def solve(self, b):
        """
        Return the solution of the linear system created by the matrix and RHS vector b.
        
        Args:
            b: a StateVectorNumpy making up the RHS of the linear system.
        
        """
        raise NotImplementedError
        #

    def lu_factor(self):
        """
        Compute pivoted LU decomposition of a matrix.
        The decomposition is::

            A = P L U

        where P is a permutation matrix, L lower triangular with unit
        diagonal elements, and U upper triangular.

        Args:

        Returns:
            A tuple (lu, piv) to be used by lu_factor.
            lu : (N, N) ndarray
                Matrix containing U in its upper triangle, and L in its lower triangle.
                The unit diagonal elements of L are not stored.
            piv : (N,) ndarray
                Pivot indices representing the permutation matrix P:
                row i of matrix was interchanged with row piv[i]
            
        """
        raise NotImplementedError
        #

    def lu_solve(self, b, lu_and_piv=None):
        """
        Return the solution of the linear system created by the matrix and RHS vector b.

        Args:
            lu_and_piv: (lu, piv) the output of lu_factor
            b: a StateVectorNumpy making up the RHS of the linear system.

        Returns:
            a StateVectorNumpy containing the solution to the linear system.
        
        """
        raise NotImplementedError
        #

    def svd(self):
        """
        Return the singular value decomposition of the matrix.
        
        Args:
        
        Returns:
        
        """
        raise NotImplementedError
        #

    def inverse(self, in_place=True):
        """
        Invert the matrix.
        
        Args:
            in_place: If True, inverse of the matrix is carried out (in-place) to (self)
                If False, a new copy will be returned.
        
        Returns:
        
        """
        raise NotImplementedError
        #
    # add alias to inverse method
    inv = inverse
        #

    def diagonal(self, k=0):
        """
        Return the matrix diagonal as numpy 1D array.
        
        Args:
            k: int, optional
                Diagonal in question. The default is 0. 
                Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
        
        Returns:
            The extracted diagonal as numpy array
        
        """
        raise NotImplementedError
        #
    # add alias to diag method
    diag = diagonal
        #

    def set_diagonal(self, diagonal):
        """
        Update the matrix diagonal to the passed diagonal.
        
        Args:
            diagonal: a one dimensional numpy array, or a scalar
        
        """
        raise NotImplementedError
        #
    # add alias to set_diagonal method
    set_diag = set_diagonal
        #

    def norm2(self):
        """
        Return the 2-norm of the matrix.
        
        """
        raise NotImplementedError
        #

    def det(self):
        """
        Returns the determinate of the matrix
        
        """
        raise NotImplementedError
        #

    def cond(self):
        """
        Return the condition number of the matrix.
        
        """
        raise NotImplementedError
        #

    def eig(self):
        """
        Return the eigenvalues of the matrix.
        
        """
        raise NotImplementedError
        #

    def schur_product(self, other, in_place=True):
        """
        Perform elementwise (Hadamard) multiplication with another state matrix.
        
        Args:
            other: another StateMatrix object of the same type as this object
            in_place: If True, inverse of the matrix is carried out (in-place) to (self)
                If False, a new copy will be returned.
        
        Returns:
            The result of Hadamard prodect of self with other
        
        """
        raise NotImplementedError
        #

    def cholesky(self, in_place=True):
        """
        Return a Cholesky decomposition of self._raw_matrix
        
        Args:
            other: another StateMatrix object of the same type as this object
            in_place: If True, inverse of the matrix is carried out (in-place) to (self)
                If False, a new copy will be returned.

        Returns:        
            Cholesky decomposition of self (or a copy of it). 
        
        """
        raise NotImplementedError
        #
    
        
    
    #
    # Properties' Setters, and Getters:
    # ------------------------------------
    #
    @property
    def str_type(self):
        """
        Get the size of the state matrix.
        """
        return self.__str_type
        #
    @str_type.setter
    def str_type(self, value):
        """
        Set the str_type of the state matrix.
        """
        self.__str_type = value
        #
    
    @property
    def size(self):
        """
        Get the size of the state matrix.
        """
        return self.__size
        #
    @size.setter
    def size(self, value):
        """
        Set the size of the state matrix.
        """
        self.__size = value
        #
    
    #
    @property
    def shape(self):
        """
        Get the size of the state matrix.
        """
        return self.__shape
        #
    @shape.setter
    def shape(self, value):
        """
        Set the shape of the state matrix.
        """
        self.__shape = value
        #
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
    
