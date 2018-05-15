
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
import scipy.linalg as slinalg

from observation_vector_numpy import ObservationVectorNumpy
from observation_matrix_base import ObservationMatrixBase


class ObservationMatrixNumpy(ObservationMatrixBase):
    """
    Numpy implementation of model observation matrix operations.
    
    A model object can choose this implementation to present it's ObservationMatrix, and the associated operations/functionalities.
    
    Args:
        model_obs_matrix_ref: a reference to the model's matrix object to be wrapped.
            This could be a reference to a numpy 2d array for example.
            It could even be a reference to the initial index of some bizzare model-based structure.
            All is needed is that the implemented methods be aware of this structure and handle it properly.
            
    """

    def __init__(self, model_obs_matrix_ref):
        
        assert isinstance(model_obs_matrix_ref, np.ndarray)
        if model_obs_matrix_ref.ndim != 2:
            raise AssertionError("model_obs_matrix_ref passed is not a two-dimensional NumpyArray.")

        self._raw_matrix = model_obs_matrix_ref
        #   
        # update object's attributes
        self._update_attributes()
        #

    def _update_attributes(self):
        """
        Update the attributes of the reference matrix.
        
        """
        # set basic attributes of the reference matrix
        self.__str_type = 'numpy.ndarray'
        self.__shape = self._raw_matrix.shape
        self.__size = self._raw_matrix.size
        #        

    def set_raw_matrix_ref(self, model_obs_matrix_ref):
        """
        Set the model matrix reference to a new object.
        This should do what the constructor is doing. This however, can be called if the reference of the 
        underlying data structure is to be updated after initialization.
        
        Args:
            model_obs_matrix_ref: a reference (pointer) to the model's observation matrix object.
                This could be a reference to a numpy array for example.
                It could even be a reference to the initial index of some bizzare model-based structure.
                All is needed is that the implemented methods be aware of this structure and handle it properly.   
                
        """
        assert isinstance(model_obs_matrix_ref, np.ndarray)
        if model_obs_matrix_ref.ndim != 2:
            raise AssertionError("model_obs_matrix_ref passed is not a two-dimensional NumpyArray.")

        self._raw_matrix = model_obs_matrix_ref
        #
        # update object's attributes
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
        Returns the observation matrix (a copy) as a 2D NxN Numpy array.
        
        Returns:
            a Numpy representation (2D NxN Numpy array) of the nuderlying observation matrix data structure.
        
        """
        return copy.deepcopy(self._raw_matrix)
        #

    def copy(self):
        """
        Return a (deep) copy of this object (ObservationMatrixNumpy).
        """
        return copy.deepcopy(self)
        #

    def scale(self, alpha, in_place=True):
        """
        BLAS-like method; scale the underlying ObservationMatrix by the constant (scalar) alpha.
        
        Args:
            alpha: scalar
            in_place: If true scaling is applied (in-place) to (self).
                If False, a new copy will be returned.
            
        Returns:
            The ObservationMatrix object scaled by the constant (scalar) alpha.
        
        """
        if not in_place:
            result_matrix = self.copy()
        else:
            result_matrix = self
        result_matrix._raw_matrix = result_matrix._raw_matrix * alpha
        return result_matrix
        #

    def vector_product(self, vector, in_place=True):
        """
        Return the vector resulting from the right multiplication of the matrix and vector.
        
        Args:
            vector: ObservationVector object
            in_place: If True, matrix-vector product is applied (in-place) to the passed ObservationVector
                If False, a new copy will be returned.
            
        Returns:
            The product of self by the passed vector.
            
        """
        assert isinstance(vector, ObservationVectorNumpy)
        #
        if not in_place:
            result_vector = vector.copy()
        else:
            result_vector = vector
        result_vector._raw_vector = np.dot(self._raw_matrix, vector._raw_vector)
        return result_vector
        #

    def transpose_vector_product(self, vector, in_place=True):
        """
        Right multiplication of the matrix-transpose.
        
        Args:
            vector: ObservationVector object
            in_place: If True, matrix-transpose-vector product is applied (in-place) to the passed vector
                If False, a new copy will be returned.
        
        Returns:
            The product of self-transposed by the passed vector.
            
        """
        if not in_place:
            result_vector = vector.copy()
        else:
            result_vector = vector
        result_vector._raw_vector = np.dot(vector._raw_vector, self._raw_matrix)
        return result_vector
        #

    def matrix_product(self, other, in_place=True):
        """
        Right multiplication by a matrix (other).
        
        Args:
            other: another ObservationMatrix object of the same type as this object
            in_place: If True, matrix-matrix product is applied (in-place) to  (self)
            If False, a new copy will be returned.
            
        Returns:
            The product of self by the passed ObservationMatrix (other).
            
        """
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        result_mat._raw_matrix = np.dot(result_mat._raw_matrix, other._raw_matrix)
        return result_mat
        #

    def transpose_matrix_product(self, other, in_place=True):
        """
        Right multiplication of the matrix-transpose by a matrix (other).
        
        Args:
            other: another ObservationMatrix object of the same type as this object
            in_place: If True, matrix-transpose-matrix product is applied (in-place) to  (self)
                If False, a new copy will be returned.
            
        Returns:
            The product of self-transposed by the passed ObservationMatrix (other).
            
        """
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        result_mat._raw_matrix = np.dot(result_mat._raw_matrix.T, other._raw_matrix)
        return result_mat
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
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        result_mat._raw_matrix = result_mat._raw_matrix.T
        return result_mat
        #

    def matrix_product_matrix_transpose(self, other, in_place=True):
        """
        Right multiplication of the by a matrix-transpose.
        
        Args:
            other: another ObservationMatrix object of the same type as this object
        
        Returns:
            The product of self by the passed ObservationMatrix-transposed (other).
        
        """
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        result_mat._raw_matrix = np.dot(result_mat._raw_matrix, other._raw_matrix.T)
        return result_mat
        #

    def add(self, other, in_place=True):
        """
        Add a matrix.
        
        Args:
            other: another ObservationMatrix object of the same type as this object
            in_place: If True, matrix addition is applied (in-place) to (self)
                If False, a new copy will be returned.
        
        Returns:
            The sum of self with the passed ObservationMatrix (other).
        
        """
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        result_mat._raw_matrix = result_mat._raw_matrix + other._raw_matrix
        return result_mat
        #

    def addAlphaI(self, alpha, in_place=True):
        """
        Add a scaled identity matrix.
        
        Args:
            alpha: scalar
            in_place: If True, scaled diagonale is added (in-place) to (self)
                If False, a new copy will be returned.
        
        Returns:
            self + alpha * Identity matrix
        
        """
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        # for i in xrange(result_mat._raw_matrix.shape[0]):
        #     result_mat._raw_matrix[i,i] += alpha
        result_mat._raw_matrix[np.diag_indices_from(result_matresult_mat._raw_matrix)] += alpha
        return result_mat
        #
    
    def presolve(self):
        """
        Prepare for performing a linear solve (by, for example, performing an LU factorization).
        """
        # Do Nothing
        raise NotImplementedError
        #

    def solve(self, b):
        """
        Return the solution of the linear system created by the matrix and RHS vector b.

        Args:
            b: a ObservationVectorNumpy making up the RHS of the linear system.

        Returns:
            a ObservationVectorNumpy containing the solution to the linear system.
        
        """
        return ObservationVectorNumpy(np.linalg.solve(self._raw_matrix, b._raw_vector))
        #

    def lu_factor(self):
        """
        Compute pivoted LU decomposition of a matrix.
        The decomposition is::

            A = P L U

        where P is a permutation matrix, L lower triangular with unit
        diagonal elements, and U upper triangular.

        Returns:
            A tuple (lu, piv) to be used by lu_factor.
            lu : (N, N) ndarray
                Matrix containing U in its upper triangle, and L in its lower triangle.
                The unit diagonal elements of L are not stored.
            piv : (N,) ndarray
                Pivot indices representing the permutation matrix P:
                row i of matrix was interchanged with row piv[i]
                
        """
        lu, piv = slinalg.lu_factor(self._raw_matrix)
        return (lu, piv)
        #

    def lu_solve(self, b, lu_and_piv=None):
        """
        Return the solution of the linear system created by the matrix and RHS vector b.

        Args:
            lu_and_piv: (lu, piv) the output of lu_factor
            b: a ObservationVectorNumpy making up the RHS of the linear system.

        Returns:
            a ObservationVectorNumpy containing the solution to the linear system.
        
        """
        if lu_and_piv is None:
            lu_and_piv = self.lu_factor()
        else:
            assert isinstance(lu_and_piv, tuple)
            assert len(lu_and_piv) == 2
            assert isinstance(lu_and_piv[0], np.ndarray)
            assert isinstance(lu_and_piv[1], np.ndarray)
        assert isinstance(b, ObservationVectorNumpy)

        return ObservationVectorNumpy(slinalg.lu_solve(lu_and_piv, b._raw_vector))
        #

    def svd(self, full_matrices=False, compute_uv=False):
        """
        Return the singular value decomposition of the matrix.
        
        Returns:
            U, s, V such that the self._raw_matrix = U s V;
            U : ndarray, shape=(M, k)
                Unitary matrix having left singular vectors as columns.            
            s : ndarray, shape=(k,)
                The singular values.
            V : ndarray, shape=(k, N)
               Unitary matrix having right singular vectors as rows.Only returned when compute_uv is True.
        
        """
        U, s, V = np.linalg.svd(self._raw_matrix, full_matrices=full_matrices, compute_uv=compute_uv )
        return U, s, V
        #

    def inverse(self, in_place=True):
        """
        Invert the matrix.
        
        Args:
            in_place: If True, inverse of the matrix is carried out (in-place) to (self)
            If False, a new copy will be returned.
            
        Returns:
            Inverse of self._raw_matrix.
        
        """
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        result_mat._raw_matrix = np.linalg.inv(result_mat._raw_matrix)
        return result_mat
        #
    # add alias to inverse method
    inv = inverse
        #

    def diagonal(self, k=0):
        """
        Return the matrix diagonal as numpy 1D array.
        
        Args:
            k : int, optional diagonal in question. The default is 0. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
        
        Returns:
            The extracted diagonal as numpy array
        
        """
        return np.diag(self._raw_matrix, k).copy()
        #
    # add alias to diag method
    diag = diagonal
        #

    def set_diagonal(self, diagonal):
        """
        Update the matrix diagonal to the passed diagonal.
        
        Args:
            diagonal: a one dimensional numpy array to set on the diagonal. It can be a scalar as well. size and type must be conformable.
        """
        self._raw_matrix[np.diag_indices_from(self._raw_matrix)] = diagonal
        #
    # add alias to set_diagonal method
    set_diag = set_diagonal
        #

    def norm2(self):
        """
        Return the 2-norm of the matrix.
        
        Returns:
            a scalar containing the norm of the matrix.
        
        """
        # return np.linalg.norm(self._raw_matrix, 2)
        return np.linalg.norm(self._raw_matrix)
        #

    def det(self):
        """
        Returns the determinant of the matrix
        
        Returns:
            a scalar containing the determinant of the matrix.
        
        """
        return np.linalg.det(self._raw_matrix)
        #

    def cond(self):
        """
        Return the condition number of the matrix.
        
        Returns:
            a scalar containing the condition number of the matrix.
        
        """
        return np.linalg.cond(self._raw_matrix)
        #

    def eig(self):
        """
        Return the eigenvalues of the matrix.
        
        Returns:
            a Numpy array of the eigenvalues of the matrix.
        
        """
        return np.linalg.eigvals(self._raw_matrix)
        #
        
    def schur_product(self, other, in_place=True):
        """
        Perform elementwise (Hadamard) multiplication with another observation matrix.
        
        Args:
            other: another ObservationMatrix object of the same type as this object
            in_place: If True, inverse of the matrix is carried out 'in-place' to 'self'
                If False, a new copy will be returned.
        
        Returns:
            The result of Hadamard prodect of self with other. 
        """
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        result_mat._raw_matrix *= other._raw_matrix
        return result_mat
        #

    def cholesky(self, in_place=True):
        """
        Return a Cholesky decomposition of self._raw_matrix
        
        Args:
            other: another ObservationMatrix object of the same type as this object
            in_place: If True, inverse of the matrix is carried out 'in-place' to 'self'
                If False, a new copy will be returned.

        Returns:        
            Cholesky decomposition of self (or a copy of it). 
        
        """
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        result_mat._raw_matrix = np.linalg.cholesky(result_mat._raw_matrix)
        return result_mat
        #
    
        
    
    #
    # Properties' Setters, and Getters:
    # ------------------------------------
    #
    @property
    def str_type(self):
        """
        Get the size of the observation matrix.
        """
        return self.__str_type
        #
    @str_type.setter
    def str_type(self, value):
        """
        Set the str_type of the observation matrix.
        """
        self.__str_type = value
        #
    
    @property
    def size(self):
        """
        Get the size of the observation matrix.
        """
        return self.__size
        #
    @size.setter
    def size(self, value):
        """
        Set the size of the observation matrix.
        """
        self.__size = value
        #
    
    #
    @property
    def shape(self):
        """
        Get the size of the observation matrix.
        """
        return self.__shape
        #
    @shape.setter
    def shape(self, value):
        """
        Set the shape of the observation matrix.
        """
        self.__shape = value
        #
    
    
    #
    # Emulate Python Descriptors/Decorators:
    # --------------------------------------
    #
    def __repr__(self):
        return repr(self._raw_matrix)

    def __str__(self):
        return str(self._raw_matrix)

    def __iter__(self):
        return self._raw_matrix.__iter__()

    def __contains__(self, item):
        return self._raw_matrix.__contains__(item)

    def __getitem__(self, item):
        return self._raw_matrix.__getitem__(item)

    def __setitem__(self, key, value):
        return self._raw_matrix.__setitem__(key, value)

    def __delitem__(self, key):
        return self._raw_matrix.__delitem__(key)

    def __getslice__(self, i, j):
        return self._raw_matrix.__getslice__(i, j)

    def __setslice__(self, i, j, sequence):
        return self._raw_matrix.__setslice__(i, j, sequence)

    def __delslice__(self, i, j, sequence):
        return self._raw_matrix.__delslice__(i, j, sequence)

    def __len__(self):
        return self._raw_matrix.__len__()

    def __add__(self, other):
        return self._raw_matrix.__add__(self, other)

    def __sub__(self, other):
        return self._raw_matrix.__sub__(self, other)

    def __mul__(self, other):
        return self._raw_matrix.__mul__(self, other)
    #
    #
    # ----------------------------------------------------------------------------------------------------- #
    #
    
