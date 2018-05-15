
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
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from state_vector_numpy import StateVectorNumpy
from state_matrix_numpy import StateMatrixNumpy
from state_matrix_base import StateMatrixBase


class StateMatrixSpSciPy(StateMatrixBase):
    """
    Sparse SciPy-based implementation of StateMatrix operations.
    A model object can choose this implementation to present it's StateMatrix, and the associated operations/functionalities.
    
    Sparse SciPy-based implementation of StateMatrix operations.
    
    Args:
        model_matrix_ref: a reference to the model's SPARSE matrix object to be wrapped.
            This could be a reference to a numpy 2d array for example.
            It could even be a reference to the initial index of some bizzare model-based structure.
            All is needed is that the implemented methods be aware of this structure and handle it properly.

    """

    def __init__(self, model_matrix_ref):
        
        assert sparse.issparse(model_matrix_ref)
        if model_matrix_ref.ndim != 2:
            raise AssertionError("model_matrix_ref passed is not a two-dimensional.")
        #
        self._raw_matrix = model_matrix_ref
        # update object's attributes
        self._update_attributes()
        #

    def _update_attributes(self):
        """
        Update the attributes of the reference matrix
        
        """
        # set basic attributes of the reference matrix
        self.__format = self._raw_matrix.format
        self.__shape = self._raw_matrix.shape
        #
        
    def set_raw_matrix_ref(self, model_matrix_ref):
        """
        Set the model sparse matrix reference to a new object.
        This should do what the constructor is doing. This however, can be called if the reference of the 
        underlying data structure is to be updated after initialization.
        
        Args:
            model_matrix_ref: a reference (pointer) to the model's sparse state matrix object.
                This could be a reference to a numpy array for example.
                It could even be a reference to the initial index of some bizzare model-based structure.
                All is needed is that the implemented methods be aware of this structure and handle it properly.

        """
        assert sparse.issparse(model_matrix_ref)
        if model_matrix_ref.ndim != 2:
            raise AssertionError("model_matrix_ref passed is not a two-dimensional NumpyArray.")
        #
        self._raw_matrix = model_matrix_ref
        self._update_attributes()
        #

    def get_raw_matrix_ref(self):
        """
        Return a reference to the enclosed matrix object constructed by the model.

        Returns:
            a reference to the model's sparse matrix wrapped by this class.
        
        """
        return self._raw_matrix
        #

    def get_numpy_array(self):
        """
        Return the sparse state matrix (a copy) as a 2D NxN dense Numpy array.
        
        Returns:
            a Numpy representation (2D NxN Numpy array) of the nuderlying sparse state matrix data structure.
        
        """
        return copy.deepcopy(self._raw_matrix.toarray())
        #

    def copy(self):
        """
        Return a (deep) copy of this object (StateMatrixSpSciPy).
        
        """
        return copy.deepcopy(self)
        #

    def scale(self, alpha, in_place=True):
        """
        Scale the underlying StateMatrixSpSciPy by the constant (scalar) alpha.
        
        Args:
            alpha: scalar
            in_place: If true scaling is applied (in-place) to (self).
                If False, a new copy will be returned.
        
        Returns:
            The StateMatrixSpSciPy object scaled by the constant (scalar) alpha.
        
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
            vector: StateVector object
            in_place: If True, matrix-vector product is applied (in-place) to the passed StateVectorNumpy
                If False, a new copy will be returned.
        
        Returns:
            The product of self by the passed vector.
        
        """
        assert isinstance(vector, StateVectorNumpy)
        #
        if not in_place:
            result_vector = vector.copy()
        else:
            result_vector = vector
        result_vector._raw_vector = self._raw_matrix.dot(vector._raw_vector)
        return result_vector
        #

    def transpose_vector_product(self, vector, in_place=True):
        """
        Right multiplication of the matrix-transpose.
        
        Args:
            vector: StateVectorNumpy object
            in_place: If True, matrix-transpose-vector product is applied (in-place) to the passed vector
                If False, a new copy will be returned.
            
        Returns:
            The product of self-transposed by the passed vector.
        
        """
        assert isinstance(vector, StateVectorNumpy)
        #
        if not in_place:
            result_vector = vector.copy()
        else:
            result_vector = vector
        result_vector._raw_vector = self._raw_matrix.T.dot(vector._raw_vector)
        return result_vector
        #

    def matrix_product(self, other, in_place=True):
        """
        Right multiplication by a matrix (other). 
        We need to handle the cases where other is numpy or scipy.sparse
        
        Args:
            other: another StateMatrix object that can be of the same type as this object, or a StateMatrixNumpy,
            or even a regular Numpy array.
            in_place: If True, matrix-matrix product is applied (in-place) to  (self)
                If False, a new copy will be returned.
        
        Returns:
            The product of self by the passed StateMatrix (other).
        
        """
        raise NotImplementedError
        #

    def transpose_matrix_product(self, other, in_place=True):
        """
        Right multiplication of the matrix-transpose by a matrix (other).
        We need to handle the cases where other is numpy or scipy.sparse
        
        Args:
            other: another StateMatrix object that can be of the same type as this object, or a StateMatrixNumpy,
            or even a regular Numpy array.
            in_place: If True, matrix-matrix product is applied (in-place) to  (self)
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
        if not in_place:
            result_mat = self.copy()
        else:
            result_mat = self
        result_mat._raw_matrix = result_mat._raw_matrix.T
        return result_mat
        #

    def matrix_product_matrix_transpose(self, other, in_place=True):
        """
        Right multiplication by the transpose of a matrix (other). 
        We need to handle the cases where other is numpy or scipy.sparse
        
        Args:
            other: another StateMatrix object that can be of the same type as this object, or a StateMatrixNumpy,
            or even a regular Numpy array.
            in_place: If True, matrix-matrix product is applied (in-place) to  (self)
            If False, a new copy will be returned.
        
        Returns:
            The product of self by the transpose of the passed StateMatrix (other).
        
        """
        raise NotImplementedError
        #

    def add(self, other, in_place=True, write_to_self=False):
        """
        Add a matrix. The sum of self._raw_matrix with the passed StateMatrix (other).
        
        Args:
            other: another StateMatrix object of the same type as this object
            in_place: If True, matrix addition is applied (in-place) to (self)
                If False, a new copy will be returned.
            
        Returns:
            The sum of self with the passed StateMatrix (other).
        
        """
        if write_to_self:
            # update the reference matrix inside this object.
            if not in_place:
                result_mat = self.copy()
            else:
                result_mat = self
            if isinstance(other, (StateMatrixNumpy, self.__class__)):
                source_matrix = other
                source_matrix_ref = other._raw_matrix
            elif isinstance(other, np.ndarray):
                source_matrix = other
                source_matrix_ref = other
            else:
                raise TypeError("matrix has to be either 'StateMatrixNumpy', or 'StateMatrixSpSciPy', or 'np.ndarray' ")
        else:
            # the target is the input matrix or a copy of it
            if not in_place:
                result_mat = other.copy()
            else:
                result_mat = other
            source_matrix = self
            source_matrix_ref = self._raw_matrix
        #
        # Check the result matrix format
        if isinstance(result_mat, self.__class__):
            result_mat_ref = result_mat.get_raw_matrix_ref()
            #
            # frmt = result_mat._raw_matrix.getformat()
            # print('\n xxxxxxxxxxxxxxxxx \n %s \n xxxxxxxxxxxxxxxxx \n' % frmt)
            #
            if sparse.isspmatrix_bsr(result_mat._raw_matrix):
                result_mat_ref = sparse.bsr_matrix(result_mat_ref + source_matrix_ref)
            elif sparse.isspmatrix_coo(result_mat._raw_matrix):
                result_mat_ref = sparse.coo_matrix(result_mat_ref + source_matrix_ref)
            elif sparse.isspmatrix_csc(result_mat._raw_matrix):
                result_mat_ref = sparse.csc_matrix(result_mat_ref + source_matrix_ref)
            elif sparse.isspmatrix_csr(result_mat._raw_matrix):
                result_mat_ref = sparse.csr_matrix(result_mat_ref + source_matrix_ref)
                # print(result_mat._raw_matrix)
                # print("is sparse: ", sparse.issparse(result_mat._raw_matrix))
            elif sparse.isspmatrix_dia(result_mat._raw_matrix):
                result_mat_ref = sparse.dia_matrix(result_mat_ref + source_matrix_ref)
            elif sparse.isspmatrix_dok(result_mat._raw_matrix):
                result_mat_ref = sparse.dok_matrix(result_mat_ref + source_matrix_ref)
            elif sparse.isspmatrix_lil(result_mat._raw_matrix):
                result_mat_ref = sparse.lil_matrix(result_mat_ref + source_matrix_ref)
            else:
                raise TypeError("Unsupported Format! My format has been tapered with!")
            result_mat.set_raw_matrix_ref(result_mat_ref)
            result_mat._update_attributes()
            
        elif isinstance(result_mat, StateMatrixNumpy):
            result_mat_ref = result_mat.get_raw_matrix_ref()
            if isinstance(source_matrix, self.__class__):
                result_mat_ref = result_mat_ref + source_matrix_ref
                try:
                    result_mat_ref = result_mat_ref.toarray()
                except AttributeError:
                    result_mat_ref = np.asarray(result_mat_ref)
            elif isinstance(source_matrix, (np.ndarray, StateMatrixNumpy)):
                result_mat_ref = result_mat_ref + source_matrix_ref
            
            result_mat.set_raw_matrix_ref(result_mat_ref)
                
        elif isinstance(result_mat, np.ndarray):
            result_mat_ref = result_mat
            if isinstance(source_matrix, self.__class__):
                result_mat_ref = result_mat_ref + source_matrix_ref
                try:
                    result_mat_ref = result_mat_ref.toarray()
                except AttributeError:
                    result_mat_ref = np.asarray(result_mat_ref)
            elif isinstance(source_matrix, (np.ndarray, StateMatrixNumpy)):
                result_mat_ref = result_mat_ref + source_matrix_ref
            
        else:
            type.mro(type(other))
            print(type.mro(type(other)))
            print(other)
            raise TypeError("matrix has to be either 'StateMatrixNumpy', or 'StateMatrixSpSciPy', or 'np.ndarray' ")
            # raise TypeError("matrix has to be either 'StateMatrixNumpy', or 'StateMatrixSpSciPy'! ")
        #
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
        diagonal = self.get_raw_matrix_ref().diagonal()
        diagonal += alpha
        result_mat._raw_matrix.setdiag(diagonal)
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
            b: a StateVectorNumpy making up the RHS of the linear system.

        Returns:
            a StateVectorNumpy containing the solution to the linear system.
        
        """
        return StateVectorNumpy(splinalg.spsolve(self._raw_matrix, b._raw_vector))
        #

    def lu_factor(self):
        """
        Compute pivoted LU decomposition of a matrix.
        The decomposition is:

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

    def svd(self, k=6, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True):
        """
        Return the singular value decomposition of the matrix.
        
        Args:
            Same as scipy.sparse.linalg.svds inputs

        Returns:
            U : ndarray, shape=(M, k)
                Unitary matrix having left singular vectors as columns.            
            s : ndarray, shape=(k,)
                The singular values.
            Vt : ndarray, shape=(k, N)
                Unitary matrix having right singular vectors as rows.
        """
        U, s, Vt = splinalg.svds(self._raw_matrix, 
                                 k=6, 
                                 ncv=None, 
                                 tol=0, 
                                 which='LM', 
                                 v0=None, 
                                 maxiter=None, 
                                 return_singular_vectors=True
                                 )
        return U, s, Vt
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
        result_mat._raw_matrix = splinalg.inv(result_mat._raw_matrix)
        return result_mat
        #
    # add alias to inverse method
    inv = inverse
    #

    def diagonal(self, k=0):
        """
        Return the matrix diagonal as numpy 1D array.        
        
        Args:
            k : int, optional diagonal in question. 
            The default is 0. Use k>0 for diagonals above the main diagonal, and k < 0 for diagonals below the main diagonal.
        
        Returns:
            The extracted diagonal as numpy array
        
        """
        if k==0:
            return self._raw_matrix.diagonal().copy()
        else:
            raise ValueError('only main diagonal is supported now!')
        #
    # add alias to diag method
    diag = diagonal
        #

    def set_diagonal(self, diagonal, k=0):
        """
        Update the matrix diagonal to the passed diagonal.
        
        Args:
            diagonal: a one dimensional numpy array
            k : int, optional diagonal in question. 
                The default is 0. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
        """
        raise NotImplementedError
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
        raise NotImplementedError
        #

    def det(self):
        """
        Returns the determinant of the matrix
        
        Args:
        
        Returns:
            a scalar containing the determinant of the matrix.
        
        """
        return np.linalg.det(self._raw_matrix.toarray())
        #

    def cond(self):
        """
        Return the condition number of the matrix.
        
        Returns:
            a scalar containing the condition number of the matrix.
        
        """
        raise NotImplementedError
        #

    def eig(self):
        """
        Return the eigenvalues of the matrix.
        
        Returns:
            a Numpy array of the eigenvalues of the matrix.
        
        """
        return splinalg.eigs(self._raw_matrix, return_eigenvectors=False)
        #
        
    def schur_product(self, other, in_place=True):
        """
        Perform elementwise (Hadamard) multiplication with another StateMatrix.
        
        Args:
            other: another StateMatrix object that can be of the same type as this object, or a StateMatrixNumpy,
            or even a regular Numpy array.
            in_place: If True, matrix-matrix product is applied (in-place) to  (self)
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
    def format(self):
        """
        Get the format of the wraped sparse StateMatrix.
        
        """
        return self._raw_matrix.format
        #
    @format.setter
    def str_type(self, value):
        """
        Set the format of the wraped sparse StateMatrix.
        
        """
        self.__format = self._raw_matrix.format
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
    
