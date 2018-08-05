#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel Discriminant Analysis (KDA)

Created on Sun Feb 18 18:38:16 2018

@author: jlsuarezdiaz
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_X_y, check_array

from numpy.linalg import inv, pinv
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from .dml_algorithm import DML_Algorithm, KernelDML_Algorithm
from .dml_utils import pairwise_sq_distances_from_dot


class KDA(KernelDML_Algorithm):
    """
    Kernel Discriminant Analysis (KDA)

    Discriminant Analysis in high dimensionality using the kernel trick.

    Parameters
    ----------

    solver : string, default='eigen'.

        Solver to use, posible values:
            - 'eigen': Eigenvalue decomposition.

    n_components : int, default=None.

        Number of components (lower than number of classes -1) for dimensionality reduction.

    tol : float, default=1e-4

        Singularity toleration level.

    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel. Default="linear".

    gamma : float, default=1/n_features
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels.

    degree : int, default=3
        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    References
    ----------
        Sebastian Mika et al. “Fisher discriminant analysis with kernels”. In: Neural networks for signal
        processing IX, 1999. Proceedings of the 1999 IEEE signal processing society workshop. Ieee. 1999,
        pages 41-48.

    """

    def __init__(self, solver='eigen', n_components=None, tol=1e-4, kernel="linear",
                 gamma=None, degree=3, coef0=1, kernel_params=None):

        self.solver_ = solver
        self.n_components_ = n_components
        self.tol_ = tol
        self.kernel_ = kernel
        self.gamma_ = gamma
        self.degree_ = degree
        self.coef0_ = coef0
        self.kernel_params_ = kernel_params

    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        A : (d'x N) matrix, where d' is the desired output dimension, and N is the number of samples.
            To apply A to a new sample x, A must be multiplied by the kernel vector of dimension N
            obtained by taking the kernels between x and each training sample.
        """
        return self.L_

    def fit(self, X, y):
        """
        Fit the model from the data in X and the labels in y.

        Parameters
        ----------
        X : array-like, shape (N x d)
            Training vector, where N is the number of samples, and d is the number of features.

        y : array-like, shape (N)
            Labels vector, where N is the number of samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X, y = check_X_y(X, y)
        self.X_ , self.y_ = X,y
        n, d = X.shape
        
        K = self._get_kernel(X)
        
        classes, class_counts = np.unique(y,return_counts=True)
        
        # Compute N and M matrices
        M_avg = K.sum(axis=1)/n
        M = np.zeros([n,n])
        N = np.zeros([n,n])
        for i, c in enumerate(classes):
            c_mask = np.where(y==c)[0]
            K_i = K[:,c_mask]
            M_i = K_i.sum(axis=1)/class_counts[i]
            diff = (M_i - M_avg)
            M += class_counts[i]*np.outer(diff,diff)
            const_ni = np.full([class_counts[i],class_counts[i]],1.0-1.0/class_counts[i])
            N += K_i.dot(const_ni).dot(K_i.T)
            
        #Regularize
        N+=1.0*np.eye(n)
        
        evals, evecs = eigh(inv(N).dot(M))
        evecs = evecs[:,np.argsort(evals)[::-1]]
        #evecs /= np.apply_along_axis(np.linalg.norm,0,evecs)
        
        self.L_ = evecs[:,:len(classes)-1].T
        
        return self
            
            
        