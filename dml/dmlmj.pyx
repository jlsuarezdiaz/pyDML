#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance Metric Learning through the Maximization of the Jeffrey divergence (DMLMJ)

Created on Fri Feb 23 12:34:43 2018

@author: jlsuarezdiaz
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_X_y, check_array

from numpy.linalg import eig
from scipy.linalg import eigh

from .dml_algorithm import DML_Algorithm, KernelDML_Algorithm
from .dml_utils import pairwise_sq_distances_from_dot


class DMLMJ(DML_Algorithm):
    """
    Distance Metric Learning through the Maximization of the Jeffrey divergence (DMLMJ).

    A DML Algorithm that obtains a transformer that maximizes the Jeffrey divergence between
    the distribution of differences of same-class neighbors and the distribution of differences between
    different-class neighbors.

    Parameters
    ----------
    num_dims : int, default=None
        Dimension desired for the transformed data.

    n_neighbors : int, default=3
        Number of neighbors to consider in the computation of the difference spaces.

    alpha : float, default=0.001
        Regularization parameter for inverse matrix computation.

    reg_tol : float, default=1e-10
        Tolerance threshold for applying regularization. The tolerance is compared with the matrix determinant.

    References
    ----------
        Bac Nguyen, Carlos Morell and Bernard De Baets. “Supervised distance metric learning through
        maximization of the Jeffrey divergence”. In: Pattern Recognition 64 (2017), pages 215-225.
    """

    def __init__(self, num_dims=None, n_neighbors=3, alpha=0.001, reg_tol=1e-10):
        self.num_dims_ = num_dims
        self.k_ = n_neighbors
        self.alpha_ = alpha
        self.reg_tol_ = reg_tol

        # Metadata
        self.acum_eig_ = None
        self.nd_ = None

    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        L : (d'xd) matrix, where d' is the desired output dimension and d is the number of features.
        """
        return self.L_

    def metadata(self):
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            acum_eig : eigenvalue rate accumulated in the learned output respect to the total dimension.

            num_dims : dimension of the reduced data.
        """
        return {'acum_eig': self.acum_eig_, 'num_dims': self.nd_}

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
        self.X_, self.y_ = X, y

        self.n_, self.d_ = X.shape

        het_neighs, hom_neighs = DMLMJ._compute_neighborhoods(X, y, self.k_)

        if self.num_dims_ is None:
            num_dims = self.d_
        else:
            num_dims = min(self.num_dims_, self.d_)

        S, D = DMLMJ._compute_matrices(X, het_neighs, hom_neighs)

        # Regularization
        I = np.eye(self.d_)
        if np.abs(np.linalg.det(S) < self.reg_tol_):
            S = (1 - self.alpha_) * S + self.alpha_ * I
        if np.abs(np.linalg.det(D) < self.reg_tol_):
            D = (1 - self.alpha_) * D + self.alpha_ * I

        # Eigenvalues and eigenvectors of S-1D
        self.eig_vals_, self.eig_vecs_ = eigh(D, S)
        vecs_orig = self.eig_vecs_.copy()  # /np.apply_along_axis(np.linalg.norm,0,self.eig_vecs_)
        # Reordering
        self.eig_pairs_ = [(np.abs(self.eig_vals_[i]), vecs_orig[:, i]) for i in xrange(self.eig_vals_.size)]
        self.eig_pairs_ = sorted(self.eig_pairs_, key=lambda k: k[0] + 1 / k[0], reverse=True)

        for i, p in enumerate(self.eig_pairs_):
            self.eig_vals_[i] = p[0]
            self.eig_vecs_[i, :] = p[1]

        self.L_ = self.eig_vecs_[:num_dims, :]
        # print(self._compute_average_margin(self.L_,S,C))

        self.nd_ = num_dims
        self.acum_eigvals_ = np.cumsum(self.eig_vals_)
        self.acum_eig_ = self.acum_eigvals_[num_dims - 1] / self.acum_eigvals_[-1]

        return self

    def _compute_neighborhoods(X, y, k):
        n, d = X.shape
        het_neighs = np.empty([n, k], dtype=int)
        hom_neighs = np.empty([n, k], dtype=int)
        distance_matrix = pairwise_distances(X=X, n_jobs=-1)
        for i, x in enumerate(X):
            cur_class = y[i]
            mask_het = np.flatnonzero(y != cur_class)
            mask_hom = np.concatenate([np.flatnonzero(y[:i] == cur_class), (i + 1) + np.flatnonzero(y[i + 1:] == cur_class)])

            enemy_dists = [(m, distance_matrix[i, m]) for m in mask_het]
            enemy_dists = sorted(enemy_dists, key=lambda v: v[1])

            friend_dists = [(m, distance_matrix[i, m]) for m in mask_hom]
            friend_dists = sorted(friend_dists, key=lambda v: v[1])

            for j, p in enumerate(enemy_dists[:k]):
                het_neighs[i, j] = p[0]

            for j, p in enumerate(friend_dists[:k]):
                hom_neighs[i, j] = p[0]

        return het_neighs, hom_neighs

    def _compute_matrices(X, het_neighs, hom_neighs):
        n, d = X.shape
        k = het_neighs.shape[1]
        S = np.zeros([d, d])
        D = np.zeros([d, d])

        for i, x in enumerate(X):
            for j in xrange(k):
                S += np.outer(x - X[hom_neighs[i, j], :], x - X[hom_neighs[i, j], :])
                D += np.outer(x - X[het_neighs[i, j], :], x - X[het_neighs[i, j], :])

        dsize = n * k

        S /= dsize
        D /= dsize

        return S, D


class KDMLMJ(KernelDML_Algorithm):
    """
    The kernelized version of DMLMJ.

    Parameters
    ----------
    num_dims : int, default=None
        Dimension desired for the transformed data.

    n_neighbors : int, default=3
        Number of neighbors to consider in the computation of the difference spaces.

    alpha : float, default=0.001
        Regularization parameter for inverse matrix computation.

    reg_tol : float, default=1e-10
        Tolerance threshold for applying regularization. The tolerance is compared with the matrix determinant.

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
        Bac Nguyen, Carlos Morell and Bernard De Baets. “Supervised distance metric learning through
        maximization of the Jeffrey divergence”. In: Pattern Recognition 64 (2017), pages 215-225.
    """

    def __init__(self, num_dims=None, n_neighbors=3, alpha=0.001, reg_tol=1e-10,
                 kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None):
        self.num_dims_ = num_dims
        self.k_ = n_neighbors
        self.alpha_ = alpha
        self.reg_tol_ = reg_tol

        self.kernel_ = kernel
        self.gamma_ = gamma
        self.degree_ = degree
        self.coef0_ = coef0
        self.kernel_params_ = kernel_params

        # Metadata
        self.acum_eig_ = None
        self.nd_ = None

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

    def metadata(self):
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            acum_eig : eigenvalue rate accumulated in the learned output respect to the total dimension.

            num_dims : dimension of the reduced data.
        """
        return {'acum_eig': self.acum_eig_, 'num_dims': self.nd_}

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
        self.X_, self.y_ = X, y

        K = self._get_kernel(X)

        self.n_, self.d_ = X.shape

        het_neighs, hom_neighs = DMLMJ._compute_neighborhoods(X, y, self.k_)

        if self.num_dims_ is None:
            num_dims = self.d_
        else:
            num_dims = min(self.num_dims_, self.d_)

        U, V = DMLMJ._compute_matrices(K, het_neighs, hom_neighs)

        # Regularization
        I = np.eye(self.n_)
        if np.abs(np.linalg.det(U) < self.reg_tol_):
            U = (1 - self.alpha_) * U + self.alpha_ * I
        if np.abs(np.linalg.det(V) < self.reg_tol_):
            V = (1 - self.alpha_) * V + self.alpha_ * I

        # Eigenvalues and eigenvectors of S-1D
        self.eig_vals_, self.eig_vecs_ = eigh(V, U)
        vecs_orig = self.eig_vecs_.copy()  # /np.apply_along_axis(np.linalg.norm,0,self.eig_vecs_)
        # Reordering
        self.eig_pairs_ = [(np.abs(self.eig_vals_[i]), vecs_orig[:, i]) for i in xrange(self.eig_vals_.size)]
        self.eig_pairs_ = sorted(self.eig_pairs_, key=lambda k: k[0] + 1 / k[0], reverse=True)

        for i, p in enumerate(self.eig_pairs_):
            self.eig_vals_[i] = p[0]
            self.eig_vecs_[i, :] = p[1]

        self.L_ = self.eig_vecs_[:num_dims, :]
        # print(self._compute_average_margin(self.L_,S,C))

        self.nd_ = num_dims
        self.acum_eigvals_ = np.cumsum(self.eig_vals_)
        self.acum_eig_ = self.acum_eigvals_[num_dims - 1] / self.acum_eigvals_[-1]

        return self

    def _compute_neighborhoods(K, X, y, k):
        n, d = X.shape
        het_neighs = np.empty([n, k], dtype=int)
        hom_neighs = np.empty([n, k], dtype=int)
        distance_matrix = pairwise_sq_distances_from_dot(K)
        for i in xrange(n):
            cur_class = y[i]
            mask_het = np.flatnonzero(y != cur_class)
            mask_hom = np.concatenate([np.flatnonzero(y[:i] == cur_class), (i + 1) + np.flatnonzero(y[i + 1:] == cur_class)])

            enemy_dists = [(m, distance_matrix[i, m]) for m in mask_het]
            enemy_dists = sorted(enemy_dists, key=lambda v: v[1])

            friend_dists = [(m, distance_matrix[i, m]) for m in mask_hom]
            friend_dists = sorted(friend_dists, key=lambda v: v[1])

            for j, p in enumerate(enemy_dists[:k]):
                het_neighs[i, j] = p[0]

            for j, p in enumerate(friend_dists[:k]):
                hom_neighs[i, j] = p[0]

        return het_neighs, hom_neighs

    def _compute_matrices(K, het_neighs, hom_neighs):
        k = het_neighs.shape[1]
        n, _ = K.shape
        U = np.zeros([n, n])
        V = np.zeros([n, n])

        for i, k in enumerate(K):
            for j in xrange(k):
                U += np.outer(k - K[hom_neighs[i, j], :], k - K[hom_neighs[i, j], :])
                V += np.outer(k - K[het_neighs[i, j], :], k - K[het_neighs[i, j], :])

        dsize = n * k

        U /= dsize
        V /= dsize

        return U, V
