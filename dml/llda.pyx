#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Local Linear Discriminant Analysis (LLDA)

"""

from __future__ import absolute_import
import numpy as np
from sklearn.utils.validation import check_X_y


from .dml_algorithm import DML_Algorithm, KernelDML_Algorithm
from .dml_utils import neighbors_affinity_matrix, local_scaling_affinity_matrix

from six.moves import xrange
import warnings


class LLDA(DML_Algorithm):
    """
    Local Linear Discriminant Analysis (LDA).

    A local version for the Linear Discriminant Analysis.

    Parameters
    ----------

    n_components : int, default=None

        Number of components for dimensionality reduction. If None, it will be taken as n_classes - 1. Ignored if thres is provided.

    affinity : array-like or string, default="neighbors"

        The affinity matrix, that is, an (N x N) matrix with entries in [0,1], where N is the number of samples, where the (i, j) element specifies
        the affinity between samples x_i and x_j. It can be also a string. In this case, the affinity matrix will be computed in the algorithm.
        Valid strings are:

        - "neighbors" : An affinity matrix A, where A[i, j] is 1 if x_j is one of the k-nearest neighbors of x_i, will be computed. The value of k
                        is determined by the 'n_neighbors' attribute.

        - "local-scaling" : An affinity matrix is computed according to the local scaling method, using the kth-nearest neighbors. The value of k
                            is determined by the 'n_neighbors' attribute. A recommended value for this case is n_neighbors=7. See [1] for more
                            information.

    n_neighbors : int, default=1

        Number of neighbors to consider in the affinity matrix. Ignored if 'affinity' is not equal to "neighbors" or "local-scaling".

    tol : float, default=1e-4

        Singularity toleration level.

    alpha : float, default=1e-3

        Regularization term for singular within-class matrix.

    solver : string, default="sugiyama"

        The resolution method. Valid values are:

        - "classic" : the original LLDA problem will be computed (building the within-class and between-class matrices in the usual way).

        - "sugiyama" : the algorithm proposed in [1]. It is faster than the classic method and provides the same results. The solver 'classic'
                       is kept for testing, but this solver is the recommended one.


    References
    ----------
        [1] Masashi Sugiyama “Dimensionality reduction of multimodal labeled data by local fisher discriminant analysis”.
            In: Journal of Machine Learning Research, 2007, vol 8, May, pages 1027-1061.
    """

    def __init__(self, n_components=None, affinity="neighbors", n_neighbors=1, tol=1e-4, alpha=1e-3, solver="sugiyama"):
        self.num_dims_ = n_components
        self.affinity_ = affinity
        self.k_ = n_neighbors
        self.tol_ = tol
        self.alpha_ = alpha
        self.solver_ = solver

        self.nd_ = None
        self.acum_eig_ = None

    def metadata(self):
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            acum_eig : eigenvalue rate accumulated in the learned output respect to the total dimension.

            num_dims : dimension of the reduced data.
        """
        return {'num_dims': self.nd_, 'acum_eig': self.acum_eig_}

    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        L : (d'xd) matrix, where d' is the desired output dimension and d is the number of features.
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
        self.X_, self.y_ = X, y

        n, d = X.shape

        # Dimensionality reduction
        if self.num_dims_ is None:
            self.nd_ = d
        else:
            self.nd_ = min(d, self.num_dims_)

        # Affinity matrix
        if self.affinity_ == 'neighbors':
            self.A_ = neighbors_affinity_matrix(X, y, self.k_)
        elif self.affinity_ == 'local-scaling':
            self.A_ = local_scaling_affinity_matrix(X, y, self.k_)
        else:
            self.A_ = self.affinity_

        # Solver
        if self.solver_ == 'classic':
            self._solve_classic(X, y)
        elif self.solver_ == 'sugiyama':
            self._solve_sugiyama(X, y)
        else:
            ValueError("Invalid option for 'solver': " + self.solver_)

        return self

    def _solve_classic(self, X, y):
        n, d = X.shape
        classes = np.unique(y)
        nc = {}
        for c in classes:
            nc[c] = len(y[y == c])

        # A_w = np.empty([n, n])
        # A_b = np.empty([n, n])

        S_w = np.zeros([d, d])
        S_b = np.zeros([d, d])

        # Scatter matrices construction
        for i in xrange(n):
            for j in xrange(n):
                xij = X[i, :] - X[j, :]
                oij = np.outer(xij, xij)
                if y[i] == y[j]:
                    ny = nc[y[i]]
                    A_w_ij = self.A_[i, j] / ny
                    A_b_ij = self.A_[i, j] * (1 / n - 1 / ny)
                else:
                    A_w_ij = 0
                    A_b_ij = 1 / n

                S_w += A_w_ij * oij
                S_b += A_b_ij * oij

        # Regularization
        if abs(np.linalg.det(S_w)) < self.tol_:
            warnings.warn("Variables are collinear.")
            S_w += self.alpha_ * np.eye(d)

        evals, evecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))

        evecs = evecs[:, np.argsort(evals)[::-1]]
        # evecs /= np.apply_along_axis(np.linalg.norm,0,evecs)

        self.L_ = evecs[:, :self.nd_].T

        self.acum_eigvals_ = np.cumsum(evals)
        self.acum_eig_ = self.acum_eigvals_[self.nd_ - 1] / self.acum_eigvals_[-1]

    def _solve_sugiyama(self, X, y):
        n, d = X.shape
        classes = np.unique(y)
        A = self.A_
        A = (A + A.T) / 2
        ones_n = np.ones([n, 1])

        S_b = np.zeros([d, d])
        S_w = np.zeros([d, d])

        for c in classes:
            Xc = X[y == c]
            yc = y[y == c]
            Ac = A[y == c, :][:, y == c]
            nc = len(yc)
            ones_c = np.ones([nc, 1])

            G = Xc.T.dot(np.diag(Ac.dot(ones_c).ravel())).dot(Xc) - Xc.T.dot(Ac).dot(Xc)

            Xt1c = Xc.T.dot(ones_c)
            S_b += (G / n + (1 - nc / n) * (Xc.T.dot(Xc)) + Xt1c.dot(Xt1c.T) / n)
            S_w += (G / nc)

        Xt1n = X.T.dot(ones_n)
        S_b -= (Xt1n.dot(Xt1n.T) / n + S_w)

        # Regularization
        if abs(np.linalg.det(S_w)) < self.tol_:
            warnings.warn("Variables are collinear.")
            S_w += (self.alpha_ / 2) * np.eye(d)  # alpha / 2 to keep the results in classic

        evals, evecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))

        evecs = evecs[:, np.argsort(evals)[::-1]]
        # evecs /= np.apply_along_axis(np.linalg.norm,0,evecs)

        self.L_ = evecs[:, :self.nd_].T

        self.acum_eigvals_ = np.cumsum(evals)
        self.acum_eig_ = self.acum_eigvals_[self.nd_ - 1] / self.acum_eigvals_[-1]


class KLLDA(KernelDML_Algorithm):
    """
    The kernelized version of LLDA.

    Parameters
    ----------

    num_dims : int, default=None

        Number of components for dimensionality reduction. If None, it will be taken as n_classes - 1. Ignored if thres is provided.

    affinity : array-like or string, default="neighbors"

        The affinity matrix, that is, an (N x N) matrix with entries in [0,1], where N is the number of samples, where the (i, j) element specifies
        the affinity between samples x_i and x_j. It can be also a string. In this case, the affinity matrix will be computed in the algorithm.
        Valid strings are:

        - "neighbors" : An affinity matrix A, where A[i, j] is 1 if x_j is one of the k-nearest neighbors of x_i, will be computed. The value of k
                        is determined by the 'n_neighbors' attribute.

        - "local-scaling" : An affinity matrix is computed according to the local scaling method, using the kth-nearest neighbors. The value of k
                            is determined by the 'n_neighbors' attribute. A recommended value for this case is n_neighbors=7. See [1] for more
                            information.

    n_neighbors : int, default=1

        Number of neighbors to consider in the affinity matrix. Ignored if 'affinity' is not equal to "neighbors" or "local-scaling".

    tol : float, default=1e-4

        Singularity toleration level.

    alpha : float, default=1e-3

        Regularization term for singular within-class matrix.

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
        [1] Masashi Sugiyama “Dimensionality reduction of multimodal labeled data by local fisher discriminant analysis”.
            In: Journal of Machine Learning Research, 2007, vol 8, May, pages 1027-1061.
    """

    def __init__(self, n_components=None, affinity="neighbors", n_neighbors=1, tol=1e-4, alpha=1e-3, kernel="linear",
                 gamma=None, degree=3, coef0=1, kernel_params=None):

        self.num_dims_ = n_components
        self.affinity_ = affinity
        self.k_ = n_neighbors
        self.tol_ = tol
        self.alpha_ = alpha

        self.kernel_ = kernel
        self.gamma_ = gamma
        self.degree_ = degree
        self.coef0_ = coef0
        self.kernel_params_ = kernel_params

        self.nd_ = None
        self.acum_eig_ = None

    def metadata(self):
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            acum_eig : eigenvalue rate accumulated in the learned output respect to the total dimension.

            num_dims : dimension of the reduced data.
        """
        return {'num_dims': self.nd_, 'acum_eig': self.acum_eig_}

    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        L : (d'xd) matrix, where d' is the desired output dimension and d is the number of features.
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
        self.X_, self.y_ = X, y

        n, d = X.shape

        # Dimensionality reduction
        if self.num_dims_ is None:
            self.nd_ = d
        else:
            self.nd_ = min(d, self.num_dims_)

        # Affinity matrix
        if self.affinity_ == 'neighbors':
            self.A_ = neighbors_affinity_matrix(X, y, self.k_)
        elif self.affinity_ == 'local-scaling':
            self.A_ = local_scaling_affinity_matrix(X, y, self.k_)
        else:
            self.A_ = self.affinity_

        self._solve_sugiyama(X, y)

        return self

    def _solve_sugiyama(self, X, y):
        n, d = X.shape
        classes = np.unique(y)
        A = self.A_
        A = (A + A.T) / 2
        ones_n = np.ones([n, 1])
        K = self._get_kernel(X)

        S_b = np.zeros([n, n])
        S_w = np.zeros([n, n])

        for c in classes:
            Kc = K[y == c]
            yc = y[y == c]
            Ac = A[y == c, :][:, y == c]
            nc = len(yc)
            ones_c = np.ones([nc, 1])

            G = Kc.T.dot(np.diag(Ac.dot(ones_c).ravel())).dot(Kc) - Kc.T.dot(Ac).dot(Kc)

            Kt1c = Kc.T.dot(ones_c)
            S_b += (G / n + (1 - nc / n) * (Kc.T.dot(Kc)) + Kt1c.dot(Kt1c.T) / n)
            S_w += (G / nc)

        Kt1n = K.T.dot(ones_n)
        S_b -= (Kt1n.dot(Kt1n.T) / n + S_w)

        # Regularization
        if abs(np.linalg.det(S_w)) < self.tol_:
            warnings.warn("Variables are collinear.")
            S_w += (self.alpha_ / 2) * np.eye(n)  # alpha / 2 to keep the results in classic

        evals, evecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))

        evecs = evecs[:, np.argsort(evals)[::-1]]
        # evecs /= np.apply_along_axis(np.linalg.norm,0,evecs)

        self.L_ = evecs[:, :self.nd_].T

        self.acum_eigvals_ = np.cumsum(evals)
        self.acum_eig_ = self.acum_eigvals_[self.nd_ - 1] / self.acum_eigvals_[-1]
