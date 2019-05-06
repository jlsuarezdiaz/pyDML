#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for different DML algoritms
"""

from __future__ import print_function, absolute_import
import numpy as np
import warnings
from six.moves import xrange
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix


def metric_to_linear(M):
    """
    Converts a metric PSD matrix into an associated linear transformation matrix, so the distance defined by the
    metric matrix is the same as the euclidean distance after projecting by the linear transformation.
    This implementation takes the linear transformation corresponding to the square root of the matrix M.

    Parameters
    ----------
    M : 2D-Array or Matrix

        A positive semidefinite matrix.

    Returns
    -------
    L : 2D-Array

        The matrix associated to the linear transformation that computes the same distance as M.
    """
    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = eigvals.astype(float)  # Remove residual imaginary part
    eigvals[eigvals < 0.0] = 0.0  # MEJORAR ESTO (no debería hacer falta, pero está bien para errores de precisión)
    sqrt_diag = np.sqrt(eigvals)
    return eigvecs.dot(np.diag(sqrt_diag)).T


def SDProject(M):
    """
    Projects a symmetric matrix onto the positive semidefinite cone (considering the Frobenius norm).
    The projection is made by taking the non negative eigenvalues after diagonalizing.

    Parameters
    ----------
    M : 2D-Array or Matrix

        A symmetric matrix.

    Returns
    -------
    Mplus : 2D-Array

        The projection of M onto the positive semidefinite cone.
    """
    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = eigvals.astype(float)  # Remove residual imaginary part
    eigvecs = eigvecs.astype(float)
    eigvals[eigvals < 0.0] = 0.0  # MEJORAR ESTO
    diag_sdp = np.diag(eigvals)
    return eigvecs.dot(diag_sdp).dot(eigvecs.T)


def calc_outers(X, Y=None):
    """
    Calculates the outer products between two datasets. All outer products are calculated, so memory may be not enough.
    To avoid memory errors the output of this function should be used in the input of :func:`~calc_outers_i`.

    Parameters
    ----------

    X : Numpy array, shape (N x d)

        A 2D-array, where N is the number of samples and d is the number of features.

    Y : Numpy array, shape (M x d), default=None

        A 2D-array, where M is the number of samples in Y and d is the number of features.
        If None, Y is taken as X.

    Returns
    -------

    outers : A 4D-array, of shape (N x M x d x d), where outers[i,j] is the outer product between X[i] and Y[j].
             It can also be None, if memory was not enough. In this case, outers will be calculated in :func:`~calc_outers_i`.
    """
    n, d = X.shape
    if Y is None:
        Y = X
    m, e = Y.shape
    if n * m * d * e > 600000000:
        return None
    try:
        diff = X[:, None] - Y[None]
        return np.einsum("...i,...j->...ij", diff, diff)

        # outers = np.empty([n, m, d, d], dtype=float)  # !!!!

        # for i in xrange(n):
        #     for j in xrange(m):
        #         outers[i, j] = np.outer(X[i, :] - Y[j, :], X[i, :] - Y[j, :])

    except:
        warnings.warn("Memory is not enough to calculate all outer products at once. "
                      "Algorithm will be slower.")
        outers = None

    return outers


def calc_outers_i(X, outers, i, Y=None):
    """
    Obtains a subset of outer products from the calculated in :func:`~calc_outers`.
    If memory was enough, this function just returns a row of outer products from the calculated matrix of outer products.
    Else, this method calculates this row.

    Parameters
    ----------

    X : Numpy array, shape (N x d)

        A 2D-array, where N is the number of samples and d is the number of features.

    outers : Numpy array, or None

        The output of the function :func:`~calc_outers`.

    i : int

        The row to fetch from outers, from 0 to N-1.

    Y : Numpy array, shape (M x d), default=None

        A 2D-array, where M is the number of samples in Y and d is the number of features.
        If None, Y is taken as X.

    Returns
    -------

    outers_i : A 3D-Array, of shape (M x d x d), where outers_i[j] is the outer product between X[i] and Y[j].
               It can also be None, if memory was not enough. In this case, outers will be calculated in :func:`~calc_outers_ij`.
    """
    if outers is not None:
        return outers[i, :]
    else:
        n, d = X.shape
        if Y is None:
            Y = X
        m, e = Y.shape
        outers_i = np.empty([n, d, d], dtype=float)

        for j in xrange(m):
            outers_i[j] = np.outer(X[i, :] - Y[j, :], X[i, :] - Y[j, :])
        return outers_i


def calc_outers_ij(X, outers_i, i, j, Y=None):
    """
    Obtains an outer product between two elements in datasets, from the output calculated in :func:`~calc_outers`.

    Parameters
    ----------

    X : Numpy array, shape (N x d)

        A 2D-array, where N is the number of samples and d is the number of features.

    outers_i : Numpy array, or None

        The output of the function :func:`~calc_outers_i`.

    i : int

        The row to fetch from outers, from 0 to N-1.

    j : int

        The column to fetch from outers, from 0 to M-1.

    Y : Numpy array, shape (M x d), default=None

        A 2D-array, where M is the number of samples in Y and d is the number of features.
        If None, Y is taken as X.

    Returns
    -------

    outers_i : A 2D-Array, of shape (d x d), with the outer product between X[i] and Y[j].

    """
    if outers_i is not None:
        return outers_i[j]
    else:
        if Y is None:
            Y = X
        return np.outer(X[i, :] - Y[j, :], X[i, :] - Y[j, :])


def metric_sq_distance(M, x, y):
    """
    Calculates a distance between two points given a metric PSD matrix.

    Parameters
    ----------

    M : 2D-Array or Matrix

        A positive semidefinite matrix defining the distance.

    x : Array.

        First argument for the distance. It must have the same length as y and the order of M.

    y : Array.

        Second argument for the distance. It must have the same length as x and the order of M.
    """
    d = (x - y).reshape(1, -1)
    return d.dot(M).dot(d.T)


def unroll(A):
    """
    Returns a column vector from a matrix with all its columns concatenated.

    Parameters
    ----------

    A : 2D-Array or Matrix.

        The matrix to unroll.

    Returns
    -------

    v : 1D-Array

        The vector with the unrolled matrix.
    """
    n, m = A.shape
    v = np.empty([n * m, 1])
    for i in xrange(m):
        v[(i * n):(i + 1) * n, 0] = A[:, i]
    return v


def matpack(v, n, m):
    """
    Returns a matrix that takes by columns the elements in the vector v.

    Parameters
    ----------

    v : 1D-Array

        The vector to fit in a matrix.

    n : int

        The matrix rows.

    m : int

        The matrix columns.

    Returns
    -------

    A : 2D-Array, shape (n x m)

        The matrix that takes by columns the elements in v.
    """
    A = np.empty([n, m], dtype=float)
    for i in xrange(m):
        A[:, i] = v[(i * n):(i + 1) * n, 0]
    return A


# Pairwise distance for two datasets given their dot products
def pairwise_sq_distances_from_dot(K):
    """
    Calculates the pairwise squared distance between two datasets given the matrix of dot products.

    Parameters
    ----------

    K : 2D-Array or Matrix

        A matrix with the dot products between two datasets. It verifies
        ..math:: K[i,j] = \\langle x_i, y_j \\rangle

    Returns
    -------

    dists : 2D-Array

        A matrix with the squared distances between the elements in both datasets. It verifies
        ..math:: dists[i,j] = d(x_i, y_j)
    """
    n, m = K.shape
    dists = np.empty([n, m])
    for i in xrange(n):
        for j in xrange(m):
            dists[i, j] = K[i, i] + K[j, j] - 2 * K[i, j]
    return dists


def neighbors_affinity_matrix(X, y, k):
    """
    Neighbors affinity matrix.

    Computes a neighbors affinity matrix A for the dataset (X, y), where A[i, j] = 1 if x_j is a k-nearest neighbor of the same class
    as x_i, and 0 otherwise.

    Parameters
    ----------

    X : array-like, shape (N x d)
            Training vector, where N is the number of samples, and d is the number of features.

    y : array-like, shape (N)
        Labels vector, where N is the number of samples.

    k : int
        The number of neighbors to consider as nearest neighbors.

    Returns
    -------

    A : 2D-array
        The affinity matrix.
    """
    n = X.shape[0]
    # neigh = KNeighborsClassifier(n_neighbors=k)
    # neigh.fit(X, y)
    # neigh_list = neigh.kneighbors(return_distance=False)

    classes = np.unique(y)
    neigh_list = np.empty([n, k], dtype=int)
    for c in classes:
        inds, = np.where(y == c)
        dists = pairwise_distances(X[inds])

        np.fill_diagonal(dists, np.inf)
        neigh_inds = np.argsort(dists)[..., :k]
        neigh_list[inds] = inds[neigh_inds]

    A = np.zeros([n, n], dtype=int)

    for i in xrange(n):
        for j in neigh_list[i]:
            A[i, j] = 1

    return A


def local_scaling_affinity_matrix(X, y, k):
    """
    Local scaling affinity matrix.

    Computes a local scaling affinity matrix A for the dataset (X, y), where
    .. math::
        A[i, j] = exp(-\\|x_i - x_j\\|^2/\\sigma_i \\sigma_j).
    The sigma values represent the local scaling of the samples around x_i, and are given by
    .. math::
        \\sigma_i = \\|x_i - x_i^{(K)}\\|,
    where :math:`x_i^{(K)}` is the K-th nearest neighbor of :math:`x_i`.

    Parameters
    ----------

    X : array-like, shape (N x d)
            Training vector, where N is the number of samples, and d is the number of features.

    y : array-like, shape (N)
        Labels vector, where N is the number of samples.

    k : int
        The value for the k-th nearest neighbor to consider in the local scaling.

    Returns
    -------

    A : 2D-array
        The affinity matrix.
    """
    n = X.shape[0]
    # neigh = KNeighborsClassifier(n_neighbors=k)
    # neigh.fit(X, y)
    # neigh_list = neigh.kneighbors(return_distance=False)[:, k - 1]
    A = np.zeros([n, n])

    classes = np.unique(y)
    neigh_list = np.empty([n], dtype=int)
    for c in classes:
        inds, = np.where(y == c)
        dists = pairwise_distances(X[inds])

        np.fill_diagonal(dists, np.inf)
        neigh_inds = np.argsort(dists)[..., k - 1]
        neigh_list[inds] = inds[neigh_inds]

    for i in xrange(n):
        sigma_i = np.linalg.norm(X[i, :] - X[neigh_list[i], :])

        for j in xrange(i):
            if y[i] == y[j]:
                sigma_j = np.linalg.norm(X[j, :] - X[neigh_list[j], :])
                x_ij = X[i, :] - X[j, :]
                d_ij = np.inner(x_ij, x_ij)
                if d_ij == 0 and (sigma_i == 0 or sigma_j == 0):
                    A[i, j] = A[j, i] = 1
                else:
                    A[i, j] = A[j, i] = np.exp(- d_ij / (sigma_i * sigma_j))

    return A


def calc_regularized_outers(X, Y=None):
    """
    Calculates the outer products between two datasets. All outer products are calculated, so memory may be not enough.
    To avoid memory errors the output of this function should be used in the input of :func:`~calc_outers_i`.

    Parameters
    ----------

    X : Numpy array, shape (N x d)

        A 2D-array, where N is the number of samples and d is the number of features.

    Y : Numpy array, shape (M x d), default=None

        A 2D-array, where M is the number of samples in Y and d is the number of features.
        If None, Y is taken as X.

    Returns
    -------

    outers : A 4D-array, of shape (N x M x d x d), where outers[i,j] is the outer product between X[i] and Y[j].
             It can also be None, if memory was not enough. In this case, outers will be calculated in :func:`~calc_outers_i`.
    """
    n, d = X.shape
    if Y is None:
        Y = X
    m, e = Y.shape
    if n * m * d * e > 600000000:
        return None
    try:
        diff = X[:, None] - Y[None]
        return np.eye(d) + np.einsum("...i,...j->...ij", diff, diff)

        # outers = np.empty([n, m, d, d], dtype=float)  # !!!!

        # for i in xrange(n):
        #     for j in xrange(m):
        #         outers[i, j] = np.outer(X[i, :] - Y[j, :], X[i, :] - Y[j, :])

    except:
        warnings.warn("Memory is not enough to calculate all outer products at once. "
                      "Algorithm will be slower.")
        outers = None

    return outers


def calc_regularized_outers_i(X, reg_outers, i, Y=None):
    """
    Obtains a subset of outer products from the calculated in :func:`~calc_outers`.
    If memory was enough, this function just returns a row of outer products from the calculated matrix of outer products.
    Else, this method calculates this row.

    Parameters
    ----------

    X : Numpy array, shape (N x d)

        A 2D-array, where N is the number of samples and d is the number of features.

    outers : Numpy array, or None

        The output of the function :func:`~calc_outers`.

    i : int

        The row to fetch from outers, from 0 to N-1.

    Y : Numpy array, shape (M x d), default=None

        A 2D-array, where M is the number of samples in Y and d is the number of features.
        If None, Y is taken as X.

    Returns
    -------

    outers_i : A 3D-Array, of shape (M x d x d), where outers_i[j] is the outer product between X[i] and Y[j].
               It can also be None, if memory was not enough. In this case, outers will be calculated in :func:`~calc_outers_ij`.
    """
    if reg_outers is not None:
        return reg_outers[i, :]
    else:
        n, d = X.shape
        if Y is None:
            Y = X
        m, e = Y.shape
        outers_i = np.empty([n, d, d], dtype=float)

        Id = np.eye(d)

        for j in xrange(m):
            print(j)
            outers_i[j] = np.outer(X[i, :] - Y[j, :], X[i, :] - Y[j, :]) + Id
        return outers_i


def calc_regularized_outers_ij(X, reg_outers_i, i, j, Y=None):
    """
    Obtains an outer product between two elements in datasets, from the output calculated in :func:`~calc_outers`.

    Parameters
    ----------

    X : Numpy array, shape (N x d)

        A 2D-array, where N is the number of samples and d is the number of features.

    outers_i : Numpy array, or None

        The output of the function :func:`~calc_outers_i`.

    i : int

        The row to fetch from outers, from 0 to N-1.

    j : int

        The column to fetch from outers, from 0 to M-1.

    Y : Numpy array, shape (M x d), default=None

        A 2D-array, where M is the number of samples in Y and d is the number of features.
        If None, Y is taken as X.

    Returns
    -------

    outers_i : A 2D-Array, of shape (d x d), with the outer product between X[i] and Y[j].

    """
    if reg_outers_i is not None:
        return reg_outers_i[j]
    else:
        if Y is None:
            Y = X
        return np.outer(X[i, :] - Y[j, :], X[i, :] - Y[j, :]) + np.eye(X.shape[0])
