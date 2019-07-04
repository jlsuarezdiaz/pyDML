#!/usr/bin/env python
# -*- coding: utf-8 -*-
# distutils: language=c++
# cython: profile=True

"""
Condensed Neighbourhood Component Analysis (CNCA)

"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

from .dml_utils import calc_outers, calc_outers_i, calc_regularized_outers, calc_regularized_outers_i
from .dml_algorithm import DML_Algorithm
from .knn import kNN
from .base import Transformer

from libcpp cimport bool
from scipy.optimize import differential_evolution
import warnings

# from collections import defaultdict
import time

cimport numpy as np
import sys

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cimport cython


class CNCA(DML_Algorithm):
    """
    Condensed Neighborhood Components Analysis (NCA)

    An hybridization between Neighborhood Components Analysis (NCA) and Condensed Nearest Neighbors (CNN) to handle imbalanced classification problems.

    Parameters
    ----------

    num_dims : int, default=None

        Desired value for dimensionality reduction. If None, the dimension of transformed data will be the same as the original.

    learning_rate : string, default='adaptive'

        Type of learning rate update for gradient descent. Possible values are:

        - 'adaptive' : the learning rate will increase if the gradient step is succesful, else it will decrease.

        - 'constant' : the learning rate will be constant during all the gradient steps.

    eta0 : int, default=0.3

        The initial value for learning rate.

    initial_transform : 2D-Array or Matrix (d' x d), or string, default=None.

        If array or matrix that will represent the starting linear map for gradient descent, where d is the number of features,
        and d' is the dimension specified in num_dims.
        If None, euclidean distance will be used. If a string, the following values are allowed:

        - 'euclidean' : the euclidean distance.

        - 'scale' : a diagonal matrix that normalizes each attribute according to its range will be used.

    max_iter : int, default=100

        Maximum number of gradient descent iterations.

    prec : float, default=1e-8

        Precision stop criterion (gradient norm).

    tol : float, default=1e-8

        Tolerance stop criterion (difference between two iterations)

    descent_method : string, default='SGD'

        The descent method to use. Allowed values are:

        - 'SGD' : stochastic gradient descent.

    eta_thres : float, default=1e-14

        A learning rate threshold stop criterion.

    learn_inc : float, default=1.01

        Increase factor for learning rate. Ignored if learning_rate is not 'adaptive'.

    learn_dec : float, default=0.5

        Decrease factor for learning rate. Ignored if learning_rate is not 'adaptive'.

    cnn_thresh : float, default=0.5

        Probability threshold that defines a non-noisy misclassified sample. Non-noisy samples under this theshold
        will be added to the prototypes set.

    remove_cnn_thresh : float, default=0.95

        Probability threshold that defines a non-noisy highly well classified sample. Non-noisy samples over this threshold
        will be removed from the prototypes set.

    noise_thresh : float, default=0.5

        Probability threshold that defines a noisy sample. Samples under this threshold will be tagged as noisy and not
        added to the prototypes set.

    continue_if_cnn_change : boolean, default=True

        Continues the gradient optimization if any change has happened in the prototypes set, unless the maximum number of
        iterations is reached.

    keep_best : boolean, default=True

        If True, the distance and prototypes that the algorithm learns are those that obtained the highest value of the objective function.
        Otherwise, the distance and prototypes that the algorithm learns are those obtained in the last iteration of the learning process.

    keep_class_thresh: float, default=0.3

        Class distribution threshold for keeping a class in the prototypes set. If a class distribution is lower than this threshold all its
        samples will be added and never removed from the prototypes set, independently of their probabilities.

    min_class_samples: int, default=10

        Number of minimum samples to keep of each class in the prototypes set. If after the learning process any class has a lower number of samples,
        the prototypes set will be filled according the 'complete_strategy' parameter until 'min_class_samples' (or all the class samples, if not enough)
        are available in the prototypes set.

    complete_strategy: string, default='neighbors'

        Strategy used to fill the prototypes set if not enough prototypes for some class are obtained. Available values are:

        - 'neighbors' : Fills the prototypes set with the same-class neighbors of the current samples of that class in the prototypes set.

    stochastic_balance: boolean, default=False

        Whether to consider balanced class samples during the gradient optimization (True) or to consider all the dataset samples at each complete iteration (False).

    References
    ----------
        Submitted to journal.

    """
    def __init__(self, num_dims=None, learning_rate="adaptive", eta0=0.3, initial_transform=None, max_iter=100, prec=1e-8,
                 tol=1e-8, descent_method="SGD", eta_thres=1e-14, learn_inc=1.01, learn_dec=0.5, cnn_thresh=0.5,
                 remove_cnn_thresh=0.95, noise_thresh=0.5, continue_if_cnn_change=True, keep_best=True, keep_class_thresh=0.3,
                 min_class_samples=10, complete_strategy='neighbors', stochastic_balance=False):
        self.num_dims_ = num_dims
        self.L0_ = initial_transform
        self.max_it_ = max_iter
        self.eta_ = self.eta0_ = eta0
        self.learning_ = learning_rate
        self.adaptive_ = (self.learning_ == 'adaptive')
        self.method_ = descent_method
        self.eps_ = prec
        self.tol_ = tol
        self.etamin_ = eta_thres
        self.l_inc_ = learn_inc
        self.l_dec_ = learn_dec
        self.cnn_in_thresh_ = cnn_thresh
        self.cnn_out_thresh_ = remove_cnn_thresh
        self.noise_thresh_ = noise_thresh
        self.continue_if_cnn_change_ = continue_if_cnn_change
        self.keep_best_ = keep_best
        self.keep_class_thresh_ = keep_class_thresh
        self.min_class_samples_ = min_class_samples
        self.complete_strategy_ = complete_strategy
        self.stochastic_balance_ = stochastic_balance
        # Metadata initialization
        self.num_its_ = None
        # self.initial_softmax_ = None
        self.final_softmax_ = None

    def metadata(self):
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            - num_iters : Number of iterations that the descent method took.

            - final_expectance : Final value of the objective function (the expected LOO score)
        """
        return {'num_iters': self.num_its_,
                # 'initial_expectance': self.initial_softmax_,
                'final_expectance': self.final_softmax_}

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
        self.n_, self.d_ = X.shape
        if self.num_dims_ is not None:
            self.nd_ = min(self.d_, self.num_dims_)
        else:
            self.nd_ = self.d_

        self.L_ = self.L0_
        self.eta_ = self.eta0_
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if self.L_ is None or self.L_ == "euclidean":
            self.L_ = np.zeros([self.nd_, self.d_])
            np.fill_diagonal(self.L_, 1.0)  # Euclidean distance
        elif self.L_ == "scale":
            self.L_ = np.zeros([self.nd_, self.d_])
            np.fill_diagonal(self.L_, 1. / (np.maximum(X.max(axis=0) - X.min(axis=0), 1e-16)))  # Scaled eculidean distance

        # self.initial_softmax_ = np.nan
        if self.method_ == "SGD":  # Stochastic Gradient Descent
            self._SGD_fit(X, y)

        # self.final_softmax_ = np.nan # CNCA._compute_expected_success(self.L_, X, y, self.lambdas_)  # !!!
        return self

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def _SGD_fit(self, X, y):
        # Initialize parameters
        cdef np.ndarray[DTYPE_t, ndim=4] outers = calc_outers(X)

        cdef int n = self.n_
        cdef int d = self.d_

        cdef np.ndarray[DTYPE_t, ndim=2] L = self.L_

        cdef int num_its = 0
        cdef int max_it = self.max_it_

        cdef np.ndarray[DTYPE_t, ndim=2] grad = None

        cdef float succ_prev = 0.0
        cdef float succ = 0.0
        cdef float eta = self.eta_
        cdef float etamin = self.etamin_
        cdef float l_inc = self.l_inc_
        cdef float l_dec = self.l_dec_
        cdef float eps = self.eps_
        cdef float tol = self.tol_
        cdef float cnn_in_thresh = self.cnn_in_thresh_
        cdef float cnn_out_thresh = self.cnn_out_thresh_
        cdef float cond_in_thresh = cnn_in_thresh
        cdef float cond_out_thresh = cnn_out_thresh
        cdef float noise_thresh = self.noise_thresh_
        cdef float noise_ub = 0.5
        cdef float noise_lb = 0.005
        cdef float keep_class_thresh = self.keep_class_thresh_

        cdef bool stop = False
        cdef bool adaptive = self.adaptive_
        cdef bool cont_cnn_ch = self.continue_if_cnn_change_
        cdef bool keep_best = self.keep_best_
        cdef bool stochastic_balance = self.stochastic_balance_

        cdef np.ndarray[long, ndim=1] cnn_best
        cdef np.ndarray[DTYPE_t, ndim=2] Lbest
        cdef float succ_best = 0.0

        cdef int i, j, k, i_max

        cdef np.ndarray[DTYPE_t, ndim=2] Lx,  sum_p, sum_m, s
        cdef np.ndarray[DTYPE_t, ndim=1] Lxi, softmax, dists_i
        cdef np.ndarray[long, ndim=1] rnd, yi_mask

        cdef float pw, p_i, grad_norm

        # Get class proportions and indexes
        cdef np.ndarray classes = np.unique(y)
        cdef float nclasses = len(classes)
        cdef class_split_inds = {}
        cdef class_props = {}
        cdef majority_class = None
        cdef float max_prop = 0.0
        for c in classes:
            class_split_inds[c] = np.where(y == c)[0]
            class_props[c] = (<float>len(class_split_inds[c])) / n
            if class_props[c] > max_prop:
                majority_class = c
                max_prop = class_props[c]

        cdef cnn_set_inds = []
        cdef bool cnn_changed = False

        while not stop:
            if stochastic_balance:
                rnd = np.array([], dtype=long)
                for c in classes:
                    rnd = np.append(rnd, np.random.choice(class_split_inds[c], len(class_split_inds[majority_class])))
                rnd = np.random.permutation(rnd)
            else:
                rnd = np.random.permutation(len(y))
            cnn_changed = False

            cond_in_thresh = noise_thresh + (1 - noise_thresh) * cnn_in_thresh
            cond_out_thresh = noise_thresh + (1 - noise_thresh) * cnn_out_thresh
            for i in rnd:

                if len(cnn_set_inds) == 0:
                    cnn_set_inds.append(i)
                    cnn_changed = True
                else:
                    grad = np.zeros([d, d])
                    cnnX = X[cnn_set_inds, :]
                    cnnY = y[cnn_set_inds]
                    Lx = L.dot(cnnX.T).T

                    # Calc p_ij (softmax)

                    Lxi = L.dot(X[i, :])  # Lx[i]
                    dists_i = -np.diag((Lxi - Lx).dot((Lxi - Lx).T))

                    if i in cnn_set_inds:
                        dists_i[cnn_set_inds.index(i)] = -np.inf

                    # To avoid exponential underflow we use the identity softmax(x) = softmax(x + c) for all c, and take c = max(dists)
                    i_max = np.argmax(dists_i)
                    c = dists_i[i_max]

                    softmax = np.zeros([len(cnn_set_inds)], dtype=float)

                    for j in xrange(len(cnn_set_inds)):
                        if cnn_set_inds[j] != i:
                            if j == i_max:
                                softmax[j] = 1
                            else:
                                pw = min(0, dists_i[j] - c)
                                softmax[j] = np.exp(pw)

                    cnn_class_inds = np.where(cnnY == y[i])
                    if cnn_class_inds:
                        softmax /= softmax.sum()
                        p_i = softmax[cnn_class_inds].sum()
                    else:
                        p_i = 0.0

                    if ((p_i < cond_in_thresh and (p_i > noise_thresh or num_its <= 1)) or class_props[y[i]] < keep_class_thresh) and i not in cnn_set_inds:
                        cnn_set_inds.append(i)
                        cnn_changed = True

                    elif ((p_i > cond_out_thresh or (p_i < noise_thresh and num_its > 1)) and class_props[y[i]] >= keep_class_thresh) and i in cnn_set_inds:
                        cnn_set_inds.remove(i)
                        cnn_changed = True

                    # Gradient computing
                    sum_p = np.zeros([d, d])
                    sum_m = np.zeros([d, d])

                    outers_i = calc_outers_i(X, outers, i)

                    for k in xrange(len(cnn_set_inds)):
                        if cnn_set_inds[k] != i:
                            s = softmax[k] * outers_i[cnn_set_inds[k]]
                            sum_p += s
                            if(y[i] == cnnY[k]):
                                sum_m -= s

                    grad += (p_i * sum_p + sum_m)
                    grad = 2 * L.dot(grad)
                    L += eta * grad

            succ = CNCA._compute_expected_success(L, X, y, cnn_set_inds)
            # print(sum(y[cnn_set_inds] == 0), sum(y[cnn_set_inds] == 1))

            if adaptive:
                if succ > succ_prev:
                    eta *= l_inc
                else:
                    eta *= l_dec
                    if eta < etamin:
                        stop = True

                succ_prev = succ

            if keep_best and succ > succ_best:
                Lbest = np.array(L)
                cnn_best = np.array(cnn_set_inds, dtype=int)
                succ_best = succ

            if not cont_cnn_ch or not cnn_changed:
                grad_norm = np.max(np.abs(grad))
                if grad_norm < eps or eta * grad_norm < tol:  # Difference between two iterations is given by eta*grad
                    stop = True

            num_its += 1
            if num_its == max_it:
                stop = True
            if stop:
                break

        if keep_best:
            self.cnn_set_inds_ = cnn_best
            self.L_ = Lbest
        else:
            self.cnn_set_inds_ = np.array(cnn_set_inds, dtype=int)
            self.L_ = L

        self.num_its_ = num_its
        self.eta_ = eta

        self._complete_condensed_neighbors(self.L_, X, y, classes)

        cnX, cnY = self.get_condensed_neighbors()
        print("Initial IR: ", sum(y == 1) / sum(y == 0), sum(y == 1), sum(y == 0))
        print("Condensed IR: ", sum(cnY == 1) / sum(cnY == 0), sum(cnY == 1), sum(cnY == 0))

        return self

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @staticmethod
    def _compute_expected_success(L, X, y, cnn_set_inds):
        cdef int n, d
        n, d = X.shape
        cdef np.ndarray Lx = L.dot(X.T).T
        cdef float success = 0.0
        cdef int i, j, i_max
        cdef np.ndarray softmax, Lxi, dists_i, yi_mask
        cdef float c, pw, p_i
        for i in range(len(y)):
            softmax = np.zeros([len(cnn_set_inds)], dtype=float)

            cnnX = X[cnn_set_inds, :]
            cnnY = y[cnn_set_inds]
            Lx = L.dot(cnnX.T).T
            Lxi = L.dot(X[i, :])
            dists_i = -np.diag((Lxi - Lx).dot((Lxi - Lx).T))

            if i in cnn_set_inds:
                dists_i[cnn_set_inds.index(i)] = -np.inf

            i_max = np.argmax(dists_i)
            c = dists_i[i_max]          # TODO all distances can reach -inf
            for j in xrange(len(cnn_set_inds)):
                if cnn_set_inds[j] != i:
                    # softmax[j] = np.exp(dists_i[j])
                    # To avoid precision errors, argmax is assigned directly softmax 1
                    if j == i_max:
                        softmax[j] = 1
                    else:
                        pw = min(0, dists_i[j] - c)
                        softmax[j] = np.exp(pw)
            # softmax[i] = 0

            softmax /= softmax.sum()
            # Calc p_i
            yi_mask = np.where(cnnY == y[i])[0]
            p_i = softmax[yi_mask].sum()

            success += p_i

        # print("Success: ", success / len(y), len(cnn_set_inds))
        return success / len(y)

    def _complete_condensed_neighbors(self, L, X, y, classes):
        cnX, cnY = self.get_condensed_neighbors()
        cind = self.get_condensed_neighbor_indexes()
        for c in classes:
            nc = sum(cnY == c)
            if nc < self.min_class_samples_:
                Lx = X.dot(L.T)
                if self.complete_strategy_ == "neighbors":
                    all_c_ind = np.flatnonzero(y == c)
                    out_c = np.setdiff1d(all_c_ind, cind)
                    in_c = np.intersect1d(all_c_ind, cind)
                    if len(in_c) != 0 and len(out_c) != 0:
                        distance_matrix = pairwise_distances(X=Lx[out_c, :], Y=Lx[in_c, :], n_jobs=-1)
                        condensed_distances = np.column_stack((out_c, distance_matrix.min(axis=1).ravel()))
                        sorted_distances = np.array(sorted(condensed_distances, key=lambda v: v[1]))
                        neigh_indexes = sorted_distances[:, 0].astype(int)
                        self.cnn_set_inds_ = np.append(self.cnn_set_inds_, neigh_indexes[:(self.min_class_samples_ - nc)])

    def get_condensed_neighbor_indexes(self):
        """
        Return the obtained condensed neighbors indices.

        Returns
        -------

        indices: List

            The indices of the condensed neighbors in the original dataset X.
        """
        return np.array(self.cnn_set_inds_, dtype=int)

    def get_condensed_neighbors(self):
        """
        Return the obtained condensed neighbors.

        Returns
        -------

        Xc : 2D-array

            A dataset with the (non-transformed by the learned distance) condensed neighbors attributes.

        yc : 1D-array

            The labels of the condensed neighbors, in the same order as placed in Xc.
        """
        return self.X_[self.get_condensed_neighbor_indexes(), :], self.y_[self.get_condensed_neighbor_indexes()]
