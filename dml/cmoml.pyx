#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chain Maximizing Ordinal Metric Learning (CMOML)
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y

from numpy.linalg import eig

from .dml_algorithm import DML_Algorithm, KernelDML_Algorithm
from .dml_utils import pairwise_sq_distances_from_dot

from scipy.optimize import differential_evolution
import scipy.optimize as opt
from scipy.special import comb
from itertools import combinations

import time
import warnings

from collections import defaultdict


def _CMOML_compute_Lvec_objective(Lvec, dd, d, X, y, k, neigh_size, counting_function, n_jobs=-1):
    return CMOML._compute_Lvec_objective(Lvec, dd, d, X, y, k, neigh_size, counting_function, n_jobs)


def _CMOML_numOfIncSubseqOfSizeKEff(arr, n, k):
    return CMOML.numOfIncSubseqOfSizeKEff(arr, n, k)


def _KCMOML_compute_Lvec_objective(Lvec, dd, d, X, y, k, neigh_size, counting_function, n_jobs=-1):
    return KCMOML._compute_Lvec_objective(Lvec, dd, d, X, y, k, neigh_size, counting_function, n_jobs)


class CMOML(DML_Algorithm):
    """

    """

    evaluations = 0

    def __init__(self, n_dims=None, chain_size=3, neighborhood_size=50, workers=1, n_jobs=-1):
        self.n_dims_ = n_dims
        self.k_ = chain_size
        self.neigh_size_ = neighborhood_size
        self.workers = workers
        self.n_jobs = n_jobs

        self.initial_objective_ = None
        self.final_objective_ = None

    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        L : (d'xd) matrix, where d' is the desired output dimension and d is the number of features.
        """
        return self.L_

    def metadata(self):
        return {'initial_objective': self.initial_objective_, 'final_objective': self.final_objective_}

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

        if self.n_dims_ is None:
            n_dims = self.d_
        else:
            n_dims = min(self.n_dims_, self.d_)

        bounds = [(-100, 100) for i in xrange(n_dims * self.d_)]

        nclasses = len(np.unique(y))
        counting_function = CMOML._get_fastest_counting_function([CMOML.numOfIncSubseqOfSizeK,
                                                                 CMOML.numOfIncSubseqOfSizeKNp,
                                                                 _CMOML_numOfIncSubseqOfSizeKEff,
                                                                 CMOML.numOfIncSubseqOfSizeKEffNp],
                                                                 self.k_, self.neigh_size_, nclasses)
        self.initial_objective_ = - CMOML._compute_L_objective(np.eye(self.d_), X, y, self.k_, self.neigh_size_, counting_function)

        self.population_ = self._initial_population(100, n_dims * self.d_)
        # opt_result = differential_evolution(CMOML._compute_Lvec_objective, bounds=bounds, args=(n_dims, self.d_, X, y, self.k_, self.neigh_size_, counting_function), disp=True, maxiter=150, init=self.population_, polish=False)
        opt_result = differential_evolution(_CMOML_compute_Lvec_objective, bounds=bounds, args=(n_dims, self.d_, X, y, self.k_, self.neigh_size_, counting_function, self.n_jobs), disp=True, maxiter=150, init=self.population_, polish=False, workers=self.workers)
        self.final_objective_ = - opt_result.fun
        self.L_ = np.reshape(opt_result.x, [n_dims, self.d_])

        return self

    def _initial_population(self, popsize, dim):
        segsize = 1.0 / popsize

        samples = (segsize * np.random.random_sample((popsize, dim)) +
                   np.linspace(0., 1., popsize, endpoint=False)[:, np.newaxis])

        population = np.zeros_like(samples)

        for j in range(dim):
            order = np.random.permutation(range(popsize))
            population[:, j] = samples[order, j]

        return population

    @staticmethod
    def _count_ordered_chains(i, distance_matrix, y, k, neigh_size, counting_function):
        # dists = [(np.abs(y[i] - y[m]), distance_matrix[i, m]) for m in xrange(distance_matrix.shape[0])]
        # dists = sorted(dists, key=lambda k: k[1])
        # seq = [d[0] for d in dists[:neigh_size]]
        # return CMOML.numOfIncSubseqOfSizeK(seq, neigh_size, k)
        dists = [(y[i] - y[m], distance_matrix[i, m]) for m in xrange(distance_matrix.shape[0])]
        dists = sorted(dists, key=lambda k: k[1])
        seq = np.array([d[0] for d in dists[:neigh_size]])
        asc_seq = seq[seq >= 0]
        dsc_seq = - seq[seq <= 0]
        const_seq = seq[seq == 0]
        # print(len(asc_seq), len(dsc_seq))
        asc_count = counting_function(asc_seq, len(asc_seq), k)
        dsc_count = counting_function(dsc_seq, len(dsc_seq), k)
        const_count = comb(len(const_seq), k)   # It is counted twice in ascending and descending sequences. Cannot be removed
                                                # in them since zeros may be used in both ascending and descending sequences.

        return asc_count + dsc_count - const_count

    @staticmethod
    def _compute_L_objective(L, X, y, k, neigh_size, counting_function, n_jobs=-1):
        Lx = X.dot(L.T)
        distance_matrix = pairwise_distances(X=Lx, n_jobs=n_jobs)
        chains = [CMOML._count_ordered_chains(i, distance_matrix, y, k, neigh_size, counting_function) for i in xrange(X.shape[0])]
        obj = sum(chains)
        return - obj

    @staticmethod
    def _compute_Lvec_objective(Lvec, dd, d, X, y, k, neigh_size, counting_function, n_jobs=-1):
        CMOML.evaluations += 1
        print(CMOML.evaluations)
        return CMOML._compute_L_objective(np.reshape(Lvec, [dd, d]), X, y, k, neigh_size, counting_function, n_jobs)

    @staticmethod
    def numOfIncSubseqOfSizeKEffNp(arr, n, k):
        dp = np.zeros([n, k])

        dp[:, 0] = 1

        for p in range(1, k):
            num = defaultdict(int)

            for i in range(1, n):
                num[arr[i - 1]] += dp[i - 1, p - 1]

                for j in range(0, arr[i] + 1):
                    dp[i, p] += num[j]

        Sum = dp[:, k - 1].sum()

        return Sum

    @staticmethod
    def numOfIncSubseqOfSizeKEff(arr, n, k):
        dp = [[0 for i in range(k)] for i in range(n)]

        for i in range(n):
            dp[i][0] = 1

        for p in range(1, k):
            num = defaultdict(int)

            for i in range(1, n):
                num[arr[i - 1]] += dp[i - 1][p - 1]

                for j in range(0, arr[i] + 1):
                    dp[i][p] += num[j]

        Sum = 0
        for i in range(n):
            Sum += dp[i][k - 1]

        return Sum

    @staticmethod
    def numOfIncSubseqOfSizeKNp(arr, n, k):
        dp = np.zeros([k, n])
        dp[0, :] = 1

        for l in range(1, k):
            for i in range(l, n):
                dp[l, i] = 0
                for j in range(l - 1, i):
                    if (arr[j] <= arr[i]):
                        dp[l, i] += dp[l - 1, j]

        Sum = dp[k - 1, (k - 1):].sum()

        return Sum

    @staticmethod
    def numOfIncSubseqOfSizeK(arr, n, k):
        dp = [[0 for i in range(n)] for i in range(k)]

        for i in range(n):
            dp[0][i] = 1

        for l in range(1, k):
            for i in range(l, n):
                dp[l][i] = 0
                for j in range(l - 1, i):
                    if (arr[j] <= arr[i]):
                        dp[l][i] += dp[l - 1][j]

        Sum = 0
        for i in range(k - 1, n):
            Sum += dp[k - 1][i]

        return Sum

    @staticmethod
    def listIncSubseqOfSizeK(arr, n, k, y):
        dp = []
        dp.append([[[arr[i]]] for i in range(n)])

        for l in range(1, k):
            dp.append([[v + [arr[i + l]] for j in range(0, i + 1) if y[arr[j + l - 1]] <= y[arr[i + l]] for v in dp[l - 1][j]] for i in range(0, n - l)])
        return sum(dp[k - 1], [])

    @staticmethod
    def listIncSubseqOfSizeKComb(arr, n, k, y):
        pass

    @staticmethod
    def _get_fastest_counting_function(functions, k, K, nclasses, sample_size=10000):
        x = np.random.randint(nclasses, size=(sample_size, K))
        times = np.empty([len(functions)])
        for i, f in enumerate(functions):
            start = time.time()
            calc = [f(x[j, :], K, k) for j in range(sample_size)]
            times[i] = time.time() - start
        # print(times)
        return functions[np.argmin(times)]

class KCMOML(KernelDML_Algorithm):

    def __init__(self, n_dims=None, chain_size=5, neighborhood_size=20, popsize=100, maxiter=150,
                 kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None, workers=1, n_jobs=-1):
        self.n_dims_ = n_dims
        self.k_ = chain_size
        self.neigh_size_ = neighborhood_size
        self.popsize_ = popsize
        self.maxiter_ = maxiter

        self.initial_objective_ = None
        self.final_objective_ = None

        self.kernel_ = kernel
        self.gamma_ = gamma
        self.degree_ = degree
        self.coef0_ = coef0
        self.kernel_params_ = kernel_params

        self.workers = workers
        self.n_jobs = n_jobs

    def metadata(self):
        return {'initial_objective': self.initial_objective_, 'final_objective': self.final_objective_}

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

        K = self._get_kernel(X)

        if self.n_dims_ is None:
            n_dims = self.d_
        else:
            n_dims = min(self.n_dims_, self.d_)

        bounds = [(-100, 100) for i in xrange(n_dims * self.n_)]

        nclasses = len(np.unique(y))
        counting_function = CMOML._get_fastest_counting_function([_CMOML_numOfIncSubseqOfSizeKEff,
                                                                 ],
                                                                 self.k_, self.neigh_size_, nclasses)

        self.initial_objective_ = - KCMOML._compute_L_objective(np.eye(self.d_), X, y, self.k_, self.neigh_size_, counting_function, self.n_jobs)

        self.population_ = self._initial_population(self.popsize_, n_dims * self.n_)

        opt_result = differential_evolution(_KCMOML_compute_Lvec_objective, bounds=bounds, args=(n_dims, self.n_, K, y, self.k_, self.neigh_size_, counting_function, self.n_jobs), disp=True, maxiter=self.maxiter_, init=self.population_, polish=False, workers=self.workers)
        # opt_result = opt.basinhopping(CMOML._compute_Lvec_objective, x0=np.reshape(np.eye(n_dims, self.d_), [n_dims * self.d_]), minimizer_kwargs={'args': (n_dims, self.d_, X, y, self.k_, self.neigh_size_)}, disp=True)

        self.final_objective_ = - opt_result.fun
        self.L_ = np.reshape(opt_result.x, [n_dims, self.n_])
        # print(self.L_)

    @staticmethod
    def _compute_L_objective(L, K, y, k, neigh_size, counting_function, n_jobs):
        Lkx = K.dot(L.T)
        distance_matrix = pairwise_distances(X=Lkx, n_jobs=n_jobs)
        chains = [CMOML._count_ordered_chains(i, distance_matrix, y, k, neigh_size, counting_function) for i in xrange(K.shape[0])]
        obj = sum(chains)
        # print(time.time() - t, obj)
        return - obj

    @staticmethod
    def _compute_Lvec_objective(Lvec, dd, n, K, y, k, neigh_size, counting_function, n_jobs):
        return KCMOML._compute_L_objective(np.reshape(Lvec, [dd, n]), K, y, k, neigh_size, counting_function, n_jobs)

    def _initial_population(self, popsize, dim):
        segsize = 1.0 / popsize

        samples = (segsize * np.random.random_sample((popsize, dim)) +
                   np.linspace(0., 1., popsize, endpoint=False)[:, np.newaxis])

        population = np.zeros_like(samples)

        for j in range(dim):
            order = np.random.permutation(range(popsize))
            population[:, j] = samples[order, j]

        return population
