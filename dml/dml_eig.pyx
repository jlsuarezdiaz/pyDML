#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance Metric Learning with Eigenvalue Optimization

Created on Fri Mar  9 10:18:35 2018

@author: jlsuarezdiaz
"""

from __future__ import print_function, absolute_import
import numpy as np

from six.moves import xrange
from sklearn.utils.validation import check_X_y

from numpy.linalg import inv, eigh
import scipy.linalg as sl

from .dml_algorithm import DML_Algorithm
from .dml_utils import calc_outers, calc_outers_i


class DML_eig(DML_Algorithm):
    """
        Distance Metric Learning with Eigenvalue Optimization (DML-eig)

        A DML Algorithm that learns a metric that minimizes the minimum distance between different-class points
        constrained to the sum of distances at same-class points be non higher than a constant.

        Parameters
        ----------
        mu : float, default=1e-4
            Smoothing parameter.

        tol : float, default=1e-5
            Tolerance stop criterion (difference between two point iterations at gradient descent.)

        eps : float, default=1e-10
            Precision stop criterion (norm of gradient at gradient descent)

        max_it: int, default=25
            Number of iterations at gradient descent.

        References
        ----------
            Yiming Ying and Peng Li. “Distance metric learning with eigenvalue optimization”. In: Journal of
            Machine Learning Research 13.Jan (2012), pages 1-26.
    """

    def __init__(self, mu=1e-4, tol=1e-5, eps=1e-10, max_it=25):
        self.mu_ = mu
        # self.beta_ = beta
        self.tol_ = tol
        self.max_it_ = max_it
        self.eps_ = 1e-10
        # self.linesearch_ = linesearch

        self.initial_error_ = None
        self.final_error_ = None

    def metric(self):
        """
        Obtains the learned metric.

        Returns
        -------
        M : (dxd) positive semidefinite matrix, where d is the number of features.
        """
        return self.M_

    def metadata(self):
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            initial_error : initial value of the objective error function.

            final_error : final value of the objective error function.
        """
        return {'initial_error': self.initial_error_, 'final_error': self.final_error_}

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

        # Initialize parameters
        X, y = check_X_y(X, y)
        self.X_, self.y_ = X, y

        S, D = DML_eig._label_to_similarity_set(y)
        # ns, nd = S.shape[0], D.shape[0]

        S, D = DML_eig._similarity_set_to_iterable(S, D)

        mu = self.mu_
        # beta = self.beta_
        tol = self.tol_
        max_it = self.max_it_
        eps = self.eps_
        # linesearch = self.linesearch_

        n, d = X.shape

        M = np.zeros([d, d])
        np.fill_diagonal(M, 1.0 / d)

        outers = calc_outers(X)
        Xs = np.zeros([d, d])
        # for [i,j] in S:
        for i in S:
            outers_i = calc_outers_i(X, outers, i)
            for j in S[i]:
                Xs += outers_i[j]
            # Xs += calc_outers_i(X,outers,i)[j]
        vals, U = eigh(Xs)
        if np.linalg.det(Xs) < eps:
            I = np.eye(d)
            Xs += 1e-5 * I

        vals[vals < eps] = eps
        Xs_invsqrt = inv(U.dot(np.diag(np.sqrt(vals))).dot(U.T))

        stop = False
        its = 0

        while not stop:
            grad_sum = 0.0
            grad = np.zeros([d, d])

            # for [i,j] in D:
            for i in D:
                outers_i = calc_outers_i(X, outers, i)
                for j in D[i]:
                    Xtau = outers_i[j]  # calc_outers_i(X,outers,i)[j]
                    XT = Xs_invsqrt.dot(Xtau).dot(Xs_invsqrt)
                    inner = np.inner(XT.reshape(1, -1), M.reshape(1, -1))
                    soft = np.exp(-inner / mu)
                    grad += soft * XT
                    grad_sum += soft
                    if its == 0:
                        self.initial_error_ = -mu * np.log(grad_sum)

            grad /= grad_sum

            _, V = sl.eigh(grad, eigvals=(grad.shape[0] - 1, grad.shape[0] - 1))
            Z = V.dot(V.T)
            alphat = 1 / (its + 1)
            Mprev = M
            M = (1 - alphat) * M + alphat * Z

            tol_norm = np.max(np.abs(M - Mprev))

            if tol_norm < tol:
                stop = True

            its += 1
            if its == max_it:
                stop = True

        self.final_error_ = -mu * np.log(grad_sum)  # Error before last iteration !!
        self.M_ = M
        return self

    def _label_to_similarity_set(y):
        n = len(y)
        S = []  # np.empty([n,n],dtype=bool)
        D = []  # np.empty([n,n],dtype=bool)
        for i in xrange(n):
            for j in xrange(n):
                if y[i] == y[j]:
                    S.append([i, j])
                else:
                    D.append([i, j])
        return np.array(S), np.array(D)

    def _similarity_set_to_iterable(S, D):
        # For more efficiency
        dS = {}
        dD = {}

        for [i, j] in S:
            if i in dS:
                dS[i].append(j)
            else:
                dS[i] = []

        for i, j in D:
            if i in dD:
                dD[i].append(j)
            else:
                dD[i] = []

        return dS, dD

    def _SODW(x, a, b, w):
        nn = len(a)
        d = x.shape[0]
        res = np.zeros([d, d])
        for i in xrange(nn):
            res += w[i] * (x[:, a[i]] - x[:, b[i]]).dot((x[:, a[i]] - x[:, b[i]]).T)

        return res
