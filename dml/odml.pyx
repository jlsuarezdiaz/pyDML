#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y

from numpy.linalg import eig
import numpy.matlib as mat

from .dml_algorithm import DML_Algorithm, KernelDML_Algorithm
from .dml_utils import pairwise_sq_distances_from_dot, calc_outers, calc_outers_i, calc_outers_ij

from scipy.optimize import differential_evolution


class LODML(DML_Algorithm):

    def __init__(self, k=5, v=15, alpha=0.01, maxit=1000, stepsize=0.1):
        self.k_ = k
        self.v_ = v
        self.alpha_ = alpha
        self.maxit_ = maxit
        self.eta_ = stepsize

    def metric(self):
        return self.M_

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_, self.y_ = X, y
        self.n_, self.d_ = X.shape
        n, d = X.shape

        T = self._get_all_constraints(X, y, self.k_, self.v_)
        # print(T)
        x0 = np.eye(d).reshape(-1, 1)
        self.M_, self.final_objective_ = self._gradproj(x0, self.maxit_, self._objfunction, T, X, self.alpha_, self.eta_)
        self.M_ = self.M_.reshape(d, d)
        # print(self.M_)
        return self

    def _get_all_constraints(self, X, y, k, v):
        n, d = X.shape
        l = np.unique(y)

        D = np.sum(X**2, axis=1).reshape(1, -1)
        D = D + D.T - 2 * (X.dot(X.T))  # Squared distances
        np.fill_diagonal(D, np.inf)

        T = np.zeros([3, n * (v + k) * (v + k)], dtype=int)
        m = 0

        for i in xrange(len(l)):
            print("Searching constraints in class ", l[i])

            # find targets
            inds = np.flatnonzero(y == l[i])
            k1 = min(len(inds) - 1, k)
            print(k, len(inds))
            if k1 >= 1:
                tars = np.argsort(D[inds, :][:, inds], axis=0)
                tars = inds[tars[:k1, :]]
                # find impostors
                indd = np.flatnonzero(y != l[i])
                k2 = min(len(indd), v)

                if k2 >= 1:
                    imps = np.argsort(D[indd, :][:, inds], axis=0)
                    imps = indd[imps[:k2, :]]
                    # print(imps)
                    [C, length] = self._join_triplets(inds, y, tars, imps, k1, k2)
                    T[:, m:(m + length)] = C
                    m = m + length

        T = T[:, :m]
        print("---------------")
        return T

    def _join_triplets(self, inds, y, tars, imps, k1, k2):
        inds = inds.reshape(-1, 1).T  # (vec = reshape(-1,1))
        n = len(inds.flatten())
        T = np.zeros([3, n * k2 * k2], dtype=int)

        T[0, :] = mat.repmat(inds, k2 * k2, 1).T.flatten()  # .reshape(-1, 1)
        T[1, :] = mat.repmat(imps.T.reshape(-1, 1), 1, k2).flatten()  # .reshape(-1, 1)
        T[2, :] = mat.repmat(imps, k2, 1).T.flatten()   # .reshape(-1, 1)

        G = y[T]
        ind = np.logical_or(np.logical_and(G[2, :] > G[1, :], G[1, :] > G[0, :]), np.logical_and(G[0, :] > G[1, :], G[1, :] > G[2, :]))
        length = sum(ind)
        T[:, :length] = T[:, ind]

        print(k1, k2, T.shape, length, n, inds.shape)
        print(T[0, length:(length + n * k1 * k2)].shape)
        T[0, length:(length + n * k1 * k2)] = mat.repmat(inds, k1 * k2, 1).T.flatten()  # reshape(-1, 1)
        T[1, length:(length + n * k1 * k2)] = mat.repmat(tars.T.reshape(-1, 1), 1, k2).flatten()  # reshape(-1, 1)
        T[2, length:(length + n * k1 * k2)] = mat.repmat(imps, k1, 1).T.flatten()  # reshape(-1, 1)

        T = T[:, :(length + n * k1 * k2)]
        length = length + n * k1 * k2

        return T, length

    def _gradproj(self, x0, maxit, objfunction, T, X, alpha, eta):
        X = X.T  # (d x n) like matlab
        d, n = X.shape
        beta = 1.0 / T.shape[1]
        L = np.eye(d)

        # eta = stepsize
        min_iter = 50
        tol = 1e-7
        quiet = False

        xc = x0
        it = 1
        C = np.inf
        prev_C = np.inf
        best_C = np.inf
        best_x = x0

        grad = (alpha * np.eye(d)).reshape(-1, 1)

        slack = np.zeros([1, T.shape[1]], dtype=float)
        outers = calc_outers(X.T)

        while (abs(prev_C - C) > tol or it < min_iter) and (it < maxit):
            prev_C = C

            C, grad, slack = objfunction(xc, X, alpha, T, L.T.dot(X), beta, grad, slack, outers)  # Evaluar función objetivo

            if not quiet and it % 50 == 0:
                print("iter = ", it, ", C = ", C, ", Active = ", sum(slack > 0))

            if C < best_C:
                best_C = C
                best_x = xc

            [xc, L] = self._kk_proj(xc - eta * grad, d)
            if L.size == 0:
                break

            if prev_C > C:
                eta = eta * 1.01
            else:
                eta = eta * 0.5

            it = it + 1

        return best_x, best_C

    def _kk_proj(self, xc, d):
        xc = xc.reshape(d, d)
        D, L = eig(xc)
        D = D.astype(float)
        L = L.astype(float)
        ind = np.flatnonzero(D > 0.0)
        D = np.matrix(np.diag(D)[ind, :][:, ind])
        xc = (L[:, ind].dot(D).dot(L[:, ind].T)).reshape(-1, 1)
        L = L[:, ind].dot(np.sqrt(D))
        return np.asarray(xc), np.asarray(L)

    def _objfunction(self, x, X, alpha, T, Lx, beta, grad, old_slack, outers):
        d, n = X.shape
        slack = np.maximum(0, 1 + np.sum((Lx[:, T[0, :]] - Lx[:, T[1, :]]) ** 2, axis=0) -
                           np.sum((Lx[:, T[0, :]] - Lx[:, T[2, :]]) ** 2, axis=0))

        val = alpha * np.trace(x.reshape(d, d)) + beta * np.sum(slack)

        ind = np.flatnonzero(np.logical_and(slack > 0, old_slack == 0))
        for i in ind:
            if outers is not None:
                grad = grad + beta * (outers[T[0, i], T[1, i], :, :]).reshape(-1, 1) - beta * (outers[T[0, i], T[2, i], :, :]).reshape(-1, 1)
            else:
                outers_i = calc_outers_i(X.T, outers, T[0, i])
                outers_ij1 = calc_outers_ij(X.T, outers_i, T[0, i], T[1, i])
                outers_ij2 = calc_outers_ij(X.T, outers_i, T[0, i], T[2, i])
                grad = grad + beta * outers_ij1.reshape(-1, 1) - beta * outers_ij2.reshape(-1, 1)

        ind = np.flatnonzero(np.logical_and(slack == 0, old_slack > 0))
        for i in ind:
            if outers is not None:
                grad = grad - beta * (outers[T[0, i], T[1, i], :, :]).reshape(-1, 1) + beta * (outers[T[0, i], T[2, i], :, :]).reshape(-1, 1)
            else:
                outers_i = calc_outers_i(X.T, outers, T[0, i])
                outers_ij1 = calc_outers_ij(X.T, outers_i, T[0, i], T[1, i])
                outers_ij2 = calc_outers_ij(X.T, outers_i, T[0, i], T[2, i])
                grad = grad - beta * outers_ij1.reshape(-1, 1) + beta * outers_ij2.reshape(-1, 1)

        return val, grad, slack


class KODML(KernelDML_Algorithm):

    def __init__(self, k=5, v=15, alpha=0.01, maxit=1000, stepsize=0.1,
                 kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None):
        self.k_ = k
        self.v_ = v
        self.alpha_ = alpha
        self.maxit_ = maxit
        self.eta_ = stepsize

        self.kernel_ = kernel
        self.gamma_ = gamma
        self.degree_ = degree
        self.coef0_ = coef0
        self.kernel_params_ = kernel_params

    def metric(self):
        return self.M_

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_, self.y_ = X, y
        self.n_, self.d_ = X.shape
        n, d = X.shape

        K = self._get_kernel(X)

        T = self._get_all_constraints(K, y, self.k_, self.v_)
        x0 = np.eye(n).reshape(-1, 1)
        self.M_, self.final_objective_ = self._gradproj(x0, self.maxit_, self._objfunction, T, K, self.alpha_, self.eta_)
        self.M_ = self.M_.reshape(n, n)
        print(self.M_)
        return self

    def _get_all_constraints(self, K, y, k, v):
        n = K.shape[0]
        l = np.unique(y)

        D = np.sum(K**2, axis=1).reshape(1, -1)
        D = D + D.T - 2 * (K.dot(K.T))  # Squared distances
        np.fill_diagonal(D, np.inf)

        T = np.zeros([3, n * (v + k) * (v + k)], dtype=int)
        m = 0

        for i in xrange(len(l)):
            print("Searching constraints in class ", l[i])

            # find targets
            inds = np.flatnonzero(y == l[i])
            k1 = min(len(inds) - 1, k)

            if k1 >= 1:
                tars = np.argsort(D[inds, :][:, inds], axis=0)
                tars = inds[tars[:k1, :]]
                # find impostors
                indd = np.flatnonzero(y != l[i])
                k2 = min(len(indd), v)

                if k2 >= 1:
                    imps = np.argsort(D[indd, :][:, inds], axis=0)
                    imps = indd[imps[:k2, :]]
                    # print(imps)
                    [C, length] = self._join_triplets(inds, y, tars, imps, k1, k2)
                    T[:, m:(m + length)] = C
                    m = m + length

        T = T[:, :m]
        print("---------------")
        return T

    def _join_triplets(self, inds, y, tars, imps, k1, k2):
        inds = inds.reshape(-1, 1).T  # (vec = reshape(-1,1))
        n = len(inds.flatten())
        T = np.zeros([3, n * k2 * k2], dtype=int)

        T[0, :] = mat.repmat(inds, k2 * k2, 1).T.flatten()  # .reshape(-1, 1)
        T[1, :] = mat.repmat(imps.T.reshape(-1, 1), 1, k2).flatten()  # .reshape(-1, 1)
        T[2, :] = mat.repmat(imps, k2, 1).T.flatten()   # .reshape(-1, 1)

        G = y[T]
        ind = np.logical_or(np.logical_and(G[2, :] > G[1, :], G[1, :] > G[0, :]), np.logical_and(G[0, :] > G[1, :], G[1, :] > G[2, :]))
        length = sum(ind)
        T[:, :length] = T[:, ind]

        T[0, length:(length + n * k1 * k2)] = mat.repmat(inds, k1 * k2, 1).T.flatten()  # reshape(-1, 1)
        T[1, length:(length + n * k1 * k2)] = mat.repmat(tars.T.reshape(-1, 1), 1, k2).flatten()  # reshape(-1, 1)
        T[2, length:(length + n * k1 * k2)] = mat.repmat(imps, k1, 1).T.flatten()  # reshape(-1, 1)

        T = T[:, :(length + n * k1 * k2)]
        length = length + n * k1 * k2

        return T, length

    def _gradproj(self, x0, maxit, objfunction, T, K, alpha, eta):
        K = K.T  # (d x n) like matlab
        # d, n = X.shape
        n = K.shape[0]
        beta = 1.0 / T.shape[1]
        L = np.eye(n)

        # eta = stepsize
        min_iter = 50
        tol = 1e-7
        quiet = False

        xc = x0
        it = 1
        C = np.inf
        prev_C = np.inf
        best_C = np.inf
        best_x = x0

        grad = (alpha * K).reshape(-1, 1)  # (alpha * np.eye(d)).reshape(-1, 1)

        slack = np.zeros([1, T.shape[1]], dtype=float)
        # outers = calc_outers(K.T)

        while (abs(prev_C - C) > tol or it < min_iter) and (it < maxit):
            prev_C = C

            C, grad, slack = objfunction(xc, K, alpha, T, L.T.dot(K), beta, grad, slack, None)  # Evaluar función objetivo

            if not quiet and it % 50 == 0:
                print("iter = ", it, ", C = ", C, ", Active = ", sum(slack > 0))

            if C < best_C:
                best_C = C
                best_x = xc

            [xc, L] = self._kk_proj(xc - eta * grad, n)
            if L.size == 0:
                break

            if prev_C > C:
                eta = eta * 1.01
            else:
                eta = eta * 0.5

            it = it + 1

        return best_x, best_C

    def _kk_proj(self, xc, d):
        xc = xc.reshape(d, d)
        D, L = eig(xc)
        D = D.astype(float)
        L = L.astype(float)
        ind = np.flatnonzero(D > 0.0)
        D = np.matrix(np.diag(D)[ind, :][:, ind])
        xc = (L[:, ind].dot(D).dot(L[:, ind].T)).reshape(-1, 1)
        L = L[:, ind].dot(np.sqrt(D))
        return np.asarray(xc), np.asarray(L)

    def _objfunction(self, x, K, alpha, T, Lx, beta, grad, old_slack, outers): # !!! outers not being used. TODO remove this parameter.
        # d, n = K.shape
        # n = K.shape[0]
        slack = np.maximum(0, 1 + np.sum((Lx[:, T[0, :]] - Lx[:, T[1, :]]) ** 2, axis=0) -
                           np.sum((Lx[:, T[0, :]] - Lx[:, T[2, :]]) ** 2, axis=0))

        val = alpha * + x.T.dot(K.reshape(-1, 1)) + beta * np.sum(slack)  # alpha * np.trace(x.reshape(d, d)) + beta * np.sum(slack)

        ind = np.flatnonzero(np.logical_and(slack > 0, old_slack == 0))
        for i in ind:
            xij1 = K[T[0, i]] - K[T[1, i]]
            xij2 = K[T[0, i]] - K[T[2, i]]
            grad = grad + beta * np.outer(xij1, xij1).reshape(-1, 1) - beta * np.outer(xij2, xij2).reshape(-1, 1)
            # grad = grad + beta * (outers[T[0, i], T[1, i], :, :]).reshape(-1, 1) - beta * (outers[T[0, i], T[2, i], :, :]).reshape(-1, 1)

        ind = np.flatnonzero(np.logical_and(slack == 0, old_slack > 0))
        for i in ind:
            xij1 = K[T[0, i]] - K[T[1, i]]
            xij2 = K[T[0, i]] - K[T[2, i]]
            grad = grad - beta * np.outer(xij1, xij1).reshape(-1, 1) + beta * np.outer(xij2, xij2).reshape(-1, 1)
            # grad = grad - beta * (outers[T[0, i], T[1, i], :, :]).reshape(-1, 1) + beta * (outers[T[0, i], T[2, i], :, :]).reshape(-1, 1)

        return val, grad, slack
