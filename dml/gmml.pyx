#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Geometric Mean Metric Learning (GMML)

"""

from __future__ import print_function, absolute_import
import numpy as np
from sklearn.utils.validation import check_X_y

from .dml_algorithm import DML_Algorithm


class GMML(DML_Algorithm):
    """
    Geometric Mean Metric Learning (GMML)

    A geometric approach for distance metric learning based on the Riemannian geometry of positive definite matrices.

    Parameters
    ----------

    geodesic_step: float, default=0.5

        The geodesic step for the GMML solution. This value should be in [0,1] and determines the position in the geodesic between the similarity matrices
        that will determine the solution.

    reg: float, default=1e-6

        The regularization parameter to avoid the singularity of the covariance matrix.

    constraint_factor: int, default=40

        The coefficient of the number of constraints

    prior: string or 2D-array or None, default=None

        A positive definite matrix with the prior knowledge for the regularization term. If None, no regularization is applied.
        If string, the following values are allowed:

        - "identity": the identiy matrix.

        If 2D-array, the array will be taken as the regularization matrix.

    auto_thresh: float, default=1e-9

        Threshold for auto-regularization (if no prior is specified).
        Auto-regularization is performed by adding the identity matrix to the similarity matrices.

    References
    ----------
        Pourya Zadeh et al.. "Geometric mean metric learning".
        In: International conference on machine learning. 2016, pages 2464-2471.

    """
    def __init__(self, geodesic_step=0.5, reg=1e-6, constraint_factor=40, prior=None, auto_thresh=1e-9):
        self.t_ = geodesic_step
        self.lmbda_ = reg
        self.const_factor_ = constraint_factor
        self.A0_ = prior
        self.thresh_ = auto_thresh

    def metric(self):
        """
        Obtains the learned metric.

        Returns
        -------
        M : (dxd) positive semidefinite matrix, where d is the number of features.
        """
        return self.M_

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

        t = self.t_
        lmbda = self.lmbda_
        const_factor = self.const_factor_
        A0 = self.A0_
        thresh = self.thresh_

        if A0 == "identity":
            A0 = np.eye(d)

        classes = np.unique(y)
        nclas = len(classes)

        num_const = const_factor * (nclas * (nclas - 1))

        # Constraint generation
        S, D = GMML._generate_constraints(X, y, num_const)

        # Regularization
        if A0 is not None:
            S = S + lmbda * np.linalg.inv(A0)
            D = D + lmbda * A0
        elif 1.0 / np.linalg.cond(S) < thresh or 1.0 / np.linalg.cond(D) < thresh:
            S = S + lmbda * np.eye(d)
            D = D + lmbda * np.eye(d)

        # Geodesic computation
        self.M_ = GMML._compute_geodesic_point(np.linalg.inv(S), D, t).astype(float)

        return self

    @staticmethod
    def _generate_constraints(X, y, num_const):
        n, d = X.shape

        k1 = np.random.randint(n, size=[num_const, 1])
        k2 = np.random.randint(n, size=[num_const, 1])

        ss = y[k1] == y[k2]
        dd = ~ss

        SD = X[k1[ss], :] - X[k2[ss], :]
        DD = X[k1[dd], :] - X[k2[dd], :]

        S = SD.T.dot(SD)
        D = DD.T.dot(DD)

        return S, D

    @staticmethod
    def _compute_geodesic_point(A, B, t):
        """
        Computes a point in the geodesic between A and B using the Cholesky-Schur method.

        Parameters
        ----------

        A: 2D-array

            Positive definite matrix with the starting point of the geodesic.

        B: 2D-array

            Positive definite matrix with the ending point of the geodesic.

        t: float

            Time snapshot in the geodesic to return. Between 0 and 1.

        Returns
        -------

            The point in the geodesic at the time t.
        """

        mA = np.linalg.cond(A)
        mB = np.linalg.cond(B)

        if mA > mB:
            C = A
            A = B
            B = C
            t = 1 - t

        RA = np.linalg.cholesky(A).T
        RB = np.linalg.cholesky(B).T

        Z = RB.dot(np.linalg.inv(RA))
        D, U = np.linalg.eig(Z.T.dot(Z))

        idx = D.argsort()
        D = D[idx]
        U = U[:, idx]

        T = np.diag(D ** (t / 2)).dot(U.T).dot(RA)
        G = T.T.dot(T)

        return G
