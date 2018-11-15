#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some basic DML implementations.

Created on Fri Mar 30 19:13:58 2018

@author: jlsuarezdiaz
"""

from __future__ import print_function, absolute_import

from numpy import eye

from .dml_algorithm import DML_Algorithm

import warnings
import numpy as np


class Metric(DML_Algorithm):
    """
    A DML algorithm that defines a distance given a PSD metric matrix.

    Parameters
    ----------

    metric : (d x d) matrix. A positive semidefinite matrix, to define a pseudodistance in euclidean d-dimensional space.
    """

    def __init__(self, metric):
        self.M_ = metric

    def fit(self, X=None, y=None):
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
        self.X_, self.y_ = X, y
        return self

    def metric(self):
        """
        Obtains the learned metric.

        Returns
        -------
        M : (dxd) positive semidefinite matrix, where d is the number of features.
        """
        return self.M_


class Transformer(DML_Algorithm):
    """
    A DML algorithm that defines a distance given a linear transformation.

    Parameters
    ----------

    transformer : (d' x d) matrix, representing a linear transformacion from d-dimensional euclidean space
                  to d'-dimensional euclidean space.
    """

    def __init__(self, transformer):
        self.L_ = transformer

    def fit(self, X=None, y=None):
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
        self.X_, self.y_ = X, y
        return self

    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        L : (d'xd) matrix, where d' is the desired output dimension and d is the number of features.
        """
        return self.L_


class Euclidean(DML_Algorithm):
    """
    A basic transformer that represents the euclidean distance.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
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
        self.X_ = X
        _, d = X.shape
        self.I_ = eye(d)

    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        L : (d'xd) matrix, where d' is the desired output dimension and d is the number of features.
        """
        return self.I_

    def metric(self):
        """
        Obtains the learned metric.

        Returns
        -------
        M : (dxd) positive semidefinite matrix, where d is the number of features.
        """
        return self.I_

    def transform(self, X=None):
        """Applies the metric transformation.

        Parameters
        ----------
        X : (N x d) matrix, optional
            Data to transform. If not supplied, the training data will be used.

        Returns
        -------
        transformed : (N x d') matrix
            Input data transformed to the metric space by :math:`XL^{\\top}`
        """
        return self.X_ if X is None else X


class Covariance(DML_Algorithm):
    """
    A distance metric learning algorithm that learns the distance given by the inverse covariance matrix
    (the original concept of Mahalanobis distance).

    Parameters
    ----------

    tol: float, default=1e-15

        The toleration level to consider the covariance matrix invertible.

    reg_method: string, default='pseudoinverse'

        The strategy to deal with singular matrix. Valid options are:

        - 'pseudoinverse' : calculates the pseudoinverse matrix when covariance is singular.

        - 'addid' : adds a multiple of identity to the covariance matrix to make it invertible.
                    The value added is given by the 'alpha' parameter.

    alpha : regularization constant to calculate the inverse covariance matrix.
            Ignored if reg_method is not 'addid'.
    """

    def __init__(self, tol=1e-15, reg_method='pseudoinverse', alpha=1e-3):
        self.tol_ = tol
        self.reg_method_ = reg_method
        self.alpha_ = alpha

    def fit(self, X, y=None):
        """
        Fit the model from the data in X and the labels in y.

        Parameters
        ----------
        X : array-like, shape (N x d)
            Training vector, where N is the number of samples, and d is the number of features.

        y : array-like, shape (N)
            Labels vector, where N is the number of samples. Added for compatibility, but not used.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        self.cov_ = np.cov(X, rowvar=False)
        det = abs(np.linalg.det(self.cov_))

        if det < self.tol_:
            warnings.warn("Variables are collinear.")
            if self.reg_method_ == 'pseudoinverse':
                self.M_ = np.linalg.pinv(self.cov_)
            elif self.reg_method_ == 'addid':
                regcov = self.cov_ + self.alpha_ * np.eye(self.cov_.shape[0])
                self.M_ = np.linalg.inv(regcov)
            else:
                raise ValueError("Invalid option for 'reg_method': " + self.reg_method_)
        else:
            self.M_ = np.linalg.inv(self.cov_)

        return self

    def metric(self):
        """
        Obtains the learned metric.

        Returns
        -------
        M : (dxd) positive semidefinite matrix, where d is the number of features.
        """
        return self.M_
