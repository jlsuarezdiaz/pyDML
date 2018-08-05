#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linear Discriminant Analysis (LDA)

"""

from __future__ import absolute_import
import numpy as np
from sklearn.utils.validation import check_X_y

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA


from .dml_algorithm import DML_Algorithm


class LDA(DML_Algorithm):
    """
    Linear Discriminant Analysis (LDA).

    A distance metric learning algorithm for supervised dimensionality reduction, maximizing the ratio of variances between classes and within classes.
    This class is a wrapper for :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.

    Parameters
    ----------

    num_dims : int, default=None

        Number of components (< n_classes - 1) for dimensionality reduction. If None, it will be taken as n_classes - 1. Ignored if thres is provided.

    thres : float

        Fraction of variability to keep, from 0 to 1. Data dimension will be reduced until the lowest dimension that keeps 'thres' explained variance.
    """

    def __init__(self, num_dims=None, thres=None):

        self.nd_init = num_dims
        self.thres_init = thres
         
        
        #Metadata
        self.nd_ = None
        self.acum_eig_ = None

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
        return {'num_dims': self.nd_, 'acum_eig': self.acum_eig_}

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
        self.X_ = X
        self.y_ = y
        self.n_, self.d_ = X.shape

        self.num_dims = self.nd_init
        self.thres = self.thres_init
        self.sklda = skLDA(n_components=self.num_dims)

        self.sklda.fit(X, y)
        self.explained_variance = self.sklda.explained_variance_ratio_
        self.acum_variance = np.cumsum(self.explained_variance)
        
        

        if self.thres is None and self.num_dims is None:
            self.num_dims = self.d_
        elif not self.thres is None:
            for i, v in enumerate(self.acum_variance):
                if v >= self.thres:
                    self.num_dims = i+1
                    break
        self.num_dims = min(self.num_dims,len(self.explained_variance))

        self.L_ = self.sklda.scalings_.T[:self.num_dims,:]
        
        self.nd_ = self.num_dims
        self.acum_eig_ = self.acum_variance[self.num_dims-1]/self.acum_variance[-1]

        return self

    def transform(self,X=None):
        if X is None:
            X = self.X_

        Xcent=X-self.sklda.xbar_

        return DML_Algorithm.transform(self,Xcent)