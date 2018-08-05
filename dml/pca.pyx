#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Principal Component Analysis (PCA)

"""

from __future__ import absolute_import
import numpy as np

from sklearn.utils.validation import check_array
from sklearn.decomposition import PCA as skPCA

from .dml_algorithm import DML_Algorithm


class PCA(DML_Algorithm):
    """
    Principal Component Analysis (PCA)

    A distance metric learning algorithm for unsupervised dimensionality reduction, obtaining orthogonal directions that maximize the variance.
    This class is a wrapper for :class:`~sklearn.decomposition.PCA`.

    Parameters
    ----------

    num_dims : int, default=None

        Number of components for dimensionality reduction. If None, all the principal components will be taken. Ignored if thres is provided.

    thres : float

        Fraction of variability to keep, from 0 to 1. Data dimension will be reduced until the lowest dimension that keeps 'thres' explained variance.
    """

    def __init__(self, num_dims=None, thres=None):

        self.num_dims_ = num_dims
        self.thres_ = thres
        if thres is not None:
            if thres == 1.0:
                self.skpca_ = skPCA()
            else:
                self.skpca_ = skPCA(n_components=thres)  # Thres ignores num_dims
        else:
            self.skpca_ = skPCA(n_components=num_dims)
            
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
            num_dims : dimension of the reduced data.

            acum_eig : eigenvalue rate accumulated in the learned output respect to the total dimension.
        """
        return {'num_dims':self.nd_,'acum_eig':self.acum_eig_}

    def fit(self,X,y=None):
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
        X = check_array(X)
        self.X_ = X
        self.n_, self.d_ = X.shape

        self.skpca_.fit(X)   
        self.explained_variance_ = self.skpca_.explained_variance_ratio_
        self.acum_variance_ = np.cumsum(self.explained_variance_)

        
        self.L_ = self.skpca_.components_
        self.nd_ = self.L_.shape[0]
        self.acum_eig_ = self.acum_variance_[self.nd_-1]

        return self 

    def transform(self,X=None):
        """Applies the kernel transformation.

        Parameters
        ----------
        X : (N x d) matrix, optional
            Data to transform. If not supplied, the training data will be used.

        Returns
        -------
        transformed: (N x d') matrix.
            Input data transformed by the learned mapping.
        """
        if X is None:
            X = self.X_

        Xcent=X-self.skpca_.mean_

        return DML_Algorithm.transform(self,Xcent)