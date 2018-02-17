#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Principal Component Analysis (NCA)

"""

from __future__ import absolute_import
import numpy as np

from sklearn.utils.validation import check_array
from sklearn.decomposition import PCA as skPCA



from .dml_algorithm import DML_Algorithm

class PCA(DML_Algorithm):
    def __init__(self,num_dims=None, thres=None):
        """
            num_dims: Number of dimensions for transformed data (ignored if num_dims > num_classes or thres specified)
            thres: Fraction of variability to keep. Data dimension will be reduced until the lowest dimension that keeps 'thres' explained variance.
        """

        self.num_dims_ = num_dims
        self.thres_ = thres
        if thres is not None:
            if thres == 1.0:
                self.skpca_ = skPCA()
            else:
                self.skpca_ = skPCA(n_components=thres) # Thres ignores num_dims
        else:
            self.skpca_ = skPCA(n_components=num_dims)
            
        self.nd_ = None
        self.acum_eig_ = None
        

    def transformer(self):
        return self.L_

    def metadata(self):
        return {'num_dims':self.nd_,'acum_eig':self.acum_eig_}

    def fit(self,X,y=None):
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
        if X is None:
            X = self.X_

        Xcent=X-self.skpca_.mean_

        return DML_Algorithm.transform(self,Xcent)