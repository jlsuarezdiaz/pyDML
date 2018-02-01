#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Principal Component Analysis (NCA)

"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y
from sklearn.decomposition import PCA as skPCA
import time



from .dml_algorithm import DML_Algorithm

class PCA(DML_Algorithm):
    def __init__(self,num_dims=None, thres=None):
        """
            num_dims: Number of dimensions for transformed data (ignored if num_dims > num_classes or thres specified)
            thres: Fraction of variability to keep. Data dimension will be reduced until the lowest dimension that keeps 'thres' explained variance.
        """
        #if not num_dims is None and not thres is None:
        #    warnings.warn("Arguments 'num_dims' and 'thres' are mutually exclusive."
        #                  "Argument 'num_dims' will be ignored. Using 'thres'.")
        #    num_dims = None

        self.num_dims = num_dims
        self.thres = thres
        if thres is not None:
            self.skpca = skPCA(n_components=thres) # Thres ignores num_dims
        else:
            self.skpca = skPCA(n_components=num_dims)

    def transformer(self):
        return self.L_


    def fit(self,X,y=None):
        self.X_ = X
        self.n_, self.d_ = X.shape

        self.skpca.fit(X)   
        self.explained_variance_ = self.skpca.explained_variance_ratio_
        self.acum_variance_ = np.cumsum(self.explained_variance_)

        #if self.thres is None and self.num_dims is None:
        #    self.num_dims = self.d_
        #elif not self.thres is None:
        #    for i, v in enumerate(self.acum_variance):
        #        if v >= self.thres:
        #            self.num_dims = i+1
        #            break


        #self.L_ = self.skpca.components_[:self.num_dims,:]
        self.L_ = self.skpca.components_

        return self 

    def transform(self,X=None):
        if X is None:
            X = self.X_

        Xcent=X-self.skpca.mean_

        return DML_Algorithm.transform(self,Xcent)