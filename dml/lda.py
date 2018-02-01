#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linear Discriminant Analysis (LDA)

A DML that tries to minimize kNN expected error.
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y
import time
from numpy.linalg import(
    inv,eig)
import warnings


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA


from .dml_algorithm import DML_Algorithm


class LDA(DML_Algorithm):
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
        self.sklda = skLDA(n_components=num_dims) 

    def transformer(self):
        return self.L_

    def _process_data(self,X,y):
        """
        Deprecated
        """
        self.X_ = X
        self.y_ = y
        self.classes_, self.label_inds_ = np.unique(y, return_inverse=True)
        self.n_, self.d_ = X.shape

        self.mean_vectors = []

        for c in self.classes_:
            self.mean_vectors.append(np.mean(X[y==c], axis=0))

        self.overall_mean = np.mean(X,axis=0)


    def _compute_within_class_matrix(self):
        """
        Deprecated
        """
        X = self.X_
        y = self.y_
        d = self.d_

        self.within_class_matrix = np.zeros((d,d))

        for c, mu in zip(self.classes_, self.mean_vectors):
            class_mat = np.zeros((d,d))

            for row in X[y==c]:
                fact = (row-mu)[:,None]
                class_mat += np.outer(fact,fact)

            self.within_class_matrix += class_mat

        self.within_class_matrix /= self.n_


    def _compute_between_class_matrix(self):
        """
        Deprecated
        """
        X = self.X_
        y = self.y_

        self.between_class_matrix = np.zeros((self.d_,self.d_))
        glmu = self.overall_mean

        for i, mu in enumerate(self.mean_vectors):
            fact = (mu-glmu)[:,None]
            self.between_class_matrix += np.outer(fact,fact)

        self.between_class_matrix /= self.classes_.size

    def _compute_eig(self):
        """
        Deprecated
        """
        X = self.X_
        y = self.y_
        C_b, C_w = self.between_class_matrix, self.within_class_matrix

        self.eig_vals, self.eig_vecs = eig(inv(C_w).dot(C_b))

        self.eig_pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[:,i]) for i in xrange(self.eig_vals.size)]
        self.eig_pairs = sorted(self.eig_pairs, key = lambda k: k[0], reverse=True)

        # Ordered and separated eigs
        #self.eig_vals[i] = self.eig_pairs[i][0] for i in xrange(self.eig_vals.size)
        #self.eig_vecs[i] = self.eig_pairs[i][1] for i in xrange(self.eig_vals.size)
        for i, p in enumerate(self.eig_pairs):
            self.eig_vals[i] = p[0]
            self.eig_vecs[i,:] = p[1]


    def fit(self,X,y):
        self.X_ = X
        self.y_ = y
        self.n_, self.d_ = X.shape

        self.sklda.fit(X,y)
        self.explained_variance = self.sklda.explained_variance_ratio_
        self.acum_variance = np.cumsum(self.explained_variance)

        if self.thres is None and self.num_dims is None:
            self.num_dims = self.d_
        elif not self.thres is None:
            for i, v in enumerate(self.acum_variance):
                if v >= self.thres:
                    self.num_dims = i+1
                    break


        self.L_ = self.sklda.scalings_.T[:self.num_dims,:]



        return self

    def transform(self,X=None):
        if X is None:
            X = self.X_

        Xcent=X-self.sklda.xbar_

        return DML_Algorithm.transform(self,Xcent)