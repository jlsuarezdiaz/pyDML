#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neighbourhood Component Analysis (NCA)

A DML that tries to minimize kNN expected error.
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y

from .dml_algorithm import DML_Algorithm


class NCA(DML_Algorithm):
    def __init__(self,num_dims=None, max_iter=100, learning_rate=0.01, initial_transform=None):
        self.num_dims = num_dims
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.initial_transform = initial_transform

    def transformer(self):
        return self.L_

    def fit(self,X,y):
        X, y = check_X_y(X,y) # Consistency check
        n, d = X.shape

        num_dims = self.num_dims # If no dimensions are specified dataset dimensions are used.
        if num_dims is None:
            num_dims = d

        A = self.initial_transform # If starting transformation is not specified, diagonal ones will be used.
        if A is None:
            A = np.zeros((num_dims,d))
            np.fill_diagonal(A,1)

        grad_scale = 2*self.learning_rate # Gradient constant 2 is added to the learning rate.

        # TOO MUCH MEMORY!!!
        dX = X[:,None] - X[None] # Difference vectors (nxnxd). Each pair of indices represent the vector dX[i,j] = X[i,] - X[j,]
        outers = np.einsum('...i,...j->...ij',dX,dX) # Outer product of difference vectors (nxnxdxd). Each pair of indices represent the matrix dX[i,j]*dX[i,j].T
        belongs = y[:,None] == y[None] # Belonging matrix (nxn). belongs[i,j] == true iff y[i] == y[j] 

        for it in xrange(self.max_iter):
            for i, label in enumerate(y):
                friends = belongs[i] # Same class indices
    
                # Gradient computation (stochastic)
                Ax = A.dot(X.T).T
    
                softmax = np.exp(-((Ax[i]-Ax)**2).sum(axis=1)) # Cálculo de p_{ij} (suma por columnas de la matriz de columnas A[,j] = (Ax[i]-Ax[j])^2)
                softmax[i]=0
                softmax /= softmax.sum()
    
                p_outer = softmax[:, None, None] * outers[i] # pik*xik*xik.T (i fixed, nxdxd)
                grad = softmax[friends].sum() * p_outer.sum(axis=0) - p_outer[friends].sum(axis=0)
    
                A += grad_scale* A.dot(grad)

                #print(self.compute_expected_success(A,X,y))

        self.X_ = X
        self.L_ = A
        self.n_iter = it

        # Improve this
        self.expected_success = self.compute_expected_success(A,X,y)

        return self

    def compute_expected_success(self,A,X,y):
        Ax = A.dot(X.T).T
        success = 0
        belongs = y[:,None] == y[None]
        for i, label in enumerate(y):
            softmax = np.exp(-((Ax[i]-Ax)**2).sum(axis=1)) 
            softmax[i] = 0
            softmax/= softmax.sum()

            success += softmax[belongs[i]].sum()

        return success/y.size  # Normalized between 0 and 1






