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
import time

from .dml_algorithm import DML_Algorithm


class NCA(DML_Algorithm):
    def __init__(self,num_dims=None, max_iter=100, learning_rate=0.01, initial_transform=None, eps = 1e-16):
        self.num_dims = num_dims
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.initial_transform = initial_transform
        self.eps_ = eps

    def transformer(self):
        return self.L_

    def _fit(self,X,y):
        X, y = check_X_y(X,y) # Consistency check
        n, d = X.shape

        num_dims = self.num_dims # If no dimensions are specified dataset dimensions are used.
        if num_dims is None:
            num_dims = d

        A = self.initial_transform # If starting transformation is not specified, diagonal ones will be used.
        if A is None:
            A = np.zeros((num_dims,d))
            #np.fill_diagonal(A,1./(np.maximum(X.max(axis=0)-X.min(axis=0))))
            np.fill_diagonal(A, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16)))


        grad_scale = 2*self.learning_rate # Gradient constant 2 is added to the learning rate.

        #cinitial=couter=cAx=csoftmax=cgr1=cgr2=cgr=0
        # TOO MUCH MEMORY!!!
        dX = X[:,None] - X[None] # Difference vectors (nxnxd). Each pair of indices represent the vector dX[i,j] = X[i,] - X[j,]
        #outers = np.einsum('...i,...j->...ij',dX,dX) # Outer product of difference vectors (nxnxdxd). Each pair of indices represent the matrix dX[i,j]*dX[i,j].T
        belongs = y[:,None] == y[None] # Belonging matrix (nxn). belongs[i,j] == true iff y[i] == y[j] 

        #cinitial+=(time.time()-start); start=time.time()  # 0,6 %
        self.n_iter = 0

        for it in xrange(self.max_iter):
            for i, label in enumerate(y):
                friends = belongs[i] # Same class indices
                outers_i = np.einsum('...i,...j->...ij',dX[i],dX[i])

                #couter+=(time.time()-start); start=time.time() # 37,11 %
    
                # Gradient computation (stochastic)
                Ax = A.dot(X.T).T
    
                #cAx+=(time.time()-start); start=time.time() # 5,47 %

                softmax = np.exp(-((Ax[i]-Ax)**2).sum(axis=1)) # Cálculo de p_{ij} (suma por columnas de la matriz de columnas A[,j] = (Ax[i]-Ax[j])^2)
                #print softmax
                softmax[i]=0
                #print softmax.sum()
                softmax /= softmax.sum()
                #print softmax

                # csoftmax+=(time.time()-start); start=time.time() # 1 %
    
                #p_outer = softmax[:, None, None] * outers[i] # pik*xik*xik.T (i fixed, nxdxd)
                p_outer = softmax[:,None,None] * outers_i
                # cgr1+=(time.time()-start); start=time.time() # 37,9 %
                grad = softmax[friends].sum() * p_outer.sum(axis=0) - p_outer[friends].sum(axis=0)
                norm = np.amax(abs(grad))
                
                # cgr2+=(time.time()-start); start=time.time() # 17,48 %
    
                A += grad_scale* A.dot(grad)
                A /= np.amax(abs(A))
                # cgr+=(time.time()-start); start=time.time() # 0,23 %

                print "Norm: ", norm
                print "Succ:" , self.compute_expected_success(A,X,y)


                if norm < self.eps_:
                    break


            self.n_iter += i
            if norm < self.eps_:
                break


        #print "Initial: ", cinitial
        #print "Outer: ", couter
        #print "Ax: ", cAx
        #print "Softmax: ", csoftmax
        #print "gr1: ", cgr1
        #print "gr2: ", cgr2
        #print "gr: ", cgr
        #print "Total: ", time.time() -st


        self.X_ = X
        self.L_ = A
        

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
            #print softmax
            #print softmax.sum()
            softmax/= softmax.sum()

            success += softmax[belongs[i]].sum()

        return success/y.size  # Normalized between 0 and 1






