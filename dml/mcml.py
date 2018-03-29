#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:47:23 2018

@author: jlsuarezdiaz
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y

from .dml_utils import calc_outers, calc_outers_i, SDProject
from .dml_algorithm import DML_Algorithm

class MCML(DML_Algorithm):


    def __init__(self, num_dims = None, learning_rate = "adaptive", eta0 = 0.01, initial_metric = None, max_iter = 100, prec = 1e-3, 
                tol = 1e-3, descent_method = "SDP", eta_thres = 1e-14, learn_inc = 1.01, learn_dec = 0.5):
        self.num_dims_ = num_dims
        self.initial_ = initial_metric
        self.max_it_ = max_iter
        self.eta_ = self.eta0_ = eta0
        self.learning_ = learning_rate
        self.adaptive_ = (self.learning_ == 'adaptive')
        self.method_ = descent_method
        self.eps_ = prec
        self.tol_ = tol
        self.etamin_ = eta_thres
        self.l_inc_ = learn_inc
        self.l_dec_ = learn_dec
        
        # Metadata initialization
        self.num_its_ = None
        self.initial_error_ = None
        self.final_error_ = None
        
    def metadata(self):
        return {'num_iters':self.num_its_,'initial_error':self.initial_error_,'final_error':self.final_error_}

    def metric(self):
        return self.M_

    def fit(self,X,y): 
        self.n_, self.d_ = X.shape
        if self.num_dims_ is not None:
            self.nd_ = min(self.d_,self.num_dims_)
        else:
            self.nd_ = self.d_

        self.eta_ = self.eta0_
        X, y = check_X_y(X,y)
        self.X_ = X
        self.y_ = y      
        
        

        
        if self.method_ == "SDP": # Semidefinite Programming
            self._SDP_fit(X,y)
            
        return self

    def _SDP_fit(self,X,y):
        # Initialize parameters
        outers = calc_outers(X)
        n,d= self.n_, self.d_

        M = self.initial_
        if M is None or M == "euclidean":
            M= np.zeros([d,d])
            np.fill_diagonal(M,1.0) #Euclidean distance 
        elif M == "scale":
            M = np.zeros([self.nd_,self.d_])
            np.fill_diagonal(M, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance

        
        self.num_its_ = 0

        grad = None

        
        
        stop = False
        err_prev = err = self.initial_error_ = MCML._compute_error(M,X,y)
        
        while not stop:
            grad = np.zeros([d,d])
            
            for i, yi in enumerate(y):
                outers_i = calc_outers_i(X,outers,i)
                softmax = np.empty([n],dtype=float)
                softmax_sum = 0.0
                for k in xrange(n):
                    if i != k:
                        xik = (X[i]-X[k]).reshape(1,-1)
                        pik = np.exp(-xik.dot(M).dot(xik.T))
                        
                        softmax_sum += pik
                        softmax[k] = pik
                    #else:
                    #    softmax[k] = 0.0
                 
                softmax /= softmax_sum[0,0]
                        
                for j, yj in enumerate(y):
                    p0 = 1.0 if yi == yj else 0.0
                    grad += (p0 - softmax[j])*outers_i[j]
                    
            Mprev = M   
            M = M - self.eta_*grad
            M = SDProject(M)    
            
            err = MCML._compute_error(M,X,y)
            print(err)
            print("ETA: ", self.eta_)
            if self.adaptive_:
                if err < err_prev:
                    self.eta_ *= self.l_inc_                    
                else:
                    self.eta_ *= self.l_dec_
                    if self.eta_ < self.etamin_:
                        stop = True
                
                err_prev = err
            
            grad_norm = np.max(np.abs(grad))
            tol_norm = np.max(np.abs(M-Mprev)) 
            if grad_norm < self.eps_ or tol_norm < self.tol_:
                stop=True

            self.num_its_+=1
            if self.num_its_ == self.max_it_:
                stop=True
            
        self.final_error_ = MCML._compute_error(M,X,y)
        self.M_ = M
        
        return self
    
    def _compute_error(M,X,y):
        sdij = 0.0
        slog = 0.0
        for i, yi in enumerate(y):
            sexp = 0.0
            for j, yj in enumerate(y):
                if j != i:
                    xij = (X[i]-X[j]).reshape(1,-1)
                    dij = xij.dot(M).dot(xij.T)
                    if yi == yj:
                        sdij += dij
                    sexp += np.exp(dij)
            slog += np.log(sexp)
        return sdij + slog
                    

        
