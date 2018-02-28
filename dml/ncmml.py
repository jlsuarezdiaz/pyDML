#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:07:43 2018

@author: jlsuarezdiaz
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder

from numpy.linalg import eig
from scipy.linalg import eigh

from .dml_algorithm import DML_Algorithm, KernelDML_Algorithm
from .dml_utils import calc_outers, calc_outers_i

class NCMML(DML_Algorithm):
    
    def __init__(self, num_dims=None, learning_rate = "adaptive", eta0 = 0.01, initial_transform = None, max_iter = 100,
                 tol = 1e-3, prec = 1e-3, descent_method = "SGD", eta_thres = 1e-14, learn_inc = 1.01, learn_dec = 0.5):
        
        self.num_dims_ = num_dims
        self.L0_ = initial_transform
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
        self.initial_expectance_ = None
        self.final_expectance_ = None
        
    def metadata(self):
        return {'num_iters':self.num_its_, 'initial_expectance':self.initial_expectance_, 'final_expectance':self.final_expectance_}
    
    def transformer(self):
        return self.L_
        
    def fit(self,X,y): 
        self.n_, self.d_ = X.shape
        if self.num_dims_ is not None:
            self.nd_ = min(self.d_,self.num_dims_)
        else:
            self.nd_ = self.d_


        self.L_ = self.L0_
        self.eta_ = self.eta0_
        X, y = check_X_y(X,y)
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.X_ = X
        self.y_ = y      

        if self.L_ is None or self.L_ == "euclidean":
            self.L_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.L_,1.0) #Euclidean distance 
        elif self.L_ == "scale":
            self.L_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.L_, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance

        self.centroids_ = NCMML._compute_class_centroids(X,y)
        
        self.initial_expectance_ = NCMML._compute_expected_success(self.L_,X,y,self.centroids_)
        if self.method_ == "SGD": # Stochastic Gradient Descent
            self._SGD_fit(X,y)
            
        elif self.method_ == "BGD": # Batch Gradient Desent
            self._BGD_fit(X,y)
        
        self.final_expectance_ = NCMML._compute_expected_success(self.L_,X,y,self.centroids_)
        return self
    
    def _SGD_fit(self,X,y):
        # Initialize parameters
        
        n,d= self.n_, self.d_

        L = self.L_
        centroids = self.centroids_
        classes, _ = centroids.shape
        
        outers = calc_outers(X,centroids)
        
        self.num_its_ = 0

        grad = None

        succ_prev = succ = 0.0
        
        stop = False
        
        
        while not stop:
            X, y, outers = NCMML._shuffle(X,y,outers)            
            
            for i,yi in enumerate(y):
               
                Lxi = L.dot(X[i,:].T).T
                Lmu = L.dot(centroids.T).T
                
                mu_diff = Lxi - Lmu
                dists_i = -0.5*np.diag(mu_diff.dot(mu_diff.T))
                softmax = np.exp(dists_i)
                softmax /= softmax.sum()
    
                grad_sum = 0.0
                outers_i = calc_outers_i(X,outers,i,centroids)
                for c in xrange(classes):
                    mask = 1 if yi == c else 0
                    grad_sum += (softmax[c]-mask)*outers_i[c]
                    
                grad = L.dot(grad_sum)
                L += self.eta_*grad
            
            succ = NCMML._compute_expected_success(L,X,y,centroids)
            
            if self.adaptive_:
                if succ > succ_prev:
                    self.eta_ *= self.l_inc_                    
                else:
                    self.eta_ *= self.l_dec_
                    if self.eta_ < self.etamin_:
                        stop = True
                
                succ_prev = succ
                    
            
            grad_norm = np.max(np.abs(grad))
            if grad_norm < self.eps_ or self.eta_*grad_norm < self.tol_: # Difference between two iterations is given by eta*grad
                stop=True

            self.num_its_+=1
            if self.num_its_ == self.max_it_:
                stop=True
            if stop:
                break
            
        self.L_ = L
        
        return self
        
    def _BGD_fit(self,X,y):
        # Initialize parameters
        
        n,d= self.n_, self.d_

        L = self.L_
        centroids = self.centroids_
        classes, _ = centroids.shape
        
        outers = calc_outers(X,centroids)
        
        self.num_its_ = 0

        grad = None

        succ_prev = succ = 0.0
        
        stop = False
        
        
        while not stop:
            Lx = L.dot(X.T).T            
            Lmu = L.dot(centroids.T).T
            grad_sum = 0.0
            
            for i,yi in enumerate(y):
                grad = np.zeros([d,d])
                Lxi = Lx[i]
                
                mu_diff = Lxi - Lmu
                dists_i = -0.5*np.diag(mu_diff.dot(mu_diff.T))
                softmax = np.exp(dists_i)
                softmax /= softmax.sum()
    
                
                outers_i = calc_outers_i(X,outers,i,centroids)
                for c in xrange(classes):
                    mask = 1 if yi == c else 0
                    grad_sum += (softmax[c]-mask)*outers_i[c]
                    
            grad = L.dot(grad_sum)/len(y)
            L += self.eta_*grad
            
            succ = NCMML._compute_expected_success(L,X,y,centroids)
            
            
            if self.adaptive_:
                if succ > succ_prev:
                    self.eta_ *= self.l_inc_                    
                else:
                    self.eta_ *= self.l_dec_
                    if self.eta_ < self.etamin_:
                        stop = True
                
                succ_prev = succ
                    
            
            grad_norm = np.max(np.abs(grad))
            if grad_norm < self.eps_ or self.eta_*grad_norm < self.tol_: # Difference between two iterations is given by eta*grad
                stop=True

            self.num_its_+=1
            if self.num_its_ == self.max_it_:
                stop=True
            if stop:
                break
            
        self.L_ = L
        
        return self



    def _shuffle(X,y,outers=None):
        rnd = np.random.permutation(len(y))
        X = X[rnd,:]
        y = y[rnd]
        if outers is not None:
            outers = outers[rnd,:]
        else:
            outers = None

        return X,y, outers
    
    def _compute_expected_success(L,X,y,centroids):
        n,d = X.shape
        classes, _ = centroids.shape
        Lx = L.dot(X.T).T
        Lmu = L.dot(centroids.T).T
        success = 0.0
        
        for i, yi in enumerate(y):
            
            Lxi = Lx[i]
            mu_diff = Lxi - Lmu
            dists_i = -0.5*np.diag(mu_diff.dot(mu_diff.T))
            
            softmax = np.exp(dists_i)
            
            success += np.log(softmax[yi]/softmax.sum())
            
        return success/len(y)
    
    
    def _compute_class_centroids(X,y):
        classes = np.unique(y)
        n, d = X.shape
        centroids = np.empty([len(classes),d])
        
        for i,c in enumerate(classes):
            Xc = X[y==c]
            mu_c = np.mean(Xc,axis=0)
            centroids[i,:] = mu_c
            
        return centroids
        
    
        
    