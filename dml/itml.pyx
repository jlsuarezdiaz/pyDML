#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information Theoretic Metric Learning (ITML)

Created on Thu Feb  1 17:19:12 2018

@author: jlsuarezdiaz
"""

from __future__ import print_function, absolute_import
import numpy as np

from six.moves import xrange
from sklearn.utils.validation import check_X_y

from numpy.linalg import norm

from .dml_algorithm import DML_Algorithm


class ITML(DML_Algorithm):
    """
    Information Theoretic Metric Learning (ITML).

    A DML algorithm that learns a metric associated to the nearest gaussian distribution satisfying similarity constraints.
    The nearest gaussian distribution is obtained minimizing the Kullback-Leibler divergence.

    Parameters
    ----------

    initial_metric : 2D-Array or Matrix

            A positive definite matrix that defines the initial metric used to compare.

    upper_bound : float, default=None

            Bound for dissimilarity constraints. If None, it will be estimated from upper_perc.

    lower_bound : float, default=None

            Bound for similarity constraints. If None, it will be estimated from lower_perc.

    num_constraints : int, default=None

            Number of constraints to generate. If None, it will be taken as 40 * k * (k-1), where k is the number of classes.

    gamma : float, default=1.0

            The gamma value for slack variables.

    tol : float, default=0.001

            Tolerance stop criterion for the algorithm.

    max_iter : int, default=100000

            Maximum number of iterations for the algorithm.

    low_perc : int, default=5

            Lower percentile (from 0 to 100) to estimate the lower bound from the dataset. Ignored if lower_bound is provided.

    up_perc : int, default=95

            Upper percentile (from 0 to 100) to estimate the upper bound from the dataset. Ignored if upper_bound is provided.

    References
    ----------
        Jason V Davis et al. “Information-theoretic metric learning”. In: Proceedings of the 24th
        international conference on Machine learning. ACM. 2007, pages. 209-216.

    """

    def __init__(self, initial_metric=None, upper_bound=None, lower_bound=None, num_constraints=None, gamma=1.0, tol=0.001, max_iter=100000, low_perc=5, up_perc=95):
        self.M0_ = initial_metric
        self.u_ = upper_bound
        self.l_ = lower_bound
        self.gamma_ = gamma
        self.tol_ = tol
        self.max_its_ = max_iter
        self.num_constraints_ = num_constraints
        self.low_perc_ = low_perc
        self.up_perc_ = up_perc

    def metric(self):
        """
        Obtains the learned metric.

        Returns
        -------
        M : (dxd) positive semidefinite matrix, where d is the number of features.
        """
        return self.M_

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
        self.X_, self.y_ = X, y

        # Initialize parameters
        n, d = X.shape
        M = self.M0_
        u = self.u_
        l = self.l_
        num_constraints = self.num_constraints_
        tol = self.tol_
        gamma = self.gamma_
        max_its = self.max_its_

        if M is None or M == "euclidean":
            M = np.zeros([d,d])
            np.fill_diagonal(M,1.0) #Euclidean distance 
        elif M == "scale":
            M = np.zeros([d,d])
            np.fill_diagonal(M, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance
        
        if u is None or l is None:
            [l,u] = ITML._compute_distance_extremes(X,self.low_perc_,self.up_perc_,M)
            
        if num_constraints is None:
            k = len(np.unique(y))
            num_constraints = 40 * k * (k-1) # 40 is the default parameter in the original algorithm
            
        # Obtain constraints matrix
        C = ITML._get_constraints_y(y,num_constraints,l,u)
       
        # Remove too similar vectors from constraints
        valid = np.empty([num_constraints],dtype=bool)
        for k in xrange(num_constraints):
            i = C[k,0].astype(int)
            j = C[k,1].astype(int)
            xij = X[i,:]-X[j,:]
            valid[k] = (np.max(np.abs(xij)) > 1e-10)
        C = C[valid,:]
        
        # Pre-loopind declarations
        i = 1
        num_iters = 0
        c = C.shape[0]
        lmbda = np.zeros([c]) # lambdas
        bhat = C[:,3]         # bounds (xi_c(i.j))
        lmbdaold = np.zeros([c])
        conv = np.inf
        
        # Main loop
        while True:
            i1 = C[i,0].astype(int)
            i2 = C[i,1].astype(int)
            v = (X[i1,:]-X[i2,:]).reshape(-1,1)
            wtw = v.T.dot(M).dot(v)  # p
            
            if np.abs(bhat[i]) < 1e-10:
                raise ValueError('Some bounds are too low.')
                
            if gamma == np.inf or gamma == 'inf':
                gamma_proj = 1
            else:
                gamma_proj = gamma/(gamma+1)
                
            # Bregman projection computations (?)
            if C[i,2] == 1: # TODO necesarios estos if?
                #print(1, wtw, bhat[i])
                alpha = np.min([lmbda[i],gamma_proj * (1.0/wtw - 1.0/bhat[i])])
                lmbda[i] = lmbda[i] - alpha
                beta = alpha/(1.0 - alpha*wtw)
                bhat[i] = 1.0/( (1.0/bhat[i]) + (alpha/gamma) )
                
            elif C[i,2] == -1:
                #print(-1, wtw, bhat[i])
                alpha = np.min([lmbda[i],gamma_proj * (1.0/bhat[i] - 1.0/wtw)])
                lmbda[i] = lmbda[i] - alpha
                beta = - alpha/(1.0 + alpha*wtw)
                bhat[i] = 1.0/( (1.0/bhat[i]) - (alpha/gamma) )
            #print("ALPHA:  ",alpha)
            #print("BETA:   ",beta)
            #print("LAMBDA: ",lmbda[i])
            #print("BHAT:   ",bhat[i])
            #print("P:      ",wtw)
            #import time
            #time.sleep(1)
            
            # Matrix update
            M += (beta*M.dot(v).dot(v.T).dot(M))
           
            # Stop conditions
            if i == 0:
                normsum = norm(lmbda)+norm(lmbdaold)
                if normsum == 0.0:
                    break
                else:
                    conv = norm(lmbdaold - lmbda,1) / normsum
                    if conv < tol or num_iters > max_its:
                        break
                    
            i = (i + 1) % c
            num_iters += 1
            
        self.M_ = M
        return self
            
        
    def _get_constraints_y(y,num_constraints,l,u):
        n = len(y)
        C = np.zeros([num_constraints,4])
        for k in xrange(num_constraints):
            i = np.random.randint(0,n)
            j = np.random.randint(0,n)
            if y[i] == y[j]:
                C[k,:] = [i,j,1,l]
            else:
                C[k,:] = [i,j,-1,u]
                
        return C
    
    def _compute_distance_extremes(X,a,b,M):
        #    X: data matrix
        #    a: lower bound percentile (between 1 and 100)
        #    b: upper bound percentile (between 1 and 100)
        #    M: distance matrix
        
        if a < 0 or a > 100:
            raise ValueError('a must be between 0 and 100')
        if b < 0 or b > 100:
            raise ValueError('b must be between 0 and 100')
            
        n, d = X.shape
        
        num_trials = min(100, n*(n-1)/2)
        
        dists = np.empty([num_trials])
        for i in xrange(num_trials):
            j1 = np.random.randint(0,n)
            j2 = np.random.randint(0,n)
            xij = (X[j1,:]-X[j2,:]).reshape(-1,1)
            dists[i] = xij.T.dot(M).dot(xij)
            
        hist, edges = np.histogram(dists,100)
        l = edges[np.floor(a).astype(int)]
        u = edges[np.floor(b).astype(int)]
        #print(edges[0],l,u,edges[100])
    
        return l, u
