#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Discriminant Metric Learning (LDML)

Created on Mon Mar 12 18:26:53 2018

@author: jlsuarezdiaz
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y

from .dml_utils import calc_outers, calc_outers_i, SDProject
from .dml_algorithm import DML_Algorithm


class LDML(DML_Algorithm):
    """
    Logistic Discriminant Metric Learning (LDML).

    Distance Metric Learning through the likelihood maximization of a logistic based probability distribution.

    Parameters
    ----------

    num_dims : int, default=None.

        Number of dimensions for dimensionality reduction. Not supported yet.

    b : float, default=1e-3

        Logistic function positive threshold.

    learning_rate : string, default='adaptive'

        Type of learning rate update for gradient descent. Possible values are:

        - 'adaptive' : the learning rate will increase if the gradient step is succesful, else it will decrease.

        - 'constant' : the learning rate will be constant during all the gradient steps.

    eta0 : float, default=0.3

        The initial value for learning rate.

    initial_metric : 2D-Array or Matrix (d x d), or string, default=None.

        If array or matrix, it must be a positive semidefinite matrix with the starting metric for gradient descent, where d is the number of features.
        If None, euclidean distance will be used. If a string, the following values are allowed:

        - 'euclidean' : the euclidean distance.

        - 'scale' : a diagonal matrix that normalizes each attribute according to its range will be used.

    max_iter : int, default=10

        Maximum number of iterations of gradient descent.

    prec : float, default=1e-3

        Precision stop criterion (gradient norm).

    tol : float, default=1e-3

        Tolerance stop criterion (difference between two iterations)

    descent_method : string, default='SDP'

        The descent method to use. Allowed values are:

        - 'SDP' : semidefinite programming, consisting of gradient descent with projections onto the PSD cone.

    eta_thres : float, default=1e-14

        A learning rate threshold stop criterion.

    learn_inc : float, default=1.01

        Increase factor for learning rate. Ignored if learning_rate is not 'adaptive'.

    learn_dec : float, default=0.5

        Decrease factor for learning rate. Ignored if learning_rate is not 'adaptive'.


    References
    ----------
        Matthieu Guillaumin, Jakob Verbeek and Cordelia Schmid. “Is that you? Metric learning approaches
        for face identification”. In: Computer Vision, 2009 IEEE 12th international conference on. IEEE.
        2009, pages 498-505.
    """

    def __init__(self, num_dims=None, b=1e-3, learning_rate="adaptive", eta0=0.3, initial_metric=None, max_iter=10, prec=1e-3,
                 tol=1e-3, descent_method="SDP", eta_thres=1e-14, learn_inc=1.01, learn_dec=0.5):
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
        self.b_ = b

        # Metadata initialization
        self.num_its_ = None
        self.initial_error_ = None
        self.final_error_ = None

    def metadata(self):
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            - 'num_iters' : Number of iterations that the descent method took.

            - 'initial_error' : Initial value of the objective function.

            - 'final_error' : Final value of the objective function.
        """
        return {'num_iters': self.num_its_, 'initial_error': self.initial_error_, 'final_error': self.final_error_}

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
            np.fill_diagonal(M,1.0) # Euclidean distance 
        elif M == "scale":
            M = np.zeros([self.nd_,self.d_])
            np.fill_diagonal(M, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) # Scaled eculidean distance

        b = self.b_
        self.num_its_ = 0

        grad = None

        
        
        stop = False
        lklh_prev = lklh = self.initial_error_ = LDML._compute_error(M,X,y,outers,b)
        
        while not stop:
            grad = np.zeros([d,d])
            
            for i, yi in enumerate(y):
                outers_i = calc_outers_i(X,outers,i)
                for j, yj in enumerate(y):
                    outer_ij = outers_i[j]
                    wtoij = np.inner(M.reshape(1,-1),outer_ij.reshape(1,-1))
                    z = b - wtoij
                    pij = 1.0/(1.0 + np.exp(-z))
                    yij = 1.0 if yi == yj else 0.0
                    
                    grad += (yij - pij)*outer_ij
                
                    
                    
            Mprev = M   
            M = M - self.eta_*grad
            M = SDProject(M)    
            
            lklh = LDML._compute_error(M,X,y,outers,b)
            
            if self.adaptive_:
                if lklh > lklh_prev:
                    self.eta_ *= self.l_inc_                    
                else:
                    self.eta_ *= self.l_dec_
                    if self.eta_ < self.etamin_:
                        stop = True
                
                lklh_prev = lklh
            
            grad_norm = np.max(np.abs(grad))
            tol_norm = np.max(np.abs(M-Mprev)) 
            
            if grad_norm < self.eps_ or tol_norm < self.tol_:
                stop=True

            self.num_its_+=1
            if self.num_its_ == self.max_it_:
                stop=True
            
        self.final_error_ = LDML._compute_error(M,X,y,outers,b)
        self.M_ = M
        
        return self
    
    def _compute_error(M,X,y,outers,b):
        f_obj = 0.0
        for i, yi in enumerate(y):
            outers_i = calc_outers_i(X,outers,i)
            for j, yj in enumerate(y):
                yij = 1 if yi == yj else 0
                outer_ij = outers_i[j]
                wtoij = np.inner(M.reshape(1,-1),outer_ij.reshape(1,-1))
                z = b - wtoij
                pij = 1.0/(1.0 + np.exp(-z))
                
                f_obj += yij * np.log(pij) + (1-yij)*np.log(1-pij)
                
        return f_obj