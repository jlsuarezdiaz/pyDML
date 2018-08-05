#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maximally collapsing metric learning (MCML)

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

    """
    Maximally Collapsing Metric Learning (MCML)

    A distance metric learning algorithm that learns minimizing the KL divergence to the maximally collapsing distribution.

    Parameters
    ----------

    num_dims : int, default=None.

        Number of dimensions for dimensionality reduction. Not supported yet.

    learning_rate : string, default='adaptive'

        Type of learning rate update for gradient descent. Possible values are:

        - 'adaptive' : the learning rate will increase if the gradient step is succesful, else it will decrease.

        - 'constant' : the learning rate will be constant during all the gradient steps.

    eta0 : float, default=0.01

        The initial value for learning rate.

    initial_metric : 2D-Array or Matrix (d x d), or string, default=None.

        If array or matrix, it must be a positive semidefinite matrix with the starting metric for gradient descent, where d is the number of features.
        If None, euclidean distance will be used. If a string, the following values are allowed:

        - 'euclidean' : the euclidean distance.

        - 'scale' : a diagonal matrix that normalizes each attribute according to its range will be used.

    max_iter : int, default=20

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
        Amir Globerson and Sam T Roweis. “Metric learning by collapsing classes”. In: Advances in neural
        information processing systems. 2006, pages 451-458.
    """

    def __init__(self, num_dims=None, learning_rate="adaptive", eta0=0.01, initial_metric=None, max_iter=20, prec=0.01,
                 tol=0.01, descent_method="SDP", eta_thres=1e-14, learn_inc=1.01, learn_dec=0.5):
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
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            - 'num_iters' : Number of iterations that the descent method took.

            - 'initial_error' : Initial value of the objective function.

            - 'final_error' : Final value of the objective function.
        """
        return {'num_iters':self.num_its_,'initial_error':self.initial_error_,'final_error':self.final_error_}

    def metric(self):
        """
        Obtains the learned metric.

        Returns
        -------
        M : (dxd) positive semidefinite matrix, where d is the number of features.
        """
        return self.M_

    def fit(self,X,y):
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
            np.fill_diagonal(M,1.0) #Euclidean distance 
        elif M == "scale":
            M = np.zeros([self.nd_,self.d_])
            np.fill_diagonal(M, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance

        
        self.num_its_ = 0

        grad = None

        
        
        stop = False
        invalid=False
        err_prev = err = self.initial_error_ = MCML._compute_error(M,X,y)
        
        while not stop:
            grad = np.zeros([d,d])
            
            #for i, yi in enumerate(y):
            #    outers_i = calc_outers_i(X,outers,i)
            #    softmax = np.empty([n],dtype=float)
            #    softmax_sum = 0.0
            #    for k in xrange(n):
            #        if i != k:
            #            xik = (X[i]-X[k]).reshape(1,-1)
            #            pik = np.exp(-xik.dot(M).dot(xik.T))
                        
            #            softmax_sum += pik
            #            softmax[k] = pik
                    #else:
                    #    softmax[k] = 0.0
                 
            #    softmax /= softmax_sum[0,0]
                        
            #    for j, yj in enumerate(y):
            #        p0 = 1.0 if yi == yj else 0.0
            #        grad += (p0 - softmax[j])*outers_i[j]
            
            for i, yi in enumerate(y):
                outers_i = calc_outers_i(X,outers,i)
                #softmax = np.empty([n],dtype=float)
                softmax_sum = 0.0
                softout_sum = np.zeros([d,d])
                for k in xrange(n):
                    if i != k:
                        xik = (X[i]-X[k]).reshape(1,-1)
                        pik = np.exp(-xik.dot(M).dot(xik.T))
                        softmax_sum += pik
                        softout_sum += pik*outers_i[k]
                        
                if softmax_sum > 1e-16:
                    const_grad = softout_sum / softmax_sum
                    invalid=False
                else:
                    invalid=True
                
                const_count = 0
                for j, yj in enumerate(y):
                    if yi == yj:
                        grad += outers_i[j]
                        const_count+=1
                        
                if not invalid:        
                    grad -= const_count*const_grad
            
            if stop:
                grad = np.zeros([d,d])
                
            Mprev = M   
            #print("M");print(M);print("G");print(grad);input()
            
            M = M - self.eta_*grad
            M = SDProject(M)    
            
            if not invalid:
                err = MCML._compute_error(M,X,y)
                
            else:
                err = err_prev
            
            #print(err)
            #print("ETA: ", self.eta_)
            if self.adaptive_:
                if err < err_prev and not invalid:
                    self.eta_ *= self.l_inc_                    
                else:
                    self.eta_ *= self.l_dec_
                    if self.eta_ < self.etamin_:
                        stop = True
                
                err_prev = err
            
            grad_norm = np.max(np.abs(grad))
            tol_norm = np.max(np.abs(M-Mprev)) 
            
            if (not invalid and (grad_norm < self.eps_ or tol_norm < self.tol_)) or (invalid and not self.adaptive_):
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
            nexp = 0
            for j, yj in enumerate(y):
                if j != i:
                    xij = (X[i]-X[j]).reshape(1,-1)
                    dij = xij.dot(M).dot(xij.T)
                    if yi == yj:
                        sdij += dij
                        nexp += 1
                    sexp += np.exp(-dij)
            slog += nexp*np.log(sexp)
        return sdij + slog
                    

        
