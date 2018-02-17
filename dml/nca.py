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

from .dml_utils import calc_outers, calc_outers_i
from .dml_algorithm import DML_Algorithm


class NCA(DML_Algorithm):


    def __init__(self, num_dims = None, learning_rate = "adaptive", eta0 = 0.001, initial_transform = None, max_iter = 100, prec = 1e-3, 
                tol = 1e-3, descent_method = "SGD", eta_thres = 1e-14, learn_inc = 1.01, learn_dec = 0.5):
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
        self.initial_softmax_ = None
        self.final_softmax_ = None
        
    def metadata(self):
        return {'num_iters':self.num_its_,'initial_softmax':self.initial_softmax_,'final_softmax':self.final_softmax_}

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
        self.X_ = X
        self.y_ = y      

        if self.L_ is None or self.L_ == "euclidean":
            self.L_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.L_,1.0) #Euclidean distance 
        elif self.L_ == "scale":
            self.L_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.L_, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance

        self.initial_softmax_ = self._compute_expected_success(self.L_,X,y)/len(y)
        if self.method_ == "SGD": # Stochastic Gradient Descent
            self._SGD_fit(X,y)
            
        elif self.method_ == "BGD": # Batch Gradient Desent
            self._BGD_fit(X,y)
        
        self.final_softmax_ = self._compute_expected_success(self.L_,X,y)/len(y)
        return self

    def _SGD_fit(self,X,y):
        # Initialize parameters
        outers = calc_outers(X)
        n,d= self.n_, self.d_

        L = self.L_

        
        self.num_its_ = 0

        grad = None

        succ_prev = succ = 0.0
        
        stop = False
        
        
        while not stop:
            X, y, outers = self._shuffle(X,y,outers)            
            
            for i,yi in enumerate(y):
                grad = np.zeros([d,d])
                Lx = L.dot(X.T).T
                
                # Calc p_ij (softmax)
                
                # To avoid exponential underflow we use the identity softmax(x) = softmax(x + c) for all c, and take c = max(dists)
                Lxi = Lx[i]
                dists_i = -np.diag((Lxi-Lx).dot((Lxi-Lx).T))
                dists_i[i] = -np.inf
    
                i_max = np.argmax(dists_i)
                c = dists_i[i_max]
                
                softmax = np.empty([n],dtype=float)
                for j in xrange(n):
                    if j != i:
                        # To avoid precision errors, argmax is assigned directly softmax 1
                        if j==i_max:
                            softmax[j] = 1
                        else:
                            pw = min(0,-((Lx[i]-Lx[j]).dot(Lx[i]-Lx[j]))-c)
                            softmax[j] = np.exp(pw)
                        
                softmax[i] = 0
                softmax/=softmax.sum()

                #Calc p_i
                yi_mask = np.where(y == yi)
                p_i = softmax[yi_mask].sum()
                
                #Gradient computing
                sum_p = sum_m = 0.0
                outers_i = calc_outers_i(X,outers,i)

                #sum_p = (softmax*outers_i.T).T.sum(axis=0)
                #sum_m = -(softmax[yi_mask]*outers_i[yi_mask].T).T.sum(axis=0)
                for k in xrange(n):
                    s = softmax[k]*outers_i[k]
                    sum_p += s
                    if(yi == y[k]):
                        sum_m -= s
                
                grad += p_i*sum_p + sum_m
                grad = 2*L.dot(grad)
                L+= self.eta_*grad
                
            
            succ = self._compute_expected_success(L,X,y)
            
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
        outers = calc_outers(X)
        
        n,d  = self.n_, self.d_

        L = self.L_

        
        self.num_its_ = 0

        grad = None

        succ_prev = succ = 0.0
        
        stop = False
        
        while not stop:
            grad = np.zeros([d,d])
            Lx = L.dot(X.T).T
            
            succ = 0.0 # Expected error can be computed directly in BGD
            
            for i,yi in enumerate(y):

                # Calc p_ij (softmax)
                
                # To avoid exponential underflow we use the identity softmax(x) = softmax(x + c) for all c, and take c = max(dists)
                Lxi = Lx[i]
                dists_i = -np.diag((Lxi-Lx).dot((Lxi-Lx).T))
                dists_i[i] = -np.inf
                i_max = np.argmax(dists_i)
                c = dists_i[i_max]
                
                
                softmax = np.empty([n],dtype=float)
                for j in xrange(n):
                    if j != i:
                        # To avoid precision errors, argmax is assigned directly softmax 1
                        if j==i_max:
                            softmax[j] = 1
                        else:
                            pw = min(0,-((Lx[i]-Lx[j]).dot(Lx[i]-Lx[j]))-c)
                            softmax[j] = np.exp(pw)
                        
                softmax[i] = 0
                softmax/=softmax.sum()
                

                #Calc p_i
                yi_mask = np.where(y == yi)
                p_i = softmax[yi_mask].sum()
                
                #Gradient computing
                sum_p = sum_m = 0.0
                outers_i = calc_outers_i(X,outers,i)

                for k in xrange(n):
                    s = softmax[k]*outers_i[k]
                    sum_p += s
                    if(yi == y[k]):
                        sum_m -= s
                
                grad += p_i*sum_p + sum_m
                succ += p_i
                
            
            succ /= len(y)
            
            update=True
            if self.adaptive_:
                if succ > succ_prev:
                    self.eta_ *= self.l_inc_                    
                else:
                    self.eta_ *= self.l_dec_
                    update=False
                    if self.eta_ < self.etamin_:
                        stop = True
                
                succ_prev = succ
                
            if update:
                grad = 2*L.dot(grad)
                L+= self.eta_*grad
                grad_norm = np.max(np.abs(grad))
                if grad_norm < self.eps_ or self.eta_*grad_norm < self.tol_: # Difference between two iterations is given by eta*grad
                    stop=True

            self.num_its_+=1
            if self.num_its_ == self.max_it_:
                stop=True

        self.L_ = L

      

        return self



    def _shuffle(self,X,y,outers=None):
        rnd = np.random.permutation(len(y))
        X = X[rnd,:]
        y = y[rnd]
        if outers is not None:
            for i in xrange(len(y)):
                outers[:,i]=outers[rnd,i]
            for i in xrange(len(y)):
                outers[i,:]=outers[i,rnd]
            #outers = outers[rnd,:][:,rnd]
        else:
            outers = None

        return X,y, outers

    def _compute_expected_success(self,L,X,y):
        n,d = X.shape
        Lx = L.dot(X.T).T
        success = 0
        for i, yi in enumerate(y):
            softmax = np.empty([n],dtype=float)
            
            Lxi = Lx[i]
            dists_i = -np.diag((Lxi-Lx).dot((Lxi-Lx).T)) ## TODO improve efficiency of dists_i
            dists_i[i] = -np.inf
            i_max = np.argmax(dists_i)          
            c = dists_i[i_max]          ## TODO all distances can reach -inf
            for j in xrange(n):
                if j != i:
                    if j == i_max:
                        softmax[j] = 1
                    else:
                        pw = min(0,-((Lx[i]-Lx[j]).dot(Lx[i]-Lx[j]))-c)
                        softmax[j] = np.exp(pw)
            softmax[i] = 0
            
            softmax/=softmax.sum()
            

            #Calc p_i
            yi_mask = np.where(y == yi)
            p_i = softmax[yi_mask].sum()

            success += p_i

        return success  # Normalized between 0 and 1







