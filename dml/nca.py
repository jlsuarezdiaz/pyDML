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
import warnings
import signal
import sys

from .dml_utils import calc_outers, calc_outers_i
from .dml_algorithm import DML_Algorithm


class NCA(DML_Algorithm):

    
    #stop_signal_ = False
    """
    @staticmethod
    def _signal_handler(signal,frame):
        print("Algorithm will stop at next iteration.")
        NCA.stop_signal_ = True
        NCA._stop_handler()
    """

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

        #self.expected_success_ = 0.0
        

        if self.L_ is None or self.L_ == "euclidean":
            self.L_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.L_,1.0) #Euclidean distance 
        elif self.L_ == "scale":
            self.L_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.L_, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance

        self.initial_softmax_ = self._compute_expected_success(self.L_,X,y)

        if self.method_ == "SGD": # Stochastic Gradient Descent
            self._SGD_fit(X,y)
            
        elif self.method_ == "BGD": # Batch Gradient Desent
            self._BGD_fit(X,y)
        
        self.final_softmax_ = self._compute_expected_success(self.L_,X,y)
        return self

    def _SGD_fit(self,X,y):
        # Initialize parameters

        outers = calc_outers(X)

        n,d,nd = self.n_, self.d_, self.nd_

        L =  self.L_

        #SGD
        self.num_its_ = 0
        succ_prev = succ = 0.0
        grad = None

        while not self._stop_criterion(L,X,y,grad): # Stop criterion updates L
            X,y = self._shuffle(X,y)

            for i,yi in enumerate(y):

                # Calc p_ij (softmax)
                Lx = L.dot(X.T).T

                # To avoid exponential underflow we use the identity softmax(x) = softmax(x + c) for all c, and take c = max(dists)
                Lxi = Lx[i]
                dists_i = -np.diag((Lxi-Lx).dot((Lxi-Lx).T))
                dists_i[i] = -np.inf
                i_max = np.argmax(dists_i)
                c = dists_i[i_max]
                
                softmax = np.empty([n],dtype=float)
                for j in xrange(n):
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

                grad = 2*L.dot(p_i*sum_p - sum_m)

                L+= self.eta_*grad
                
                if self.adaptive_:
                    succ = self._compute_expected_success(L,X,y)
                    #print("SCC: ",succ)
                    if succ > succ_prev:
                        self.eta_ *= self.l_inc_
                    else:
                        self.eta_ *= self.l_dec_
                    #print("ETA: ",self.eta_)
                    succ_prev = succ
                #print(self._compute_expected_success(L,X,y))
            self.num_its_+=1
            
            
        self.L_ = L

        return self

    def _BGD_fit(self,X,y):

        # Initialize parameters
        outers = calc_outers(X)
        
        n,d,nd = self.n_, self.d_, self.nd_

        L = self.L_

        
        self.num_its_ = 0

        grad = None

        succ_prev = succ = 0.0
        
        while not self._stop_criterion(L,X,y,grad): #Stop criterion updates L
            grad = np.zeros([d,d])
            Lx = L.dot(X.T).T
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

                grad += p_i*sum_p - sum_m

                #print "Iteration ", self.num_its_, " Item ", i

            grad = 2*L.dot(grad)
            L+= self.eta_*grad
            
            if self.adaptive_:
                succ = self._compute_expected_success(L,X,y)
                #print(succ)
                if succ > succ_prev:
                    self.eta_ *= self.l_inc_
                else:
                    self.eta_ *= self.l_dec_
                #print(self.eta_)
                succ_prev = succ

            self.num_its_+=1
            #print(self._compute_expected_success(L,X,y))

        self.L_ = L

       

        return self



    def _shuffle(self,X,y):
        rnd = np.random.permutation(len(y))
        X = X[rnd,:]
        y = y[rnd]

        return X,y

    def _stop_criterion(self,L,X,y, grad = None):
        tol = self.tol_
        
        if tol is not None and self.num_its_ > 0:
            tol_norm = np.max(np.abs(self.L_ - L))
            tol_crit = (tol_norm < self.tol_)
            #print("TOL: ",tol_norm)
        else: 
            tol_crit = False
        
        self.L_ = L.copy()
 
        eps = self.eps_
        
        if eps is not None and grad is not None:
            grad_norm = np.max(np.abs(grad))
            eps_crit = grad_norm < eps
            #print("GRD: ", grad_norm)
        else:
            eps_crit = False

        
        it_crit = self.num_its_ >= self.max_it_
        
        eta_crit = self.eta_ < self.etamin_

        return tol_crit or it_crit or eps_crit or eta_crit

    def _compute_expected_success(self,L,X,y):
        n,d = X.shape
        Lx = L.dot(X.T).T
        success = 0
        #belongs = y[:,None] == y[None]
        for i, yi in enumerate(y):
            softmax = np.empty([n],dtype=float)
            
            Lxi = Lx[i]
            dists_i = -np.diag((Lxi-Lx).dot((Lxi-Lx).T)) ## TODO improve efficiency of dists_i
            dists_i[i] = -np.inf
            i_max = np.argmax(dists_i)          
            c = dists_i[i_max]          ## TODO all distances can reach -inf
            for j in xrange(n):
                #print Lx[i], Lx[j]
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

        return success/y.size  # Normalized between 0 and 1
"""
    @staticmethod
    def _start_handler():
        NCA._original_handler_  = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT,NCA._signal_handler)

    @staticmethod
    def _stop_handler():
        signal.signal(signal.SIGINT,NCA._original_handler_)

"""


#class NCA(DML_Algorithm):
#    def __init__(self,num_dims=None, max_iter=100, learning_rate=0.01, initial_transform=None, eps = 1e-16):
#        self.num_dims = num_dims
#        self.max_iter = max_iter
#        self.learning_rate = learning_rate
#        self.initial_transform = initial_transform
#        self.eps_ = eps
#
#    def transformer(self):
#        return self.L_
#
#    def fit(self,X,y):
#        n,d = X.shape
#
#        num_dims = self.num_dims
#        if num_dims is None:
#            num_dims = d
#
#        L = self.initial_transform
#        if L is None:
#            np.zeros([num_dims,d])
#            np.fill_diagonal(A, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16)))
#
#        grad_scale = 2*self.learning_rate
#
#        for it in xrange(self.max_iter):
#            x=0 #####
#
#
#
#
#
#    def _fit(self,X,y):
#        X, y = check_X_y(X,y) # Consistency check
#        n, d = X.shape
#
#        num_dims = self.num_dims # If no dimensions are specified dataset dimensions are used.
#        if num_dims is None:
#            num_dims = d
#
#        A = self.initial_transform # If starting transformation is not specified, diagonal ones will be used.
#        if A is None:
#            A = np.zeros((num_dims,d))
#            #np.fill_diagonal(A,1./(np.maximum(X.max(axis=0)-X.min(axis=0))))
#            np.fill_diagonal(A, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16)))
#
#
#        grad_scale = 2*self.learning_rate # Gradient constant 2 is added to the learning rate.
#
#        #cinitial=couter=cAx=csoftmax=cgr1=cgr2=cgr=0
#        # TOO MUCH MEMORY!!!
#        dX = X[:,None] - X[None] # Difference vectors (nxnxd). Each pair of indices represent the vector dX[i,j] = X[i,] - X[j,]
#        #outers = np.einsum('...i,...j->...ij',dX,dX) # Outer product of difference vectors (nxnxdxd). Each pair of indices represent the matrix dX[i,j]*dX[i,j].T
#        belongs = y[:,None] == y[None] # Belonging matrix (nxn). belongs[i,j] == true iff y[i] == y[j] 
#
#        #cinitial+=(time.time()-start); start=time.time()  # 0,6 %
#        self.n_iter = 0
#
#        for it in xrange(self.max_iter):
#            for i, label in enumerate(y):
#                friends = belongs[i] # Same class indices
#                outers_i = np.einsum('...i,...j->...ij',dX[i],dX[i])
#
#                #couter+=(time.time()-start); start=time.time() # 37,11 %
#    
#                # Gradient computation (stochastic)
#                Ax = A.dot(X.T).T
#    
#                #cAx+=(time.time()-start); start=time.time() # 5,47 %
#
#                softmax = np.exp(-((Ax[i]-Ax)**2).sum(axis=1)) # Cálculo de p_{ij} (suma por columnas de la matriz de columnas A[,j] = (Ax[i]-Ax[j])^2)
#                #print softmax
#                softmax[i]=0
#                #print softmax.sum()
#                softmax /= softmax.sum()
#                #print softmax
#
#                # csoftmax+=(time.time()-start); start=time.time() # 1 %
#    
#                #p_outer = softmax[:, None, None] * outers[i] # pik*xik*xik.T (i fixed, nxdxd)
#                p_outer = softmax[:,None,None] * outers_i
#                # cgr1+=(time.time()-start); start=time.time() # 37,9 %
#                grad = softmax[friends].sum() * p_outer.sum(axis=0) - p_outer[friends].sum(axis=0)
#                norm = np.amax(abs(grad))
#                
#                # cgr2+=(time.time()-start); start=time.time() # 17,48 %
#    
#                A += grad_scale* A.dot(grad)
#                A /= np.amax(abs(A))
#                # cgr+=(time.time()-start); start=time.time() # 0,23 %
#
#                print "Norm: ", norm
#                print "Succ:" , self.compute_expected_success(A,X,y)
#
#
#                if norm < self.eps_:
#                    break
#
#
#            self.n_iter += i
#            if norm < self.eps_:
#                break
#
#
#        #print "Initial: ", cinitial
#        #print "Outer: ", couter
#        #print "Ax: ", cAx
#        #print "Softmax: ", csoftmax
#        #print "gr1: ", cgr1
#        #print "gr2: ", cgr2
#        #print "gr: ", cgr
#        #print "Total: ", time.time() -st
#
#
#        self.X_ = X
#        self.L_ = A
#        
#
#        # Improve this
#        self.expected_success = self.compute_expected_success(A,X,y)
#
#        return self
#
#    def compute_expected_success(self,A,X,y):
#        Ax = A.dot(X.T).T
#        success = 0
#        belongs = y[:,None] == y[None]
#        for i, label in enumerate(y):
#            softmax = np.exp(-((Ax[i]-Ax)**2).sum(axis=1)) 
#            softmax[i] = 0
#            #print softmax
#            #print softmax.sum()
#            softmax/= softmax.sum()
#
#            success += softmax[belongs[i]].sum()
#
#        return success/y.size  # Normalized between 0 and 1






