#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learning with Side Information (LSI)

A DML that optimizes the distance between pairs of similar data
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y, check_array

from math import sqrt

from numpy.linalg import norm

from .dml_utils import unroll, matpack, calc_outers, calc_outers_i, SDProject
from .dml_algorithm import DML_Algorithm

class LSI(DML_Algorithm):

    def __init__(self,initial_metric = None, learning_rate = 'adaptive', eta0 = 0.1, max_iter = 100, max_proj_iter = 100, itproj_err = 0.01, err = 0.01, supervised = False):
        self.M0_ = initial_metric
        self.eta0_ = eta0
        self.learning_ = learning_rate
        self.max_it_ = max_iter
        self.max_projit_ = max_proj_iter
        self.itproj_err_ = itproj_err
        self.err_ = err
        self.supv_ = supervised
        
        # Metadata
        self.iterative_projections_conv_exp_ = None
        self.initial_objective_ = None
        self.initial_constraint_ = None
        self.final_objective_ = None
        self.final_constraint_ = None
        self.projection_iterations_avg_ = None
        self.num_its_ = None
        
    def metric(self):
        return self.M_
    
    def metadata(self):
        return {'initial_objective':self.initial_objective_,
                'initial_constraint':self.initial_constraint_,
                'final_objective':self.final_objective_,
                'final_constraint':self.final_constraint_,
                'iterative_projections_conv_exp':self.iterative_projections_conv_exp_,
                'projection_iterations_avg': self.projection_iterations_avg_,
                'num_its':self.num_its_}
        
    def fit(self,X,side):
        """
            X: data
            side: side information. A list of sets of pairs of indices. Options:
                - side = y, the label set (only if supervised = True)
                - side = [S,D], where S is the set of indices of similar data and D is the set of indices of dissimilar data.
                - side = [S], where S is the set of indices of similar data. The set D will be the complement of S.
                Sets S and D are represented as a boolean matrix (S[i,j]==True iff (i,j) in S)
        """
        
        # Obtain similarity sets
        if self.supv_:
            X, side = check_X_y(X,side)
            self.y_ = side
            side = LSI.label_to_similarity_set(side)
        else:
            X = check_array(X)
        
        self.X_ = X
        self.side_ = side
            
        if len(side) == 1:
            S = side[0]
            D = self._compute_complement(S)
        else:
            S = side[0]
            D = side[1]
            
        N, d = X.shape
        
        # Init parameters
        M = self.M0_
        if M is None or M == 'euclidean':
            M = np.zeros([d,d])
            np.fill_diagonal(M,1.0)
        elif M == 'scale':
            M = np.zeros([d,d])
            np.fill_diagonal(M, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance
        
        outers = calc_outers(X)
        
        W = np.zeros([d,d])
        for i in xrange(N):
            outers_i = calc_outers_i(X,outers,i)
            for j in xrange(i+1,N):
                if S[i,j]:
                    #d_ij = X[i,:]-X[j,:]
                    W += 2*(outers_i[j])
                
        w = unroll(W)
        t = w.T.dot(unroll(M))/100
        
        nw = norm(w)
        w1 = w/nw
        t1 = t/nw
        
        #Metadata
        self.initial_objective_ = LSI.fD(X,D,M,N,d)
        self.initial_constraint_ = LSI.fS(X,S,M,N,d)
        self.projection_iterations_avg_ = 0
        self.iterative_projections_conv_exp_ = 0
        
        num_its = 0
        max_iter = self.max_it_
        max_projit = self.max_projit_
        err = self.err_
        itproj_err = self.itproj_err_
        eta = self.eta0_
        
        grad2 = LSI.fS1(X, S, M, N, d, outers)              # Similarity gradient (CONSTANT)
        grad1 = LSI.fD1(X, D, M, N, d, outers)              # Dissimilarity gradient
        
        G = LSI.grad_projection(grad1, grad2, d)    # Gradient of fD1 orthogonal to fS1
        
        M_last = M
        done = False
        
        while not done:
            #M_update_cycle = num_its
            projection_iters = 0
            satisfy = False
            
            #print("g(M) = ",LSI.fD(X,D,M,N,d))
            #print("f(M) = ",LSI.fS(X,D,M,N,d))
            #print("t = ",t)
            
            while projection_iters < max_projit and not satisfy:
                M1 = M  ## TODO copy (?)
                
                # First constraint
                x1 = unroll(M1)
                if w.T.dot(x1) <= t:
                    M = M1
                else:
                    x = x1 + (t1-w1.T.dot(x1))*w1
                    M = matpack(x,d,d)
                    
                #fDC1 = w.T.dot(x) After this projection, w.T.dot(X)=t
                
                #M2 = M ## TODO copy (?)
                
                # Second constraint
                M = (M + M.T)/2.0
                M = SDProject(M).astype(float)
                
                fDC2 = w.T.dot(unroll(M))
                #M3 = M
                
                err2 = (fDC2-t)/t

                projection_iters+=1
                satisfy = (err2 <= itproj_err)
                #print("P: ",fDC2,err2,t)
                #fs=LSI.fS(X,S,M,N,d);print(fDC2,fs,t,fDC2/fs)
                
            self.projection_iterations_avg_ += projection_iters
        
            # Gradient ascent
            obj_previous = LSI.fD(X,D,M_last,N,d)
            obj = LSI.fD(X,D,M,N,d)
            #print(obj,obj_previous)
            #print(M)
            #print(M_last)
            #print("SAT: ",satisfy)
            if (obj > obj_previous or num_its == 0) and satisfy:
                # Projection successful and improves objective function. Increase learning rate.
                # Take gradient step
                eta *= 1.05
                M_last = M.copy()
                #grad2 = fS1(X,S,M,N,d,outers) # CONSTANT
                grad1 = LSI.fD1(X,D,M,N,d,outers)
                G = grad1#LSI.grad_projection(grad1,grad2,d)
                M += eta*G
                
                self.iterative_projections_conv_exp_ += 1
                #print("[OK]")
                
            else:
                # Projection failed or objective function not improved. Shrink learning rate and take last M
                eta *= 0.5
                M = M_last + eta*G
                
            delta = norm(eta*G,'fro')/norm(M_last,'fro')
            num_its += 1
            
            if num_its == max_iter or delta < err:
                done = True
            
            
                
        self.M_ = M/t
        
        #Metadata
        self.final_objective_ = LSI.fD(X,D,self.M_,N,d)
        self.final_constraint_ = LSI.fS(X,S,self.M_,N,d)
        self.num_its_ = num_its
        self.iterative_projections_conv_exp_ /= num_its
        self.projection_iterations_avg_ /= num_its
        return self
    
    
    def label_to_similarity_set(y):
        n = len(y)
        S = np.empty([n,n],dtype=bool)
        D = np.empty([n,n],dtype=bool)
        for i in xrange(n):
            for j in xrange(n):
                if i != j:
                    S[i,j] = (y[i] == y[j])
                    D[i,j] = (y[i] != y[j])
                else:
                    S[i,j] = D[i,j] = False
        return S,D
    
    def fS1(X, S, M, N, d, outers):
        f_sum = np.zeros([d,d],dtype=float)
        
        for i in xrange(N):
            outers_i = calc_outers_i(X,outers,i)
            for j in xrange(N):
                if S[i,j]:
                    f_sum += outers_i[j]
                    
        return f_sum
        
    def fD1(X, D, M, N, d, outers):
        g_sum = np.zeros([d,d],dtype=float)
        #d_sum = 0.0
        for i in xrange(N):
            outers_i = calc_outers_i(X,outers,i)
            for j in xrange(N):
                if D[i,j]:
                    xij = (X[i,:]-X[j,:]).reshape(1,-1)
                    d_ij = sqrt(xij.dot(M).dot(xij.T))
                    #d_sum += d_ij
                    g_sum += outers_i[j]/d_ij
        return 0.5*g_sum
    
    def fD(X, D, M, N, d):
        g_sum = 0
        for i in xrange(N):
            for j in xrange(N):
                if D[i,j]:
                    xij = X[i,:]-X[j,:]
                    d_ij = sqrt(xij.dot(M).dot(xij.T))
                    g_sum += d_ij
        return g_sum   
    
    def fS(X, S, M, N, d):
        f_sum = 0
        for i in xrange(N):
            for j in xrange(N):
                if S[i,j]:
                    xij = X[i,:]-X[j,:]
                    d_ij = xij.dot(M).dot(xij.T)
                    f_sum += d_ij
        return f_sum  
        
    def grad_projection(grad1, grad2, d):
        g1 = unroll(grad1)
        g2 = unroll(grad2)
        g2 = g2/norm(g2,2)
        gtemp = g1 - (g2.T.dot(g1))*g2
        gtemp = gtemp/norm(gtemp,2)
        return matpack(gtemp,d,d)
    
    def _compute_complement(self,S):
        n, m = S.shape # (n = m)
        D = np.empty([n,m],dtype=bool)
        for i in xrange(n):
            for j in xrange(m):
                if i != j:
                    D[i,j] = not S[i,j]
                else:
                    D[i,j] = False
        return D
    