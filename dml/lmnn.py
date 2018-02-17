#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Large Margin Nearest Neighbors

A DML that obtains a metric with target neighbors as near as possible and impostors as far as possible
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y

from .dml_utils import SDProject, calc_outers, calc_outers_i, metric_sq_distance
from .dml_algorithm import DML_Algorithm


class LMNN(DML_Algorithm):

    def __init__(self, num_dims = None, learning_rate = "adaptive", eta0 = 0.001, initial_metric = None, max_iter = 100, prec = 1e-3,
                tol = 1e-6, k = 3, mu = 0.5, soft_comp_interval= 1, learn_inc = 1.01, learn_dec = 0.5, eta_thres=1e-14):
        self.num_dims_ = num_dims
        self.M0_ = initial_metric
        self.max_it_ = max_iter
        self.eta0_ = eta0
        self.eta_ = eta0
        self.learning_ = learning_rate
        self.adaptive_ = (self.learning_ == 'adaptive')
        self.eps_ = prec
        self.tol_ = tol
        self.mu_ = mu
        self.k_ = k
        self.soft_comp_interval_ = soft_comp_interval
        self.l_inc_ = learn_inc
        self.l_dec_ = learn_dec
        self.etamin_ = eta_thres
        
        # Metadata
        self.num_its_ = None
        self.initial_error_ = None
        self.final_error_ = None
        
    def metadata(self):
        return {'num_iters':self.num_its_,
                'initial_error':self.initial_error_,
                'final_error':self.final_error_}

    def metric(self):
        return self.M_

    def fit(self,X,y):
        self._set_initial_parameters(X,y)
        n,d = X.shape

        self.num_its_ = 0
        self.eta_ = self.eta0_

        N_up = set()       #Active set
        N_down = set()     #"Exact" set
        N_old = set()      #Exact set of last iteration

        outers = calc_outers(X)
        target_neighbors = self._target_neighbors(X,y)

        M = self.M_
        G = self._compute_not_imposter_gradient(X,target_neighbors,outers)
        Mprev = None

        
        impostors = self._impostors(M,X,y,target_neighbors)
        self.initial_error_ = self._compute_error(self.mu_,M,X,y,target_neighbors,impostors)
        err_prev =  err = self.initial_error_
        
        stop = False
        
        while not stop:    
            if self.num_its_ % self.soft_comp_interval_ == 0:
                N_down_new = self._compute_N_triplets(n,target_neighbors,impostors)
                N_up_new = N_up | N_down # Union
            else:
                N_down_new = N_down & N_up # intersection
                
            Mprev = M
            
            # Gradient update
            grad_imp = self._compute_imposter_gradient(X,outers,N_down_new,N_old)
            Gnew = G + grad_imp
            
            # Gradient step
            Mnew = M - self.eta_*Gnew
            Mnew = SDProject(Mnew)
            imp_new = None
            update=True  
            
            # Adaptive update
            if self.adaptive_ and (self.num_its_+1) % self.soft_comp_interval_ == 0:
                imp_new = self._impostors(Mnew,X,y,target_neighbors)
                err = self._compute_error(self.mu_,Mnew,X,y,target_neighbors,imp_new)
                if err < err_prev:
                    self.eta_ *= self.l_inc_
                else:
                    self.eta_ *= self.l_dec_
                    update=False
                    if self.eta_ < self.etamin_:
                        stop=True
                        
            # Update and stop conditions
            if update:
                self.M_ = M = Mnew
                G = Gnew
                N_down = N_down_new
                N_up = N_up_new
                N_old = N_down 
                if imp_new is not None:
                    impostors=imp_new
                if imp_new is None and (self.num_its_+1) % self.soft_comp_interval_ == 0:
                    impostors = self._impostors(M,X,y,target_neighbors)
                
                tol_norm = np.max(np.abs(M-Mprev)) 
                grad_norm = np.max(np.abs(G))
                if tol_norm < self.tol_ or grad_norm < self.eps_:
                    stop=True
                err_prev = err
                
            self.num_its_+=1
            if self.num_its_ == self.max_it_:
                stop=True

        self.M_ = M
        self.final_error_ = self._compute_error(self.mu_,M,X,y,target_neighbors,self._impostors(M,X,y,target_neighbors))
        return self



    def _set_initial_parameters(self,X,y):
        self.n_, self.d_ = X.shape
        if self.num_dims_ is not None:
            self.nd_ = min(self.d_,self.num_dims_)
        else:
            self.nd_ = self.d_

        self.M_ = self.M0_

        if self.M_ is None or self.M_ == "euclidean":
            self.M_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.M_,1.0) #Euclidean distance 
        elif self.M_ == "scale":
            self.M_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.M_, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance

        
        X, y = check_X_y(X,y)
        self.X_ = X
        self.y_ = y


    def _target_neighbors(self,X,y):
        """
        Returns a matrix nxk, where n is the amount of data, and each row contains
        the target neighbors indexes for each data index.
        """
        n,d = X.shape

        unique_labels = np.unique(y)
        target_neighbors = np.empty([n,self.k_],dtype=int)

        for label in unique_labels:
            inds, = np.where(y == label)
            dists = pairwise_distances(X[inds])

            np.fill_diagonal(dists, np.inf)
            target_inds = np.argsort(dists)[..., :self.k_]
            target_neighbors[inds] = inds[target_inds]

        return target_neighbors


    def _impostors(self,M,X,y,target_neighbors):
        impostors = []
        for i, yi in enumerate(y):
            out_inds, = np.where(y != yi)
            target_inds = target_neighbors[i,:]
            inds = np.concatenate([target_inds,out_inds])
            target_len = len(target_inds)
            impostors_i = []
            
            dists = self._pairwise_metric_distances(X[i,:],X[inds,:],M)
            target_limit = np.amax(dists[0,0:target_len])
            margin = (1+target_limit)
            
            for l in xrange(len(out_inds)):
                ldist = dists[0,target_len+l]
                if ldist < margin:
                    impostors_i.append(out_inds[l])

            impostors.append(impostors_i)

        return impostors

    def _pairwise_metric_distances(self,xi,X,M):
        return pairwise_distances(xi.reshape(1,-1),X,metric=LMNN._distance,M=M)
        #n,d = X.shape
        #xi = xi.reshape(1,-1)
        #dists = np.empty([n],dtype=float)
        #for j in xrange(n):
        #    xj = X[j,:].reshape(1,-1)
        #    xij = xi - xj
        #    dists[j] = xij.dot(M).dot(xij.T).astype('float')
        #return dists
        
    def _distance(x,y,M):
        xy=(x-y).reshape(1,-1)
        return xy.dot(M).dot(xy.T)

    def _compute_error(self,mu,M,X,y,target_neighbors,impostors):
        n,d = X.shape
        non_imposter_err = 0.0
        imposter_err = 0.0
        for i in xrange(n):
            for j in target_neighbors[i,:]:
                non_imposter_err += metric_sq_distance(M,X[i,:],X[j,:])
                for l in impostors[i]:
                    i_err = (1 + metric_sq_distance(M,X[i,:],X[j,:]) - metric_sq_distance(M,X[i,:],X[l,:]))
                    if(i_err > 0):
                        imposter_err += i_err 
                        
        return (1-mu)*non_imposter_err + mu*imposter_err


    def _compute_N_triplets(self,n,target_neighbors,impostors):
        triplets = set()
        for i in xrange(n):
            for j in target_neighbors[i,:]:
                for l in impostors[i]:
                    #if(1 + metric_sq_distance(self.M_,self.X_[i,:],self.X_[j,:]) - metric_sq_distance(self.M_,self.X_[i,:],self.X_[l,:]) > 0):
                    triplets.add((i,j,l))
                        

        return triplets

    def _compute_not_imposter_gradient(self,X,target_neighbors,outers):
        n,d = X.shape
        grad = 0
        for i in xrange(n):
            outers_i = calc_outers_i(X,outers,i)
            for j in target_neighbors[i,:]:
                grad += outers_i[j]
                
        return (1-self.mu_)*grad



    def _compute_imposter_gradient(self,X,outers,N_down,N_old):
        grad = 0
        new_old = N_down - N_old
        old_new = N_old - N_down

        for (i,j,l) in new_old:
            outers_i = calc_outers_i(X,outers,i)
            grad += (outers_i[j]-outers_i[l])

        for (i,j,l) in old_new:
            outers_i = calc_outers_i(X,outers,i)
            grad -= (outers_i[j]-outers_i[l])

        return self.mu_ * grad
               


