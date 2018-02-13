#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Average Neighbor Margin Maximization (ANMM)

A DML that obtains a metric that maximizes the distance between the nearest friend and the nearest enemy for each example.
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_X_y, check_array

from numpy.linalg import eig

from .dml_algorithm import DML_Algorithm
from .dml_utils import pairwise_sq_distances_from_dot

class ANMM(DML_Algorithm):

    def __init__(self, num_dims = None, n_friends = 3, n_enemies = 1):
        self.num_dims_ = num_dims
        self.n_fr_ = n_friends
        self.n_en_ = n_enemies

    def fit(self,X,y):
        X, y = check_X_y(X,y)
        self.X_, self.y_ = X, y
        
        self.distance_matrix_ = pairwise_distances(X = X, n_jobs = -1)
        self.n_,self.d_ = X.shape

        het_neighs = self._compute_heterogeneous_neighborhood(X,y)
        hom_neighs = self._compute_homogeneous_neighborhood(X,y)

        if self.num_dims_ is None:
            num_dims = self.d_
        else:
            num_dims = self.num_dims_

        S,C = self._compute_matrices(X,het_neighs,hom_neighs)

        #Id =  np.zeros([self.d_,self.d_])
        #np.fill_diagonal(Id,1.0)
        #print(self._compute_average_margin(Id,S,C))
        
        # Eigenvalues and eigenvectors of S - C
        self.eig_vals_, self.eig_vecs_ = eig(S-C)
        vecs_orig = self.eig_vecs_.copy()
        
        # Reordering
        self.eig_pairs_ = [(np.abs(self.eig_vals_[i]), vecs_orig[:,i]) for i in xrange(self.eig_vals_.size)]
        self.eig_pairs_ = sorted(self.eig_pairs_, key = lambda k: k[0], reverse=True)

        for i, p in enumerate(self.eig_pairs_):
            self.eig_vals_[i] = p[0]
            self.eig_vecs_[i,:] = p[1]

        self.L_ = self.eig_vecs_[:num_dims,:]
        #print(self._compute_average_margin(self.L_,S,C))
        return self
    
    def transformer(self):
        return self.L_
    
    
    def _compute_average_margin(self,L,S,C):
        return np.trace(L.dot(S-C).dot(L.T))


    def _compute_heterogeneous_neighborhood(self,X,y):
        het_neighs = np.empty([self.n_,self.n_en_],dtype=int)
        for i, x in enumerate(X):
            mask = np.flatnonzero(y != y[i])
            enemy_dists = [(m,self.distance_matrix_[i,m]) for m in mask]
            enemy_dists = sorted(enemy_dists, key = lambda k: k[1])

            for j, p in enumerate(enemy_dists[:self.n_en_]):
                het_neighs[i,j] = p[0]
            #het_neighs[i,:] = enemy_dists[0,:self.n_en_]

        return het_neighs


    def _compute_homogeneous_neighborhood(self,X,y):
        hom_neighs = np.empty([self.n_,self.n_fr_],dtype=int)
        for i, x in enumerate(X):
            cur_class = y[i] ### Para no contar el propio indice de forma eficiente (mejorar)
            #y[i]+=1 ###
            #mask = np.flatnonzero(y == cur_class)
            #y[i]-=1 ###
            mask=np.concatenate([np.flatnonzero(y[:i]==cur_class),(i+1)+np.flatnonzero(y[i+1:]==cur_class)])

            friend_dists = [(m,self.distance_matrix_[i,m]) for m in mask]
            friend_dists = sorted(friend_dists, key = lambda k: k[1])

            for j, p in enumerate(friend_dists[:self.n_fr_]):
                hom_neighs[i,j] = p[0]

        return hom_neighs

    def _compute_matrices(self,X,het_neighs,hom_neighs):
        S = np.zeros([self.d_,self.d_])
        C = np.zeros([self.d_,self.d_])

        for i,x in enumerate(X):
            for j in xrange(self.n_en_):
                S += np.outer(x-X[het_neighs[i,j],:],x-X[het_neighs[i,j],:])
            for j in xrange(self.n_fr_):
                C += np.outer(x-X[hom_neighs[i,j],:],x-X[hom_neighs[i,j],:])
        
        S /= self.n_en_
        C /= self.n_fr_

        return S,C



class KANMM(DML_Algorithm):
    
    def __init__(self, num_dims = None, n_friends = 3, n_enemies = 1, kernel = "linear", 
                 gamma=None, degree=3, coef0=1, kernel_params=None):
        self.num_dims_ = num_dims
        self.n_fr_ = n_friends
        self.n_en_ = n_enemies
        
        self.kernel_ = kernel
        self.gamma_ = gamma
        self.degree_ = degree
        self.coef0_ = coef0
        self.kernel_params_ = kernel_params
        
    def _get_kernel(self,X,Y=None):
        if callable(self.kernel_):
            params = self.kernel_params or {}
        else:
            params = {'gamma':self.gamma_,
                      'degree':self.degree_,
                      'coef0':self.coef0_}
            
        return pairwise_kernels(X,Y,metric=self.kernel_,filter_params=True,**params)
    
    @property
    def _pairwise(self):
        return self.kernel_ == "precomputed"

    def fit(self,X,y):
        X, y = check_X_y(X,y)
        self.X_, self.y_ = X, y
        
        K = self._get_kernel(X)
    
        self.distance_matrix_ = pairwise_sq_distances_from_dot(K)
        self.n_,self.d_ = X.shape

        het_neighs = self._compute_heterogeneous_neighborhood(X,y)
        hom_neighs = self._compute_homogeneous_neighborhood(X,y)
        
        if self.num_dims_ is None:
            num_dims = self.d_
        else:
            num_dims = self.num_dims_

        S,C = self._compute_matrices(K,het_neighs,hom_neighs)
        print(S,C)
        #Id =  np.zeros([self.n_,self.n_])
        #np.fill_diagonal(Id,1.0)
        #print(self._compute_average_margin(Id,S,C))
        
        # Eigenvalues and eigenvectors of S - C
        self.eig_vals_, self.eig_vecs_ = eig(S-C)
        vecs_orig = self.eig_vecs_.copy()
        
        # Reordering
        self.eig_pairs_ = [(np.abs(self.eig_vals_[i]), vecs_orig[:,i]) for i in xrange(self.eig_vals_.size)]
        self.eig_pairs_ = sorted(self.eig_pairs_, key = lambda k: k[0], reverse=True)

        for i, p in enumerate(self.eig_pairs_):
            self.eig_vals_[i] = p[0]
            self.eig_vecs_[i,:] = p[1]

        self.L_ = self.eig_vecs_[:num_dims,:]
        #print(np.trace(self.L_.dot(S-C).dot(self.L_.T)))
        return self
    
    def transformer(self):
        return self.L_
    
    def transform(self,X=None):
        if X is None:
            X=self.X_
        else:
            X=check_array(X,accept_sparse=True)
            
        L=self.transformer()
        K=self._get_kernel(X,self.X_)
        return K.dot(L.T)
    
    
    def _compute_average_margin(self,L,S,C):
        return np.trace(L.dot(S-C).dot(L.T))


    def _compute_heterogeneous_neighborhood(self,X,y):
        het_neighs = np.empty([self.n_,self.n_en_],dtype=int)
        for i, x in enumerate(X):
            mask = np.flatnonzero(y != y[i])
            enemy_dists = [(m,self.distance_matrix_[i,m]) for m in mask]
            enemy_dists = sorted(enemy_dists, key = lambda k: k[1])

            for j, p in enumerate(enemy_dists[:self.n_en_]):
                het_neighs[i,j] = p[0]
            #het_neighs[i,:] = enemy_dists[0,:self.n_en_]

        return het_neighs


    def _compute_homogeneous_neighborhood(self,X,y):
        hom_neighs = np.empty([self.n_,self.n_fr_],dtype=int)
        for i, x in enumerate(X):
            cur_class = y[i] ### Para no contar el propio indice de forma eficiente (mejorar)
            #y[i]+=1 ###
            #mask = np.flatnonzero(y == cur_class)
            #y[i]-=1 ###
            mask=np.concatenate([np.flatnonzero(y[:i]==cur_class),(i+1)+np.flatnonzero(y[i+1:]==cur_class)])

            friend_dists = [(m,self.distance_matrix_[i,m]) for m in mask]
            friend_dists = sorted(friend_dists, key = lambda k: k[1])

            for j, p in enumerate(friend_dists[:self.n_fr_]):
                hom_neighs[i,j] = p[0]

        return hom_neighs

    def _compute_matrices(self,K,het_neighs,hom_neighs):
        S = np.zeros([self.n_,self.n_])
        C = np.zeros([self.n_,self.n_])

        for i,k in enumerate(K):
            for j in xrange(self.n_en_):
                S += np.outer(k-K[het_neighs[i,j],:],k-K[het_neighs[i,j],:])
            for j in xrange(self.n_fr_):
                C += np.outer(k-K[hom_neighs[i,j],:],k-K[hom_neighs[i,j],:])
        
        S /= self.n_en_
        C /= self.n_fr_

        return S,C

        



