#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Large Margin Nearest Neighbors

A DML that obtains a metric with target neighbors as near as possible and impostors as far as possible
"""

from __future__ import print_function, absolute_import
import numpy as np
import warnings
from collections import Counter
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y, check_array

from .dml_utils import SDProject, calc_outers, calc_outers_i, metric_sq_distance
from .dml_algorithm import DML_Algorithm

class LMNN(DML_Algorithm):

    def __init__(self, num_dims = None, learning_rate = "adaptive", eta0 = 0.001, initial_transform = None, max_iter = 100, prec = 1e-3,
                tol = 1e-6, k = 3, mu = 0.5, soft_comp_interval= 1, learn_inc = 1.01, learn_dec = 0.5):
        self.num_dims_ = num_dims
        self.M0_ = initial_transform
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
        
        # Metadata
        self.num_its_ = None
        self.initial_error_ = None
        self.final_error_ = None
        
    def metadata(self):
        return {'num_iters':self.num_its_,'initial_error':self.initial_error_,'final_error':self.final_error_}

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
        err_prev =  err = np.inf
        
        self.initial_error_ = self._compute_error(self.mu_,M,X,y,target_neighbors,self._impostors(X,y,target_neighbors
                                                                                                  ))
        #while not self._stop_criterion():
        while self.num_its_ < self.max_it_ and  (Mprev is None or np.max(np.abs(M-Mprev)) > self.tol_) and np.max(np.abs(G)) > self.eps_:
            if self.num_its_ % self.soft_comp_interval_ == 0:
                impostors = self._impostors(X,y,target_neighbors)
                print(impostors)
                N_down = self._compute_N_triplets(n,target_neighbors,impostors)
                N_up |= N_down # Union
            else:
                N_down = N_down & N_up # intersection
                
            Mprev = M.copy()
            err_prev = err

            grad_imp = self._compute_imposter_gradient(X,outers,N_down,N_old)
            G += grad_imp

            M -= self.eta_*G
            self.M_ = M = SDProject(M)

            N_old = N_down
            self.num_its_+=1
            
            #if self.num_its_ == 1: 
            #    err = self._compute_error(self.mu_,M,X,y,target_neighbors,impostors)
            #    self.initial_error_ = err
            #print("ERR: ",self._compute_error(self.mu_,M,X,y,target_neighbors,impostors))
            #print("ETA: ",self.eta_)
            #print("GRD: ",np.max(np.abs(G)))
            #if Mprev is not None:
            #    print("CCH: ",np.max(np.abs(M-Mprev)))
                
            if self.adaptive_: # TODO En el primer paso el error puede subir drásticamente. Controlar si el error es muy grande y no actualizar matriz
                err = self._compute_error(self.mu_,M,X,y,target_neighbors,impostors)
                if err < err_prev:
                    self.eta_ *= self.l_inc_
                else:
                    self.eta_ *= self.l_dec_
            #if(err > err_prev):
            #    break
            

        self.M_ = M
        self.final_error_ = self._compute_error(self.mu_,M,X,y,target_neighbors,self._impostors(X,y,target_neighbors))
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

        

        self.X_ = X
        self.y_ = y

    def _stop_criterion(self):

        it_crit = self.num_its_ >= self.max_it_

        return it_crit

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


    def _impostors(self,X,y,target_neighbors):
        
        #Lx = self.transform()
        
        impostors = []
        for i, yi in enumerate(y):
            out_inds, = np.where(y != yi)
            target_inds = target_neighbors[i,:]
            
            inds = np.concatenate([target_inds,out_inds])
            target_len = len(target_inds)
            impostors_i = []
            
            #Lxi = Lx[i].reshape(1,-1) # Convert single row to matrix
            #dists = pairwise_distances(Lxi,Lx[inds])
            dists = self._pairwise_metric_distances(X[i,:],X[inds,:])
            #print(dists)
            margin = (1+np.amax(dists[0:target_len]))*(1+np.amax(dists[0:target_len]))

            for l in xrange(len(out_inds)):
                ldist = dists[target_len+l]
                print(i,X[i],target_len+l,X[target_len+l],ldist,margin)
                import time
                time.sleep(1)
                if ldist < margin:
                    impostors_i.append(out_inds[l])

            impostors.append(impostors_i)

        return impostors

    def _pairwise_metric_distances(self,xi,X):
        n,d = X.shape
        xi = xi.reshape(1,-1)
        dists = np.empty([n],dtype=float)
        for j in xrange(n):
            xj = X[j,:].reshape(1,-1)
            xij = xi - xj
            dists[j] = xij.dot(self.M_).dot(xij.T)
        return dists

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
                    #print(metric_sq_distance(M,X[i,:],X[j,:]),"  ",metric_sq_distance(M,X[i,:],X[l,:]))
                    
        return (1-mu)*non_imposter_err + mu*imposter_err


    def _compute_N_triplets(self,n,target_neighbors,impostors):
        triplets = set()
        for i in xrange(n):
            for j in target_neighbors[i,:]:
                for l in impostors[i]:
                    if(1 + metric_sq_distance(self.M_,self.X_[i,:],self.X_[j,:]) - metric_sq_distance(self.M_,self.X_[i,:],self.X_[l,:]) > 0):
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
               


# commonality between LMNN implementations
#class _base_LMNN(DML_Algorithm):
#  def __init__(self, k=3, min_iter=50, max_iter=1000, learn_rate=1e-7,
#               regularization=0.5, convergence_tol=0.001, use_pca=True,
#               verbose=False):
#    """Initialize the LMNN object.
#
#    Parameters
#    ----------
#    k : int, optional
#        Number of neighbors to consider, not including self-edges.
#
#    regularization: float, optional
#        Weighting of pull and push terms, with 0.5 meaning equal weight.
#    """
#    self.k = k
#    self.min_iter = min_iter
#    self.max_iter = max_iter
#    self.learn_rate = learn_rate
#    self.regularization = regularization
#    self.convergence_tol = convergence_tol
#    self.use_pca = use_pca
#    self.verbose = verbose
#
#  def transformer(self):
#    return self.L_
#
#
## slower Python version
#class LMNN(_base_LMNN):
#
#  def _process_inputs(self, X, labels):
#    self.X_ = check_array(X, dtype=float)
#    num_pts, num_dims = self.X_.shape
#    unique_labels, self.label_inds_ = np.unique(labels, return_inverse=True)
#    if len(self.label_inds_) != num_pts:
#      raise ValueError('Must have one label per point.')
#    self.labels_ = np.arange(len(unique_labels))
#    if self.use_pca:
#      warnings.warn('use_pca does nothing for the python_LMNN implementation')
#    self.L_ = np.eye(num_dims)
#    required_k = np.bincount(self.label_inds_).min()
#    if self.k > required_k:
#      raise ValueError('not enough class labels for specified k'
#                       ' (smallest class has %d)' % required_k)
#
#  def fit(self, X, y):
#    k = self.k
#    reg = self.regularization
#    learn_rate = self.learn_rate
#    self._process_inputs(X, y)
#
#    target_neighbors = self._select_targets()
#    impostors = self._find_impostors(target_neighbors[:,-1])
#    if len(impostors) == 0:
#        # L has already been initialized to an identity matrix
#        return
#
#    # sum outer products
#    dfG = _sum_outer_products(self.X_, target_neighbors.flatten(),
#                              np.repeat(np.arange(self.X_.shape[0]), k))
#    df = np.zeros_like(dfG)
#
#    # storage
#    a1 = [None]*k
#    a2 = [None]*k
#    for nn_idx in xrange(k):
#      a1[nn_idx] = np.array([])
#      a2[nn_idx] = np.array([])
#
#    # initialize gradient and L
#    G = dfG * reg + df * (1-reg)
#    L = self.L_
#    objective = np.inf
#
#    # main loop
#    for it in xrange(1, self.max_iter):
#      df_old = df.copy()
#      a1_old = [a.copy() for a in a1]
#      a2_old = [a.copy() for a in a2]
#      objective_old = objective
#      # Compute pairwise distances under current metric
#      Lx = L.dot(self.X_.T).T
#      g0 = _inplace_paired_L2(*Lx[impostors])
#      Ni = 1 + _inplace_paired_L2(Lx[target_neighbors], Lx[:,None,:])
#      g1,g2 = Ni[impostors]
#
#      # compute the gradient
#      total_active = 0
#      for nn_idx in reversed(xrange(k)):
#        act1 = g0 < g1[:,nn_idx]
#        act2 = g0 < g2[:,nn_idx]
#        total_active += act1.sum() + act2.sum()
#
#        if it > 1:
#          plus1 = act1 & ~a1[nn_idx]
#          minus1 = a1[nn_idx] & ~act1
#          plus2 = act2 & ~a2[nn_idx]
#          minus2 = a2[nn_idx] & ~act2
#        else:
#          plus1 = act1
#          plus2 = act2
#          minus1 = np.zeros(0, dtype=int)
#          minus2 = np.zeros(0, dtype=int)
#
#        targets = target_neighbors[:,nn_idx]
#        PLUS, pweight = _count_edges(plus1, plus2, impostors, targets)
#        df += _sum_outer_products(self.X_, PLUS[:,0], PLUS[:,1], pweight)
#        MINUS, mweight = _count_edges(minus1, minus2, impostors, targets)
#        df -= _sum_outer_products(self.X_, MINUS[:,0], MINUS[:,1], mweight)
#
#        in_imp, out_imp = impostors
#        df += _sum_outer_products(self.X_, in_imp[minus1], out_imp[minus1])
#        df += _sum_outer_products(self.X_, in_imp[minus2], out_imp[minus2])
#
#        df -= _sum_outer_products(self.X_, in_imp[plus1], out_imp[plus1])
#        df -= _sum_outer_products(self.X_, in_imp[plus2], out_imp[plus2])
#
#        a1[nn_idx] = act1
#        a2[nn_idx] = act2
#
#      # do the gradient update
#      assert not np.isnan(df).any()
#      G = dfG * reg + df * (1-reg)
#
#      # compute the objective function
#      objective = total_active * (1-reg)
#      objective += G.flatten().dot(L.T.dot(L).flatten())
#      assert not np.isnan(objective)
#      delta_obj = objective - objective_old
#
#      if self.verbose:
#        print(it, objective, delta_obj, total_active, learn_rate)
#
#      # update step size
#      if delta_obj > 0:
#        # we're getting worse... roll back!
#        learn_rate /= 2.0
#        df = df_old
#        a1 = a1_old
#        a2 = a2_old
#        objective = objective_old
#      else:
#        # update L
#        L -= learn_rate * 2 * L.dot(G)
#        learn_rate *= 1.01
#
#      # check for convergence
#      if it > self.min_iter and abs(delta_obj) < self.convergence_tol:
#        if self.verbose:
#          print("LMNN converged with objective", objective)
#        break
#    else:
#      if self.verbose:
#        print("LMNN didn't converge in %d steps." % self.max_iter)
#
#    # store the last L
#    self.L_ = L
#    self.n_iter_ = it
#    return self
#
#  def _select_targets(self):
#    target_neighbors = np.empty((self.X_.shape[0], self.k), dtype=int)
#    for label in self.labels_:
#      inds, = np.nonzero(self.label_inds_ == label)
#      dd = pairwise_distances(self.X_[inds])
#      np.fill_diagonal(dd, np.inf)
#      nn = np.argsort(dd)[..., :self.k]
#      target_neighbors[inds] = inds[nn]
#    return target_neighbors
#
#  def _find_impostors(self, furthest_neighbors):
#    Lx = self.transform()
#    margin_radii = 1 + _inplace_paired_L2(Lx[furthest_neighbors], Lx)
#    impostors = []
#    for label in self.labels_[:-1]:
#      in_inds, = np.nonzero(self.label_inds_ == label)
#      out_inds, = np.nonzero(self.label_inds_ > label)
#      dist = pairwise_distances(Lx[out_inds], Lx[in_inds])
#      i1,j1 = np.nonzero(dist < margin_radii[out_inds][:,None])
#      i2,j2 = np.nonzero(dist < margin_radii[in_inds])
#      i = np.hstack((i1,i2))
#      j = np.hstack((j1,j2))
#      if i.size > 0:
#        # get unique (i,j) pairs using index trickery
#        shape = (i.max()+1, j.max()+1)
#        tmp = np.ravel_multi_index((i,j), shape)
#        i,j = np.unravel_index(np.unique(tmp), shape)
#      impostors.append(np.vstack((in_inds[j], out_inds[i])))
#    if len(impostors) == 0:
#        # No impostors detected
#        return impostors
#    return np.hstack(impostors)
#
#
#def _inplace_paired_L2(A, B):
#  '''Equivalent to ((A-B)**2).sum(axis=-1), but modifies A in place.'''
#  A -= B
#  return np.einsum('...ij,...ij->...i', A, A)
#
#
#def _count_edges(act1, act2, impostors, targets):
#  imp = impostors[0,act1]
#  c = Counter(zip(imp, targets[imp]))
#  imp = impostors[1,act2]
#  c.update(zip(imp, targets[imp]))
#  if c:
#    active_pairs = np.array(list(c.keys()))
#  else:
#    active_pairs = np.empty((0,2), dtype=int)
#  return active_pairs, np.array(list(c.values()))
#
#
#def _sum_outer_products(data, a_inds, b_inds, weights=None):
#  Xab = data[a_inds] - data[b_inds]
#  if weights is not None:
#    return np.dot(Xab.T, Xab * weights[:,None])
#  return np.dot(Xab.T, Xab)
#
#
#
#