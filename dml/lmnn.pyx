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
from sklearn.base import BaseEstimator, ClassifierMixin

from .dml_utils import SDProject, calc_outers, calc_outers_i, metric_sq_distance, pairwise_sq_distances_from_dot
from .dml_algorithm import DML_Algorithm, KernelDML_Algorithm



class LMNN(DML_Algorithm, ClassifierMixin):

    def __init__(self, num_dims = None, learning_rate = "adaptive", eta0 = 0.3, initial_metric = None, max_iter = 100, prec = 1e-8,
                tol = 1e-8, k = 3, mu = 0.5, soft_comp_interval= 1, learn_inc = 1.01, learn_dec = 0.5, eta_thres=1e-14, solver = "SDP"):
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
        self.solver_ = solver
        
        # Metadata
        self.num_its_ = None
        self.initial_error_ = None
        self.final_error_ = None
        
    def metadata(self):
        return {'num_iters':self.num_its_,
                'initial_error':self.initial_error_,
                'final_error':self.final_error_}


    def fit(self,X,y):
        
        if self.solver_ == "SDP":
            self._SDP_fit(X,y)
        elif self.solver_ == "SGD":
            self._SGD_fit(X,y)
        
        return self

    def _SDP_fit(self,X,y):
        X, y = self._set_initial_parameters(X,y)
        n,d = X.shape

        self.num_its_ = 0
        self.eta_ = self.eta0_

        N_up = set()       #Active set
        N_down = set()     #"Exact" set
        N_old = set()      #Exact set of last iteration

        outers = calc_outers(X)
        self.target_neighbors_ = target_neighbors = self._target_neighbors(X,y)

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

    def _SGD_fit(self,X,y):
        # Initializing parameters
        X, y = check_X_y(X,y)
        self.X_, self.y_ = X,y
        n, d = X.shape
        self.n_, self.d_ = n,d
        
        if self.num_dims_ is not None:
            self.nd_ = min(self.d_,self.num_dims_)
        else:
            self.nd_ = self.d_
            
        self.L_ = self.M0_
        
        if self.L_ is None or self.L_ == "euclidean":
            self.L_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.L_,1.0) #Euclidean distance
        elif self.L_ == "scale":
            self.L_ = np.zeros([self.nd_, self.d_])
            np.fill_diagonal(self.L_,1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16)))
            
        self.X_ = X
        self.y_ = y
        
        # Initializing algorithm
        self.num_its_ = 0
        self.eta_ = self.eta0_
        
        outers = calc_outers(X)
        self.target_neighbors_ = target_neighbors = self._target_neighbors(X,y)
    
        L = self.L_

        Lx = L.dot(X.T).T
        
        impostors = self._euc_impostors(Lx,y,target_neighbors)
        self.initial_error_ = self._compute_euc_error(self.mu_,Lx,y,target_neighbors,impostors)
        err_prev =  err = self.initial_error_
        stop = False
        
        while not stop:    
            #X, y, outers, target_neighbors, Lx = LMNN._shuffle(X,y,outers,target_neighbors,Lx)
            rnd = np.random.permutation(len(y))
            for i in rnd:
                non_imp_grad = np.zeros([d,d])
                imp_grad = np.zeros([d,d])
                outers_i = calc_outers_i(X,outers,i)
                for j in target_neighbors[i]:
                    lxij = Lx[i]-Lx[j]
                    margin = 1 +  np.inner(lxij,lxij)
                    oij = outers_i[j]
                    non_imp_grad += outers_i[j]
                    
                    for l in rnd:
                        if y[i] != y[l] and margin > np.inner(Lx[i]-Lx[l],Lx[i]-Lx[l]):
                            imp_grad += (oij - outers_i[l])
                            
                grad = (1-self.mu_)*non_imp_grad + self.mu_*imp_grad
                grad = 2*L.dot(grad)
                L -= self.eta_*grad
                Lx = L.dot(X.T).T  
                
                
            # Update and stop conditions
            if self.adaptive_:
                impostors = self._euc_impostors(Lx,y,target_neighbors)
                err = self._compute_euc_error(self.mu_,Lx,y,target_neighbors,impostors)
    
                if err < err_prev:
                    self.eta_ *= self.l_inc_                    
                else:
                    self.eta_ *= self.l_dec_
                    if self.eta_ < self.etamin_:
                        stop = True
                
                err_prev = err
            
            grad_norm = np.max(np.abs(grad))
            if grad_norm < self.eps_ or self.eta_*grad_norm < self.tol_: # Difference between two iterations is given by eta*grad
                stop=True

            self.num_its_+=1
            if self.num_its_ == self.max_it_:
                stop=True
            if stop:
                break

        self.L_ = L
        self.final_error_ = self._compute_euc_error(self.mu_,Lx,y,target_neighbors,self._euc_impostors(X,y,target_neighbors))
            

    def predict(self,X=None):
        if X is None:
            X = self.X_
        Xtr, ytr = self.X_, self.y_
        M = self.metric()
        
        y = np.empty([X.shape[0]],dtype=ytr.dtype)
        classes = np.unique(ytr)
        
        for t, xt in enumerate(X):
            emin = np.inf
            argmin = None
            for c in classes:
                target_energy = 0.0
                imposter_energy = 0.0
                self_imposter_energy = 0.0
                
                target = self._test_target_neighbors(xt,c)
                
                for j in target:
                    djt = LMNN._distance(Xtr[j,:],xt,M)
                    target_energy += djt
                    target_margin = 1+djt
                    for l, xl in enumerate(Xtr):  
                        dtl = LMNN._distance(xl,xt,M)
                
                        if ytr[l] != c and dtl < target_margin:
                            imposter_energy += (target_margin - dtl)
                
                for i, xi in enumerate(Xtr):
                    dit = LMNN._distance(xi,xt,M)
                    for j in self.target_neighbors_[i]:
                        margin = 1 + LMNN._distance(xi,Xtr[j,:],M)
                        if ytr[i] != c and dit < margin:
                            self_imposter_energy += (margin - dit)
                            
                energy = (1-self.mu_)*target_energy + self.mu_*(imposter_energy + self_imposter_energy)
                
                if energy < emin:
                    emin = energy
                    argmin = c
                
            y[t] = argmin
                    
        return y
                        
                

    def _set_initial_parameters(self,X,y):
        self.n_, self.d_ = X.shape
        if self.num_dims_ is not None:
            self.nd_ = min(self.d_,self.num_dims_)
        else:
            self.nd_ = self.d_

        self.M_ = self.M0_

        if self.M_ is None or self.M_ == "euclidean":
            self.M_= np.zeros([self.d_,self.d_]) # TODO reduce dimension with PCA in case of SDP (at the moment num_dims is ignored)
            np.fill_diagonal(self.M_,1.0) #Euclidean distance 
        elif self.M_ == "scale":
            self.M_= np.zeros([self.d_,self.d_]) # TODO reduce dimension with PCA in case of SDP
            np.fill_diagonal(self.M_, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance

        
        X, y = check_X_y(X,y)
        self.X_ = X
        self.y_ = y
        return X, y


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
    
    def _test_target_neighbors(self,xtest,ytest):
        Xtr, ytr = self.X_, self.y_
        
        inds, = np.where(ytr == ytest)
        dists = pairwise_distances(xtest.reshape(1,-1),Xtr[inds])[0]
        return np.argsort(dists)[...,:self.k_]


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
    
    def _euc_impostors(self,X,y,target_neighbors):
        impostors = []
        for i, yi in enumerate(y):
            out_inds, = np.where(y != yi)
            target_inds = target_neighbors[i,:]
            inds = np.concatenate([target_inds,out_inds])
            target_len = len(target_inds)
            impostors_i = []
            
            xi = X[i,:].reshape(1,-1)
            dists = pairwise_distances(xi,X[inds,:])
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

    def _compute_euc_error(self,mu,X,y,target_neighbors,impostors):
        n,d = X.shape
        non_imposter_err = 0.0
        imposter_err = 0.0
        for i in xrange(n):
            for j in target_neighbors[i,:]:
                xij = X[i,:]-X[j,:]
                non_imposter_err += np.linalg.norm(xij)
                for l in impostors[i]:
                    i_err = (1 + np.inner(xij,xij) - np.inner(X[i,:]-X[l,:],X[i,:]-X[l,:]))
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
               
    def _shuffle(X,y,outers,target_neighbors,Lx):
        rnd = np.random.permutation(len(y))
        invrnd = np.empty([len(y)],dtype=int)
        for i, p in enumerate(rnd):
            invrnd[p]=i
            
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
        target_neighbors = invrnd[target_neighbors[rnd,:]]
        Lx = Lx[rnd,:]

        return X,y, outers, target_neighbors, Lx


class KLMNN(KernelDML_Algorithm):
    
    def __init__(self, num_dims = None, learning_rate = "adaptive", eta0 = 0.3, initial_metric = None, max_iter = 100, prec = 1e-8,
                tol = 1e-8, k = 3, mu = 0.5, learn_inc = 1.01, learn_dec = 0.5, eta_thres=1e-14,
                kernel = "linear", gamma=None, degree=3, coef0=1, kernel_params=None, target_selection="kernel"):
        
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
        #self.soft_comp_interval_ = soft_comp_interval
        self.l_inc_ = learn_inc
        self.l_dec_ = learn_dec
        self.etamin_ = eta_thres
        self.target_selection_ = target_selection
        
        self.kernel_ = kernel
        self.gamma_ = gamma
        self.degree_ = degree
        self.coef0_ = coef0
        self.kernel_params_ = kernel_params
        
        # Metadata
        self.num_its_ = None
        self.initial_error_ = None
        self.final_error_ = None
        
    def transformer(self):
        return self.L_
    
    def metadata(self):
        return {'num_iters':self.num_its_,
                'initial_error':self.initial_error_,
                'final_error':self.final_error_}
    
    def fit(self,X,y):
        # Initializing parameters
        X, y = check_X_y(X,y)
        n, d = X.shape
        self.n_, self.d_ = n,d
        
        K = self._get_kernel(X)
        
        if self.num_dims_ is not None:
            self.nd_ = min(self.d_,self.num_dims_)
        else:
            self.nd_ = self.d_
            
        self.L_ = self.M0_
        
        if self.L_ is None or self.L_ == "euclidean":
            self.L_= np.zeros([self.nd_,self.n_])
            np.fill_diagonal(self.L_,1.0) #Euclidean distance
        elif self.L_ == "scale":
            self.L_ = np.zeros([self.nd_, self.n_])
            np.fill_diagonal(self.L_,1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16)))
            
        self.X_ = X
        self.y_ = y
        
        # Initializing algorithm
        self.num_its_ = 0
        self.eta_ = self.eta0_
        
        #outers = calc_outers(X)
        if self.target_selection_ == "original":    
            target_neighbors = self._target_neighbors(X,y)
        elif self.target_selection_ == "kernel":
            target_neighbors = self._kernel_target_neighbors(X,y,K)
        else:
            raise ValueError("'target_selection' must be on of the following: 'original', 'kernel'.")
    
        L = self.L_

        Lkx = K.dot(L.T)
        
        impostors = self._euc_impostors(Lkx,y,target_neighbors)
        self.initial_error_ = self._compute_euc_error(self.mu_,Lkx,y,target_neighbors,impostors)
        err_prev =  err = self.initial_error_
        stop = False
        
        while not stop:    
            #X, y, K, target_neighbors, Lkx, Kt = KLMNN._shuffle(X,y,K,target_neighbors,Lkx)
            rnd = np.random.permutation(len(y))
            for i in rnd:
                #print("L:",L);print("X:",X);print("y:",y);print("K:",K);print("target:",target_neighbors);print("Lkx",Lkx)  
                non_imp_grad = np.zeros([n,n])
                imp_grad = np.zeros([n,n])

                for j in target_neighbors[i]:
                    lxij = Lkx[i]-Lkx[j]
                    margin = 1 +  np.inner(lxij,lxij)
                    kij = K[i,:]-K[j,:]
                    Eij = np.zeros([n,n])
                    Eij[:,i] = kij
                    Eij[:,j] = -kij
                    non_imp_grad += Eij
                    #print("Eij:",Eij)
                    for l, yl in enumerate(y):
                        if y[i] != yl and margin > np.inner(Lkx[i]-Lkx[l],Lkx[i]-Lkx[l]):
                            kil = K[i,:]-K[l,:]
                            Eil = np.zeros([n,n])
                            Eil[:,i] = kil
                            Eil[:,l] = -kil
                            #print("Eil:",Eil)
                            #print(margin);print(np.inner(Lkx[i]-Lkx[l],Lkx[i]-Lkx[l]))
                            imp_grad += (Eij - Eil)
                            
                grad = (1-self.mu_)*non_imp_grad + self.mu_*imp_grad
                grad = 2*L.dot(grad)
                #print("GRAD: ");print(grad)
                L -= self.eta_*grad
                #print("L:");print(L);input()
                Lkx = K.dot(L.T)  
                
                
            # Update and stop conditions
            if self.adaptive_:
                impostors = self._euc_impostors(Lkx,y,target_neighbors)
                err = self._compute_euc_error(self.mu_,Lkx,y,target_neighbors,impostors)
                
                if err < err_prev:
                    self.eta_ *= self.l_inc_                    
                else:
                    self.eta_ *= self.l_dec_
                    if self.eta_ < self.etamin_:
                        stop = True
                
                err_prev = err
                    
            
            grad_norm = np.max(np.abs(grad))
            if grad_norm < self.eps_ or self.eta_*grad_norm < self.tol_: # Difference between two iterations is given by eta*grad
                stop=True

            self.num_its_+=1
            if self.num_its_ == self.max_it_:
                stop=True
            if stop:
                break

        self.L_ = L
        self.final_error_ = self._compute_euc_error(self.mu_,Lkx,y,target_neighbors,self._euc_impostors(X,y,target_neighbors))

        return self


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
    
    def _kernel_target_neighbors(self,X,y,K):
        """
        Returns a matrix nxk, where n is the amount of data, and each row contains
        the target neighbors indexes for each data index.
        """
        n,d = X.shape

        unique_labels = np.unique(y)
        target_neighbors = np.empty([n,self.k_],dtype=int)
        
        kernel_distances = pairwise_sq_distances_from_dot(K)

        for label in unique_labels:
            inds, = np.where(y == label)
            dists = kernel_distances[inds,:][:,inds]
            np.fill_diagonal(dists, np.inf)
            target_inds = np.argsort(dists)[..., :self.k_]
            target_neighbors[inds] = inds[target_inds]

        return target_neighbors
    
    def _euc_impostors(self,X,y,target_neighbors):
        impostors = []
        for i, yi in enumerate(y):
            out_inds, = np.where(y != yi)
            target_inds = target_neighbors[i,:]
            inds = np.concatenate([target_inds,out_inds])
            target_len = len(target_inds)
            impostors_i = []
            
            xi = X[i,:].reshape(1,-1)
            dists = pairwise_distances(xi,X[inds,:])
            target_limit = np.amax(dists[0,0:target_len])
            margin = (1+target_limit)
            
            for l in xrange(len(out_inds)):
                ldist = dists[0,target_len+l]
                if ldist < margin:
                    impostors_i.append(out_inds[l])

            impostors.append(impostors_i)

        return impostors
    
    
    def _compute_euc_error(self,mu,X,y,target_neighbors,impostors):
        n,d = X.shape
        non_imposter_err = 0.0
        imposter_err = 0.0
        for i in xrange(n):
            for j in target_neighbors[i,:]:
                xij = X[i,:]-X[j,:]
                non_imposter_err += np.linalg.norm(xij)
                for l in impostors[i]:
                    i_err = (1 + np.inner(xij,xij) - np.inner(X[i,:]-X[l,:],X[i,:]-X[l,:]))
                    if(i_err > 0):
                        imposter_err += i_err
                        
        return (1-mu)*non_imposter_err + mu*imposter_err
    
    
    ## Deprecated
    def _shuffle(X,y,K,target_neighbors,Lkx):
        rnd = np.random.permutation(len(y))
        invrnd = np.empty([len(y)],dtype=int)
        for i, p in enumerate(rnd):
            invrnd[p]=i
            
        X = X[rnd,:]
        y = y[rnd]
        Kt = K[rnd,:]
       
        for i in xrange(len(y)):
            K[:,i]=K[rnd,i]
        for i in xrange(len(y)):
            K[i,:]=K[i,rnd]
        
            
        target_neighbors = invrnd[target_neighbors[rnd,:]]
        Lkx = Lkx[rnd,:]

        return X,y, K, target_neighbors, Lkx, Kt
    
"""    
class LMNN_Energy(BaseEstimator, ClassifierMixin):
    def __init__(self, k = 3, mu = 0.5):
        self.k_ = k
        self.mu_ = mu
        
    def fit(self,X,y):
        X, y = check_X_y(X,y)
        self.X_, self.y_ = X, y
        self.target_neighbors = self._target_neighbors(X,y,self.k_)
        
    def predict(self,X=None):
        if X is None:
            X = self.X_
    
    def _target_neighbors(X,y,k,Xt=None,yt=None):
        ""
        Calculate target neighbors from test data to train data
        ""
        n,d = X.shape
        
        if Xt is None or yt is None:
            Xt = X
            yt = y

        unique_labels = np.unique(y)
        target_neighbors = np.empty([n,k],dtype=int)

        for label in unique_labels:
            inds, = np.where(y == label)
            inds_t = np.where(yt == label)
            dists = pairwise_distances(X[inds],Xt[inds_t])

            np.fill_diagonal(dists, np.inf)
            target_inds = np.argsort(dists)[..., :self.k_]
            target_neighbors[inds] = inds[target_inds]

        return target_neighbors
    
    
    
    def _distance(x,y,M):
        xy=(x-y).reshape(1,-1)
        return xy.dot(M).dot(xy.T)
"""