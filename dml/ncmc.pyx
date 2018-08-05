#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nearest Class with Multiple Centroids (NCMC)

Created on Wed Feb 28 16:18:39 2018

@author: jlsuarezdiaz
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestCentroid

from sklearn.cluster import KMeans

from numpy.linalg import eig
from scipy.linalg import eigh

from .dml_algorithm import DML_Algorithm, KernelDML_Algorithm
from .dml_utils import calc_outers, calc_outers_i


class NCMC(DML_Algorithm):
    """
    Nearest Class with Multiple Centroids distance metric learner (NCMC).

    A distance metric learning algorithm to improve the nearest class with multiple centroids classifier.

    Parameters
    ----------

    num_dims : int, default=None

        Desired value for dimensionality reduction. If None, the dimension of transformed data will be the same as the original.

    centroids_num : int, or list of int, default=3

        If it is a list, it must have the same size as the number of classes. In this case, i-th item will be the number of
        centroids to take in the i-th class. If it is an int, every class will have the same number of centroids.

    learning_rate : string, default='adaptive'

        Type of learning rate update for gradient descent. Possible values are:

        - 'adaptive' : the learning rate will increase if the gradient step is succesful, else it will decrease.

        - 'constant' : the learning rate will be constant during all the gradient steps.

    eta0 : int, default=0.3

        The initial value for learning rate.

    initial_transform : 2D-Array or Matrix (d' x d), or string, default=None.

        If array or matrix that will represent the starting linear map for gradient descent, where d is the number of features,
        and d' is the dimension specified in num_dims.
        If None, euclidean distance will be used. If a string, the following values are allowed:

        - 'euclidean' : the euclidean distance.

        - 'scale' : a diagonal matrix that normalizes each attribute according to its range will be used.

    max_iter : int, default=300

        Maximum number of gradient descent iterations.

    prec : float, default=1e-15

        Precision stop criterion (gradient norm).

    tol : float, default=1e-15

        Tolerance stop criterion (difference between two iterations)

    descent_method : string, default='SGD'

        The descent method to use. Allowed values are:

        - 'SGD' : stochastic gradient descent.

        - 'BGD' : batch gradient descent.

    eta_thres : float, default=1e-14

        A learning rate threshold stop criterion.

    learn_inc : float, default=1.01

        Increase factor for learning rate. Ignored if learning_rate is not 'adaptive'.

    learn_dec : float, default=0.5

        Decrease factor for learning rate. Ignored if learning_rate is not 'adaptive'.

    References
    ----------
        Thomas Mensink et al. “Metric learning for large scale image classification: Generalizing to new
        classes at near-zero cost”. In: Computer Vision–ECCV 2012. Springer, 2012, pages 488-501.

    """

    def __init__(self, num_dims=None, centroids_num=3, learning_rate="adaptive", eta0=0.3, initial_transform=None, max_iter=300,
                 tol=1e-15, prec=1e-15, descent_method="SGD", eta_thres=1e-14, learn_inc=1.01, learn_dec=0.5, **kmeans_kwargs):
        
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
        self.centroids_num_init_ = centroids_num
        self.kmeans_kwargs_ = kmeans_kwargs
        
        # Metadata initialization
        self.num_its_ = None
        self.initial_expectance_ = None
        self.final_expectance_ = None
        
    def metadata(self):
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            - num_iters : Number of iterations that the descent method took.

            - initial_expectance : Initial value of the objective function (the expected score)

            - final_expectance : Final value of the objective function (the expected score)
        """
        return {'num_iters':self.num_its_, 'initial_expectance':self.initial_expectance_, 'final_expectance':self.final_expectance_}
    
    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        L : (d'xd) matrix, where d' is the desired output dimension and d is the number of features.
        """
        return self.L_

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


        self.L_ = self.L0_
        self.eta_ = self.eta0_
        X, y = check_X_y(X,y)
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.X_ = X
        self.y_ = y      

        self.centroids_num_ = self.centroids_num_init_
        
        if self.L_ is None or self.L_ == "euclidean":
            self.L_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.L_,1.0) #Euclidean distance 
        elif self.L_ == "scale":
            self.L_= np.zeros([self.nd_,self.d_])
            np.fill_diagonal(self.L_, 1./(np.maximum(X.max(axis=0 )-X.min(axis=0),1e-16))) #Scaled eculidean distance

        if isinstance(self.centroids_num_,int):
            self.centroids_num_ = [self.centroids_num_]*len(np.unique(y))
            
        self.centroids_, self.cs_ = NCMC._compute_class_centroids(X,y,self.centroids_num_,**self.kmeans_kwargs_)
        
        self.initial_expectance_ = NCMC._compute_expected_success(self.L_,X,y,self.centroids_,self.cs_)
        if self.method_ == "SGD": # Stochastic Gradient Descent
            self._SGD_fit(X,y)
            
        elif self.method_ == "BGD": # Batch Gradient Desent
            self._BGD_fit(X,y)
        
        self.final_expectance_ = NCMC._compute_expected_success(self.L_,X,y,self.centroids_,self.cs_)
        return self
    
    def _SGD_fit(self,X,y):
        # Initialize parameters
        
        n,d= self.n_, self.d_

        L = self.L_
        centroids = self.centroids_
        cs = self.cs_
        cn = self.centroids_num_
        
        classes = len(cn)
        
        outers = calc_outers(X,centroids)
        
        self.num_its_ = 0

        grad = None

        succ_prev = succ = 0.0
        
        stop = False
        
        
        while not stop:
            #X, y, outers = NCMC._shuffle(X,y,outers)            
            rnd = np.random.permutation(len(y))
            for i in rnd:
               
                Lxi = L.dot(X[i,:].T).T
                Lm = L.dot(centroids.T).T
                
                mu_diff = Lxi - Lm
                dists_i = -0.5*np.diag(mu_diff.dot(mu_diff.T))
                i_max = np.argmax(dists_i)
                cmax = dists_i[i_max]
                softmax = np.exp(dists_i-cmax)
                softmax[i_max]=1.0
                softmax /= softmax.sum()
                
                grad_sum = 0.0
                outers_i = calc_outers_i(X,outers,i,centroids)
                for c in xrange(classes):
                    for k in xrange(cn[c]):
                        mask = softmax[cs[c]+k]/(softmax[cs[c]:cs[c+1]].sum()) if y[i] == c else 0.0
                        grad_sum += (softmax[cs[c]+k]-mask)*outers_i[cs[c]+k]
                grad = L.dot(grad_sum)
                
                if np.isnan(grad.sum()):
                    stop=True
                else:
                    L += self.eta_*grad   
            
            if self.adaptive_:
                succ = NCMC._compute_expected_success(L,X,y,centroids,cs)
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
        
        n,d= self.n_, self.d_

        L = self.L_
        centroids = self.centroids_
        cs = self.cs_
        cn = self.centroids_num_
        
        classes = len(cn)
        
        outers = calc_outers(X,centroids)
        
        self.num_its_ = 0

        grad = None

        succ_prev = succ = 0.0
        
        stop = False
        
        
        while not stop:
            Lx = L.dot(X.T).T
            Lm = L.dot(centroids.T).T
            grad_sum = 0.0
            
            for i,yi in enumerate(y):
               
                Lxi = Lx[i]
                
                mu_diff = Lxi - Lm
                dists_i = -0.5*np.diag(mu_diff.dot(mu_diff.T))
                i_max = np.argmax(dists_i)
                c = dists_i[i_max]
                softmax = np.exp(dists_i-c)
                softmax[i_max]=1.0
                softmax /= softmax.sum()
                
                grad_sum = 0.0
                outers_i = calc_outers_i(X,outers,i,centroids)
                for c in xrange(classes):
                    for k in xrange(cn[c]):
                        mask = softmax[cs[c]+k]/(softmax[cs[c]:cs[c+1]].sum()) if yi == c else 0.0
                        grad_sum += (softmax[cs[c]+k]-mask)*outers_i[cs[c]+k]
            
            grad = L.dot(grad_sum)/len(y)
            L += self.eta_*grad   
            
            if self.adaptive_:
                succ = NCMC._compute_expected_success(L,X,y,centroids,cs)
                #print("SUCC: ",succ)
                #print("ETA: ",self.eta_)
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
                #print(grad_norm)

            self.num_its_+=1
            if self.num_its_ == self.max_it_:
                stop=True
            if stop:
                break
            
        self.L_ = L
        
        return self



    def _shuffle(X,y,outers=None):
        rnd = np.random.permutation(len(y))
        X = X[rnd,:]
        y = y[rnd]
        if outers is not None:
            outers = outers[rnd,:]
        else:
            outers = None

        return X,y, outers
    
    def _compute_expected_success(L,X,y,centroids,cs):
        n,d = X.shape
        classes, _ = centroids.shape
        Lx = L.dot(X.T).T
        Lm = L.dot(centroids.T).T
        success = 0.0
        for i, yi in enumerate(y):
            
            Lxi = Lx[i]
            mu_diff = Lxi - Lm
            dists_i = -0.5*np.diag(mu_diff.dot(mu_diff.T))
            softmax = np.exp(dists_i)
            success += np.sum(np.log(softmax[cs[yi]:(cs[yi+1])]/softmax.sum()))
            # TODO Check if success calculation is good for more than one centroid
        return success/len(y)
    

    
    
    def _compute_class_centroids(X,y,centroids_num,**kmeans_kwargs):
        classes = np.unique(y)
        n, d = X.shape
        centroids = np.empty([sum(centroids_num),d])
        class_start = np.cumsum([0]+list(centroids_num))
        for i,c in enumerate(classes):
            print(i, centroids_num)
            k = centroids_num[i]
            #class_cent = np.empty([k,d])
            Xc = X[y==c]
            kmeans = KMeans(n_clusters=k,**kmeans_kwargs)
            kmeans.fit(Xc)
            centroids[class_start[i]:(class_start[i+1]),:]=kmeans.cluster_centers_
            
        return centroids, class_start


class NCMC_Classifier(BaseEstimator, ClassifierMixin):
    """
    Nearest Class with Multiple Centroids classifier.

    A classifier that makes its predictions by choosing the class who has a centroid the nearest to the point.
    For each class, an arbitrary number of centroids can be set. This centroids are calculated using k-Means
    over each class sub-dataset.

    Parameters
    ----------

    centroids_num : int, or list of int, default=3

        If it is a list, it must have the same size as the number of classes. In this case, i-th item will be the number of
        centroids to take in the i-th class. If it is an int, every class will have the same number of centroids.

    kmeans_args : dictionary

        Additional keyword args for k-Means.
    """
    def __init__(self, centroids_num=3, **kmeans_kwargs):
        self.centroids_num_ = centroids_num
        self.kmeans_args_ = kmeans_kwargs

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
        X,y = check_X_y(X,y)
        self.X_, self.y_ = X,y
        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(y)
        self.classes_ = np.unique(y)
        le = LabelEncoder()
        le.fit_transform(y)
        
        if isinstance(self.centroids_num_,int):
            self.centroids_num_ = [self.centroids_num_]*len(self.classes_)
            
        self.class_map_ = {}
        last_c = 0
        ygroups = np.array(y)
        for i,c in enumerate(self.classes_):
            c_mask = y==c
            Xc = X[c_mask]
            kmeans = KMeans(self.centroids_num_[c],**self.kmeans_args_)
            ymeans = kmeans.fit_predict(Xc)
            ymeans += last_c
            ymeans_groups = np.unique(ymeans)
            for j in ymeans_groups:
                self.class_map_[j] = c
            last_c = ymeans.max()+1
            ygroups[c_mask] = ymeans 
        
        self.ncm_ = NearestCentroid()
        self.ncm_.fit(X,ygroups)
        
        return self

    def predict(self, X):
        """
        Predicts the labels for the given data. Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        Returns
        -------

        y : 1D-Array

            The vector with the label predictions.
        """
        y = self.ncm_.predict(X)
        for i in range(len(y)):
            y[i] = self.class_map_[y[i]]
        return self.le_.inverse_transform(y)