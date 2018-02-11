#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 19:29:06 2018

@author: jlsuarezdiaz
"""

from __future__ import absolute_import
import numpy as np
from sklearn import neighbors
import pandas as pd
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from .dml_algorithm import DML_Algorithm
from six.moves import xrange
from itertools import product
import time

def cross_validate(alg,X,y,n_folds=5,n_reps=1,verbose=False,seed=None):
    kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps,random_state=seed)
    n = n_folds*n_reps
    results = np.empty([n + 2,3])
    rownames = ["SPLIT "+str(i+1) for i in range(n)]+["MEAN","STD"]
    colnames = ["SCORE", "FIT TIME", "PREDICT TIME"]
    
    for i, [train_index, test_index] in enumerate(kf.split(X,y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit
        st = time.time()
        alg.fit(X_train,y_train)
        end = time.time()
        elapsed_fit = end - st
        
        # Predict
        st = time.time()
        score = alg.score(X_test,y_test)
        end = time.time()
        elapsed_sc = end - st
        
        results[i,:] = [score,elapsed_fit,elapsed_sc]
        
    results[n,:] = np.mean(results[0:n,:],axis=0)
    results[n+1,:] = np.std(results[0:n,:],axis=0)
    
    results = pd.DataFrame(data = results, index = rownames, columns = colnames)
    
    return results

def tune_knn(dml,X,y,n_neighbors,dml_params,tune_args,n_folds=5,n_reps=1,verbose=False,seed=None,**knn_args):
    args = []
    prod_size = 1
    for key in tune_args:
        args.append(tune_args[key])
        prod_size *= len(tune_args[key])
        
    prod = product(*args)
    
    rownames = []
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,**knn_args)
    
    tune_results = np.empty([prod_size,1])
    detailed_results = {}
    
    for j, p in enumerate(prod):
        tuned_args = {}
        # Obtain each dictionary of params to tune
        for i, key in enumerate(tune_args):
            tuned_args[key] = p[i]
            
        tune_case = str(tuned_args)
        rownames.append(tuned_args)
        
        # DML algorthm with each tune params
        dml_alg = dml(**tuned_args,**dml_params)
        alg = Pipeline([("DML",dml_alg),("KNN",knn)])
        
        results = cross_validate(alg,X,y,n_folds=n_folds,n_reps=n_reps,verbose=verbose,seed=seed)
        
        detailed_results[tune_case] = results
        tune_results[j,:] = results['SCORE'].loc['MEAN']
        
    tune_results = pd.DataFrame(data = tune_results, index = rownames, columns = ['SCORE'])
    best_performance = (tune_results['SCORE'].argmax(),tune_results['SCORE'].max())
    best_dml = dml(**(best_performance[0]),**dml_params)
    
    return tune_results, best_performance, best_dml, detailed_results
    
        