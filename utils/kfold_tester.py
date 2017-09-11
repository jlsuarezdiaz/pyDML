"""
K-Fold tester.
Functions to test a DML algorithm with cross validation.
"""

from __future__ import absolute_import
import numpy as np
import pandas as pd

import warnings
from six.moves import xrange

from sklearn.model_selection import(
    StratifiedKFold, KFold)

from dml import(
    kNN, MultiDML_kNN)

def kfold_tester_supervised_knn(X,y,k,n_neigh,dml):
    """
    X: data
    y: labels
    k: k for validation
    n_neigh: k for nearest neighbors
    dml: DML algorithm
    """
    skf = StratifiedKFold(n_splits = k,shuffle = True)

    m = np.empty([k+1,4])

    for i, [train_index, test_index] in enumerate(skf.split(X,y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dml.fit(X_train, y_train)

        knn = kNN(n_neighbors = n_neigh, dml_algorithm= dml)
        knn.fit(X_train, y_train)

        m[i,0] = knn.score_orig()
        m[i,1] = knn.score_orig(X_test,y_test)
        m[i,2] = knn.score()
        m[i,3] = knn.score(X_test,y_test)

#        print "FOLD ", i
#
#        print "Train - Train [ORIG]:", knn.score_orig()
#        print "Train - Test [ORIG]:", knn.score_orig(X_test,y_test)
#        print "Train - Train [DML]:", knn.score()
#        print "Train - Test [DML]:", knn.score(X_test,y_test)

    m[k,:] = np.mean(m[0:(k-1),:],axis = 0)

    rownames = []
    for i in range(k):
        rownames.append("FOLD "+str(i+1))

    rownames.append("MEAN")
    colnames = ["Train [ORIG]", "Test [ORIG]", "Train [DML]", "Test [DML]"]
    m = pd.DataFrame(data = m, index = rownames, columns = colnames)

    return m

def kfold_multitester_supervised_knn(X,y,k,n_neigh,dmls):
    """
    X: data
    y: labels
    k: k for validation
    n_neigh: k for nearest neighbors
    dmls: DML algorithms list
    """
    skf = StratifiedKFold(n_splits = k,shuffle = True)

    dmls_size = len(dmls)+1
    m = np.empty([k+1,2*dmls_size])

    for i, [train_index, test_index] in enumerate(skf.split(X,y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        mknn = MultiDML_kNN(n_neighbors = n_neigh, dmls = dmls)
        mknn.fit(X_train, y_train)

        m[i,:dmls_size] = mknn.score_all()
        m[i,dmls_size:] = mknn.score_all(X_test,y_test)
        


    m[k,:] = np.mean(m[0:(k-1),:],axis = 0)

    rownames = []
    for i in range(k):
        rownames.append("FOLD "+str(i+1))

    rownames.append("MEAN")

    dml_names = mknn.dmls_string()
    colnamestrain = [s + " [TRAIN]" for s in dml_names]
    colnamestest = [s + " [TEST]" for s in dml_names]
    colnames = colnamestrain + colnamestest

    m = pd.DataFrame(data = m, index = rownames, columns = colnames)

    return m