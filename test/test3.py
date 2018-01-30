#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pandas as pd
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.datasets import(
    load_iris, load_digits, load_diabetes)
from numpy.testing import assert_array_almost_equal
from sklearn import neighbors

from sklearn.preprocessing import normalize

from utils import(
    read_ARFF, kfold_multitester_supervised_knn)

from dml import(
    NCA,LDA,RCA,PCA,LMNN,ANMM,LSI,kNN)

np.random.seed(28)

### Python included dataframes ###
def iris_data():
    data=load_iris()  # IRIS
    X=data['data']
    y=data['target']

    return X,y  

def simetria_hor(A):
    nrow, ncol= A.shape
    A = np.abs(A-A[:,::-1]) # Diferencia con la imagen simétrica
    return np.mean(A)                  # Media de las diferencias (grado de simetría)

def simetria_ver(A):
    nrow, ncol= A.shape
    A = np.abs(A-A[::-1,:]) # Diferencia con la imagen simétrica
    return np.mean(A)                  # Media de las diferencias (grado de simetría)


def digits_data():
    data=load_digits()     # DIGITS
    X=data['data']
    y=data['target']

    return X,y

def digits_red_data():
    data=load_digits()
    XX = data['data']
    y = data['target']
    nn,dd = XX.shape
    XX = XX.reshape([nn,8,8])

    X = np.empty([nn,3])
    for i in xrange(nn):
        X[i,0] = simetria_hor(XX[i,:,:])
        X[i,1] = simetria_ver(XX[i,:,:])
        X[i,2] = np.mean(XX[i,:])
    
    return X,y

### ARFF dataframes ###
def sonar_data():
    X,y,m = read_ARFF("./data/sonar.arff",-1)          # SONAR

    return X,y

def wdbc_data():
    X,y,m = read_ARFF("./data/wdbc.arff",0)            # WDBC

    return X,y

def spambase_data():
    X,y,m = read_ARFF("./data/spambase-460.arff",-1)   # SPAMBASE-460

    return X,y


### CSV dataframes ###
def wine_data():
    data = pd.read_csv("./data/wine.data")              # WINE
    X = data.iloc[:,1:].values        
    y=data.iloc[:,0].values

    return X,y





#X,y = iris_data()
#X,y = digits_data()
#X,y = digits_red_data()
#X,y = sonar_data()
#X,y = wdbc_data()
#X,y = spambase_data()
X,y = wine_data()

#X = normalize(X,axis=0,norm='max')



n,d = X.shape

print("Data dimensions: ", n, d)

nca = NCA(max_iter=500, learning_rate = "adaptive",eta0=0.001, initial_transform = "scale", descent_method = "BGD")
lda = LDA(thres = 0.95)
#rca = RCA()
pca = PCA(thres = 0.95)
lmnn = LMNN(max_iter=1000,learning_rate = "adaptive", eta0 = 0.001, k = 5, mu = 0.5,soft_comp_interval = 10)
anmm = ANMM(n_friends = 5,n_enemies = 1)

lsi = LSI(supervised=True)

dmls = [lmnn,lda]
#dmls = [nca,lda,pca,anmm]
results = kfold_multitester_supervised_knn(X,y,k = 5, n_neigh = 3, dmls = dmls, verbose = True)

print(results)