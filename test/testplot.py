#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:43:34 2018

@author: jlsuarezdiaz
"""

from utils import(
        read_ARFF, kfold_multitester_supervised_knn, toy_datasets)

from dml import(
    NCA,LDA,RCA,PCA,LMNN,ANMM,LSI,kNN, knn_plot)

from sklearn.datasets import load_digits

import numpy as np

seed=28
np.random.seed(seed)

def digits_data():
    data=load_digits()     # DIGITS
    X=data['data']
    y=data['target']

    return X,y

#X,y = toy_datasets.digits_toy_dataset(dims=[0,1],numbers=[7,8])
#X,y = toy_datasets.circular_toy_dataset(rads=[1,2,3],samples=[200,200,200],noise=[0.4,0.4,0.4],seed=seed)
#X,y = toy_datasets.hiperplane_toy_dataset(seed=seed)
#X,y = toy_datasets.iris2d_toy_dataset()
X,y = toy_datasets.balls_toy_dataset(seed=seed)
#X,y = toy_datasets.single_toy_dataset(seed=seed)

#X = np.array([[3,1],[4,2],[3,4],[5,4],[-1,1],[1,2],[2,2],[3,3]])
#y = np.array([1,1,1,1,-1,-1,-1,-1])
#y = np.array(['OK','OK','OK','OK','ERR','ERR','ERR','ERR'])

#X = np.array([[0,0],[0,1],[1.1,0],[2,0]])
#y = np.array([1,1,-1,-1])
    
#X, y = digits_data()
#selected = np.where(np.isin(y,[0,1,3,4,6,9]))[0]
#X, y = X[selected,:], y[selected]
knn_plot(X,y,k=3,cmap="gist_rainbow",figsize=(10,8))

nca = NCA(max_iter=10000, learning_rate = 0.1, initial_transform = "scale", descent_method = "BGD")
lda = LDA(num_dims = 2)
#rca = RCA()
pca = PCA(thres = 0.95)
lmnn = LMNN(max_iter=100000,learning_rate = "adaptive",eta0=0.001, k = 1, mu = 0.5,tol=0,prec=0)
anmm = ANMM(num_dims = 2, n_friends = 1,n_enemies = 1)
lsi = LSI(supervised=True, err = 1e-10, itproj_err = 1e-10)

alg = lda

alg.fit(X,y)
Xn = alg.transform(X)
if(Xn.shape[1] < 2):
    X2 = np.empty([Xn.shape[0],2])
    for i in range(Xn.shape[0]):
        X2[i,:] = [Xn[i,0],0.0]
    Xn = X2

#knn_plot(Xn,y,k=3,cmap="gist_rainbow",region_intensity=0.4,legend_plot_points=True,figsize=(15,8))
