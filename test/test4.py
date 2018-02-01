#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:25:50 2018

@author: jlsuarezdiaz
"""

from utils import(
        read_ARFF, kfold_multitester_supervised_knn, toy_datasets)

from dml import(
    NCA,LDA,RCA,PCA,LMNN,ANMM,LSI,kNN)

import numpy as np

seed=28
#X,y = toy_datasets.digits_toy_dataset(dims=[0,1],numbers=[7,8])
#X,y = toy_datasets.circular_toy_dataset(rads=[1,2,3],samples=[200,200,200],noise=[0.4,0.4,0.4],seed=seed)
#X,y = toy_datasets.hiperplane_toy_dataset(seed=seed)
#X,y = toy_datasets.iris2d_toy_dataset()
#X,y = toy_datasets.balls_toy_dataset(seed=seed)
#X,y = toy_datasets.single_toy_dataset(seed=seed)

X = np.array([[3,1],[4,2],[3,4],[5,4],[-1,1],[1,2],[2,2],[3,3]])
y = np.array([1,1,1,1,-1,-1,-1,-1])

toy_datasets.toy_plot(X,y)

nca = NCA(max_iter=100, learning_rate = 0.1, initial_transform = "scale", descent_method = "BGD")
lda = LDA(thres = 0.546)
#rca = RCA()
pca = PCA(thres = 0.5)
lmnn = LMNN(max_iter=100,learning_rate = "adaptive",eta0=0.001, k = 4, mu = 0.5)
anmm = ANMM(num_dims = 1, n_friends = 1,n_enemies = 1)
lsi = LSI(supervised=True, err = 1e-10, itproj_err = 1e-10)

alg = lmnn

alg.fit(X,y)
Xn = alg.transform(X)
if(Xn.shape[1] < 2):
    X2 = np.empty([Xn.shape[0],2])
    for i in range(Xn.shape[0]):
        X2[i,:] = [Xn[i,0],0.0]
    Xn = X2

toy_datasets.toy_plot(Xn,y)


