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

seed=28
#X,y = toy_datasets.digits_toy_dataset(dims=[0,1],numbers=[7,8])
X,y = toy_datasets.circular_toy_dataset(rads=[1,2,3],samples=[200,200,200],noise=[0.4,0.4,0.4],seed=seed)
#X,y = toy_datasets.hiperplane_toy_dataset(seed=seed)
#X,y = toy_datasets.iris2d_toy_dataset()
#X,y = toy_datasets.balls_toy_dataset(seed=seed)
#X,y = toy_datasets.single_toy_dataset(seed=seed)

toy_datasets.toy_plot(X,y)

nca = NCA(max_iter=100, learning_rate = 0.1, initial_transform = "scale", descent_method = "BGD")
lda = LDA(thres = 1.0)
#rca = RCA()
pca = PCA(thres = 1.0)
lmnn = LMNN(max_iter=100,learning_rate = "adaptive",eta0=0.001, k = 5, mu = 0.5)
anmm = ANMM(n_friends = 5,n_enemies = 1)
lsi = LSI(supervised=True)

alg = lmnn

alg.fit(X,y)
Xn = alg.transform(X)

toy_datasets.toy_plot(Xn,y)


