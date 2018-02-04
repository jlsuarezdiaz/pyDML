#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:43:34 2018

@author: jlsuarezdiaz
"""

from utils import(
        read_ARFF, kfold_multitester_supervised_knn, toy_datasets)

from dml import(
    NCA,LDA,RCA,PCA,LMNN,ANMM,LSI,kNN, knn_plot, classifier_plot, knn_multiplot)

from sklearn.datasets import load_digits
from sklearn import svm

import numpy as np

seed=28
np.random.seed(seed)

def digits_data():
    data=load_digits()     # DIGITS
    X=data['data']
    y=data['target']

    return X,y

#X,y = toy_datasets.digits_toy_dataset(dims=[0,1],numbers=[7,8])
    

X,y = toy_datasets.circular_toy_dataset(rads=[1,2,3],samples=[200,200,200],noise=[0.4,0.4,0.4],seed=seed)
knn_plot(X,y,k=3,cmap="gist_rainbow",figsize=(15,8))


X,y = toy_datasets.hiperplane_toy_dataset(seed=seed)
knn_plot(X,y,k=3,cmap="gist_rainbow",figsize=(15,8))


X,y = toy_datasets.iris2d_toy_dataset()
iris_labels = ['Setosa' if yi == 0 else 'Versicolor' if yi == 1 else 'Virginica' for yi in y]
knn_plot(X,iris_labels,region_intensity=0.2,k=3,label_colors=["red", "green", "blue"],figsize=(15,8))


X,y = toy_datasets.balls_toy_dataset(seed=seed)
knn_plot(X,y,k=3,label_colors=['red','blue','green','orange','purple'],figsize=(15,8))

X = np.array([[3,1],[4,2],[3,4],[5,4],[-1,1],[1,2],[2,2],[3,3]])
y = np.array(['OK','OK','OK','OK','ERR','ERR','ERR','ERR'])
knn_plot(X,y,k=3,label_colors=["red","blue"],figsize=(15,8))

X = np.array([[0,0],[0,1],[1.1,0],[2,0]])
y = np.array(['RED','RED','BLUE','BLUE'])
knn_plot(X,y,k=3,label_colors=["red","blue"],figsize=(15,8))
    
X, y = digits_data()
selected = np.where(np.isin(y,[0,1,3,4,6,9]))[0]
X, y = X[selected,:], y[selected]
lda = LDA(num_dims = 2)
anmm = ANMM(num_dims = 2, n_friends = 1,n_enemies = 1)
alg = lda
alg.fit(X,y)
Xn = alg.transform(X)
if(Xn.shape[1] < 2):
    X2 = np.empty([Xn.shape[0],2])
    for i in range(Xn.shape[0]):
        X2[i,:] = [Xn[i,0],0.0]
    Xn = X2
knn_plot(Xn,y,k=3,cmap="gist_rainbow",region_intensity=0.4,legend_plot_points=True,figsize=(15,8))

svmc = svm.SVC()
X = np.array([[3,1],[4,2],[3,4],[5,4],[-1,1],[1,2],[2,2],[3,3]])
y = np.array(['OK','OK','OK','OK','ERR','ERR','ERR','ERR'])
classifier_plot(X,y,svmc,label_colors=['red','blue'],figsize=(15,8))

X, y = digits_data()
selected = np.where(np.isin(y,[0,1,3,4,6,9]))[0]
#X, y = X[selected,:], y[selected]
lda = LDA(num_dims = 2)
knn_plot(X,y,k=3,dml=lda,cmap="gist_rainbow",figsize=(15,8))

anmm = ANMM(num_dims = 2, n_friends = 1,n_enemies = 1)
f = knn_multiplot(X,y,nrow=2,ncol=2,ks=[1,1,11,11],dmls=[lda,anmm,lda,anmm],title="Comparación de DMLS",subtitles=["k=1, LDA", "k=1, ANMM", "k=11, LDA", "k=11, ANMM"],
              cmap="gist_rainbow",plot_points=True,figsize=(20,16))

