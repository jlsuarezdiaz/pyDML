#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:25:50 2018

@author: jlsuarezdiaz
"""

from utils import(
        read_ARFF, kfold_multitester_supervised_knn, toy_datasets, datasets)

from dml import(
    NCA,LDA,PCA,LMNN,ANMM,LSI,kNN, knn_plot, dml_multiplot, classifier_plot, ITML, KANMM,KDA, DMLMJ, KDMLMJ, NCMML, NCMC, KLMNN, DML_eig, MCML, LDML)

import numpy as np

from sklearn.preprocessing import normalize, MinMaxScaler

seed=28
np.random.seed(seed)

#X,y = toy_datasets.digits_toy_dataset(dims=[0,1],numbers=[0,1,3,4,6,9])
#X,y = datasets.digits(numbers=[0,1,3,4,6,9])
#X,y = toy_datasets.circular_toy_dataset(rads=[1,2,3],samples=[200,200,200],noise=[0.2,0.2,0.2],seed=seed)
#X,y = toy_datasets.hiperplane_toy_dataset(seed=seed)
#X,y = toy_datasets.iris2d_toy_dataset()
#X,y = toy_datasets.balls_toy_dataset(seed=seed)
#X,y = toy_datasets.single_toy_dataset(seed=seed)

#X, y = datasets.balance()
#mms = MinMaxScaler()
#X = mms.fit_transform(X)

X = np.array([[3,1],[4,2],[3,4],[5,4],[-1,1],[1,2],[2,2],[3,3]])
y = np.array([1,1,1,1,-1,-1,-1,-1])

#X = np.array([[0.0,0.0],[1.0,0.0],[2.0,0.0],[3.0,0.0],[4.0,0.0],[5.0,0.0],[6.0,0.0],[7.0,0.0],[8.0,0.0],[9.0,0.0],
#             [0.0,0.01],[1.0,0.01],[2.0,0.01],[3.0,0.01],[4.0,0.01],[5.0,0.01],[6.0,0.01],[7.0,0.01],[8.0,0.01],[9.0,0.01]])
#y = np.array([1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

#X = np.concatenate([np.array([(i/100,-0.01,0) for i in range(100)]),np.array([(i/100,-0.02,0) for i in range(100)])])
#y = np.array([1 for i in range(100)]+[-1 for i in range(100)])
#sq22 = np.sqrt(2)/2
#L = np.array([[-sq22, sq22],[sq22, sq22]])

#X = np.array([[0,0],[0,1],[2,0],[3,0]])
#y = np.array(["A","A","B","B"])
#X = X.dot(L.T) #+ np.array([2,2])

#X, y = datasets.iris()

#X=np.array([[0.0,0.1],[0.5,0.1],[-0.5,-0.1],[1.0,0.2],[-1.0,-0.1],[0.1,1.0],[-0.1,-1.0]])
#y=np.array([0,0,0,0,0,1,2])

#X=np.array([[0.0,1],[1,0.0]])
#y=np.array([0,1])
#toy_datasets.toy_plot(X,y)
#knn_plot(X,y,k=5,figsize=(15,8),cmap="gist_rainbow")

#K = np.array([[20.11, -3.02],[70.22,0.63]])
#X = X.dot(K.T)
knn_plot(X,y,k=1,figsize=(15,8),cmap="gist_rainbow")

nca = NCA(max_iter=10000, learning_rate = "adaptive",eta0=0.1, descent_method = "BGD", tol=1e-3)
lda = LDA()
#rca = RCA()
pca = PCA(thres = 0.95)
lmnn = LMNN(max_iter=1000,learning_rate = "adaptive",eta0=1.0, k = 5, mu = 0.5,tol=1e-15,prec=1e-15)
lmnn_sgd = LMNN(max_iter=300,learning_rate = "adaptive", eta0 = 0.001, k = 1, mu = 0.5,soft_comp_interval = 1,tol=1e-15,prec=1e-10,eta_thres=1e-15,solver="SGD")
klmnn = KLMNN(max_iter=100,learning_rate = "adaptive", eta0 = 0.001, k=1, mu = 0.5, tol=1e-15, prec=1e-15,eta_thres=1e-15,kernel='rbf',target_selection="kernel")#,initial_metric=np.array([[0,0],[0,1.0],[0.5,0.0],[0,0]]).T)
anmm = ANMM(num_dims = 2, n_friends = 1,n_enemies = 1)
kanmm = KANMM(num_dims = 2, kernel='linear',n_friends = 1, n_enemies=1)
lsi = LSI(supervised=True, err = 1e-10, itproj_err = 1e-3)
kda = KDA(kernel='linear',degree=2,coef0=0)
dmlmj = DMLMJ(num_dims=3,n_neighbors=5,alpha=0.001)
kdmlmj = KDMLMJ(num_dims=2,n_neighbors=5,alpha=0.001,kernel='rbf')
ncmml = NCMML(max_iter=300, learning_rate="adaptive", eta0=0.3, descent_method="SGD", tol=1e-15,prec=1e-15)
ncmc = NCMC(max_iter=300, learning_rate="adaptive",eta0=0.01,descent_method="SGD",centroids_num=2,tol=1e-15,prec=1e-15)
itml = ITML(gamma=1.0)
dml_eig = DML_eig()
mcml = MCML()
ldml = LDML(learning_rate='adaptive')

alg = mcml

#alg.fit(X,y)
#Xn = alg.transform(X) 
#if(Xn.shape[1] < 2):
#    X2 = np.empty([Xn.shape[0],2])
#    for i in range(Xn.shape[0]):
#        X2[i,:] = [Xn[i,0],0.0]
#    Xn = X2

#toy_datasets.toy_plot(Xn,y)
knn_plot(X,y,k=1,figsize=(15,8),cmap="gist_rainbow",transform=False,dml=alg)
knn_plot(X,y,k=1,figsize=(15,8),cmap="gist_rainbow",dml=alg)
#dml_multiplot(X,y,ks=[1,1],dmls=[lmnn,anmm],figsize=(15,8),cmap="rainbow")

#classifier_plot(X,y,lmnn,figsize=(15,8),cmap="gist_rainbow")
#results = kfold_multitester_supervised_knn(X,y,k = 3, n_neigh = 1, dmls = [dmlmj], verbose = True,seed = 28)

#print(results['time'])
#print(results['train'])
#print(results['test'])
