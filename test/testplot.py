#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:43:34 2018

@author: jlsuarezdiaz
"""

from utils import(
        read_ARFF, kfold_multitester_supervised_knn, toy_datasets,datasets)

from dml import(
    NCA,LDA,PCA,LMNN,ANMM,LSI, NCMML,kNN, knn_plot, dml_plot, classifier_plot, classifier_pairplots, knn_pairplots, dml_pairplots, dml_multiplot)

from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
plt.savefig('plots/plot1.png')

X,y = toy_datasets.hiperplane_toy_dataset(seed=seed)
knn_plot(X,y,k=3,cmap="gist_rainbow",figsize=(15,8))
plt.savefig('plots/plot2.png')

X,y = toy_datasets.iris2d_toy_dataset()
iris_labels = ['Setosa' if yi == 0 else 'Versicolor' if yi == 1 else 'Virginica' for yi in y]
knn_plot(X,iris_labels,region_intensity=0.2,k=3,label_colors=["red", "green", "blue"],figsize=(15,8))
plt.savefig('plots/plot3.png')

X,y = toy_datasets.balls_toy_dataset(seed=seed)
knn_plot(X,y,k=3,label_colors=['red','blue','green','orange','purple'],figsize=(15,8))
plt.savefig('plots/plot4.png')

X = np.array([[3,1],[4,2],[3,4],[5,4],[-1,1],[1,2],[2,2],[3,3]])
y = np.array(['OK','OK','OK','OK','ERR','ERR','ERR','ERR'])
knn_plot(X,y,k=3,label_colors=["red","blue"],figsize=(15,8))
plt.savefig('plots/plot5.png')

X = np.array([[0,0],[0,1],[1.1,0],[2,0]])
y = np.array(['RED','RED','BLUE','BLUE'])
knn_plot(X,y,k=3,label_colors=["red","blue"],figsize=(15,8))
plt.savefig('plots/plot6.png')    

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
plt.savefig('plots/plot7.png')

svmc = svm.SVC()
X = np.array([[3,1],[4,2],[3,4],[5,4],[-1,1],[1,2],[2,2],[3,3]])
y = np.array(['OK','OK','OK','OK','ERR','ERR','ERR','ERR'])

classifier_plot(X,y,svmc,label_colors=['red','blue'],figsize=(15,8))
plt.savefig('plots/plot8.png')

X, y = digits_data()
selected = np.where(np.isin(y,[0,1,3,4,6,9]))[0]
X, y = X[selected,:], y[selected]
lda = LDA(num_dims = 2)
knn_plot(X,y,k=3,dml=lda,cmap="gist_rainbow",figsize=(15,8))
plt.savefig('plots/plot9.png')

anmm = ANMM(num_dims = 2, n_friends = 1,n_enemies = 1)

f = dml_multiplot(X,y,nrow=2,ncol=2,ks=[1,1,11,11],dmls=[lda,anmm,lda,anmm],title="Comparación de DMLS",subtitles=["k=1, LDA", "k=1, ANMM", "k=11, LDA", "k=11, ANMM"],
              cmap="gist_rainbow",plot_points=True,figsize=(20,16))
plt.savefig('plots/plot10.png')

X,y = digits_data()
lda = LDA(num_dims = 5)
anmm = ANMM(num_dims=5)
X = anmm.fit_transform(X,y)
knn = KNeighborsClassifier()
f1 = classifier_pairplots(X,y,knn,sections="zeros",cmap="gist_rainbow",figsize=(25,25))
plt.savefig('plots/plot11.png')
f2 = classifier_pairplots(X,y,knn,sections="mean",cmap="gist_rainbow",figsize=(25,25))
plt.savefig('plots/plot12.png')
#f1.savefig("./a.png")
#f2.savefig("./b.png")

knn_pairplots(X,y,k=3,dml=lda,cmap="gist_rainbow",figsize=(25,25))
plt.savefig('plots/plot13.png')

X, y = datasets.iris()
iris_labels = ['Setosa' if yi == 0 else 'Versicolor' if yi == 1 else 'Virginica' for yi in y]
X = pd.DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
svmc = svm.SVC()

classifier_pairplots(X,iris_labels,svmc,cmap="gist_rainbow",figsize=(20,20))
plt.savefig('plots/plot14.png')
classifier_pairplots(X,iris_labels,svmc,xattrs=['Sepal Length','Sepal Width'],yattrs=['Petal Length','Petal Width'],cmap="gist_rainbow",figsize=(20,20))
plt.savefig('plots/plot15.png')

X=np.array([[0.0,0.1],[0.5,0.1],[-0.5,-0.1],[1.0,0.2],[-1.0,-0.1],[0.1,1.0],[-0.1,1.0],[0.1,-1.0],[-0.1,-1.0]])
y=np.array([0,0,0,0,0,1,1,2,2])
lmnn = LMNN(max_iter=1000,learning_rate = "adaptive",eta0=1.0, k = 1, mu = 0.5,tol=1e-15,prec=1e-15)
dml_multiplot(X,y,2,2,ks=[1,1,1],dmls=[None,lmnn,lmnn],transforms=[True,True,False],title="LMNN",subtitles=["Original","Transformados","Región LMNN+KNN"],cmap="rainbow",figsize=(20,20))
plt.savefig('plots/plot16.png')

knn_plot(X,y,k=1,cmap="gist_rainbow",figsize=(15,8))
plt.savefig('plots/plot17.png')
knn_plot(X,y,k=1,cmap="gist_rainbow",figsize=(15,8),transformer=np.array([[0.0, 0.0],[0.0,3.0]]),transform=False)
plt.savefig('plots/plot18.png')

X,y = datasets.iris()
X=X[:,[0,2]]
dml=NCMML()
clf=NearestCentroid()
dml_plot(X,y,clf,cmap="gist_rainbow",figsize=(15,8))
plt.savefig('plots/plot19.png')
dml_plot(X,y,dml=dml,clf=clf,cmap="gist_rainbow",figsize=(15,8))
plt.savefig('plots/plot20.png')
dml_pairplots(X,y,dml=dml,clf=clf,cmap="gist_rainbow",figsize=(15,8))
plt.savefig('plots/plot21.png')