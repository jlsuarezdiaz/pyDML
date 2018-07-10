#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:24:21 2018

@author: jlsuarezdiaz
"""

import numpy as np
import pandas as pd
from dml import PCA, LDA, MCML, Transformer, dml_plot, classifier_plot, classifier_pairplots, knn_pairplots, dml_pairplots, dml_multiplot, NCMC_Classifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
from utils import toy_datasets
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

def draw_vector(v0, v1, ax=None,col='black'):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0,color=col)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

np.random.seed(28)
L = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2], [np.sqrt(2)/2, np.sqrt(2)/2]])

X = [[i,0.5*np.random.random()-0.25] for i in np.linspace(-1,1,50)]
X = np.array(X)

y = np.array([1 for i in np.linspace(-1,1,50)])

f, ax = plt.subplots(nrows=2,ncols=2,sharex='col',sharey='row',figsize=(12,12))
f.tight_layout()
LX = X.dot(L.T)
ax[0,0].set_xlim((-1,1))
ax[0,0].set_ylim((-1,1))
ax[0,0].scatter(LX[:,0],LX[:,1])  



pca = PCA(num_dims = 2)
pca.fit(LX,y)
LL = pca.transformer()
vv = pca.skpca_.explained_variance_
draw_vector([0,0], 3*LL[:,0]*vv[0],ax[0,0],'red')
draw_vector([0,0], 3*LL[:,1]*vv[1],ax[0,0],'green')

ax[0,1].set_xlim((-1,1))
ax[0,1].set_ylim((-1,1))
LLX = pca.transform()
ax[0,1].scatter(LLX[:,0],LLX[:,1])

pca2 = PCA(num_dims=1)
pca2.fit(LX,y)
LLX2 = pca2.transform()
LLXy = [0 for i in range(LLX2.size)]
ax[1,0].set_xlim((-1,1))
ax[1,0].set_ylim((-1,1))
ax[1,0].scatter(LLX2[:,0],LLXy)

UWX = pca2.skpca_.inverse_transform(LLX2)
ax[1,1].set_xlim((-1,1))
ax[1,1].set_ylim((-1,1))
ax[1,1].scatter(UWX[:,0],UWX[:,1])
ax[1,1].scatter(LX[:,0],LX[:,1],c='lightblue')

#plt.savefig('plots/pca.png')

np.random.seed(28)
#X = [[i,0.5*np.random.random() + 0.2] for i in np.linspace(-1,1,50)]+ [[i,0.5*np.random.random()-0.7] for i in np.linspace(-1,1,50)]
#X = np.array(X)
#y = np.array([1 for i in range(50)]+[-1 for i in range(50)])
X,y = toy_datasets.balls_toy_dataset(centers=[[0.0,0.35],[0.0,-0.35]],rads=[0.3,0.3],samples=[50,50],noise=[0.0,0.0])

M = np.array([[3.0,0],[0,1]])
aux = X.dot(M.T)
LX = aux.dot(L.T)

f, ax = plt.subplots(nrows=1,ncols=2,sharex='col',sharey='row',figsize=(12,6))
f.tight_layout()
ax[0].set_xlim(-1,1)
ax[0].set_ylim(-1,1)
ax[1].set_ylim(-1,1)
ax[1].set_xlim(-6,6)

ax[0].scatter(LX[:,0],LX[:,1],c=y,cmap='rainbow')

pca = PCA()
lda = LDA()

pca.fit(LX)
lda.fit(LX,y)
LLX = lda.transform()
LL = lda.transformer()[0,:]
LLp = pca.transformer()[0,:]
LLXy = [0 for i in range(LLX.size)]
ax[1].plot([-10,10],[0,0],c='green')
sc = ax[1].scatter(LLX[:,0],LLXy,c=y,cmap='rainbow')
ax[0].plot([-LL[0],LL[0]],[-LL[1],LL[1]],c='green')
ax[0].plot([-2*LLp[0],2*LLp[0]],[-2*LLp[1],2*LLp[1]],c='orange')
handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="",marker="o", mec="k")[0] for c in [-1,1]]
f.legend(handles,['A','B'],loc = "lower right")


#plt.savefig('plots/lda.png')

np.random.seed(28)
X,y = toy_datasets.balls_toy_dataset(centers=[[-1.0,0-0],[0.0,0.0],[1.0,0.0]],rads=[0.3,0.3,0.3],samples=[50,50,50],noise=[0.1,0.1,0.1])
y[y==2]=0
y=y.astype(int)
#f, ax = plt.subplots(nrows=1,ncols=2,sharex='col',sharey='row',figsize=(12,6))
#ax[0].set_xlim(-1.5,1.5)
#ax[0].set_ylim(-1.5,1.5)
#ax[1].set_ylim(-1.5,1.5)
#ax[1].set_xlim(-1.5,1.5)
#f.tight_layout()
#ax[0].scatter(X[:,0],X[:,1],c=y,cmap='rainbow')
ncm = NearestCentroid()
ncmc = NCMC_Classifier(centroids_num=[2,1])
f=dml_multiplot(X,y,nrow=1,ncol=2,clfs=[ncm,ncmc],cmap='rainbow',subtitles=['NCM','NCMC'],figsize=(12,6))
#f.savefig('plots/ncm_problem.png')

np.random.seed(28)
svm = SVC(kernel='linear')
X,y = toy_datasets.hiperplane_toy_dataset(ws=[[1,1]],bs=[[0,0]],nsamples = 100,noise=0.0)
y=y.astype(int)
X[y==1] += np.array([0.1,0.1])
X[y==0] -= np.array([0.1,0.1])
f = classifier_plot(X,y,svm,cmap='rainbow',figsize=(6,6))
#f.savefig('plots/svm_example.png')

np.random.seed(28)
Xa = np.array([[1.8*np.random.random() + -0.9,0.0] for i in range(40)])
Xb = np.array([[0.4*np.random.random()+1.1,0.0] for i in range(20)])
Xc = np.array([[0.4*np.random.random()-1.5,0.0] for i in range(20)])
X=np.concatenate([Xa,Xb,Xc],axis=0)
y=np.empty(X.shape[0])
y[np.abs(X[:,0]) > 1] = 1
y[np.abs(X[:,0]) < 1] = -1
y = y.astype(int)
f,ax=plt.subplots(sharex='row',sharey='row',figsize=(6,3))
ax.scatter(X[:,0],X[:,1],c=y,cmap='rainbow',s=20,edgecolor='k')
handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="",marker="o", mec="k")[0] for c in [-1,1]]
ax.legend(handles,[-1,1],loc = "lower right")
#plt.savefig('plots/svm_problem.png')

X[:,1] = X[:,0]*X[:,0]
L=np.array([[1,0],[0,0]])
proj = Transformer(L)
svq = SVC(kernel='poly',degree=2)
f=dml_multiplot(X,y,nrow=1,ncol=2,clfs=[svm,svq],transformers=[None,L],transforms=[False,True],cmap='rainbow',figsize=(12,6))
f.savefig('plots/svm_solution.png')