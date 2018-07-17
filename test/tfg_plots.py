#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:24:21 2018

@author: jlsuarezdiaz
"""

import numpy as np
import pandas as pd
from dml import PCA, LDA, NCA, kNN, ANMM, DMLMJ, MCML, Transformer, dml_plot, knn_plot, classifier_plot, classifier_pairplots, knn_pairplots, dml_pairplots, dml_multiplot, NCMC_Classifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib import rcParams
import seaborn as sns
from utils import toy_datasets, datasets
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.svm import SVC
from scipy.spatial import Voronoi, voronoi_plot_2d


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
#f.savefig('plots/svm_solution.png')

X,y = datasets.digits([0,1,3,4,6,9])
lda = LDA(num_dims=2)
anmm = ANMM(num_dims=2)
dmlmj = DMLMJ(num_dims=2)

alg=lda
f = knn_plot(X,y,k=3,dml=alg,cmap="gist_rainbow",figsize=(12,8))
#f.savefig('plots/ex_red_dim.png')

np.random.seed(28)
Xa = np.array([[i,0.0] for i in np.linspace(-10.0,10.0,40)])
Xb = np.array([[i,0.2] for i in np.linspace(-10.0,10.0,20)])
Xc = np.array([[i,-0.2] for i in np.linspace(-10.0,10.0,20)])
ya = ['A' for i in range(40)]
yb = ['B' for i in range(20)]
yc = ['C' for i in range(20)]
X=np.concatenate([Xa,Xb,Xc],axis=0)
y=np.concatenate([ya,yb,yc])

nca = NCA()
#f = knn_plot(X,y,k=1,dml=nca,cmap="gist_rainbow",figsize=(12,8),transform=False)
rcParams['text.usetex']='True'
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
#nca.fit(X,y)
#f = dml_multiplot(X,y,nrow=1,ncol=3,ks=[1,1,1],fitted=True,dmls=[None,nca,nca],transforms=[False,False,True],cmap="gist_rainbow",subtitles=[r'$M=\begin{pmatrix}1 & 0 \\ 0 & 1 \end{pmatrix}$',r'$M \approx \begin{pmatrix} 0 & -0.004 \\ -0.004 & 27.5 \end{pmatrix}$',r'$L \approx \begin{pmatrix} -0.0001 & 0.073 \\ -0.0008 & 5.24 \end{pmatrix}$'],figsize=(18,6))
#f.savefig('plots/ex_learning_nca.png')

np.random.seed(28)

X = [[i,0.2*np.random.random()-0.1] for i in np.linspace(-1,1,50)]
X = np.array(X)
L = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2], [np.sqrt(2)/2, np.sqrt(2)/2]])
y = np.array([1 for i in np.linspace(-1,1,50)])

f, ax = plt.subplots(nrows=1,ncols=3,sharex='col',sharey='row',figsize=(18,6))

LX = X.dot(L.T)
ax[0].set_xlim((-1.2,1.2))
ax[0].set_ylim((-1.2,1.2))
ax[0].scatter(LX[:,0],LX[:,1])  
ax[0].set_title(r'$L=\begin{pmatrix}1 & 0 \\ 0 & 1 \end{pmatrix}$')


pca = PCA(num_dims = 2)
pca.fit(LX,y)
#LL = pca.transformer()
#vv = pca.skpca_.explained_variance_
#draw_vector([0,0], 3*LL[:,0]*vv[0],ax[0,0],'red')
#draw_vector([0,0], 3*LL[:,1]*vv[1],ax[0,0],'green')

ax[1].set_xlim((-1.2,1.2))
ax[1].set_ylim((-1.2,1.2))
ax[1].set_title(r'$L=\begin{pmatrix}\sqrt{2}/2 & \sqrt{2}/2 \\ \sqrt{2}/2 & -\sqrt{2}/2 \end{pmatrix}$')
LLX = pca.transform()
ax[1].scatter(LLX[:,0],LLX[:,1])

pca2 = PCA(num_dims=1)
pca2.fit(LX,y)
LLX2 = pca2.transform()
LLXy = [0 for i in range(LLX2.size)]
ax[2].set_xlim((-1.2,1.2))
ax[2].set_ylim((-1.2,1.2))
ax[2].set_title(r'$L=\begin{pmatrix}\sqrt{2}/2 & \sqrt{2}/2\end{pmatrix}$')
ax[2].scatter(LLX2[:,0],LLXy)
#f.savefig('plots/ex_mover_ejes.png')

np.random.seed(28)
X,y = toy_datasets.circular_toy_dataset(rads = [1,2], samples = [200,200], noise = [0.0,0.0])
yy = y.astype(str)
yy[40:200] = '?'
yy[220:] = '?'
knn1 = KNeighborsClassifier(1)
knn1.fit(X[np.isin(yy,['0','1'])],yy[np.isin(yy,['0','1'])])
knn2 = KNeighborsClassifier(1)
knn2.fit(X,y)
f = dml_multiplot(X,yy,clfs=[knn1,knn2],label_colors=['red','blue','lightgray'],fitted=True,figsize=(12,6))
#f.savefig('plots/ssl.png')

np.random.seed(28)
X, y = datasets.iris()
y = ['Setosa' if yi == 0 else 'Versicolor' if yi == 1 else 'Virginica' for yi in y]
X = pd.DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
arr = ['Sepal Length','Sepal Width']
X = X[arr]
f = dml_multiplot(X,y,ks=[1,30],cmap='rainbow',subtitles=[r'$k=1$',r'$k=30$'],figsize=(12,6))
#f.savefig('plots/compare_knn.png')

np.random.seed(28)
X = [[np.random.random(),np.random.random()] for i in range(25)]
vor = Voronoi(X)
voronoi_plot_2d(vor)
#plt.savefig('plots/voronoi.png')