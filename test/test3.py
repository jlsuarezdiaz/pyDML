#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from six.moves import xrange
from sklearn.datasets import(load_iris, load_digits)


from sklearn.preprocessing import normalize, MinMaxScaler

from utils import(
    read_ARFF, kfold_multitester_supervised_knn, datasets)

from dml import(
    NCA,LDA,PCA,LMNN, KLMNN, ANMM,LSI,ITML,kNN,KANMM,KDA, DMLMJ, KDMLMJ, NCMML,NCMC, DML_eig,MCML,LDML)

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


#from utils import datasets
#X, y = datasets.iris()

#X,y = iris_data()
#X,y = digits_data()
#X,y = digits_red_data()
#X,y = sonar_data()
#X,y = wdbc_data()
#X,y = spambase_data()
#X,y = wine_data()
    
#X,y = datasets.iris()
X,y = datasets.sonar()
#X,y = datasets.balance()
#X,y = datasets.letters()
#X,y = datasets.wine()
#X,y = datasets.isolet('train')
#X,y = datasets.wdbc()
#X,y = datasets.spambase()
#X,y = datasets.digits(numbers=[0,1,3,4,6,9])
#X,y = datasets.digits()

#X = normalize(X,axis=0,norm='max')
mms = MinMaxScaler()
X = mms.fit_transform(X)

n,d = X.shape

print("Data dimensions: ", n, d)

#nca = NCA(max_iter=100, learning_rate = "adaptive",eta0=0.1, initial_transform = "scale", descent_method = "BGD")
lda = LDA()
#rca = RCA()
pca = PCA(thres=0.95)
lmnn = LMNN(max_iter=300,learning_rate = "adaptive", eta0 = 0.3, k = 5, mu = 0.5,soft_comp_interval = 1,tol=1e-8,prec=1e-8,eta_thres=1e-15)
lmnn_sgd = LMNN(num_dims=20,max_iter=300,learning_rate = "adaptive", eta0 = 0.001, k = 5, mu = 0.5,soft_comp_interval = 1,tol=1e-15,prec=1e-10,eta_thres=1e-15,solver="SGD")
klmnn = KLMNN(max_iter=100,learning_rate = "adaptive", eta0 = 0.3, k=5, mu = 0.5, tol=1e-15, prec=1e-15,eta_thres=1e-15,kernel='rbf',target_selection="kernel")
anmm = ANMM(num_dims = 10,n_friends = 5,n_enemies = 5)
itml = ITML(max_iter=100000,gamma=1.0, low_perc = 5, up_perc = 95)
nca_bgd = NCA(max_iter=100, learning_rate = "adaptive", eta0=0.3, descent_method = "BGD")
nca_sgd = NCA(max_iter=100, learning_rate = "adaptive", eta0=0.3, descent_method = "SGD",tol=1e-8,prec=1e-8)
lsi = LSI(supervised=True, err = 1e-4, itproj_err = 1e-4,max_proj_iter=20000)
kanmm = KANMM(num_dims=10,kernel='cosine',n_friends=5,n_enemies=3)
kda = KDA(kernel='rbf')
dmlmj = DMLMJ(num_dims=20,n_neighbors=5,alpha=0.001)
kdmlmj = KDMLMJ(num_dims=20,n_neighbors=5,alpha=0.001,kernel='rbf')
ncmml_sgd = NCMML(max_iter=300, learning_rate="adaptive", eta0=0.3, descent_method="SGD", tol=1e-15,prec=1e-15)
ncmml_bgd = NCMML(max_iter=300, learning_rate="adaptive", eta0=0.3, descent_method="BGD")
ncmc_sgd = NCMC(max_iter=300, learning_rate="adaptive",eta0=0.3,descent_method="SGD",centroids_num=2,tol=1e-15,prec=1e-15)
ncmc_bgd = NCMC(max_iter=300, learning_rate="adaptive",eta0=0.3,descent_method="BGD",centroids_num=2,tol=1e-15,prec=1e-15)
dml_eig = DML_eig(max_it=25)
mcml = MCML(eta0=0.01)
ldml = LDML(b=0.001,learning_rate='adaptive')
lmnn_sgd = LMNN(num_dims=2,solver="SGD",eta0=0.001)
#dmls = [itml,pca,lda,anmm,lsi,nca_bgd,nca_sgd,lmnn]
dmls = [lmnn_sgd]

results = kfold_multitester_supervised_knn(X,y,k = 5, n_neigh = 1, dmls = dmls, verbose = True,seed = 28)

print(results['time'])
print(results['train'])
print(results['test'])
