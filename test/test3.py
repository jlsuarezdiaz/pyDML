import unittest
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.datasets import(
    load_iris, load_digits, load_diabetes)
from numpy.testing import assert_array_almost_equal
from sklearn import neighbors

from utils import(
    read_ARFF, kfold_multitester_supervised_knn)

from dml import(
    NCA,LDA,RCA,PCA,LMNN,ANMM,kNN)


#data=load_iris()
#data=load_digits()
#
#X=data['data']
#y=data['target']

#X,y,m = read_ARFF("./data/sonar.arff",-1)
#X,y,m = read_ARFF("./data/wdbc.arff",0)
X,y,m = read_ARFF("./data/spambase-460.arff",-1)

n,d = X.shape

print "Data dimensions: ", n, d

#nca = NCA(max_iter=10,num_dims=10)
lda = LDA(thres = 0.95)
#rca = RCA()
pca = PCA(thres = 0.95)
lmnn = LMNN()
anmm = ANMM(n_friends = 1,n_enemies = 1)

results = kfold_multitester_supervised_knn(X,y,k = 5, n_neigh = 3, dmls = [lda,pca,lmnn,anmm])

print results