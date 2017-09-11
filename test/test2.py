import unittest
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.datasets import(
    load_iris, load_digits, load_diabetes)
from numpy.testing import assert_array_almost_equal
from sklearn import neighbors

from utils import(
    read_ARFF, kfold_tester_supervised_knn)

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

#dml = NCA(max_iter=10,num_dims=10)
#dml = LDA(thres = 0.95)
#dml = RCA()
#dml = PCA()
#dml = LMNN()
dml = ANMM(n_friends = 3,n_enemies = 3)

m = kfold_tester_supervised_knn(X,y,k = 5, n_neigh = 3, dml = dml)

print m