import unittest
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.datasets import(
    load_iris, load_digits, load_diabetes)
from numpy.testing import assert_array_almost_equal
from sklearn import neighbors

from src import(
    NCA,kNN)


data=load_iris()
#data=load_digits()
#data=load_diabetes()
X=data['data']
y=data['target']

n,d = X.shape

print "Data dimensions: ", n, d

dml = NCA(max_iter=100)
dml.fit(X,y)

knn = kNN(n_neighbors=5,dml_algorithm=dml)
knn.fit(X,y)

print "Before learning metric kNN score [Train]: ", knn.score_orig()
print "After learning metric kNN score [Train]: ", knn.score()

#XX = nca.transform(iris_points)

#knn = neighbors.KNeighborsClassifier(n_neighbors=3)
#knn.fit(XX,iris_labels)

