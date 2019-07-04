#!/usr/bin/env python
# -*- coding: utf-8 -*-
# distutils: language=c++
# cython: profile=True

"""
Iterative Metric Learning and Samples selection (IMLS)

"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.base import BaseEstimator, ClassifierMixin

from .dml_utils import calc_outers, calc_outers_i
from .dml_algorithm import DML_Algorithm
from .lmnn import LMNN
from .knn import kNN

from libcpp cimport bool

# from collections import defaultdict
import time

cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cimport cython


def jaccard_similarity(x, y):
    """
        Jaccard similarity for two sets.

        Parameters
        ----------

        x: 1D array-like

            First set.

        y: 1D array-like

            Second set.


        Returns
        -------

        Jaccard similarity between x and y, given as the length of the union divided by the length of the intersection.

    """
    cap = np.intersect1d(x, y, assume_unique=True)
    cup = np.union1d(x, y)
    return (< float > len(cap)) / len(cup)


class IMLS(DML_Algorithm, BaseEstimator, ClassifierMixin):
    """
        Iterative Metric Learning and Samples selection (IMLS)

        A Distance Metric Learning that learns a metric iteratively and then selects an appropiate subset of training samples for classification.
    """

    def __init__(self, dml=LMNN(), cache_iterations=10, max_iterations=10, similarity_thresh=0.8, sample_selection_instances=10, learner=None):
        self.dml_ = dml
        self.cache_its_ = cache_iterations
        self.max_its_ = max_iterations
        self.sim_thresh_ = similarity_thresh
        self.samples_ = sample_selection_instances
        self.clf_ = learner

    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        L : (d'xd) matrix, where d' is the desired output dimension and d is the number of features.
        """
        return self.L_

    def fit(self, X, y):
        cached_transforms = []
        X, y = check_X_y(X, y)
        n, d = X.shape
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        if isinstance(self.samples_, int):
            val = self.samples_
            self.samples_ = {c: val for c in self.classes_}
        cdef L = np.eye(d)
        cdef Li = L
        cdef iX = X
        cdef int cache_its = self.cache_its_
        cdef dml = self.dml_

        for i in xrange(cache_its):
            # print("Iteration " + str(i))
            dml.fit(iX, y)
            Li = dml.transformer()
            L = Li.dot(L)
            cached_transforms.append(L)
            iX = dml.transform(iX)

        self.L_ = L
        self.cached_transforms_ = cached_transforms
        return self

    def predict(self, X=None):
        if self.L_ is None:
            raise RuntimeError("The distance metric must be fitted before predicting.")
        if self.clf_ is None:
            raise RuntimeError("A classifier must be specified to make predictions.")

        if X is None:
            X = self.X_

        cdef max_its = self.max_its_
        cdef float sim_thresh = self.sim_thresh_
        cdef bool stop = False
        cdef int num_its = 0
        cdef iold = None
        Xtra, ytra = self.X_, self.y_
        n, d = X.shape
        ypred = np.empty([n])

        for i, xi in enumerate(X):
            stop = False
            num_its = 0
            iold = None
            while not stop:
                # print("Iteration " + str(num_its))
                xi = xi.reshape(1, -1)
                if num_its < self.cache_its_:
                    Li = self.cached_transforms_[num_its]
                    iX = Xtra.dot(Li.T)
                    Lxi = xi.dot(Li.T)
                else:  # We train another dml iteration and cache the result.
                    L = self.L_
                    iX = Xtra.dot(L.T)
                    self.dml_.fit(iX, self.y_)
                    Li = self.dml_.transformer()
                    self.L_ = Li.dot(L)
                    self.cached_transforms_.append(L)
                    iX = self.dml_.transform(iX)
                    Lxi = xi.dot(L.T)
                    self.cache_its_ = num_its + 1

                iloc = self._get_sample_selection(iX, Lxi, ytra)
                Xloc, yloc = iX[iloc, :], ytra[iloc]
                # print(Xloc)
                self.clf_.fit(Xloc, yloc)
                ypred[i] = self.clf_.predict(Lxi)
                # print(num_its, ypred[i])
                num_its += 1
                # print(ytra[iloc])
                stop = ((iold is not None) and jaccard_similarity(iloc, iold) >= sim_thresh) or num_its == max_its
                iold = iloc

        return ypred

    def predict_proba(self, X=None):
        if self.L_ is None:
            raise RuntimeError("The distance metric must be fitted before predicting.")
        if self.clf_ is None:
            raise RuntimeError("A classifier must be specified to make predictions.")

        if X is None:
            X = self.X_

        cdef max_its = self.max_its_
        cdef float sim_thresh = self.sim_thresh_
        cdef bool stop = False
        cdef int num_its = 0
        cdef iold = None
        Xtra, ytra = self.X_, self.y_
        n, d = X.shape
        yprob = np.empty([n, len(self.classes_)])

        for i, xi in enumerate(X):
            stop = False
            num_its = 0
            iold = None
            while not stop:
                # print("Iteration " + str(num_its))
                xi = xi.reshape(1, -1)
                if num_its < self.cache_its_:
                    Li = self.cached_transforms_[num_its]
                    iX = Xtra.dot(Li.T)
                    Lxi = xi.dot(Li.T)
                else:  # We train another dml iteration and cache the result.
                    L = self.L_
                    iX = Xtra.dot(L.T)
                    self.dml_.fit(iX, self.y_)
                    Li = self.dml_.transformer()
                    self.L_ = Li.dot(L)
                    self.cached_transforms_.append(L)
                    iX = self.dml_.transform(iX)
                    Lxi = xi.dot(L.T)
                    self.cache_its_ = num_its + 1

                iloc = self._get_sample_selection(iX, Lxi, ytra)
                Xloc, yloc = iX[iloc, :], ytra[iloc]
                self.clf_.fit(Xloc, yloc)
                yprob[i, :] = self.clf_.predict_proba(Lxi)
                #print(self.clf_.predict_proba(xi))
                #print(self.clf_.predict(xi))
                num_its += 1
                stop = ((iold is not None) and jaccard_similarity(iloc, iold) >= sim_thresh) or num_its == max_its
                iold = iloc

        return yprob

    def score(self, X=None, y=None):
        if y is None:
            y = self.y_
        return np.mean(self.predict(X) == y)

    def _get_sample_selection(self, Lx, xi, y):
        distance_matrix = pairwise_distances(X=Lx, Y=xi, n_jobs=-1)
        sample = np.array([], dtype=int)
        for c in self.classes_:
            mask_c = np.flatnonzero(y == c)
            dists_c = np.array([mask_c, distance_matrix[mask_c, 0], y[y == c]]).T
            dists_c = np.array(sorted(dists_c, key=lambda v: v[1]))
            #print(dists_c)
            sample = np.append(sample, dists_c[:min(self.samples_[c], dists_c.shape[0]), 0].astype(int))

        return sample
