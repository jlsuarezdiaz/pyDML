#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Condensed Nearest Neighbors (CNN)

A module with undersampling utilities based on condensed neighbors.
"""

from __future__ import absolute_import
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y
from six.moves import xrange


class CondensedNearestNeighbors:

    def __init__(self):
        self.cnn_ = set()

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_, self.y_ = X, y
        n, d = X.shape
        additions = True
        self.cnn_ = {0}

        while additions:
            additions = False
            for i in xrange(1, n):
                xi, yi = X[i, :], y[i]
                cnn_arr = np.array(list(self.cnn_))
                cnnX = X[cnn_arr, :]
                cnnY = y[cnn_arr]
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(cnnX, cnnY)
                yi_pred = knn.predict(xi.reshape(1, -1))

                if yi != yi_pred:
                    self.cnn_.add(i)
                    additions = True

        return self

    def get_condensed_neighbors(self):
        cnn_arr = self.get_condensed_neighbor_indexes()
        return self.X_[cnn_arr, :], self.y_[cnn_arr]

    def get_condensed_neighbor_indexes(self):
        return np.array(list(self.cnn_))


class ReducedNearestNeighbors:

    def __init__(self):
        self.cnn_ = set()

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_, self.y_ = X, y
        n, d = X.shape

        cnn = CondensedNearestNeighbors()
        cnn.fit(X, y)
        self.rnn_ = set(cnn.get_condensed_neighbor_indexes())

        for i in self.rnn_:
            rnn_candidate = self.rnn_ - {i}
            rnn_arr = np.array(list(rnn_candidate))
            rnnX, rnnY = X[rnn_arr, :], y[rnn_arr]
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(rnnX, rnnY)
            ypred = knn.predict(X)
            if sum(ypred == y) == len(y):
                self.rnn_ = rnn_candidate

    def get_reduced_neighbors(self):
        rnn_arr = self.get_reduced_neighbor_indexes()
        return self.X_[rnn_arr, :], self.y_[rnn_arr]

    def get_reduced_neighbor_indexes(self):
        return np.array(list(self.rnn_))
