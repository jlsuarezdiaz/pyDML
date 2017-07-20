#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
k-Nearest Neighbors (kNN)

An interface for kNN adapted to distance metric learning algorithms.
"""

from __future__ import absolute_import
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from .dml_algorithm import DML_Algorithm


class kNN:
    def __init__(self,n_neighbors,dml_algorithm):
        self.nn_ = n_neighbors
        self.dml = dml_algorithm
        self.knn = neighbors.KNeighborsClassifier(n_neighbors)
        self.knn_orig = neighbors.KNeighborsClassifier(n_neighbors)


    def fit(self,X,y):
        """
        Adjusts a data set to be used on predictions.
        """
        self.X_ = X    # Training set.
        self.trX = self.dml.transform(X)   # Transformed training set.
        self.y_ = y # Training labels

        self.knn.fit(self.trX,self.y_)    # kNN with learnt metric
        self.knn_orig.fit(self.X_,self.y_) # kNN with euclidean metric

        self.num_labels=len(set(y))

        return self

    def predict(self,X=None):
        """
        Predicts the labels for the given data. If no set is specified, training set will be used. Model needs to be fitted.
        """
        if X is None:
            return self.loo_pred(self.trX)
        else:
            X= self.dml.transform(X)

        return self.knn.predict(X)


    def predict_orig(self,X=None):
        """
        Predicts the labels for the given data without transformations. If no set is specified, training set will be used. Model needs to be fitted.
        """
        if X is None:
            return self.loo_pred(self.X_)

        return self.knn_orig.predict(X)

    def predict_proba(self,X=None):
        """
        Predicts label probabilities for the given data. If no set is specified, training set will be used. Model needs to be fitted.
        """
        if X is None:
            return self.loo_prob(self.trX)
        else:
            X= self.dml.transform(X)

        return self.knn.predict_proba(X)

    def predict_proba_orig(self,X=None):
        """
        Predicts label probabilities for the given data without transformations. If no set is specified, training set will be used. Model needs to be fitted.
        """
        if X is None:
            return self.loo_prob(self.X_)

        return self.knn_orig.predict_proba(X)

    def score(self,X=None,y=None):
        """
        Obtains a classification score for the given data. If no set is specified, label will be ignored and training set and labels will be used.
        """
        if X is None:
            return self.loo_score(self.trX)
        else:
            X=self.dml.transform(X)

        return self.knn.score(X,y)

    def score_orig(self,X=None,y=None):
        """
        Obtains a classification score for the given data without transformations. If no set is specified, label will be ignored and training set and labels will be used.
        """
        if X is None:
            return self.loo_score(self.X_)

        return self.knn_orig.score(X,y)

    def loo_prob(self,X):
        """
        Obtains labels probability matrix for data in the training set using Leave One Out.
        """
        loo= LeaveOneOut()
        probs = np.empty([self.y_.size,self.num_labels])

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = self.y_[train_index], self.y_[test_index]

            knnloo = neighbors.KNeighborsClassifier(self.nn_)
            knnloo.fit(X_train,y_train)

            probs[test_index,:]=knnloo.predict_proba(X_test)

        return probs

    def loo_pred(self,X):
        """
        Obtains predicted labels for data in the training set using Leave One Out.
        """
        loo = LeaveOneOut()
        preds = np.empty(self.y_.size)

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = self.y_[train_index], self.y_[test_index]

            knnloo = neighbors.KNeighborsClassifier(self.nn_)
            knnloo.fit(X_train,y_train)

            preds[test_index]=knnloo.predict(X_test)

        return preds

    def loo_score(self,X):
        """
        Obtains score of the training set using leave one out.
        """
        preds = self.loo_pred(X)

        return np.mean(preds == self.y_)







