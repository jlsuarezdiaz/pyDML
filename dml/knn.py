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
    """
        k-Nearest Neighbors (kNN)
        The nearest neighbors classifier adapted to be used with distance metric learning algorithms.

        Parameters
        ----------

        n_neighbors : int

            Number of neighbors to consider in classification.

        dml_algorithm : DML_Algorithm

            The distance metric learning algorithm that will provide the distance in kNN.
    """
    def __init__(self,n_neighbors,dml_algorithm):
        self.nn_ = n_neighbors
        self.dml = dml_algorithm
        self.knn = neighbors.KNeighborsClassifier(n_neighbors)
        self.knn_orig = neighbors.KNeighborsClassifier(n_neighbors)


    def fit(self,X,y):
        """
        Fit the model from the data in X and the labels in y.

        Parameters
        ----------
        X : array-like, shape (N x d)
            Training vector, where N is the number of samples, and d is the number of features.

        y : array-like, shape (N)
            Labels vector, where N is the number of samples.

        Returns
        -------
        self : object
            Returns the instance itself.
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
        Predicts the labels for the given data. Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        Returns
        -------

        y : 1D-Array

            The vector with the label predictions.
        """
        if X is None:
            return self.loo_pred(self.trX)
        else:
            X = self.dml.transform(X)

        return self.knn.predict(X)

    def predict_orig(self, X=None):
        """
        Predicts the labels for the given data with the Euclidean distance (with no dml transformations). Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        Returns
        -------

        y : 1D-Array

            The vector with the label predictions.
        """
        if X is None:
            return self.loo_pred(self.X_)

        return self.knn_orig.predict(X)

    def predict_proba(self, X=None):
        """
        Predicts the probabilities for the given data. Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        Returns
        -------

        T : 2D-Array, shape (N x c)

            A matrix with the probabilities for each class. N is the number of samples and c is the number of classes.
            The element i, j shows the probability of sample X[i] to be in class j.
        """
        if X is None:
            return self.loo_prob(self.trX)
        else:
            X = self.dml.transform(X)

        return self.knn.predict_proba(X)

    def predict_proba_orig(self, X=None):
        """
        Predicts the probabilities for the given data with euclidean distance (with no dml transformations). Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        Returns
        -------

        T : 2D-Array, shape (N x c)

            A matrix with the probabilities for each class. N is the number of samples and c is the number of classes.
            The element i, j shows the probability of sample X[i] to be in class j.
        """
        if X is None:
            return self.loo_prob(self.X_)

        return self.knn_orig.predict_proba(X)

    def score(self, X=None, y=None):
        """
        Obtains the classification score for the given data. Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        y : 1D-Array, default=None

            The real labels for the dataset. It can be None only if X is None.

        Returns
        -------

        score : float

            The classification score at kNN. It is calculated as
            ..math:: card(y_pred == y_real) / n_samples
        """
        if X is None:
            return self.loo_score(self.trX)
        else:
            X = self.dml.transform(X)

        return self.knn.score(X, y)

    def score_orig(self, X=None, y=None):
        """
        Obtains the classification score for the given data with euclidean distance (with no dml transformation). Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        y : 1D-Array, default=None

            The true labels for the dataset. It can be None only if X is None.

        Returns
        -------

        score : float

            The classification score at kNN. It is calculated as
            ..math:: card(y_pred == y_real) / n_samples
        """
        if X is None:
            return self.loo_score(self.X_)

        return self.knn_orig.score(X, y)

    def loo_prob(self, X):
        """
        Predicts the probabilities for the given data using them as a training and with Leave One Out.

        X : 2D-Array or Matrix, default=None

            The dataset to be used.

        Returns
        -------

        T : 2D-Array, shape (N x c)

            A matrix with the probabilities for each class. N is the number of samples and c is the number of classes.
            The element i, j shows the probability of sample X[i] to be in class j.
        """
        loo = LeaveOneOut()
        probs = np.empty([self.y_.size, self.num_labels])

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = self.y_[train_index], self.y_[test_index]

            knnloo = neighbors.KNeighborsClassifier(self.nn_)
            knnloo.fit(X_train,y_train)

            probs[test_index,:]=knnloo.predict_proba(X_test)

        return probs

    def loo_pred(self, X):
        """
        Obtains the predicted for the given data using them as a training and with Leave One Out.

        X : 2D-Array or Matrix, default=None

            The dataset to be used.

        Returns
        -------

        y : 1D-Array

            The vector with the label predictions.
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

    def loo_score(self, X):
        """
        Obtains the score for the given data using them as a training and with Leave One Out.

        X : 2D-Array or Matrix, default=None

            The dataset to be used.

        Returns
        -------

        score : float

            The classification score at kNN. It is calculated as
            ..math:: card(y_pred == y_real) / n_samples
        """
        preds = self.loo_pred(X)

        return np.mean(preds == self.y_)
