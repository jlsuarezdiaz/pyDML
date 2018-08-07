#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiple-DML k-Nearest Neighbors (kNN)

"""

from __future__ import absolute_import
import numpy as np
from sklearn import neighbors
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from .dml_algorithm import DML_Algorithm
from six.moves import xrange
import time


class MultiDML_kNN:
    """
    Multi-DML k-NN

    An interface that allows learning k-NN with different distance metric learners simultaneously.

    Parameters
    ----------

    n_neighbors : int

        The number of neighbors for k-NN.

    dmls : list, default=None

        A list of distance metric learning algorithms to be learned for k-NN. By default, euclidean distance will be added at the first
        place of the dml list.

    verbose : boolean, default=False

        If True, console message about the algorithms execution will be printed.

    """

    def __init__(self, n_neighbors, dmls=None, verbose=False, **knn_args):
        self.nn_ = n_neighbors
        self.knn_args_ = knn_args
        self.knns_ = [neighbors.KNeighborsClassifier(n_neighbors, **knn_args)]  # EUC
        self.verbose_ = verbose
        self.dmls_ = [None]

        if dmls is not None:
            if isinstance(dmls, list):
                for dml in dmls:
                    self.knns_.append(neighbors.KNeighborsClassifier(n_neighbors,**knn_args))
               
                self.dmls_ += dmls
            else:
                self.dmls_ = [None,dmls]
                self.knns_.append(neighbors.KNeighborsClassifier(n_neighbors,**knn_args))

    def add(self,dmls):
        """
        Adds a new distance metric learning algorithm to the list.

        Parameters
        ----------

        dmls : DML_Algorithm, or list of DMÑ_Algorithm

            The DML algorithm or algorithms to add.

        """
        if isinstance(dmls, list):
            for dml in dmls:
                self.knns_.append(neighbors.KNeighborsClassifier(self.nn_,**self.knn_args_))
           
            self.dmls_.append(dmls)
        else:
            self.dmls_.append(dmls)
            self.knns_.append(neighbors.KNeighborsClassifier(self.nn_,**self.knn_args_))

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
        self.X_ = X
        self.y_ = y
        self.num_labels_ = len(set(y))
        self.elapsed_ = []

        for i,dml in enumerate(self.dmls_):
            transf = X
            if dml is not None:
                if self.verbose_:
                    print("* Training DML ",type(dml).__name__,"...")
                start = time.time()
                dml.fit(X,y)
                end = time.time()
                transf = dml.transform(X)
                self.elapsed_.append(end - start)
            else:
                self.elapsed_.append(0.0)

            self.knns_[i].fit(transf,y)

        return self

    def elapsed(self):
        """
        Obtains the elapsed time of each DML algorithm

        Returns
        -------

        elapsed : A list of float with the time of each DML.
        """
        return self.elapsed_

    def _predict(self,dml=None,knn=None,X=None):
        trans = X
        if X is None:
            trans = self.X_
            if dml is not None:
                trans = dml.transform(trans)
            return self._loo_pred(trans)
        else:
            if dml is not None:
                trans = dml.transform(trans)
            return knn.predict(trans)

    def _predict_proba(self,dml=None,knn=None,X=None):
        trans = X
        if X is None:
            trans = self.X_
            if dml is not None:
                trans = dml.transform(trans)
            return self._loo_prob(trans)
        else:
            if dml is not None:
                trans = dml.transform(trans)
            return knn.predict_proba(trans)        

    def _score(self,dml=None,knn=None,X=None,y=None):
        trans = X
        if X is None:
            trans = self.X_
            if dml is not None:
                trans = dml.transform(trans)
            return self._loo_score(trans)
        else:
            if dml is not None:
                trans = dml.transform(trans)
            return knn.score(trans,y)

    def _loo_pred(self,X):
        loo = LeaveOneOut()
        preds = np.empty([self.y_.size],dtype=self.y_.dtype)

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train  = self.y_[train_index]

            knnloo = neighbors.KNeighborsClassifier(self.nn_)
            knnloo.fit(X_train,y_train)

            preds[test_index]=knnloo.predict(X_test)

        return preds

    def _loo_score(self, X):
        preds = self._loo_pred(X)

        return np.mean(preds == self.y_)

    def predict_all(self, X=None):
        """
        Predicts the labels for the given data. Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        Returns
        -------

        y : list of 1D-Arrays

            A list with the vectors with the label predictions for each DML.
        """
        pred_list = []
        for i in xrange(len(self.dmls_)):
            pred_list.append(self._predict(self.dmls_[i],self.knns_[i],X))

        return pred_list

    def predict_proba_all(self,X=None):
        """
        Predicts the probabilities for the given data. Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        Returns
        -------

        T : list of 2D-Arrays

            A list with the matrices with the label probabilities for each class, for each DML.
        """
        pred_list = []
        for i in xrange(len(self.dmls_)):
            pred_list.append(self._predict(self.dmls_[i],self.knns_[i],X))

        return pred_list

    def score_all(self,X=None,y=None):
        """
        Obtains the scores for the given data. Model needs to be already fitted.

        X : 2D-Array or Matrix, default=None

            The dataset to be used. If None, the training set will be used. In this case, the prediction will be made
            using Leave One Out (that is, the sample to predict will be taken away from the training set).

        Returns
        -------

        s : list of float

            A list with the k-NN scores for each DML.
        """
        score_array = np.empty([len(self.dmls_)])

        for i in xrange(len(self.dmls_)):
            score_array[i] = self._score(self.dmls_[i],self.knns_[i],X,y)

        return score_array

    def dmls_string(self):
        """
        Obtains the strings with the dml names.

        Returns
        -------

        strings : A list with the names of each dml.
        """
        strings=[]
        for dml in self.dmls_:
            if dml is None:
                strings.append("EUCLIDEAN")
            else:
                strings.append(type(dml).__name__)
        return strings
