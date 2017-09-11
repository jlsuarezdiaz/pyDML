#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learning with Side Information (LSI)

A DML that optimizes the distance between pairs of similar data
"""

from __future__ import print_function, absolute_import
import numpy as np
import warnings
from collections import Counter
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y, check_array

from .dml_algorithm import DML_Algorithm

class LSI(DML_Algorithm):

    def __init__(self,initial_metric = None, learning_rate = 0.1):
        self.initial_metric_ = initial_metric
        self.eta_ = learning_rate


    def fit(X,side):
        """
            X: data
            side: side information. A list of sets of pairs of indices. Options:
                - side = [S,D], where S is the set of indices of similar data and D is the set of indices of dissimilar data.
                - side = [S], where S is the set of indices of similar data. The set D will be the complement of S.
        """

        if len(side) == 1:
            S = side[0]
            D = self._compute_complement(S)
        else:
            S = side[0]
            D = side[1]

        grad = self._compute_gradient(X,D)

        #while



