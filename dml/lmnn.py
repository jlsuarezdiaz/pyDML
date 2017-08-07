#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Large Margin Nearest Neighbors

A DML that obtains a metric with target neighbors as near as possible and impostors as far as possible
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y

from .dml_algorithm import DML_Algorithm

class LMNN(DML_Algorithm):
    def __init__(self,initial_transform=None, mu = 0.5, k = 3):
        self.initial_transform = initial_transform
        self.mu_ = mu
        self.k_ = k

    def metric(self):
        return self.M_

    def fit(self,X,y):
        X, y = check_X_y(X,y) # Consistency check
        n, d = X.shape

        self.X_ = X
        self.y_ = y
        self.n_ = n
        self.d_ = d
        self.classes_, self.label_inds = np.unique(self.y_, return_inverse=True)

        #num_dims = self.num_dims # If no dimensions are specified dataset dimensions are used.
        #if num_dims is None:
        #    num_dims = d

        self.M_ = self.initial_transform # If starting transformation is not specified, diagonal ones will be used.
        if self.M_ is None:
            self.M_ = np.eye(d)

        t = 0

        N_up = {}
        N_down = {}

    def _select_targets(self):
        """
            Obtains target neighbors (using euclidean distance)
        """
        target_neighbors = np.empty((self.n_, self.k), dtype = int)
        for label in self.classes_:
            inds, = np.nonzero(self.label_inds == label) # Índices de la misma clase
            dd = pairwise_distances(self.X_[inds]) # Distancias entre todos los índices de la misma clase
            np.fill_diagonal(dd,np.inf) # Descartamos el propio elemento como target neighbor.
            nn = np.argsort(dd)[...,:self.k] # k target neighbors for each label index
            target_neighbors[inds] = inds[nn]

        return target_neighbors

    def _find_impostors(self,furthest_neighbors):
        """
            Obtains impostors.

            furthest_neighbors: Array with the k-th nearest target neighbor for each element in data array
        """
        Lx = self.transform()        
        margin = 1 + _inplace_paired_L2(Lx[furthest_neighbors],Lx) # Distances to furthest target for each data
        impostors = []

        for label in self.classes_




    def _inplace_paired_L2(A, B):
        '''Equivalent to ((A-B)**2).sum(axis=-1), but modifies A in place.'''
        A -= B
        return np.einsum('...ij,...ij->...i', A, A)