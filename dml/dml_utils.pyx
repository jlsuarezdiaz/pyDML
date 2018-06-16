#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for different DML algoritms
"""

from __future__ import print_function, absolute_import
import numpy as np
import warnings
from collections import Counter
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y, check_array


# Convert a metric matrix into an associated linear transformation
def metric_to_linear(M):
    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = eigvals.astype(float) #Remove residual imaginary part
    eigvals[eigvals<0.0]=0.0 # MEJORAR ESTO (no debería hacer falta, pero está bien para errores de precisión)
    sqrt_diag = np.sqrt(eigvals)
    return eigvecs.dot(np.diag(sqrt_diag)).T

# Semidefinite projection
def SDProject(M):
    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = eigvals.astype(float) #Remove residual imaginary part
    eigvecs = eigvecs.astype(float)
    eigvals[eigvals<0.0]=0.0 # MEJORAR ESTO
    diag_sdp = np.diag(eigvals)
    return eigvecs.dot(diag_sdp).dot(eigvecs.T)

def calc_outers(X, Y=None):
    n,d = X.shape
    if Y is None:
        Y = X
    m, e = Y.shape
    if n*m*d*e > 600000000:
	return None
    try:
        outers = np.empty([n,n,d,d],dtype=float)

        for i in xrange(n):
            for j in xrange(m):
                outers[i,j] = np.outer(X[i,:]-Y[j,:],X[i,:]-Y[j,:])

    except:
        warnings.warn("Memory is not enough to calculate all outer products at once. "
                      "Algorithm will be slower.")
        outers = None

    return outers

def calc_outers_i(X,outers,i,Y=None):
    """
        When enough memory is available, outers all calculated with calc_outers function at once.
        Else, outers will be calculted partially using this method.
    """
    if outers is not None:
        return outers[i,:]
    else:
        n,d = X.shape
        if Y is None:
            Y=X
        m, e = Y.shape
        outers_i = np.empty([n,d,d],dtype=float)
        
        for j in xrange(m):
            outers_i[j] = np.outer(X[i,:]-Y[j,:],X[i,:]-Y[j,:])
        return outers_i
    
def calc_outers_ij(X,outers_i,i,j,Y=None):
    if outers_i is not None:
        return outers_i[j]
    else:
        if Y is None:
            Y=X
        return np.outer(X[i,:]-Y[j,:],X[i,:]-Y[j,:])
    
def metric_sq_distance(M,x,y):
    """
        M: metric matrix
        x,y: column vectors
    """
    d = (x-y).reshape(1,-1)
    return d.dot(M).dot(d.T)

# Returns a column vector with all columns of A concatenated.
def unroll(A):
    n,m = A.shape
    v = np.empty([n*m,1])
    for i in xrange(m):
        v[(i*n):(i+1)*n,0] = A[:,i]
    return v

# Returns a matrix that takes by columns the elements in the vector v
def matpack(v,n,m):
    A = np.empty([n,m],dtype=float)
    for i in xrange(m):
        A[:,i] = v[(i*n):(i+1)*n,0]
    return A

# Pairwise distance for two datasets given their dot products
def pairwise_sq_distances_from_dot(K):
    n, m = K.shape
    dists = np.empty([n,m])
    for i in xrange(n):
        for j in xrange(m):
            dists[i,j] = K[i,i]+K[j,j]-2*K[i,j]
    return dists
