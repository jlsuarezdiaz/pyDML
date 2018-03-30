#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:13:58 2018

@author: jlsuarezdiaz
"""

from __future__ import print_function, absolute_import

from numpy.linalg import eig

from .dml_algorithm import DML_Algorithm

class Metric(DML_Algorithm):
    """
    A basic transformer that transforms the data given a metric PSD matrix.
    """
    def __init__(self,metric):
        self.M_ = metric
        
    def fit(self,X,y):
        return self
    
    def metric(self):
        return self.M_
    
class Transformer(DML_Algorithm):
    """
    A basic transformer that transforms the data given a linear application matrix.
    """
    def __init__(self,transformer):
        self.L_ = transformer
        
    def fit(self,X,y):
        return self
    
    def transformer(self):
        return self.L_