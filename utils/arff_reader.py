"""
ARFF reader.
Functions that implement a reader for ARFF files, obtaining a dataset and an optional label set.
"""


from __future__ import absolute_import
import numpy as np
#import pandas as pd
from scipy.io.arff import loadarff
import warnings
from six.moves import xrange
import arff
import pandas as pd

def _data_to_matrix(data, label_col):
    col_size = len(data[0])
    if label_col is None:
        y=None
    else:
        col_size-=1
        y=np.empty([len(data)],dtype=int)

    X=np.empty([len(data),col_size])

    classes = []

    for i, instance in enumerate(data):
        j=0
        for k, attr in enumerate(instance):
            if k != label_col and k != len(instance)+label_col:
                X[i][j]=attr
                j+=1
            else:
                if attr in classes:
                    y[i] = classes.index(attr)
                else:
                    y[i] = len(classes)
                    classes.append(attr)


    return X,y



def read_ARFF(file, label_col = None):
    """
    Reads and ARFF file.
    file: File path
    label_col = index where the class is specified in the file. If not supplied, no labels are returned.
    Returns: X (data), y (labels or None), m (metadata)
    """
    data, meta = loadarff(file)

    X=data
    y=None
    m=meta

    X,y=_data_to_matrix(X,label_col)
#
#    if not label_col is None:
#        y = data[:,label_col]
#        X = np.concatenate(data[:,:label_col],data[:,label_col:])

    return X,y,m

def read_ARFF2(file, label_col = None):
    data = arff.load(open(file,'r'))
    
    data = pd.DataFrame(data['data'])
    
    X=data
    y=None
    X,y =_data_to_matrix(X,label_col)

    return X,y