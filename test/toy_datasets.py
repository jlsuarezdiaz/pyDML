#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:25:25 2018

Toy datasets.

@author: jlsuarezdiaz
"""

import numpy as np
import pandas as pd
from six.moves import xrange
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import(
        load_iris, load_digits)

def toy_plot(X,y):
    f, ax = plt.subplots(figsize=(12,9))
    
    plt.axis('equal')
    plt.scatter(X[:,0],X[:,1],c=y,cmap="rainbow",label=y)
   
    #cmap = plt.get_cmap('rainbow')
    #cc = np.unique(y)
    #cn = len(cc)
    #for i,c in enumerate(cc):
    #    print(i,c)
    #    ind = np.where(y == c)[0]
    #    print(ind)
    #    XX = X[ind]
    #    print(cmap(i/(cn+1)))
    #    ax.scatter(XX[:,0],XX[:,1],c=cmap(i/(cn+1)),label=c)
    #plt.legend()
    
    plt.show()
    return plt

def circular_toy_dataset(rads = [1,2], samples = [200,200], noise = [0.2,0.2], seed = None):
    if seed is not None:
        np.random.seed(seed)
        
    n = sum(samples)
    d = 2
    X = np.empty([n,d])
    y = np.empty([n])
    le = LabelEncoder()
    le.fit(rads)
    
    acum = 0
    for j,s in enumerate(samples):
        for i in xrange(s):
            ns1 = noise[j]*np.random.randn()
            ns2 = noise[j]*np.random.randn()
            x1 = (rads[j]+ns1)*np.cos(2*np.pi*i/s)
            x2 = (rads[j]+ns2)*np.sin(2*np.pi*i/s)
            
            X[acum+i,:] = [x1,x2]
            y[acum+i] = rads[j]
            
        acum += s
    y = le.transform(y)
    
    return X,y

def hiperplane_toy_dataset(ws = [[1,1],[1,-1]],bs = [[0,0],[0,0]],nsamples=800,xrange=[-1,1],yrange=[-1,1], noise = 0.1,seed = None):
    if seed is not None:
        np.random.seed(seed)
        
    n=nsamples
    d=2
    X = np.random.rand(n,d)
    y = np.zeros([n])
    yy = np.empty([n,len(ws)])
    
    X[:,0] = (xrange[1]-xrange[0])*X[:,0]+xrange[0]
    X[:,1] = (yrange[1]-yrange[0])*X[:,1]+yrange[0]
    
    for j, (w, b) in enumerate(zip(ws,bs)):
        w = np.matrix(w)
        b = np.matrix(b)
        ns = noise*np.random.randn(n,2)
        yy[:,j] = np.sign(((X+ns)-b).dot(w.T)).reshape([n])
        
    yy[yy==-1]=0
    yy = yy.astype(int)

    for i in range(n):
        for j, u in enumerate(yy[i,:]):
            y[i] += (u << j)
            
    return X,y
    
def iris2d_toy_dataset(dims=[0,2]):
    data=load_iris()  # IRIS
    X=data['data']
    X=X[:,dims]
    y=data['target']
    return X,y

def balls_toy_dataset(centers = [[-2,-2],[0,0],[2,2],[2,-2],[-2,2]],rads = [1.4,1.4,1.4,1.4,1.4],samples=[200,200,200,200,200],noise = [0.3,0.3,0.3,0.3,0.3],seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    n = sum(samples)
    d=2
    
    X=np.empty([n,d])
    y=np.empty([n])
    
    acum=0
    for j, s in enumerate(samples):
        rs = rads[j]*np.random.rand(s)
        angs = 2*np.pi*np.random.rand(s)
        ns = noise[j]*np.random.rand(s)
        c = np.array(centers[j])
        
        for i in xrange(s):
            X[acum+i,:] = c +ns[i] + rs[i]*np.array([np.cos(angs[i]),np.sin(angs[i])])
            y[acum+i]=j
        
        acum += s
    
    return X,y

def simetria_hor(A):
    nrow, ncol= A.shape
    A = np.abs(A-A[:,::-1]) # Diferencia con la imagen simétrica
    return np.mean(A)                  # Media de las diferencias (grado de simetría)

def simetria_ver(A):
    nrow, ncol= A.shape
    A = np.abs(A-A[::-1,:]) # Diferencia con la imagen simétrica
    return np.mean(A)                  # Media de las diferencias (grado de simetría)

def digits_toy_dataset(dims=[0,2],numbers=[0,1,2,3,4,5,6,7,8,9]):
    data=load_digits()
    XX = data['data']
    y = data['target']
    nn,dd = XX.shape
    XX = XX.reshape([nn,8,8])

    X = np.empty([nn,3])
    for i in xrange(nn):
        X[i,0] = simetria_hor(XX[i,:,:])
        X[i,1] = simetria_ver(XX[i,:,:])
        X[i,2] = np.mean(XX[i,:])
    
    selected = np.where(np.isin(y,numbers))[0]
    
    return X[selected,:][:,dims],y[selected] 

def single_toy_dataset(samples=8, classes = 3, seed=None):
    X = np.empty([samples,2])
    y = np.empty([samples])
    for i in xrange(samples):
        c = np.random.randint(0,classes)
        x = np.random.rand(1,2)
        X[i,:]=x
        y[i]=c
        
    return X,y
    