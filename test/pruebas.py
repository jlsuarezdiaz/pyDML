#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:53:12 2018

@author: jlsuarezdiaz
"""
import numpy as np
import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.arange(0.0,1.0,0.1), np.arange(0.0, 1.0,0.5))
Z = 10*np.c_[xx.ravel()].reshape(xx.shape)

X = np.array([[i/10.0,0.5] for i in range(10)])
y = np.array([i for i in range(10)])

f, ax = plt.subplots()

ax.contourf(xx,yy,Z,alpha=0.4,cmap="gist_rainbow")
ax.scatter(X[:,0],X[:,1],c=y,s=20,edgecolor='k',cmap="gist_rainbow")

plt.show()