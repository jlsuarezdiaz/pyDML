#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:45:28 2018

@author: jlsuarezdiaz
"""

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

def knn_plot(X,y,k,knn_clf = None, fitted = False, title = None, subtitle=None, xrange = None, yrange = None, 
             xlabel = None, ylabel = None, grid_step = [0.1,0.1],
             label_legend = True, legend_loc="lower right", cmap=None, plot_points = True, plot_regions = True,
             region_intensity = 0.4,legend_plot_points=True, legend_plot_regions=True,**fig_kw):
    if knn_clf is None:
        fitted = False
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        
    if not fitted:
        knn_clf.fit(X,y)
        
    # Plot boundaries
    if xrange is None:
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    else:
        x_min, x_max = xrange[0], xrange[1]
        
    if yrange is None:
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    else:
        y_min, y_max = yrange[0], yrange[1]
        
    #le = LabelEncoder()
    #yn = le.fit_transform(y)
    
    # Grid
    xx, yy = np.meshgrid(np.arange(x_min,x_max,grid_step[0]), np.arange(y_min, y_max,grid_step[1]))
    
    # Plot
    f, ax = plt.subplots(sharex='col',sharey='row',**fig_kw)
    
    # Grid predictions
    Z = knn_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cls = np.unique(y)
    
    if plot_regions: # Region plot
        cont = ax.contourf(xx,yy,Z,10*len(cls),alpha=region_intensity,cmap=cmap)
        
    if plot_points: # Scatter plot
        sc = ax.scatter(X[:,0],X[:,1],c=y,s=20,edgecolor='k',cmap=cmap)
        
    if title is not None:
        f.suptitle(title)
        
    if subtitle is not None:
        ax.set_title(subtitle)
        
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    if label_legend:
        if not legend_plot_regions:
            handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="",marker="o", mec="k")[0] for c in cls]
        elif not legend_plot_points:
            handles = [mpatches.Patch(color=cont.get_cmap()(cont.norm(c)),alpha=region_intensity) for c in cls]
        else:
            handles = [(plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="",marker="o", mec="k")[0],
                    mpatches.Patch(color=sc.get_cmap()(sc.norm(c)),alpha=region_intensity)) for c in cls] # ms = 5
        ax.legend(handles,cls,loc=legend_loc)
        
    plt.show()
    