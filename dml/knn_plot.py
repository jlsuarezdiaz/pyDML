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

from .dml_utils import metric_to_linear

def classifier_plot(X,y,clf, fitted=False, f=None, ax=None, title = None, subtitle=None, xrange=None, yrange=None,
                    xlabel=None, ylabel=None, grid_split=[400,400], grid_step=[0.1,0.1], label_legend=True, legend_loc="lower right",
                    cmap=None, label_colors=None, plot_points=True, plot_regions=True,
                    region_intensity=0.4,legend_plot_points=True, legend_plot_regions=True,**fig_kw):
    
    if not fitted:
        clf.fit(X,y)
    
    # Plot boundaries
    if xrange is None:
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    else:
        x_min, x_max = xrange[0], xrange[1]
        
    if yrange is None:
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    else:
        y_min, y_max = yrange[0], yrange[1]
        
    # Grid
    if not grid_split:
        xx, yy = np.meshgrid(np.arange(x_min,x_max,grid_step[0]), np.arange(y_min, y_max,grid_step[1]))
    else:
        xx, yy = np.meshgrid(np.linspace(x_min,x_max,grid_split[0]),np.linspace(y_min,y_max,grid_split[1]))
    # Plot
    if f is None or ax is None:
        f, ax = plt.subplots(sharex='col',sharey='row',**fig_kw)   
    
    cls = np.unique(y)
    le = LabelEncoder()
    ncls = le.fit_transform(cls)
    
    # Grid predictions
    Z = le.transform(clf.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    if label_colors is not None:
        color_list = np.array(label_colors)[le.transform(y)]
    
    
    if plot_regions: # Region plot
        if label_colors is None:
            cont = ax.contourf(xx,yy,Z,20*len(ncls),alpha=region_intensity,cmap=cmap)
        else:
            cont = ax.contourf(xx,yy,Z,colors=label_colors,alpha=region_intensity,levels=[j-0.5 for j in ncls]+[max(ncls)+0.5])
        
    if plot_points: # Scatter plot
        if label_colors is None:
            sc = ax.scatter(X[:,0],X[:,1],c=le.transform(y),s=20,edgecolor='k',cmap=cmap)
        else:
            sc = ax.scatter(X[:,0],X[:,1],c=color_list,s=20,edgecolor='k')
            
    if title is not None:
        f.suptitle(title)
        
    if subtitle is not None:
        ax.set_title(subtitle)
        
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    if label_legend:
        if label_colors is None:    # Plot legend with color map.             
            if not legend_plot_regions or not plot_regions: # Only plot points
                handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="",marker="o", mec="k")[0] for c in ncls]
            elif not legend_plot_points or not plot_points: # Only plot regions
                handles = [mpatches.Patch(color=cont.get_cmap()(cont.norm(c)),alpha=region_intensity) for c in ncls]
            else: # Plot all
                handles = [(plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="",marker="o", mec="k")[0],
                        mpatches.Patch(color=sc.get_cmap()(sc.norm(c)),alpha=region_intensity)) for c in ncls] # ms = 5
        else:   # Plot legend with given colors
            if not legend_plot_regions or not plot_regions: # Only plot points
                handles = [plt.plot([],color=c,ls="",marker="o", mec="k")[0] for c in label_colors]
            elif not legend_plot_points or not plot_points: # Only plot regions
                handles = [mpatches.Patch(color=c,alpha=region_intensity) for c in label_colors]
            else: # PLot all
                handles = [(plt.plot([],color=c,ls="",marker="o", mec="k")[0],
                        mpatches.Patch(color=c,alpha=region_intensity)) for c in label_colors] # ms = 5
        
        ax.legend(handles,cls,loc=legend_loc)
        
    return f

def knn_plot(X,y,k=1,knn_clf = None, fitted = False, metric=None, transformer=None, dml=None, dml_fitted=False,
             f = None, ax = None, title = None, subtitle=None, xrange = None, yrange = None, 
             xlabel = None, ylabel = None, grid_split=[400,400], grid_step = [0.1,0.1], label_legend = True, legend_loc="lower right",
             cmap=None, label_colors=None, plot_points = True, plot_regions = True,
             region_intensity = 0.4,legend_plot_points=True, legend_plot_regions=True,**fig_kw):
    # Fitting distance metrics
    if dml is not None:
        if dml_fitted:
            X = dml.transform(X)
        else:
            X = dml.fit_transform(X,y)
    elif transformer is not None:
        X = X.dot(transformer.T)
    elif metric is not None:
        transformer = metric_to_linear(metric)
        X = X.dot(transformer.T)
    
    
    # Fiting classifier
    if knn_clf is None:
        fitted = False
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        
    if not fitted:
        knn_clf.fit(X,y)
        
    
    if f is None or ax is None:  
        f, ax = plt.subplots(sharex='col',sharey='row',**fig_kw)
    
    return classifier_plot(X,y,knn_clf,True,f,ax,title,subtitle,xrange,yrange,xlabel,ylabel,
                           grid_split, grid_step,label_legend,legend_loc,cmap,label_colors,plot_points,
                           plot_regions,region_intensity,legend_plot_points,legend_plot_regions)
        
def knn_multiplot(X,y,nrow=None,ncol=None,ks=None, clfs=None, fitted = False, metrics=None, transformers=None, dmls=None, dml_fitted=False,
                  title = None, subtitles = None, xlabels=None, ylabels=None, grid_split = [400,400], grid_step=[0.1,0.1,],
                  label_legend=True, legend_loc="lower right",cmap=None, label_colors=None,plot_points=True,plot_regions=True, region_intensity=0.4,
                  legend_plot_points=True, legend_plot_regions=True,**fig_kw):
    
    # Look for not None parameter between ks, knn_clfs, metrics, transformers and dmls
    param_list = ks if ks is not None else clfs if clfs is not None else metrics if metrics is not None else transformers if transformers is not None else dmls if dmls is not None else None
    
    # Axes array plot
    if nrow is None or ncol is None:
        ncol=2
        nrow = (len(param_list)+1)/2
        
    f, axarr = plt.subplots(nrow,ncol,**fig_kw)
    
    if param_list is not None:
        for i in range(len(param_list)):
            k = ks[i] if ks is not None else None
            clf = clfs[i] if clfs is not None else None
            metric = metrics[i] if metrics is not None else None
            transformer = transformers[i] if transformers is not None else None
            dml = dmls[i] if dmls is not None else None
            subtitle = subtitles[i] if subtitles is not None else None
            
            ix0, ix1 = i // ncol, i % ncol
            ax = axarr[ix0,ix1]
            knn_plot(X,y,k,clf,fitted,metric,transformer,dml,dml_fitted,f,ax,title,subtitle,
                     None,None,None,None,grid_split,grid_step,label_legend,legend_loc,
                     cmap,label_colors,plot_points,plot_regions,region_intensity,legend_plot_points,legend_plot_regions)
    
    return f
        
    
    
    
    
    