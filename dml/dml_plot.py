#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:45:28 2018

@author: jlsuarezdiaz
"""

from itertools import product, combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y
from sklearn.pipeline import Pipeline


from .dml_utils import metric_to_linear
from .base import Metric, Transformer

def classifier_plot(X,y,clf,attrs=None, sections="mean",fitted=False, f=None, ax=None, title = None, subtitle=None, xrange=None, yrange=None,
                    xlabel=None, ylabel=None, grid_split=[400,400], grid_step=[0.1,0.1], label_legend=True, legend_loc="lower right",
                    cmap=None, label_colors=None, plot_points=True, plot_regions=True,
                    region_intensity=0.4,legend_plot_points=True, legend_plot_regions=True,legend_on_axis=True,**fig_kw):
    
    X = pd.DataFrame(X)
    if attrs is None:
        attrs = X.columns
        
    if not fitted:
        clf.fit(X,y)
    
    # Plot boundaries
    margin_x = 0.1*(X[attrs[0]].max()-X[attrs[0]].min())
    margin_y = 0.1*(X[attrs[1]].max()-X[attrs[1]].min())
    margin = max(margin_x,margin_y)
    if xrange is None:
        x_min, x_max = X[attrs[0]].min() - margin, X[attrs[0]].max() + margin
    else:
        x_min, x_max = xrange[0], xrange[1]
        
    if yrange is None:
        y_min, y_max = X[attrs[1]].min() - margin, X[attrs[1]].max() + margin
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
    grid = np.c_[xx.ravel(),yy.ravel()]
    # Projection to plane <attrs[0],attrs[1]>
    if X.shape[1] > 2:
        if sections=="mean":
            section = np.empty([grid.shape[0],X.shape[1]])
            section[:] = np.mean(X,axis=0)
        else:
            section = np.zeros([grid.shape[0],X.shape[1]])
            
        section[:,X.columns.get_loc(attrs[0])] = grid[:,0]
        section[:,X.columns.get_loc(attrs[1])] = grid[:,1]
    else:
        section=grid
        
    Z = le.transform(clf.predict(section))
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
            sc = ax.scatter(X[attrs[0]],X[attrs[1]],c=le.transform(y),s=20,edgecolor='k',cmap=cmap)
        else:
            sc = ax.scatter(X[attrs[0]],X[attrs[1]],c=color_list,s=20,edgecolor='k')
            
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
        
        if legend_on_axis: 
            ax.legend(handles,cls,loc=legend_loc)    
        else:
            f.legend(handles,cls,loc=legend_loc)
        
    return f

def dml_plot(X,y,clf,attrs=None,sections="mean", fitted = False, metric=None, transformer=None, dml=None, dml_fitted=False,transform=True,
             f = None, ax = None, title = None, subtitle=None, xrange = None, yrange = None, 
             xlabel = None, ylabel = None, grid_split=[400,400], grid_step = [0.1,0.1], label_legend = True, legend_loc="lower right",
             cmap=None, label_colors=None, plot_points = True, plot_regions = True,
             region_intensity = 0.4,legend_plot_points=True, legend_plot_regions=True,legend_on_axis=True,**fig_kw):
    # Fitting distance metrics
    if dml is None:
        if transformer is not None:
            dml = Transformer(transformer)
        elif metric is not None:
            dml = Metric(metric)
    
    if dml is not None:
        if dml_fitted:
            X = dml.transform(X)
        else:
            if transform: # If transform, dataset will be transformed in the plot
                X = dml.fit_transform(X,y)
            # If not transform, the plotted dataset will be the original and the classifier regions will include the dml mapping (done in knn_clf initialization)            
    
    
    # Fiting classifier
    if not transform:
        clf = Pipeline([('dml',dml),('clf',clf)])
        
    if not fitted:
        clf.fit(X,y)
        
    
    if f is None or ax is None:  
        f, ax = plt.subplots(sharex='col',sharey='row',**fig_kw)
    
    return classifier_plot(X,y,clf,attrs,sections,True,f,ax,title,subtitle,xrange,yrange,xlabel,ylabel,
                           grid_split, grid_step,label_legend,legend_loc,cmap,label_colors,plot_points,
                           plot_regions,region_intensity,legend_plot_points,legend_plot_regions,legend_on_axis)
    
def knn_plot(X,y,k=1,attrs=None,sections="mean",knn_clf = None, fitted = False, metric=None, transformer=None, dml=None, dml_fitted=False,transform=True,
             f = None, ax = None, title = None, subtitle=None, xrange = None, yrange = None, 
             xlabel = None, ylabel = None, grid_split=[400,400], grid_step = [0.1,0.1], label_legend = True, legend_loc="lower right",
             cmap=None, label_colors=None, plot_points = True, plot_regions = True,
             region_intensity = 0.4,legend_plot_points=True, legend_plot_regions=True,legend_on_axis=True,**fig_kw):
    
    if dml is None:
        if transformer is not None:
            dml = Transformer(transformer)
        elif metric is not None:
            dml = Metric(metric)
    
    # Fitting distance metrics
    if dml is not None:
        if dml_fitted:
            X = dml.transform(X)
        else:
            if transform: # If transform, dataset will be transformed in the plot
                X = dml.fit_transform(X,y)
            # If not transform, the plotted dataset will be the original and the classifier regions will include the dml mapping (done in knn_clf initialization)            

    
    # Fiting classifier
    if knn_clf is None:
        fitted = False
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        if not transform:
            knn_clf = Pipeline([('dml',dml),('knn',knn_clf)])
        
    if not fitted:
        knn_clf.fit(X,y)
        
    
    if f is None or ax is None:  
        f, ax = plt.subplots(sharex='col',sharey='row',**fig_kw)
    
    return classifier_plot(X,y,knn_clf,attrs,sections,True,f,ax,title,subtitle,xrange,yrange,xlabel,ylabel,
                           grid_split, grid_step,label_legend,legend_loc,cmap,label_colors,plot_points,
                           plot_regions,region_intensity,legend_plot_points,legend_plot_regions,legend_on_axis)

        
def dml_multiplot(X,y,nrow=None,ncol=None,ks=None, clfs=None,attrs=None, sections="mean",fitted = False, metrics=None, transformers=None, dmls=None, dml_fitted=False,transforms=None,
                  title = None, subtitles = None, xlabels=None, ylabels=None, grid_split = [400,400], grid_step=[0.1,0.1,],
                  label_legend=True, legend_loc="center right",cmap=None, label_colors=None,plot_points=True,plot_regions=True, region_intensity=0.4,
                  legend_plot_points=True, legend_plot_regions=True,legend_on_axis=False,**fig_kw):
    
    # Look for not None parameter between ks, knn_clfs, metrics, transformers and dmls
    param_list = ks if ks is not None else clfs if clfs is not None else metrics if metrics is not None else transformers if transformers is not None else dmls if dmls is not None else transforms if transforms is not None else None
    
    # Axes array plot
    if nrow is None or ncol is None:
        ncol=2
        nrow = (len(param_list)+1)//2

    f, axarr = plt.subplots(nrow,ncol,**fig_kw)
    
    if param_list is not None:
        for i in range(len(param_list)):
            k = ks[i] if ks is not None else None
            clf = clfs[i] if clfs is not None else None
            metric = metrics[i] if metrics is not None else None
            transformer = transformers[i] if transformers is not None else None
            dml = dmls[i] if dmls is not None else None
            subtitle = subtitles[i] if subtitles is not None else None
            transform = transforms[i] if transforms is not None else None
            
            ix0, ix1 = i // ncol, i % ncol
            if nrow == 1:
                ax=axarr[ix1]
            elif ncol == 1:
                ax=axarr[ix0]
            else:
                ax = axarr[ix0,ix1]
            knn_plot(X,y,k,attrs,sections,clf,fitted,metric,transformer,dml,dml_fitted,transform,f,ax,title,subtitle,
                     None,None,None,None,grid_split,grid_step,label_legend,legend_loc,
                     cmap,label_colors,plot_points,plot_regions,region_intensity,legend_plot_points,legend_plot_regions,legend_on_axis)
    
    return f

      
def classifier_pairplots(X,y,clf,attrs=None,xattrs=None,yattrs=None,diag="hist",sections="mean",fitted=False, title=None,grid_split=[400,400], grid_step=[0.1,0.1],
                         label_legend=True, legend_loc="center right", cmap=None,label_colors=None,plot_points=True,
                         plot_regions=True,region_intensity=0.4,legend_plot_points=True,legend_plot_regions=True,legend_on_axis=False,**fig_kw):
    _,y = check_X_y(X,y)
    X = pd.DataFrame(X) # Adds column names if X is not a DataFrame
    
    if xattrs is None or yattrs is None:
        if attrs is None:
            attrs = X.columns
        if xattrs is None:
            xattrs = attrs
        if yattrs is None:
            yattrs = attrs
        
    n = len(xattrs)
    m = len(yattrs)
    #pairs = list(product(attrs,attrs))
    
    f, axarr = plt.subplots(m,n,**fig_kw)
    cls = np.unique(y)
    le = LabelEncoder()
    ncls = le.fit_transform(cls)
    Xc = [X.loc[np.where(y==c)[0]] for c in cls]
    
    if label_colors is None:
        cmap = cm.get_cmap(cmap)
        label_colors = [cmap(i/len(ncls)) for i in ncls]
    
    for i in range(n):
        for j in range(m):
            if xattrs[i] != yattrs[j]:
                #X_plot = pd.concat([X[attrs[i]],X[attrs[j]]],axis=1)
                ax = axarr[j,i]
                
                classifier_plot(X,y,clf,[xattrs[i],yattrs[j]],sections,fitted,f,ax,title,None,None,None,xattrs[i],yattrs[j],
                                grid_split,grid_step,label_legend,legend_loc,cmap,label_colors,plot_points,plot_regions,
                                region_intensity,legend_plot_points,legend_plot_regions,legend_on_axis)
        
            else:
                ax = axarr[j,i]
                if diag == "hist":
                    ax.set_xlabel(xattrs[i])
                    ax.set_ylabel("Hist")
                    
                    Xci = [Xc[k][xattrs[i]] for k in ncls]
                    ax.hist(Xci,color=label_colors,stacked=True)
                    
                else:
                    raise ValueError("Please provide a valid value for 'diag' parameter")
    return f

def dml_pairplots(X,y,clf,attrs=None,xattrs=None,yattrs=None,diag="hist",sections="mean",fitted=False, metric=None,transformer=None,
                   dml=None,dml_fitted=False,title=None,grid_split=[400,400], grid_step=[0.1,0.1],
                   label_legend=True, legend_loc="center right", cmap=None,label_colors=None,plot_points=True,
                   plot_regions=True,region_intensity=0.4,legend_plot_points=True,legend_plot_regions=True,legend_on_axis=False,**fig_kw):
     
    # Fitting distance metrics
    if dml is None:
        if transformer is not None:
            dml = Transformer(transformer)
        elif metric is not None:
            dml = Metric(metric)
            
    if dml is not None:
        if dml_fitted:
            X = dml.transform(X)
        else:
            X = dml.fit_transform(X,y) 
        
    if not fitted:
        clf.fit(X,y)
        
    return classifier_pairplots(X,y,clf,attrs,xattrs,yattrs,diag,sections,True,title,grid_split,grid_step,label_legend,legend_loc,
                                cmap,label_colors,plot_points,plot_regions,region_intensity,legend_plot_points,legend_plot_regions,
                                legend_on_axis,**fig_kw)
    
def knn_pairplots(X,y,k=1,attrs=None,xattrs=None,yattrs=None,diag="hist",sections="mean",knn_clf=None,fitted=False, metric=None,transformer=None,
                   dml=None,dml_fitted=False,title=None,grid_split=[400,400], grid_step=[0.1,0.1],
                   label_legend=True, legend_loc="center right", cmap=None,label_colors=None,plot_points=True,
                   plot_regions=True,region_intensity=0.4,legend_plot_points=True,legend_plot_regions=True,legend_on_axis=False,**fig_kw):
     
    # Fitting distance metrics
    if dml is None:
        if transformer is not None:
            dml = Transformer(transformer)
        elif metric is not None:
            dml = Metric(metric)
            
    if dml is not None:
        if dml_fitted:
            X = dml.transform(X)
        else:
            X = dml.fit_transform(X,y)
    
    
    # Fiting classifier
    if knn_clf is None:
        fitted = False
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        
    if not fitted:
        knn_clf.fit(X,y)
        
    return classifier_pairplots(X,y,knn_clf,attrs,xattrs,yattrs,diag,sections,True,title,grid_split,grid_step,label_legend,legend_loc,
                                cmap,label_colors,plot_points,plot_regions,region_intensity,legend_plot_points,legend_plot_regions,
                                legend_on_axis,**fig_kw)
    
def classifier_plot_3d(X,y,clf,attrs=None, sections="mean",fitted=False, f=None, ax=None, elev=0.0, azim=0.0, title = None, subtitle=None, xrange=None, yrange=None, zrange=None,
                    xlabel=None, ylabel=None, zlabel=None, grid_split=[40,40,40], grid_step=[0.1,0.1,0.1], label_legend=True, legend_loc="lower right",
                    cmap=None, label_colors=None, plot_points=True, plot_regions="all",
                    region_intensity=0.4,legend_plot_points=True, legend_plot_regions=True,legend_on_axis=True,**fig_kw):
    
    X = pd.DataFrame(X)
    if attrs is None:
        attrs = X.columns
        
    if not fitted:
        clf.fit(X,y)
    
    # Plot boundaries
    if xrange is None:
        x_min, x_max = X[attrs[0]].min() - 1, X[attrs[0]].max() + 1
    else:
        x_min, x_max = xrange[0], xrange[1]
        
    if yrange is None:
        y_min, y_max = X[attrs[1]].min() - 1, X[attrs[1]].max() + 1
    else:
        y_min, y_max = yrange[0], yrange[1]
        
    if zrange is None:
        z_min, z_max = X[attrs[2]].min() - 1, X[attrs[2]].max() + 1
    else:
        z_min, z_max = zrange[0], zrange[1]
        
    # Grid
    if not grid_split:
        xx, yy, zz = np.meshgrid(np.arange(x_min,x_max,grid_step[0]), np.arange(y_min, y_max,grid_step[1]), np.arange(z_min,z_max,grid_step[2]))
    else:
        xx, yy, zz = np.meshgrid(np.linspace(x_min,x_max,grid_split[0]),np.linspace(y_min,y_max,grid_split[1]), np.linspace(z_min,z_max,grid_split[2]))
    # Plot
    if f is None or ax is None:
        f = plt.figure(**fig_kw)
        ax = f.add_subplot(111,projection='3d')
    
    cls = np.unique(y)
    le = LabelEncoder()
    ncls = le.fit_transform(cls)
    
    # Grid predictions
    grid = np.c_[xx.ravel(),yy.ravel(),zz.ravel()]
    # Projection to plane <attrs[0],attrs[1]>
    if X.shape[1] > 3:
        if sections=="mean":
            section = np.empty([grid.shape[0],X.shape[1]])
            section[:] = np.mean(X,axis=0)
        else:
            section = np.zeros([grid.shape[0],X.shape[1]])
            
        section[:,X.columns.get_loc(attrs[0])] = grid[:,0]
        section[:,X.columns.get_loc(attrs[1])] = grid[:,1]
        section[:,X.columns.get_loc(attrs[2])] = grid[:,2]
    else:
        section=grid
        
    Z = le.transform(clf.predict(section))
    Z = Z.reshape(xx.shape)
    
    
                    
    if label_colors is not None:
        color_list = np.array(label_colors)[le.transform(y)]
    
    #print(list(zip(border_x,border_y,border_z)))
    if plot_regions is not None: # Region plot
        if plot_regions=="all":
            plot_regions = combinations(cls,2)
        # Finding frontier surface
        for c1, c2 in plot_regions:
            [c1, c2] = le.transform([c1,c2])
            border_x = []
            border_y = []
            border_z = []
            
            for i in range(Z.shape[0]-1):
                for j in range(Z.shape[1]-1):
                    for k in range(Z.shape[2]-1):
                        if (Z[i,j,k] == c1 and Z[i+1,j,k] == c2) or (Z[i,j,k] == c2 and Z[i+1,j,k] == c1):
                            border_x.append(0.5*(xx[i,j,k]+xx[i+1,j,k]))
                            border_y.append(yy[i,j,k])
                            border_z.append(zz[i,j,k])
                        if (Z[i,j,k] == c1 and Z[i,j+1,k] == c2) or (Z[i,j,k] == c2 and Z[i,j+1,k] == c1):
                            border_x.append(xx[i,j,k])
                            border_y.append(0.5*(yy[i,j,k]+yy[i,j+1,k]))
                            border_z.append(zz[i,j,k])
                        if (Z[i,j,k] == c1 and Z[i,j,k+1] == c2) or (Z[i,j,k] == c2 and Z[i,j,k+1] == c1):
                            border_x.append(xx[i,j,k])
                            border_y.append(yy[i,j,k])
                            border_z.append(0.5*(zz[i,j,k]+zz[i,j,k+1]))
            if len(border_x) >= 3:        
                ax.plot_trisurf(np.array(border_x),np.array(border_y),np.array(border_z),alpha=region_intensity)
        #if label_colors is None:
        #    cont = ax.contourf(xx,yy,Z,20*len(ncls),alpha=region_intensity,cmap=cmap)
        #else:
        #    cont = ax.contourf(xx,yy,Z,colors=label_colors,alpha=region_intensity,levels=[j-0.5 for j in ncls]+[max(ncls)+0.5])
        
    if plot_points: # Scatter plot
        if label_colors is None:
            sc = ax.scatter(X[attrs[0]],X[attrs[1]],X[attrs[2]],c=le.transform(y),s=20,edgecolor='k',cmap=cmap)
        else:
            sc = ax.scatter(X[attrs[0]],X[attrs[1]],X[attrs[2]],c=color_list,s=20,edgecolor='k')
            
    ax.view_init(elev=elev,azim=azim)
            
    if title is not None:
        f.suptitle(title)
        
    if subtitle is not None:
        ax.set_title(subtitle)
        
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    #if label_legend:
        #if label_colors is None:    # Plot legend with color map.             
            #if not legend_plot_regions or not plot_regions: # Only plot points
            #    handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="",marker="o", mec="k")[0] for c in ncls]
            #elif not legend_plot_points or not plot_points: # Only plot regions
            #    handles = [mpatches.Patch(color=cont.get_cmap()(cont.norm(c)),alpha=region_intensity) for c in ncls]
            #else: # Plot all
            #    handles = [(plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="",marker="o", mec="k")[0],
            #            mpatches.Patch(color=sc.get_cmap()(sc.norm(c)),alpha=region_intensity)) for c in ncls] # ms = 5
        #else:   # Plot legend with given colors
            #if not legend_plot_regions or not plot_regions: # Only plot points
            #    handles = [plt.plot([],color=c,ls="",marker="o", mec="k")[0] for c in label_colors]
            #elif not legend_plot_points or not plot_points: # Only plot regions
            #    handles = [mpatches.Patch(color=c,alpha=region_intensity) for c in label_colors]
            #else: # PLot all
            #    handles = [(plt.plot([],color=c,ls="",marker="o", mec="k")[0],
            #            mpatches.Patch(color=c,alpha=region_intensity)) for c in label_colors] # ms = 5
        
        #if legend_on_axis: 
        #    ax.legend(handles,cls,loc=legend_loc)    
        #else:
        #    f.legend(handles,cls,loc=legend_loc)
        
    return f