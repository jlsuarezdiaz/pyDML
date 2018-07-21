#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:46:45 2018

@author: jlsuarezdiaz
"""

import sys
import numpy as np
import pandas as pd
from utils import datasets as ds
from dml import( kNN, PCA, LDA, NCA, LMNN, KLMNN, LSI, ANMM, KANMM,
                ITML, DMLMJ, KDMLMJ, NCMML, NCMC, KDA, DML_eig, 
                MCML, LDML, MultiDML_kNN, Euclidean)

from sklearn.preprocessing import MinMaxScaler

seed=28
np.random.seed(seed)

results = {}

rownames = ["FOLD "+str(i+1) for i in range(10)]



test1 = [('sonar',1),
         ('movement_libras',1),
         ('spambase',1)]

test2 = [(('digits',1))]

datasets = [test1, test2]

mms = MinMaxScaler()

if len(sys.argv) > 1:
    selected_dataset = datasets[int(sys.argv[1])]
else:
    selected_dataset = datasets[0]

for d, f in selected_dataset:
    
    print("* DATASET ", d)
    
    folds, _ = ds.reduced_dobscv10(d,f)
    
    nclas = len(np.unique(folds[0][1]))
    print(nclas)
    maxdim = folds[0][0].shape[1]
    results[d] = {}
    
    for dim in np.unique([1,2,3,5,10,20,30,40,50,nclas-1,maxdim]):
        print("** DIM ",dim)
        results[d][dim] = {}
        
        results[d][dim]['train'] = {}
        results[d][dim]['test'] = {}
        
        pca = PCA(num_dims=dim)
        lda = LDA(num_dims=dim)
        anmm = {3: ANMM(num_dims=dim,n_friends=3,n_enemies=3),5: ANMM(num_dims=dim,n_friends=5,n_enemies=5),7: ANMM(num_dims=dim,n_friends=7,n_enemies=7)}
        lmnn = {3: LMNN(num_dims=dim,k=3,solver="SGD",eta0=0.001), 5: LMNN(num_dims=dim,k=5,solver="SGD",eta0=0.001), 7: LMNN(num_dims=dim,k=7,solver="SGD",eta0=0.001)}
        nca = NCA(num_dims=dim)
        dmlmj = {3: DMLMJ(num_dims=dim,n_neighbors=3),5: DMLMJ(num_dims=dim,n_neighbors=5),7: DMLMJ(num_dims=dim,n_neighbors=7)}
        
        
        train_dmls = [pca, lda, anmm[3], anmm[5], anmm[7], dmlmj[3], dmlmj[5], dmlmj[7], nca, lmnn[3], lmnn[5], lmnn[7]]
        
        
        dml_basic = {3:[pca,lda, anmm[3], dmlmj[3], nca, lmnn[3]],
             5:[pca,lda, anmm[5], dmlmj[5], nca, lmnn[5]],
             7:[pca,lda, anmm[7], dmlmj[7], nca, lmnn[7]]}

        names_basic = {3: [type(dml).__name__ for dml in dml_basic[3]],
                       5: [type(dml).__name__ for dml in dml_basic[5]],
                       7: [type(dml).__name__ for dml in dml_basic[7]]}
        
        norm_folds = []
        for i, (xtr, ytr, xtst, ytst) in enumerate(folds):
            print("*** NORMALIZING FOLD ",i+1)
            # Normalizing
            xtr = mms.fit_transform(xtr)
            xtst = mms.transform(xtst)
            norm_folds.append((xtr,ytr,xtst,ytst))
        
        for k in [3,5,7]:
            results[d][dim]['train'][k] = np.zeros([11,len(dml_basic[k])])
            results[d][dim]['test'][k] = np.zeros([11,len(dml_basic[k])])
            #print(results[d]['train'][k])
        
        
            
        for j, dml in enumerate(train_dmls):
            print("*** TESTING DML ",type(dml).__name__)
            
            for i, (xtr, ytr, xtst, ytst) in enumerate(norm_folds):
                print("**** FOLD ",i+1)
                np.random.seed(seed)
                
                #Learning DML
                try:
                    print("***** TRAINING")
                    dml.fit(xtr,ytr)
                    
                    for k in [3,5,7]:
                        if dml in dml_basic[k]:
                            jj = dml_basic[k].index(dml)
                            print("***** TEST K = ",k)
                            knn = kNN(k,dml)
                            knn.fit(xtr,ytr)
                            
                            results[d][dim]['train'][k][i,jj] = knn.score()
                            results[d][dim]['test'][k][i,jj] = knn.score(xtst,ytst)
                except:
                    print("Error en el DML:", sys.exc_info()[0])
                    for k in [3,5,7]:
                        if dml in dml_basic[k]:
                            jj = dml_basic[k].index(dml)
                            results[d][dim]['train'][k][i,jj] = np.nan
                            results[d][dim]['test'][k][i,jj] = np.nan
                            
            for k in [3,5,7]:
                results[d][dim]['train'][k][10,:] = np.mean(results[d][dim]['train'][k][:10,:],axis=0)
                results[d][dim]['test'][k][10,:] = np.mean(results[d][dim]['test'][k][:10,:],axis=0)
            
                rt = pd.DataFrame(results[d][dim]['train'][k],columns=names_basic[k],index=rownames+["MEAN"])
                rs = pd.DataFrame(results[d][dim]['test'][k],columns=names_basic[k],index=rownames+["MEAN"])
                
                rt.to_csv('./results/'+str(d)+'-dim-'+str(dim)+'-'+str(k)+'nn-train.csv')
                rs.to_csv('./results/'+str(d)+'-dim-'+str(dim)+'-'+str(k)+'nn-test.csv')
                
                print("RESULTS: ",d,"dim = ",dim,"k = ",k," [TRAIN] ")
                print(rt)
                print("RESULTS: ",d,"dim = ",dim,"k = ",k," [TEST] ")
                print(rs)
                        
        for k in [3,5,7]:
            results[d][dim]['train'][k][10,:] = np.mean(results[d][dim]['train'][k][:10,:],axis=0)
            results[d][dim]['test'][k][10,:] = np.mean(results[d][dim]['test'][k][:10,:],axis=0)
            
            results[d][dim]['train'][k] = pd.DataFrame(results[d][dim]['train'][k],columns=names_basic[k],index=rownames+["MEAN"])
            results[d][dim]['test'][k] = pd.DataFrame(results[d][dim]['test'][k],columns=names_basic[k],index=rownames+["MEAN"])
            
            results[d][dim]['train'][k].to_csv('./results/'+str(d)+'-dim-'+str(dim)+'-'+str(k)+'nn-train.csv')
            results[d][dim]['test'][k].to_csv('./results/'+str(d)+'-dim-'+str(dim)+'-'+str(k)+'nn-test.csv')
            
            print("RESULTS: ",d,"dim = ",dim,"k = ",k," [TRAIN] ")
            print(results[d][dim]['train'][k])
            print("RESULTS: ",d,"dim = ",dim,"k = ",k," [TEST] ")
            print(results[d][dim]['test'][k])