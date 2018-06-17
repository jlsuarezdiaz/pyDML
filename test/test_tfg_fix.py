#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:55:55 2018

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

euclidean = Euclidean()
lda = LDA() # num_dims / thres
nca = NCA() # num_dims, learning_rate, eta0, initial_transform, max_iter, prec, tol, descent_method, eta_thres, learn_inc, learn_dec
lmnn = {3: LMNN(k=3), 5: LMNN(k=5), 7: LMNN(k=7)} # num_dims, learning_rate, eta0, initial_metric, max_iter, prec, tol, K, mu, soft_comp_interval, learn_inc, learn_dec, eta_thres, solver


lsi = LSI(supervised=True) # initial_metric, learning_rate, eta0, max_iter, max_proj_iter, itproj_err, err, supervised
anmm = {3: ANMM(n_friends=3,n_enemies=3),5: ANMM(n_friends=5,n_enemies=5),7: ANMM(n_friends=7,n_enemies=7)}  # num_dims, n_friends, n_enemies

itml = ITML() # initial_metric, upper_bound, lower_bound, num_constraints, gamma, tol, max_iter, low_perc, up_perc
dmlmj = {3: DMLMJ(n_neighbors=3),5: DMLMJ(n_neighbors=5),7: DMLMJ(n_neighbors=7)} # num_dims, n_neighbors, alpha, reg_tol

ncmml = NCMML() # num_dims, learning_rate, eta0, initial_transform, max_iter, tol, prec, descent_method, eta_thres, learn_inc, learn_dec
ncmc = {2:NCMC(centroids_num=2),3:NCMC(centroids_num=3)} # num_dims, centroids_num, learning_rate, eta0, initial_transform, max_iter, tol, prec, descent_method, eta_thres, learn_inc, learn_dec

dml_eig = DML_eig() #mu, tol, eps, max_it
mcml = MCML() #num_dims, learning_rate, eta0, initial_metric, max_iter, prec, tol, descent_method, eta_thres, learn_inc, learn_dec
ldml = LDML() #num_dims, b, learning_rate, eta0, initial_metric, max_iter, prec, tol, descent_method, eta_thres, learn_inc, learn_dec

klmnn3 = KLMNN(k=3)
kanmm = KANMM()
kdmlmj = KDMLMJ()
kda = KDA()

mms = MinMaxScaler()


small = [('appendicitis',1),
         ('balance',1),
         ('bupa',1),
         ('cleveland',1),
         ('glass',1),
         ('hepatitis',1),
         ('ionosphere',1),
         ('iris',1),
         #'led7digit',
         ('monk-2',1),
         ('newthyroid',1),
         ('sonar',1),
         ('wine',1)
         ]

medium = [
          ('movement_libras',1),
          ('pima',1),
          ('vehicle',1),
          ('vowel',1),
          ('wdbc',1),
          ('wisconsin',1)
          ]


train_dmls = [lda, mcml]

dml_all = [euclidean, lda, itml, dmlmj[3], nca, lmnn[3], lsi, dml_eig, mcml, ldml]

dml_basic = {3:[lda, mcml],
             5:[lda, mcml],
             7:[lda, mcml]}

names_basic = {3: [type(dml).__name__ for dml in dml_all],
               5: [type(dml).__name__ for dml in dml_all],
               7: [type(dml).__name__ for dml in dml_all]}

results = {}

rownames = ["FOLD "+str(i+1) for i in range(10)]

large1 = [('segment',5),
          ('satimage',5),
          ('winequality-red',1),
          ('digits',1)]

large2 = [('spambase',1),
          ('optdigits',5),
          ('twonorm',5),
          ('titanic',1)]

large3 = [('banana',5),
          ('texture',5),
          ('ring',5),
          ('letter',10)]

large4 = [('phoneme',5),
          ('page-blocks',5),
          ('thyroid',5),
          ('magic',10)]

datasets = [small+medium,large1,large2,large3,large4]

if len(sys.argv) > 1:
    selected_dataset = datasets[int(sys.argv[1])]
else:
    selected_dataset = datasets[0]

for d, f in selected_dataset:
    
    print("* DATASET ", d)
    
    folds, _ = ds.reduced_dobscv10(d,f)
    results[d] = {}
    results[d]['train'] = {}
    results[d]['test'] = {}
    
    for k in [3,5,7]:
        results[d]['train'][k] = pd.read_csv('./results/'+str(d)+'-'+str(k)+'nn-train.csv',index_col=0).values
        results[d]['test'][k] = pd.read_csv('./results/'+str(d)+'-'+str(k)+'nn-test.csv',index_col=0).values
        #print(results[d]['train'][k])
    
    norm_folds = []
    for i, (xtr, ytr, xtst, ytst) in enumerate(folds):
        print("** NORMALIZING FOLD ",i+1)
        # Normalizing
        xtr = mms.fit_transform(xtr)
        xtst = mms.transform(xtst)
        norm_folds.append((xtr,ytr,xtst,ytst))
        
    for j, dml in enumerate(train_dmls):
        print("** TESTING DML ",type(dml).__name__)
        
        for i, (xtr, ytr, xtst, ytst) in enumerate(norm_folds):
            print("*** FOLD ",i+1)
            np.random.seed(seed)
            
            #Learning DML
            try:
                print("**** TRAINING")
                dml.fit(xtr,ytr)
                
                for k in [3,5,7]:
                    if dml in dml_basic[k]:
                        jj = dml_all.index(dml)
                        print("**** TEST K = ",k)
                        knn = kNN(k,dml)
                        knn.fit(xtr,ytr)
                        
                        results[d]['train'][k][i,jj] = knn.score()
                        results[d]['test'][k][i,jj] = knn.score(xtst,ytst)
            except:
                print("Error en el DML:", sys.exc_info()[0])
                for k in [3,5,7]:
                    if dml in dml_basic[k]:
                        jj = dml_all.index(dml)
                        results[d]['train'][k][i,jj] = np.nan
                        results[d]['test'][k][i,jj] = np.nan
                        
        for k in [3,5,7]:
            results[d]['train'][k][10,:] = np.mean(results[d]['train'][k][:10,:],axis=0)
            results[d]['test'][k][10,:] = np.mean(results[d]['test'][k][:10,:],axis=0)
        
            rt = pd.DataFrame(results[d]['train'][k],columns=names_basic[k],index=rownames+["MEAN"])
            rs = pd.DataFrame(results[d]['test'][k],columns=names_basic[k],index=rownames+["MEAN"])
            
            rt.to_csv('./results/'+str(d)+'-'+str(k)+'nn-train.csv')
            rs.to_csv('./results/'+str(d)+'-'+str(k)+'nn-test.csv')
            
            print("RESULTS: ",d,"k = ",k," [TRAIN] ")
            print(rt)
            print("RESULTS: ",d,"k = ",k," [TEST] ")
            print(rs)
                
    for k in [3,5,7]:
        results[d]['train'][k][10,:] = np.mean(results[d]['train'][k][:10,:],axis=0)
        results[d]['test'][k][10,:] = np.mean(results[d]['test'][k][:10,:],axis=0)
        
        results[d]['train'][k] = pd.DataFrame(results[d]['train'][k],columns=names_basic[k],index=rownames+["MEAN"])
        results[d]['test'][k] = pd.DataFrame(results[d]['test'][k],columns=names_basic[k],index=rownames+["MEAN"])
        
        results[d]['train'][k].to_csv('./results/'+str(d)+'-'+str(k)+'nn-train.csv')
        results[d]['test'][k].to_csv('./results/'+str(d)+'-'+str(k)+'nn-test.csv')
        
        print("RESULTS: ",d,"k = ",k," [TRAIN] ")
        print(results[d]['train'][k])
        print("RESULTS: ",d,"k = ",k," [TEST] ")
        print(results[d]['test'][k])