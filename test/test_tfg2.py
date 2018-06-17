#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:28:36 2018

@author: jlsuarezdiaz
"""
import sys
import numpy as np
import pandas as pd
import traceback
from utils import datasets as ds
from dml import( kNN, PCA, LDA, NCA, LMNN, KLMNN, LSI, ANMM, KANMM,
                ITML, DMLMJ, KDMLMJ, NCMML, NCMC, KDA, DML_eig, 
                MCML, LDML, MultiDML_kNN, Euclidean, NCMC_Classifier)

from sklearn.neighbors import NearestCentroid

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

#################### LARGE

train_dmls = [euclidean,ncmml, ncmc[2],ncmc[3]]

dml_basic = {#'Energy-3':[lmnn[3]],
             #'Energy-5':[lmnn[5]],
             #'Energy-7':[lmnn[7]],
             'ncm':[euclidean,ncmml],
             'ncmc-2':[euclidean,ncmc[2]],
             'ncmc-3':[euclidean,ncmc[3]]}

names_basic = {#'Energy-3': ["LMNN - Energy"],
               #'Energy-5': ["LMNN - Energy"],
               #'Energy-7': ["LMNN - Energy"],
               'ncm': ['Euclidean + NCM','NCMML'],
               'ncmc-2': ['Euclidean + NCM (2)','NCMC (2 centroids)'],
               'ncmc-3': ['Euclidean + NCM (3)', 'NCMC (3 centroids)']}

results = {}

rownames = ["FOLD "+str(i+1) for i in range(10)]

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
    
    for k in dml_basic.keys():
        results[d]['train'][k] = np.zeros([11,len(dml_basic[k])])
        results[d]['test'][k] = np.zeros([11,len(dml_basic[k])])
    
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
                
                xtr2 = dml.transform()
                xtst2 = dml.transform(xtst)
                
                for k in names_basic.keys():
                    if dml in dml_basic[k]:
                        if k == 'ncm':
                            jj = dml_basic[k].index(dml)
                            print("**** TEST Classifier ",k)
                            ncm = NearestCentroid()
                            ncm.fit(xtr2,ytr)
                            
                            results[d]['train'][k][i,jj] = ncm.score(xtr2,ytr)
                            results[d]['test'][k][i,jj] = ncm.score(xtst2,ytst)
                            
                        elif k == 'ncmc-2':
                            jj = dml_basic[k].index(dml)
                            print("**** TEST Classifier ",k)
                            ncm = NCMC_Classifier(2)
                            ncm.fit(xtr2,ytr)
                            
                            results[d]['train'][k][i,jj] = ncm.score(xtr2,ytr)
                            results[d]['test'][k][i,jj] = ncm.score(xtst2,ytst)
                            
                        elif k == 'ncmc-3':
                            jj = dml_basic[k].index(dml)
                            print("**** TEST Classifier ",k)
                            ncm = NCMC_Classifier(3)
                            ncm.fit(xtr2,ytr)
                            
                            results[d]['train'][k][i,jj] = ncm.score(xtr2,ytr)
                            results[d]['test'][k][i,jj] = ncm.score(xtst2,ytst)
                           
            except Exception as e:
                print("Error en el DML:")
                traceback.print_exc()
                for k in names_basic.keys():
                    if dml in dml_basic[k]:
                        jj = dml_basic[k].index(dml)
                        results[d]['train'][k][i,jj] = np.nan
                        results[d]['test'][k][i,jj] = np.nan
                        
        for k in names_basic.keys():
            results[d]['train'][k][10,:] = np.mean(results[d]['train'][k][:10,:],axis=0)
            results[d]['test'][k][10,:] = np.mean(results[d]['test'][k][:10,:],axis=0)
            
            rt = pd.DataFrame(results[d]['train'][k],columns=names_basic[k],index=rownames+["MEAN"])
            rs = pd.DataFrame(results[d]['test'][k],columns=names_basic[k],index=rownames+["MEAN"])
            
            rt.to_csv('./results/'+str(d)+'-'+str(k)+'-train.csv')
            rs.to_csv('./results/'+str(d)+'-'+str(k)+'-test.csv')
            
            print("RESULTS: ",d,"k = ",k," [TRAIN] ")
            print(rt)
            print("RESULTS: ",d,"k = ",k," [TEST] ")
            print(rs)
                
    for k in names_basic.keys():
        results[d]['train'][k][10,:] = np.mean(results[d]['train'][k][:10,:],axis=0)
        results[d]['test'][k][10,:] = np.mean(results[d]['test'][k][:10,:],axis=0)
        
        results[d]['train'][k] = pd.DataFrame(results[d]['train'][k],columns=names_basic[k],index=rownames+["MEAN"])
        results[d]['test'][k] = pd.DataFrame(results[d]['test'][k],columns=names_basic[k],index=rownames+["MEAN"])
        
        results[d]['train'][k].to_csv('./results/'+str(d)+'-'+str(k)+'-train.csv')
        results[d]['test'][k].to_csv('./results/'+str(d)+'-'+str(k)+'-test.csv')
        
        print("RESULTS: ",d,"k = ",k," [TRAIN] ")
        print(results[d]['train'][k])
        print("RESULTS: ",d,"k = ",k," [TEST] ")
        print(results[d]['test'][k])