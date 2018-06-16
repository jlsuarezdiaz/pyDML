#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:09:22 2018

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

"""
def bold_max(data):
    is_max = data == data.max()
    return['font-width: bold' if v else '' for v in is_max]
"""

seed=28
np.random.seed(seed)

datasets = ['appendicitis',
            'balance',
            'banana',
            'bupa',
            'cleveland',
            'glass',
            'hepatitis',
            'ionosphere',
            'iris',
            #'led7digit',
            'letter',
            'magic',
            'monk-2',
            'movement_libras',
            'newthyroid',
            'optdigits',
            'page-blocks',
            'phoneme',
            'ring',
            'segment',
            'sonar',
            'spambase',
            'texture',
            'thyroid',
            'titanic',
            'twonorm',
            'vehicle',
            'vowel',
            'wdbc',
            'wine',
            'winequality-red',
            'wisconsin']

small = ['appendicitis',
         'balance',
         'bupa',
         'cleveland',
         'glass',
         'hepatitis',
         'ionosphere',
         'iris',
         #'led7digit',
         'monk-2',
         'newthyroid',
         'sonar',
         'wine'
         ]

medium = [
          'movement_libras',
          'pima',
          'vehicle',
          'vowel',
          'wdbc',
          'wisconsin'
          ]

large = ['banana',
         'digits',
         'letter',
         'magic',
         'optdigits',
         'page-blocks',
         'phoneme',
         'ring',
         'satimage',
         'segment',
         'spambase',
         'texture',
         'thyroid',
         'titanic',
         'twonorm',
         'winequality-red']



#kNN()
#MultiDML_kNN()
#PCA()
#ds.sonar()

euclidean = Euclidean()
lda = LDA() # num_dims / thres
nca = NCA() # num_dims, learning_rate, eta0, initial_transform, max_iter, prec, tol, descent_method, eta_thres, learn_inc, learn_dec
lmnn = {3: LMNN(k=3), 5: LMNN(k=5), 7: LMNN(k=7)} # num_dims, learning_rate, eta0, initial_metric, max_iter, prec, tol, K, mu, soft_comp_interval, learn_inc, learn_dec, eta_thres, solver


lsi = LSI(supervised=True) # initial_metric, learning_rate, eta0, max_iter, max_proj_iter, itproj_err, err, supervised
anmm = {3: ANMM(n_friends=3,n_enemies=3),5: ANMM(n_friends=5,n_enemies=5),7: ANMM(n_friends=7,n_enemies=7)}  # num_dims, n_friends, n_enemies

itml = ITML() # initial_metric, upper_bound, lower_bound, num_constraints, gamma, tol, max_iter, low_perc, up_perc
dmlmj = {3: DMLMJ(n_neighbors=3),5: DMLMJ(n_neighbors=5),7: DMLMJ(n_neighbors=7)} # num_dims, n_neighbors, alpha, reg_tol

ncmml = NCMML() # num_dims, learning_rate, eta0, initial_transform, max_iter, tol, prec, descent_method, eta_thres, learn_inc, learn_dec
ncmc = NCMC() # num_dims, centroids_num, learning_rate, eta0, initial_transform, max_iter, tol, prec, descent_method, eta_thres, learn_inc, learn_dec

dml_eig = DML_eig() #mu, tol, eps, max_it
mcml = MCML() #num_dims, learning_rate, eta0, initial_metric, max_iter, prec, tol, descent_method, eta_thres, learn_inc, learn_dec
ldml = LDML() #num_dims, b, learning_rate, eta0, initial_metric, max_iter, prec, tol, descent_method, eta_thres, learn_inc, learn_dec

klmnn3 = KLMNN(k=3)
kanmm = KANMM()
kdmlmj = KDMLMJ()
kda = KDA()

mms = MinMaxScaler()

dml_basic = {3:[euclidean, lda, itml, dmlmj[3], nca, lmnn[3], lsi, dml_eig, mcml, ldml],
             5:[euclidean, lda, itml, dmlmj[5], nca, lmnn[5], lsi, dml_eig, mcml, ldml],
             7:[euclidean, lda, itml, dmlmj[7], nca, lmnn[7], lsi, dml_eig, mcml, ldml]}

names_basic = {3: [type(dml).__name__ for dml in dml_basic[3]],
               5: [type(dml).__name__ for dml in dml_basic[5]],
               7: [type(dml).__name__ for dml in dml_basic[7]]}



###### SMALL + MEDIUM
"""
results = {}

rownames = ["FOLD "+str(i+1) for i in range(10)]

for d in small+medium:
    
    print("* DATASET ", d)
    
    folds, _ = ds.dobscv10(d)
    results[d] = {}
    results[d]['train'] = {}
    results[d]['test'] = {}
    
    for k in [3,5,7]:
            results[d]['train'][k] = np.zeros([11,len(dml_basic[k])])
            results[d]['test'][k] = np.zeros([11,len(dml_basic[k])])
    
    for i, (xtr, ytr, xtst, ytst) in enumerate(folds):
        
        print("** FOLD ",i+1)
        # Normalizing
        xtr = mms.fit_transform(xtr)
        xtst = mms.transform(xtst)
        
        
        
        for k in [3,5,7]:
            print("*** K = ",k)
            
            for j, dml in enumerate(dml_basic[k]):
                print("**** DML ",type(dml).__name__)
                np.random.seed(seed)
                # Learning DML
                try:
                    knn = kNN(k,dml)
                    
                    dml.fit(xtr,ytr)
                    knn.fit(xtr,ytr)
                
                    results[d]['train'][k][i,j] = knn.score()
                    results[d]['test'][k][i,j] = knn.score(xtst,ytst)
                except:
                    print("Error en el DML:", sys.exc_info()[0])
                    results[d]['train'][k][i,j] = np.nan
                    results[d]['test'][k][i,j] = np.nan
                
                #print(results[d]['train'][k][i,j])
                #print(results[d]['test'][k][i,j])
            
            
    for k in [3,5,7]:
        results[d]['train'][k][10,:] = np.mean(results[d]['train'][k][:10,:],axis=0)
        results[d]['test'][k][10,:] = np.mean(results[d]['test'][k][:10,:],axis=0)
        
        results[d]['train'][k] = pd.DataFrame(results[d]['train'][k],columns=names_basic[k],index=rownames+["MEAN"])
        results[d]['test'][k] = pd.DataFrame(results[d]['test'][k],columns=names_basic[k],index=rownames+["MEAN"])
        
        #results[d]['train'][k].to_csv('./results/'+str(d)+'-'+str(k)+'nn-train.csv')
        #results[d]['test'][k].to_csv('./results/'+str(d)+'-'+str(k)+'nn-test.csv')
        
        print("RESULTS: ",d,"k = ",k," [TRAIN] ")
        print(results[d]['train'][k])
        print("RESULTS: ",d,"k = ",k," [TEST] ")
        print(results[d]['test'][k])
"""
##########
"""
final_results = {}
for k in [3,5,7]:
    final_results[k] = {}
    final_results[k]['train'] = np.empty([len(small+medium), len(dml_basic[k])])
    final_results[k]['test'] = np.empty([len(small+medium), len(dml_basic[k])])
    
    final_results[k]['train'] = pd.DataFrame(final_results[k]['train'],index = small+medium, columns=names_basic[k])
    final_results[k]['test'] = pd.DataFrame(final_results[k]['test'],index = small+medium, columns=names_basic[k])
    
    for d in small+medium:
        train_d = pd.read_csv('./results/'+str(d)+'-'+str(k)+'nn-train.csv',index_col=0)
        test_d = pd.read_csv('./results/'+str(d)+'-'+str(k)+'nn-test.csv',index_col=0)
        
        final_results[k]['train'].loc[d] = train_d.loc['MEAN']
        final_results[k]['test'].loc[d] = test_d.loc['MEAN']
        
    final_results[k]['train'].style.applymap(bold_max)
    final_results[k]['test'].style.applymap(bold_max)
    final_results[k]['train'].to_csv('./results/small-medium-train-'+str(k)+'nn.csv')
    final_results[k]['test'].to_csv('./results/small-medium-test-'+str(k)+'nn.csv')
    final_results[k]['train'].to_latex('./results/small-medium-train-'+str(k)+'nn.tex')
    final_results[k]['test'].to_latex('./results/small-medium-test-'+str(k)+'nn.tex')     
"""

###########################3



#################### LARGE

train_dmls = [euclidean, lda, itml, dmlmj[3], dmlmj[5], dmlmj[7], nca, lmnn[3], lmnn[5], lmnn[7], lsi, dml_eig, mcml, ldml]

dml_basic = {3:[euclidean, lda, itml, dmlmj[3], nca, lmnn[3], lsi, dml_eig, mcml, ldml],
             5:[euclidean, lda, itml, dmlmj[5], nca, lmnn[5], lsi, dml_eig, mcml, ldml],
             7:[euclidean, lda, itml, dmlmj[7], nca, lmnn[7], lsi, dml_eig, mcml, ldml]}

names_basic = {3: [type(dml).__name__ for dml in dml_basic[3]],
               5: [type(dml).__name__ for dml in dml_basic[5]],
               7: [type(dml).__name__ for dml in dml_basic[7]]}

results = {}

rownames = ["FOLD "+str(i+1) for i in range(10)]

large1 = [#('segment',5),
          #('satimage',5),
          ('winequality-red',1),
          ('digits',1)]

large2 = [#('spambase',1),
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

large = [large1,large2,large3,large4]

if len(sys.argv) > 1:
    large_datasets = large[int(sys.argv[1])]
else:
    large_datasets = large[0]

for d, f in large_datasets:
    
    print("* DATASET ", d)
    
    folds, _ = ds.reduced_dobscv10(d,f)
    results[d] = {}
    results[d]['train'] = {}
    results[d]['test'] = {}
    
    for k in [3,5,7]:
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
                
                for k in [3,5,7]:
                    if dml in dml_basic[k]:
                        jj = dml_basic[k].index(dml)
                        print("**** TEST K = ",k)
                        knn = kNN(k,dml)
                        knn.fit(xtr,ytr)
                        
                        results[d]['train'][k][i,jj] = knn.score()
                        results[d]['test'][k][i,jj] = knn.score(xtst,ytst)
            except:
                print("Error en el DML:", sys.exc_info()[0])
                for k in [3,5,7]:
                    if dml in dml_basic[k]:
                        jj = dml_basic[k].index(dml)
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
        
################################################
   
