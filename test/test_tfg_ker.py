#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 19:04:03 2018

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
from sklearn.decomposition import KernelPCA

seed=28
np.random.seed(seed)

euclidean = (Euclidean(),"EUCLIDEAN")

kpca_linear = (KernelPCA(kernel="linear"),"KPCA [Linear]")
kpca_poly2 = (KernelPCA(kernel="polynomial",degree=2), "KPCA [Poly-2]")
kpca_poly3 = (KernelPCA(kernel="polynomial",degree=3), "KPCA [Poly-3]")
kpca_rbf = (KernelPCA(kernel="rbf"), "KPCA [RBF]")
kpca_lapl = (KernelPCA(kernel="laplacian"), "KPCA [Laplacian]")

kda_linear = (KDA(kernel="linear"), "KDA [Linear]")
kda_poly2 = (KDA(kernel="polynomial",degree=2), "KDA [Poly-2]")
kda_poly3 = (KDA(kernel="polynomial",degree=3), "KDA [Poly-3]")
kda_rbf = (KDA(kernel="rbf"),"KDA [RBF]")
kda_lapl = (KDA(kernel="laplacian"),"KDA [Laplacian]")

kanmm_linear = (KANMM(kernel="linear"), "KANMM [Linear]")
kanmm_poly2 = (KANMM(kernel="polynomial",degree=2), "KANMM [Poly-2]")
kanmm_poly3 = (KANMM(kernel="polynomial",degree=3), "KANMM [Poly-3]")
kanmm_rbf = (KANMM(kernel="rbf"), "KANMM [RBF]")
kanmm_lapl = (KANMM(kernel="laplacian"), "KANMM [Laplacian]")

kdmlmj_linear = (KDMLMJ(kernel="linear"), "KDMLMJ [Linear]")
kdmlmj_poly2 = (KDMLMJ(kernel="polynomial",degree=2), "KDMLMJ [Poly-2]")
kdmlmj_poly3 = (KDMLMJ(kernel="polynomial",degree=3), "KDMLMJ [Poly-3]")
kdmlmj_rbf = (KDMLMJ(kernel="rbf"), "KDMLMJ [RBF]")
kdmlmj_lapl = (KDMLMJ(kernel="laplacian"), "KDMLMJ [Laplacian]")

klmnn_3_linear = (KLMNN(k=3,kernel="linear"), "KLMNN [Linear]")
klmnn_3_poly2 = (KLMNN(k=3,kernel="polynomial",degree=2), "KLMNN [Poly-2]")
klmnn_3_poly3 = (KLMNN(k=3,kernel="polynomial",degree=3), "KLMNN [Poly-3]")
klmnn_3_rbf = (KLMNN(k=3,kernel="rbf"), "KLMNN [RBF]")
klmnn_3_lapl = (KLMNN(k=3,kernel="laplacian"), "KLMNN [Laplacian]")

klmnn_5_linear = (KLMNN(k=5,kernel="linear"), "KLMNN [Linear]")
klmnn_5_poly2 = (KLMNN(k=5,kernel="polynomial",degree=2), "KLMNN [Poly-2]")
klmnn_5_poly3 = (KLMNN(k=5,kernel="polynomial",degree=3), "KLMNN [Poly-3]")
klmnn_5_rbf = (KLMNN(k=5,kernel="rbf"), "KLMNN [RBF]")
klmnn_5_lapl = (KLMNN(k=5,kernel="laplacian"), "KLMNN [Laplacian]")

klmnn_7_linear = (KLMNN(k=7,kernel="linear"), "KLMNN [Linear]")
klmnn_7_poly2 = (KLMNN(k=7,kernel="polynomial",degree=2), "KLMNN [Poly-2]")
klmnn_7_poly3 = (KLMNN(k=7,kernel="polynomial",degree=3), "KLMNN [Poly-3]")
klmnn_7_rbf = (KLMNN(k=7,kernel="rbf"), "KLMNN [RBF]")
klmnn_7_lapl = (KLMNN(k=7,kernel="laplacian"), "KLMNN [Laplacian]")

mms = MinMaxScaler()

#################### LARGE

train_dmls = [euclidean[0],kpca_linear[0], kpca_poly2[0],kpca_poly3[0],kpca_rbf[0],kpca_lapl[0],
                              kda_linear[0],kda_poly2[0],kda_poly3[0],kda_rbf[0],kda_lapl[0],
                              kanmm_linear[0],kanmm_poly2[0],kanmm_poly3[0],kanmm_rbf[0],kanmm_lapl[0],
                              kdmlmj_linear[0],kdmlmj_poly2[0],kdmlmj_poly3[0],kdmlmj_rbf[0],kdmlmj_lapl[0],
                              klmnn_3_linear[0],klmnn_3_poly2[0],klmnn_3_poly3[0],klmnn_3_rbf[0],klmnn_3_lapl[0],
                              klmnn_5_linear[0],klmnn_5_poly2[0],klmnn_5_poly3[0],klmnn_5_rbf[0],klmnn_5_lapl[0],
                              klmnn_7_linear[0],klmnn_7_poly2[0],klmnn_7_poly3[0],klmnn_7_rbf[0],klmnn_7_lapl[0]]


dml_str_basic = {3:[euclidean,kpca_linear, kpca_poly2,kpca_poly3,kpca_rbf,kpca_lapl,
                              kda_linear,kda_poly2,kda_poly3,kda_rbf,kda_lapl,
                              kanmm_linear,kanmm_poly2,kanmm_poly3,kanmm_rbf,kanmm_lapl,
                              kdmlmj_linear,kdmlmj_poly2,kdmlmj_poly3,kdmlmj_rbf,kdmlmj_lapl,
                              klmnn_3_linear,klmnn_3_poly2,klmnn_3_poly3,klmnn_3_rbf,klmnn_3_lapl],
             5:[euclidean,kpca_linear, kpca_poly2,kpca_poly3,kpca_rbf,kpca_lapl,
                              kda_linear,kda_poly2,kda_poly3,kda_rbf,kda_lapl,
                              kanmm_linear,kanmm_poly2,kanmm_poly3,kanmm_rbf,kanmm_lapl,
                              kdmlmj_linear,kdmlmj_poly2,kdmlmj_poly3,kdmlmj_rbf,kdmlmj_lapl,
                              klmnn_5_linear,klmnn_5_poly2,klmnn_5_poly3,klmnn_5_rbf,klmnn_5_lapl],
             7:[euclidean,kpca_linear, kpca_poly2,kpca_poly3,kpca_rbf,kpca_lapl,
                              kda_linear,kda_poly2,kda_poly3,kda_rbf,kda_lapl,
                              kanmm_linear,kanmm_poly2,kanmm_poly3,kanmm_rbf,kanmm_lapl,
                              kdmlmj_linear,kdmlmj_poly2,kdmlmj_poly3,kdmlmj_rbf,kdmlmj_lapl,
                              klmnn_7_linear,klmnn_7_poly2,klmnn_7_poly3,klmnn_7_rbf,klmnn_7_lapl]}

dml_basic = {3:[p[0] for p in dml_str_basic[3]],
             5:[p[0] for p in dml_str_basic[5]],
             7:[p[0] for p in dml_str_basic[7]]}

names_basic = {3:[p[1] for p in dml_str_basic[3]],
             5:[p[1] for p in dml_str_basic[5]],
             7:[p[1] for p in dml_str_basic[7]]}

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
    
    for k in [3]:
        results[d]['train'][k] = np.zeros([11,len(dml_basic[k])])
        results[d]['test'][k] = np.zeros([11,len(dml_basic[k])])
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
                
                for k in [3]:
                    if dml in dml_basic[k]:
                        jj = dml_basic[k].index(dml)
                        print("**** TEST K = ",k)
                        knn = kNN(k,dml)
                        knn.fit(xtr,ytr)
                        
                        results[d]['train'][k][i,jj] = knn.score()
                        results[d]['test'][k][i,jj] = knn.score(xtst,ytst)
            except:
                print("Error en el DML:", sys.exc_info()[0])
                for k in [3]:
                    if dml in dml_basic[k]:
                        jj = dml_basic[k].index(dml)
                        results[d]['train'][k][i,jj] = np.nan
                        results[d]['test'][k][i,jj] = np.nan
                        
        for k in [3]:
            results[d]['train'][k][10,:] = np.mean(results[d]['train'][k][:10,:],axis=0)
            results[d]['test'][k][10,:] = np.mean(results[d]['test'][k][:10,:],axis=0)
        
            rt = pd.DataFrame(results[d]['train'][k],columns=names_basic[k],index=rownames+["MEAN"])
            rs = pd.DataFrame(results[d]['test'][k],columns=names_basic[k],index=rownames+["MEAN"])
            
            rt.to_csv('./results/'+str(d)+'-ker-'+str(k)+'nn-train.csv')
            rs.to_csv('./results/'+str(d)+'-ker-'+str(k)+'nn-test.csv')
            
            print("RESULTS: ",d,"k = ",k," [TRAIN] ")
            print(rt)
            print("RESULTS: ",d,"k = ",k," [TEST] ")
            print(rs)
                
    for k in [3]:
        results[d]['train'][k][10,:] = np.mean(results[d]['train'][k][:10,:],axis=0)
        results[d]['test'][k][10,:] = np.mean(results[d]['test'][k][:10,:],axis=0)
        
        results[d]['train'][k] = pd.DataFrame(results[d]['train'][k],columns=names_basic[k],index=rownames+["MEAN"])
        results[d]['test'][k] = pd.DataFrame(results[d]['test'][k],columns=names_basic[k],index=rownames+["MEAN"])
        
        results[d]['train'][k].to_csv('./results/'+str(d)+'-ker-'+str(k)+'nn-train.csv')
        results[d]['test'][k].to_csv('./results/'+str(d)+'-ker-'+str(k)+'nn-test.csv')
        
        print("RESULTS: ",d,"k = ",k," [TRAIN] ")
        print(results[d]['train'][k])
        print("RESULTS: ",d,"k = ",k," [TEST] ")
        print(results[d]['test'][k])