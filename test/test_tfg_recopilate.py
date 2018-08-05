#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 18:37:48 2018

@author: jlsuarezdiaz
"""
import sys
import numpy as np
import pandas as pd
import numpy as np
from utils import datasets as ds
from dml import( kNN, PCA, LDA, NCA, LMNN, KLMNN, LSI, ANMM, KANMM,
                ITML, DMLMJ, KDMLMJ, NCMML, NCMC, KDA, DML_eig, 
                MCML, LDML, MultiDML_kNN, Euclidean, NCMC_Classifier)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA

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

final_results = {}
for k in [3,5,7]:
    final_results[k] = {}
    final_results[k]['train'] = np.empty([len(small+medium+large), len(dml_basic[k])])
    final_results[k]['test'] = np.empty([len(small+medium+large), len(dml_basic[k])])
    
    final_results[k]['train'] = pd.DataFrame(final_results[k]['train'],index = small+medium+large, columns=names_basic[k])
    final_results[k]['test'] = pd.DataFrame(final_results[k]['test'],index = small+medium+large, columns=names_basic[k])
    
    for d in small+medium+large:
        train_d = pd.read_csv('./results/'+str(d)+'-'+str(k)+'nn-train.csv',index_col=0)
        test_d = pd.read_csv('./results/'+str(d)+'-'+str(k)+'nn-test.csv',index_col=0)
        
        final_results[k]['train'].loc[d] = train_d.loc['MEAN']
        final_results[k]['test'].loc[d] = test_d.loc['MEAN']
        
    #final_results[k]['train'].style.applymap(bold_max)
    #final_results[k]['test'].style.applymap(bold_max)
    #final_results[k]['train'].to_csv('./results/basic-experiments-train-'+str(k)+'nn.csv')
    #final_results[k]['test'].to_csv('./results/basic-experiments-test-'+str(k)+'nn.csv')
    #final_results[k]['train'].to_latex('./results/basic-experiments-train-'+str(k)+'nn.tex')
    #final_results[k]['test'].to_latex('./results/basic-experiments-test-'+str(k)+'nn.tex')   
    
ncm_results = {}
ncm_results['train'] = np.empty([len(small+medium+large),6])
ncm_results['test'] = np.empty([len(small+medium+large),6])

ncm_results['train'] = pd.DataFrame(ncm_results['train'],index = small+medium+large, columns=['Euclidean + NCM', 'NCMML', 'Euclidean + NCM (2 ctrd)', 'NCMC (2 ctrd)', 'Euclidean + NCM (3 ctrd)', 'NCMC (3 ctrd)'])
ncm_results['test'] = pd.DataFrame(ncm_results['test'],index = small+medium+large, columns=['Euclidean + NCM', 'NCMML', 'Euclidean + NCM (2 ctrd)', 'NCMC (2 ctrd)', 'Euclidean + NCM (3 ctrd)', 'NCMC (3 ctrd)'])
for j, clf in enumerate(['ncm','ncmc-2','ncmc-3']):
    for d in small+medium+large:
        train_d = pd.read_csv('./results/'+str(d)+'-'+str(clf)+'-train.csv',index_col=0)
        test_d = pd.read_csv('./results/'+str(d)+'-'+str(clf)+'-test.csv',index_col=0)
        
        ncm_results['train'].loc[d][2*j:2*j+2] = train_d.loc['MEAN']
        ncm_results['test'].loc[d][2*j:2*j+2] = test_d.loc['MEAN']
        
#ncm_results['train'].to_csv('./results/ncm-experiments-train.csv')
#ncm_results['test'].to_csv('./results/ncm-experiments-test.csv')
#ncm_results['train'].to_latex('./results/ncm-experiments-train.tex')
#ncm_results['test'].to_latex('./results/ncm-experiments-test.tex') 
    
    
dim_results = {}
datasets_dim = ['sonar','movement_libras','spambase','digits']

for k in [3,5,7]:
    
    dim_results[k] = {}
    
    for d in datasets_dim:
        dim_results[k][d] = {}
        dim_results[k][d]['train'] =np.empty([11,6])
        dim_results[k][d]['test'] = np.empty([11,6])
        
        dim_results[k][d]['train'] = pd.DataFrame(dim_results[k][d]['train'],index = [1,2,3,5,10,20,30,40,50,'N. Clases - 1','Dim. máxima'], columns=['PCA','LDA','ANMM','DMLMJ','NCA','LMNN'])
        dim_results[k][d]['test'] = pd.DataFrame(dim_results[k][d]['test'],index = [1,2,3,5,10,20,30,40,50,'N. Clases - 1','Dim. máxima'], columns=['PCA','LDA','ANMM','DMLMJ','NCA','LMNN'])
        
        folds, _ = ds.reduced_dobscv10(d,1)

        nclas = len(np.unique(folds[0][1]))
        maxdim = folds[0][0].shape[1]
        
        for dim,tdim in zip([1,2,3,5,10,20,30,40,50,nclas-1,maxdim],[1,2,3,5,10,20,30,40,50,'N. Clases - 1','Dim. máxima']):
            try:
                train_d = pd.read_csv('./results/'+str(d)+'-dim-'+str(dim)+'-'+str(k)+'nn-train.csv',index_col=0)
                test_d = pd.read_csv('./results/'+str(d)+'-dim-'+str(dim)+'-'+str(k)+'nn-test.csv',index_col=0)
                
                dim_results[k][d]['train'].loc[tdim] = train_d.loc['MEAN']
                dim_results[k][d]['test'].loc[tdim] = test_d.loc['MEAN']
            except:
                dim_results[k][d]['train'].loc[tdim] = np.nan
                dim_results[k][d]['test'].loc[tdim] = np.nan
            
            
        #dim_results[k][d]['train'].to_csv('./results/dim-exp-'+str(d)+'-train-'+str(k)+'nn.csv')
        #dim_results[k][d]['test'].to_csv('./results/dim-exp-'+str(d)+'-test-'+str(k)+'nn.csv')
        #dim_results[k][d]['train'].to_csv('./results/dim-exp-'+str(d)+'-train-'+str(k)+'nn.tex')
        #dim_results[k][d]['test'].to_csv('./results/dim-exp-'+str(d)+'-test-'+str(k)+'nn.tex')
        
        
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


ker_results = {}
ker_results['train'] = np.empty([len(small+medium+large), 26])
ker_results['test'] = np.empty([len(small+medium+large), 26])

ker_results['train'] = pd.DataFrame(ker_results['train'],index = small+medium+large, columns=names_basic[3])
ker_results['test'] = pd.DataFrame(ker_results['test'],index = small+medium+large, columns=names_basic[3])

for d in small+medium+large:
    try:
        train_d = pd.read_csv('./results/'+str(d)+'-ker-3nn-train.csv',index_col=0)
        test_d = pd.read_csv('./results/'+str(d)+'-ker-3nn-test.csv',index_col=0)
        
        ker_results['train'].loc[d] = train_d.loc['MEAN']
        ker_results['test'].loc[d] = test_d.loc['MEAN']
    except:
        ker_results['train'].loc[d] = np.nan
        ker_results['test'].loc[d] = np.nan
    
ker_results['train'] = ker_results['train'].dropna()
ker_results['test'] = ker_results['test'].dropna()   
ker_results['train'] = ker_results['train'][(ker_results['train'] != 0).all(axis=1)]
ker_results['test'] = ker_results['test'][(ker_results['test'] != 0).all(axis=1)]

ker_results['train'].to_csv('./results/ker-experiments-3nn-train.csv')
ker_results['test'].to_csv('./results/ker-experiments-3nn-test.csv')
ker_results['train'].to_latex('./results/ker-experiments-3nn-train.tex')
ker_results['test'].to_latex('./results/ker-experiments-3nn-test.tex')

