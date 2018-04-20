#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:18:35 2018

@author: jlsuarezdiaz
"""

from __future__ import print_function, absolute_import
import numpy as np

from six.moves import xrange
from sklearn.utils.validation import check_X_y

from numpy.linalg import eig, inv, pinv, eigh
import scipy.linalg as sl

from .dml_algorithm import DML_Algorithm
from .dml_utils import calc_outers, calc_outers_i

class DML_eig(DML_Algorithm):
    
    def __init__(self,mu=1e-4,tol=1e-6,eps=1e-10,max_it=25):
        """
            mu: smoothing parameter
            beta: regularization parameter
        """
        self.mu_ = mu
        #self.beta_ = beta
        self.tol_ = tol
        self.max_it_ = max_it
        self.eps_ = 1e-10
        #self.linesearch_ = linesearch
        
        self.initial_error_ = None
        self.final_error_ = None
        
    def metric(self):
        return self.M_
    
    def metadata(self):
        return {'initial_error':self.initial_error_,'final_error':self.final_error_}
    
    def fit(self,X,y):
        """
        # Initialize parameters
        X, y = check_X_y(X,y)
        self.X_, self.y_ = X,y
        
        S, D = DML_eig.label_to_similarity_set(y)
        
        mu = self.mu_
        beta = self.beta_
        tol = self.tol_
        max_it = self.max_it_
        eps = self.eps_
        linesearch = self.linesearch_
        
        ns, nd = S.shape[0], D.shape[0]
        n, d = X.shape
        
        Xt = np.zeros([d*d,nd])
        ut = np.ones([ns,1])
        XS = DML_eig.SODW(X.T,S[:,0],S[:,1],ut)
        
        XS = (XS + XS.T)/2.0
        
        Sig, U = eig(XS);
        Sig = np.real(Sig)
        Sig[Sig <= eps]=0.0
        XSL = U.dot(np.diag(np.sqrt(Sig))).dot(U.T)
        
        invXS = pinv(XS)
        XXtr = pinv(XSL).dot(X.T)
        
        for i in xrange(nd):
            temp1 = (XXtr[:,D[i,1]]-XXtr[:,D[i,0]]).dot((XXtr[:,D[i,1]]-XXtr[:,D[i,0]]).T)
            Xt[:,i] = temp1
            
        Id = np.eye(d)
        MM = Id[:,1].dot(Id[1,:])
        
        count=True
        its=0
        fval=[]
        change_fval=[]
        change_M=[]
        mu = mu/np.log(nd)
        
        while its < max_it and count:
            temp = -Xt.T.dot(MM)/mu
            
            mg = np.max(temp)
            a = np.exp(temp-mg)
            print(Xt.shape,a.shape)
            gradfM = np.reshape((Xt.dot(a))/np.sum(a),[d,d])-beta*invXS
            gradfM = (gradfM + gradfM.T)/2.0
            fval.append(-mu*(np.log(np.sum(a))+mg))
            
            # Compute largest eigenvalue
            dd, V = eig(gradfM)
            V = V[:,np.argsort(dd)[::-1]]
            V=V[:,0]
            
            SM = V.dot(V.T)
            
            MMp = MM
            pd = SM - MMp
            
            if linesearch:
                line_tol = 1e-3
                alphak = 1/(its+1)
                max_linesearch=10
                linesearch_iter = 1
                flag_linesearch = True
                
                while flag_linesearch and linesearch_iter <= max_linesearch:
                    MM = MMp + alphak*pd
                    
                    temp = - Xt.T.dot(MM)/mu
                    temp[temp < -700] = -700
                    temp[temp > 700] = 700
                    mg = np.max(temp)
                    a = np.exp(temp-mg)
                    tempfval = -mu*(np.log(np.sum(a))+mg)
                    ftaylor = fval[-1]+alphak*line_tol*np.trace(gradfM.dot(pd))
                    
                    if tempfval >= ftaylor:
                        flag_linesearch=False
                    else:
                        alphak = alphak/2
                    linesearch_iter += 1
            else:
                alphak = 2/(its+2)
                MM = MMp+alphak*pd
                #change_M.append(np.sqrt(np.sum((MM-MMp)*(MM-MMp))))
                
            if its > 1:
                change_fval.append(np.abs(fval[-1]-fval[-2])/np.abs(fval[-1]+eps))
                if change_fval[-1] < tol:
                    count=False
                    
            its += 1
        """   
        # Initialize parameters
        X, y = check_X_y(X,y)
        self.X_, self.y_ = X,y
        
        S, D = DML_eig.label_to_similarity_set(y)
        
        mu = self.mu_
        #beta = self.beta_
        tol = self.tol_
        max_it = self.max_it_
        eps = self.eps_
        #linesearch = self.linesearch_
        
        ns, nd = S.shape[0], D.shape[0]
        n, d = X.shape
        
        M = np.zeros([d,d])
        np.fill_diagonal(M,1.0/d)
        
        outers = calc_outers(X)
        Xs = np.zeros([d,d])
        for [i,j] in S:
            Xs += calc_outers_i(X,outers,i)[j]
        vals, U = eigh(Xs)
        if np.linalg.det(Xs) < eps:
            I = np.eye(d)
            Xs += 1e-5*I
            
        Xs_invsqrt = inv(U.dot(np.diag(np.sqrt(vals))).dot(U.T))
        
        stop=False
        its = 0
        
        while not stop:
            grad_sum = 0.0
            grad = np.zeros([d,d])
            
            for [i,j] in D:
                Xtau = calc_outers_i(X,outers,i)[j]
                XT = Xs_invsqrt.dot(Xtau).dot(Xs_invsqrt)
                inner = np.inner(XT.reshape(1,-1),M.reshape(1,-1))
                soft = np.exp(-inner/mu)
                grad += soft*XT
                grad_sum += soft
                if its==0:
                    self.initial_error_ = -mu*np.log(grad_sum)
                
            grad /= grad_sum
            
            _, V = sl.eigh(grad,eigvals=(grad.shape[0]-1,grad.shape[0]-1))
            Z = V.dot(V.T)
            alphat = 1/(its+1)
            M = (1-alphat)*M+alphat*Z
            
            its+=1
            if its==max_it:
                stop=True
            
        
        self.final_error_ = -mu*np.log(grad_sum) # Error before last iteration !!
        self.M_ = M
        return self
        
        
    def label_to_similarity_set(y):
        n = len(y)
        S = []#np.empty([n,n],dtype=bool)
        D = []#np.empty([n,n],dtype=bool)
        for i in xrange(n):
            for j in xrange(n):
                if y[i] == y[j]:
                    S.append([i,j])
                else:
                    D.append([i,j])
        return np.array(S),np.array(D)
    
    def SODW(x,a,b,w):
        nn = len(a)
        d = x.shape[0]
        res = np.zeros([d,d])
        for i in xrange(nn):
            res += w[i]*(x[:,a[i]]-x[:,b[i]]).dot((x[:,a[i]]-x[:,b[i]]).T)
            
        return res