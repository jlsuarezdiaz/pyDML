#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:14:30 2018

@author: jlsuarezdiaz
"""

import numpy as np
from sklearn.datasets import load_iris

# Loading DML Algorithm
from dml import NCA

# Loading dataset
iris = load_iris()
X = iris['data']
y = iris['target']

# DML construction
nca = NCA()

# Fitting algorithm
nca.fit(X,y)

# We can look at the algorithm metadata after fitting it
meta = nca.metadata()
meta

# We can see the metric the algorithm has learned.
# This metric defines how the distance is measured.
M = nca.metric()
M

# Equivalently, we can see the learned linear map.
# The distance coincides with the euclidean distance after applying the linear map.
L = nca.transformer()
L

# Finally, we can obtain the transformed data ...
Lx = nca.transform()
Lx[:5,:]

# ... or transform new data.
X_ = np.array([[1.0,0.0,0.0,0.0],[1.0,1.0,0.0,0.0],[1.0,1.0,1.0,0.0]])
Lx_ = nca.transform(X_)
Lx_


import numpy as np
from sklearn.datasets import load_iris

from dml import NCA, LDA, kNN, MultiDML_kNN, NCMC_Classifier

# Loading dataset
iris = load_iris()
X = iris['data']
y = iris['target']

# Initializing transformers and predictor
nca = NCA()
knn = kNN(n_neighbors=7,dml_algorithm=nca)

# Fitting transformer and predictor
nca.fit(X,y)
knn.fit(X,y)

# Now we can predict the labels for k-NN with the learned distance.
knn.predict() # Also we can use predict(X_) for other datasets. When using the training set
              # predictions are made leaving the sample to predict out.

knn.predict_proba()[-10:,:] # Again it can be used for other datasets.

knn.score() # The classification score (score(X_,y_) for other datasets).

# We can also compare with the euclidean distance k-NN
knn.score_orig()

# With MultiDML_kNN we can test multiple dmls. In this case, dmls are fitted automatically.
lda = LDA()
mknn = MultiDML_kNN(n_neighbors=7,dmls=[lda,nca])
mknn.fit(X,y)

# And we can predict and take scores in the same way, for every dml.
# The euclidean distance will be added always in first place.
mknn.score_all()

# The NCMC Classifier works like every ClassifierMixin.
ncmc = NCMC_Classifier(centroids_num=2)
ncmc.fit(X,y)
ncmc.score(X,y)

# To learn a distance to use with NCMC Classifier, and with any other distance classifier
# we can use pipelines.
from sklearn.pipeline import Pipeline
dml_ncmc = Pipeline([('nca',nca),('ncmc',ncmc)])
dml_ncmc.fit(X,y)
dml_ncmc.score(X,y)

import numpy as np
from sklearn.datasets import load_iris
from dml import NCA, LDA, NCMC_Classifier, classifier_plot, dml_plot, knn_plot, dml_multiplot, knn_pairplots

# Loading dataset
iris = load_iris()
X = iris['data']
y = iris['target']

# Initializing transformers and predictors
nca = NCA()
lda = LDA()
ncmc = NCMC_Classifier(centroids_num=2)

# We can plot regions for different classifiers
f1 = classifier_plot(X[:,[0,1]],y,clf=ncmc,title = "NCMC Classification",cmap="rainbow",figsize=(12,6))
f2 = knn_plot(X[:,[0,1]],y,k=3,title = "3-NN Classification", cmap="rainbow",figsize=(12,6))

# We can also make with the transformation determined by a metric, a transformer or a DML Algorithm
f3 = dml_plot(X[:,[0,1]],y,clf=ncmc,dml=nca,title = "NCMC Classification + NCA",cmap="rainbow",figsize=(12,6))
f4 = knn_plot(X[:,[0,1]],y,k=2,dml=lda,title="3-NN Classification + LDA",cmap="rainbow",figsize=(12,6))

# Or we can see how the distance changes the classifier region using the option transform=False
f5 = dml_plot(X[:,[0,1]],y,clf=ncmc,dml=nca,title = "NCMC Classification + NCA",cmap="rainbow",transform=False,figsize=(12,6))
f6 = knn_plot(X[:,[0,1]],y,k=2,dml=lda,title="3-NN Classification + LDA",cmap="rainbow",transform=False,figsize=(12,6))

# We can compare different algorithms or distances together in the same figure
f7 = dml_multiplot(X[:,[0,1]],y,nrow=2,ncol=2,ks=[None,None,3,3],clfs=[ncmc,ncmc,None,None],dmls=[None,nca,None,lda],
              transforms=[False,False,False,False],title="Comparing",
              subtitles=["NCMC","NCMC + NCA","3-NN","3-NN + LDA"],cmap="rainbow",figsize=(12,12))

# Finally, we can also plot each pair of attributes. Here the classifier region is made taking a section
# in the features space.
f8 = knn_pairplots(X,y,k=3,sections="mean",dml=nca,title="pairplots",cmap="gist_rainbow",figsize=(24,24))

#f1.savefig("plots/plotdoc1.png")
#f2.savefig("plots/plotdoc2.png")
#f3.savefig("plots/plotdoc3.png")
#f4.savefig("plots/plotdoc4.png")
#f5.savefig("plots/plotdoc5.png")
#f6.savefig("plots/plotdoc6.png")
#f7.savefig("plots/plotdoc7.png")
#f8.savefig("plots/plotdoc8.png")


import numpy as np
from sklearn.datasets import load_iris
from dml import NCA, tune

# Loading dataset
iris = load_iris()
X = iris['data']
y = iris['target']

# Using cross validation we can tune parameters for the DML algorithms.
# Here, we tune the NCA algorithm, with a fixed parameter learning_rate='constant'.
# The parameters we tune are num_dims and eta0.
# The metrics we use are 3-NN and 5-NN scores, and the final expectance metadata of NCA.
# A 5-fold cross validation is done twice, to obtain the results.
results,best,nca_best,detailed = tune(NCA,X,y,dml_params={'learning_rate':'constant'},
                                      tune_args={'num_dims':[3,4],'eta0':[0.001,0.01,0.1]},
                                      metrics=[3,5,'final_expectance'],
                                      n_folds=5,n_reps=2,seed=28,verbose=True)

# Now we can compare the results obtained for each case.
results

# We can also take the best result (respect to the first metric).
best

# We also obtain the best DML algorithm already constructed to be used.
nca_best.fit(X,y)

# If we want, we can look at the detailed results of cross validation for each case.
detailed["{'num_dims': 3, 'eta0': 0.01}"]