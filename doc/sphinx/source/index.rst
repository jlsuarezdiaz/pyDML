.. pyDML documentation master file, created by
   sphinx-quickstart on Mon Aug  6 13:01:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyDML's documentation!
=================================



The need of a similarity measure is very common in many machine learning algorithms, such as nearest neighbors classification.
Usually, a standard distance, like the euclidean distance, is used to measure this similarity. The distance metric learning
paradigm tries to learn an optimal distance from the data. This package provides the classic algorithms of supervised distance 
metric learning, together with some of the newest proposals.


.. toctree::
   :maxdepth: 1
   :caption: Current Algorithms:

   dml.pca
   dml.lda
   dml.anmm

   dml.lmnn
   dml.nca

   dml.ncmml
   dml.ncmc

   dml.itml
   dml.dmlmj
   dml.mcml

   dml.lsi
   dml.dml_eig
   dml.ldml

   dml.klmnn
   dml.kanmm
   dml.kdmlmj
   dml.kda


.. toctree::
   :maxdepth: 2
   :caption: Additional functionalities

   similarity_classifiers

   plot

   tune

.. toctree::
   :maxdepth: 2
   :caption: Overview

   docindex

   modules

   applications

   examples

   installation

   references








