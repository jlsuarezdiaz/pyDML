.. pyDML documentation master file, created by
   sphinx-quickstart on Mon Aug  6 13:01:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


|Build| |Language| |License| |PyPI Package|

Welcome to pyDML's documentation!
=================================


The need of a similarity measure is very common in many machine learning algorithms, such as nearest neighbors classification.
Usually, a standard distance, like the euclidean distance, is used to measure this similarity. The distance metric learning
paradigm tries to learn an optimal distance from the data. This package provides the classic algorithms of supervised distance 
metric learning, together with some of the newest proposals.

How to learn a distance?
------------------------

There are two main ways to learn a distance in Distance Metric Learning:

* Learning a metric matrix M, that is, a positive semidefinite matrix. In this case, the distance is measured as

.. math:: 

    d(x,y) = \sqrt{(x-y)^TM(x-y)}.

* Learning a linear map L. This map is also represented by a matrix, not necessarily definite or squared. Here, the distance between two elements is the euclidean distance after applying the transformation.

Every linear map defines a single metric (:math:`M = L^TL`), and two linear maps that define the same metric only differ in an isometry. So both approaches are equivalent.


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

   stats

   references



.. |Build| image:: https://travis-ci.org/jlsuarezdiaz/pyDML.svg?branch=master
   :target: https://travis-ci.org/jlsuarezdiaz/pyDML
.. |Language| image:: https://img.shields.io/badge/language-Python-green.svg
   :target: https://www.python.org/
.. |License| image:: https://img.shields.io/badge/license-GPL-orange.svg
   :target: https://www.gnu.org/licenses/gpl.html
.. |PyPI Package| image:: https://badge.fury.io/py/pyDML.svg
   :target: http://badge.fury.io/py/pyDML 



