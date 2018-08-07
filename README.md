# pyDML

[![](https://img.shields.io/badge/language-Python-green.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/license-GPL-orange.svg)](https://www.python.org/)

Distance Metric Learning Algorithms for Python

## What is Distance Metric Learning?

Many machine learning algorithms need a similarity measure to carry out their tasks. Usually, standard distances, like euclidean distance, are used to measure this similarity. Distance Metric Learning algorithms try to learn an optimal distance from the data.

## How to learn a distance?

There are two main ways to learn a distance in Distance Metric Learning:

- Learning a metric matrix M, that is, a positive semidefinite matrix. In this case, the distance is measured as
<a href="https://www.codecogs.com/eqnedit.php?latex=d(x,y)&space;=&space;\sqrt{(x-y)^TM(x-y)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d(x,y)&space;=&space;\sqrt{(x-y)^TM(x-y)}" title="d(x,y) = \sqrt{(x-y)^TM(x-y)}" /></a>

- Learning a linear map L. This map is also represented by a matrix, not necessarily definite or squared. Here, the distance between two elements is the euclidean distance after applying the transformation.

Every linear map defines a single metric (M = L'L), and two linear maps that define the same metric only differ in an isometry. So both approaches are equivalent.

## Some applications

### Improve distance based classifiers

![](./plots/ex_learning_nca.png)
*Improving 1-NN classification.*

### Dimensionality reduction

![](./plots/ex_red_dim.png)
*Learning a projection onto a plane for the digits dataset (dimension 64).*

## Documentation

See the available algorithms, the additional functionalities and the full documentation [here](https://pydml.readthedocs.io/en/latest/).

## Installation

- PyPI latest version: `pip install pyDML`

- From github: clone or download this repository and run the command `python setup.py install` on the root directory.



## Authors

- Juan Luis Suárez Díaz ([jlsuarezdiaz](https://github.com/jlsuarezdiaz))
