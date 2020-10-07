# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Condensed and reduced nearest neighbors (CNN, RNN) undersampling algorithms.
- Iterative Metric Learning with Sample Selection (IMLS)
- Geometric Mean Metric Learning (GMML)
- Linear Ordinal Distance Metric Learning (LODML)
- Kernel Ordinal Distance Metric Learning (KODML) 

### Changed

- Improvements in efficiency in NCA.

### Fixed

- Matrix computation in ANMM when there are not enough neighbors.
- Replaced deprecated call to Pandas argmax functions in tune.
- Complex projections of ANMM, KANMM, LLDA and KLLDA due to precision errors in eigenvalue decompositions.
- Softmax overflow problems with NCMML (not completely solved).
- Robustness against duplicated dissimilar values in LSI.

## [0.1.0] - 2018-12-01

### Added

- Tests for algorithms, classifiers, tune functions and plot functions, and Travis Continuous Integration.
- Covariance distance as a distance metric learning algorithm.
- Local Linear Discriminant Analysis (LLDA).
- Changelog file.
- Kernel Local Linear Discriminant Analysis (KLLDA).

### Changed

- Distance calculation in LMNN.
- Cleaned code according to [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards.

### Fixed

- Dimensionality reduction in KDA.


## 0.0.1 - 2018-08-08

### Added

- Distance metric learning algorithms: PCA, LDA, ANMM, NCA, LMNN, NCMML, NCMC, ITML, DMLMJ, MCML, LSI, DML-eig, LDML, KLMNN, KANMM, KDMLMJ and KDA.
- Interfaces for Euclidean, Metric and Transformer distances as a distance metric learning algorithm.
- Distance-based classifiers, adapted to distance metric learning algorithms: kNN, NCMC_Classifier and MultiDML\_kNN.
- Plotting framework for classifiers and distance metric learning algorithms.
- Tune framework for parameter estimation.
- Readme file, sphinx docs and installation setup.


[Unreleased]: https://github.com/jlsuarezdiaz/pyDML/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jlsuarezdiaz/pyDML/compare/v0.0.1...v0.1.0

<!-- (Valid tags are: ADDED, CHANGED, FIXED, REMOVED, DEPRECATED, SECURITY) -->
