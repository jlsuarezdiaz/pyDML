Nearest Class with Multiple Centroids (NCMC)
============================================

A distance metric learning algorithm for nearest centroids classification. It learns a transformation that optimizes the expected score of multiple centroid classification. The associated classifier establishes a variable number of centroids for each class via k-Means, and predicts the new labels according to the class of the nearest centroid. This classifier is also available in this package.

Watch the full NCMC documentation `here <dml.html#module-dml.ncmc>`_. Watch also the `NCMC Classifier documentation <dml.html#dml.ncmc.NCMC_Classifier>`_.

References
----------

Thomas Mensink et al. “Metric learning for large scale image classification: Generalizing to new
classes at near-zero cost”. In: Computer Vision–ECCV 2012. Springer, 2012, pages 488-501.