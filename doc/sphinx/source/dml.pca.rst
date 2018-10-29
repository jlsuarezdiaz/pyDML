Principal Component Analysis (PCA)
==================================

Principal Component Analysis is one of the most popular dimensionality reduction techniques. Note that this algorithm is not supervised, but it is still important as a preprocessing algorithm for many other supervised techniques.

PCA computes the first :math:`d'` orthogonal directions for which the data variance is maximized, where `d'` is the desired dimensionality reduction.

The current PCA implementation is a wrapper for the `Scikit-Learn PCA implementation <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.

Watch the full PCA documentation `here <https://github.com/jlsuarezdiaz/pyDML-Stats>`_.

Images
------
.. image:: _static/pca.png
