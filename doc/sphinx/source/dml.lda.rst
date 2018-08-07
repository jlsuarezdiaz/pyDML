Linear Discriminant Analysis (LDA)
==================================

Linear Discriminant Analysis is a dimensionality reduction technique that finds the directions that maximize the ratio between the between-class variance and the within-class variances. This directions optimize the class separations in the projected space. The maximum number of directions this algorithm can learn is always lower than the number of classes.

The current LDA implementation is a wrapper from the `Scikit-Learn LDA implementation <http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis>`_.

Watch the full LDA documentation `here <dml.html#module-dml.lda>`_.

Images
------
.. image:: _static/lda.png