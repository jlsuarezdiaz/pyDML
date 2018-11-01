from dml import kNN, NCMC_Classifier, Euclidean
from test_utils import iris, wine, breast_cancer
from numpy.testing import assert_almost_equal


class TestClassifiers:

    def test_kNN(self):
        for d in [iris, wine, breast_cancer]:
            X, y = d()
            euc = Euclidean()
            knn = kNN(1, euc)
            euc.fit(X, y)
            knn.fit(X, y)
            score = knn.score(X, y)  # Should be 1.0 for 1-NN
            loo_score = knn.score()  # Better calculation for train (uses LOO)
            assert_almost_equal(score, 1.0)
            assert 1.0 >= loo_score >= 0.9

    def test_NCMC_Classifier(self):
        for d in [iris, wine, breast_cancer]:
            X, y = d()
            ncmc = NCMC_Classifier(centroids_num=4)
            ncmc.fit(X, y)
            score = ncmc.score(X, y)
            assert 1.0 >= score >= 0.9
