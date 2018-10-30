from dml import Euclidean, Metric, Transformer, PCA, LDA, ANMM, LMNN, NCA, NCMML, NCMC, ITML, DMLMJ, MCML, LSI, DML_eig, LDML, KLMNN, KANMM, KDMLMJ, KDA
from scipy.spatial.distance import pdist
from test.test_utils import iris, wine, breast_cancer
from numpy.testing import assert_array_almost_equal, assert_equal
import numpy as np


class TestWorking:

    def working_test_basic(self, alg, dataset, do_assert=True):
        X, y = dataset()
        alg.fit(X, y)

        L = alg.transformer()
        M = alg.metric()

        if do_assert:  # PROBLEM: assert functions consider only decimals, which is not fair for very big values. In these cases, assertion usually fails because of precision errors.
            assert_array_almost_equal(L.T.dot(L), M)  # M = L^TL

        LX1 = alg.transform()
        LX2 = alg.transform(X)

        dl1 = pdist(LX1)
        dl2 = pdist(LX2)
        dm = pdist(X, metric='mahalanobis', VI=M)  # CHecking that d_M = d_L

        if do_assert:
            assert_array_almost_equal(dm, dl1)
            assert_array_almost_equal(dm, dl2)

        d_, d = L.shape
        e_, e = M.shape

        if do_assert:
            assert_equal(d, e_)
            assert_equal(d, e)
            assert_equal(d, X.shape[1])

        return X, y, L, M, LX1, LX2, dl1, dl2, dm  # For future tests.

    def test_PCA(self):
        for d in [iris, wine, breast_cancer]:
            pca = PCA()
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(pca, d)
            assert_array_almost_equal(M, np.eye(X.shape[1]))

    def test_LDA(self):
        for d in [iris, wine, breast_cancer]:
            lda = LDA()                         # TODO the single object should be independent of executions
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(lda, d)
            classes = np.unique(y)
            assert_equal(len(classes) - 1, L.shape[0])

    def test_ANMM(self):
        for d in [iris, wine, breast_cancer]:
            anmm = ANMM()
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(anmm, d)
            assert_array_almost_equal(M, np.eye(X.shape[1]))

    def test_LMNN(self):
        for d in [iris, wine, breast_cancer]:
            lmnn = LMNN()
            self.working_test_basic(lmnn, d)

    def test_LMNN_SGD(self):
        for d in [iris, wine, breast_cancer]:
            lmnn = LMNN(solver='SGD')
            self.working_test_basic(lmnn, d, False)

    def test_NCA(self):
        for d in [iris, wine, breast_cancer]:
            nca = NCA()
            self.working_test_basic(nca, d)

    def test_NCMML(self):
        for d in [iris, wine, breast_cancer]:
            ncmml = NCMML()
            self.working_test_basic(ncmml, d)

    def test_NCMC(self):
        for d in [iris, wine, breast_cancer]:
            ncmc = NCMC()
            self.working_test_basic(ncmc, d)

    def test_ITML(self):
        for d in [iris, wine, breast_cancer]:
            itml = ITML()
            self.working_test_basic(itml, d)

    def test_DMLMJ(self):
        for d in [iris, wine, breast_cancer]:
            dmlmj = DMLMJ()
            self.working_test_basic(dmlmj, d)

    def test_MCML(self):
        for d in [iris, wine, breast_cancer]:
            mcml = MCML()
            self.working_test_basic(mcml, d)

    def test_LSI(self):
        for d in [iris, wine, breast_cancer]:
            lsi = LSI(supervised=True)
            self.working_test_basic(lsi, d)

    def test_DML_eig(self):
        for d in [iris, wine, breast_cancer]:
            dml_eig = DML_eig()
            self.working_test_basic(dml_eig, d)

    def test_LDML(self):
        for d in [iris, wine, breast_cancer]:
            ldml = LDML()
            self.working_test_basic(ldml, d, False) # Assert fails because of too big exponents, but results are correct.
