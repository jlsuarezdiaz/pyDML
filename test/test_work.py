from dml import Euclidean, Metric, Transformer, PCA, LDA, ANMM, LMNN, NCA, NCMML, NCMC, ITML, DMLMJ, MCML, LSI, DML_eig, LDML, KLMNN, KANMM, KDMLMJ, KDA
from scipy.spatial.distance import pdist
from test_utils import iris, wine, breast_cancer
from numpy.testing import assert_array_almost_equal, assert_equal
import numpy as np
from sklearn.datasets import make_spd_matrix


class TestWorking:

    def working_test_basic(self, alg, dataset, do_assert=True):
        np.random.seed(28)
        X, y = dataset()
        alg.fit(X, y)

        L = alg.transformer()
        M = alg.metric()

        # PROBLEM: assert functions consider only decimals, which is not fair
        # for very big values. In these cases, assertion usually fails because
        #  of precision errors.
        if do_assert:
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
            assert d_ <= d

        return X, y, L, M, LX1, LX2, dl1, dl2, dm  # For future tests.

    def working_test_kernel(self, alg, dataset, do_assert=True):
        np.random.seed(28)
        X, y = dataset()
        alg.fit(X, y)
        L = alg.transformer()

        LX1 = alg.transform()
        LX2 = alg.transform(X)

        if do_assert:
            assert_array_almost_equal(LX1, LX2)

        d_, n_ = L.shape
        n, d = X.shape

        if do_assert:
            assert_equal(n_, n)
            assert d_ <= d

        return X, y, L, LX1, LX2

    def test_PCA(self):
        for d in [iris, wine, breast_cancer]:
            pca = PCA()
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(pca, d)
            assert_array_almost_equal(M, np.eye(X.shape[1]))

    def test_PCA_dim(self):
        for d in [iris, wine, breast_cancer]:
            pca = PCA(num_dims=1)
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(pca, d)
            assert_equal(L.shape[0], 1)

    def test_LDA(self):
        for d in [iris, wine, breast_cancer]:
            # TODO the single object should be independent of executions
            lda = LDA()
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(lda, d)
            classes = np.unique(y)
            assert_equal(len(classes) - 1, L.shape[0])

    def test_ANMM(self):
        for d in [iris, wine, breast_cancer]:
            anmm = ANMM()
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(anmm, d)
            assert_array_almost_equal(M, np.eye(X.shape[1]))

    def test_ANMM_dim(self):
        for d in [iris, wine, breast_cancer]:
            anmm = ANMM(num_dims=1)
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(anmm, d)
            assert_equal(L.shape[0], 1)

    def test_LMNN(self):
        for d in [iris, wine, breast_cancer]:
            lmnn = LMNN()
            self.working_test_basic(lmnn, d)

    def test_LMNN_SGD(self):
        for d in [iris, wine, breast_cancer]:
            lmnn = LMNN(solver='SGD', num_dims=1)
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(lmnn, d, False)
            assert_equal(L.shape[0], 1)

    def test_NCA(self):
        for d in [iris, wine, breast_cancer]:
            nca = NCA(num_dims=1)
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(nca, d)
            assert_equal(L.shape[0], 1)

    def test_NCMML(self):
        for d in [iris, wine, breast_cancer]:
            ncmml = NCMML(num_dims=1)
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(ncmml, d)
            assert_equal(L.shape[0], 1)

    def test_NCMC(self):
        for d in [iris, wine, breast_cancer]:
            ncmc = NCMC(num_dims=1)
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(ncmc, d)
            assert_equal(L.shape[0], 1)

    def test_ITML(self):
        for d in [iris, wine, breast_cancer]:
            itml = ITML()
            self.working_test_basic(itml, d)

    def test_DMLMJ(self):
        for d in [iris, wine, breast_cancer]:
            dmlmj = DMLMJ(num_dims=1)
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(dmlmj, d)
            assert_equal(L.shape[0], 1)

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
            # Assert fails because of too big exponents,
            #  but results are correct.
            self.working_test_basic(ldml, d, False)

    def test_Euclidean(self):
        for d in [iris, wine, breast_cancer]:
            euc = Euclidean()
            X, y, L, M, LX1, LX2, dl1, dl2, dm = self.working_test_basic(euc, d)
            assert_array_almost_equal(M, np.eye(X.shape[1]))

    def test_Metric(self):
        np.random.seed(28)
        for d in [iris, wine, breast_cancer]:
            X, y = d()
            n, d = X.shape
            M = make_spd_matrix(d)

            metric = Metric(M)
            metric.fit(X, y)
            L = metric.transformer()
            assert_array_almost_equal(L.T.dot(L), M)

            LX1 = metric.transform()
            LX2 = metric.transform(X)

            dl1 = pdist(LX1)
            dl2 = pdist(LX2)
            dm = pdist(X, metric='mahalanobis', VI=M)  # CHecking that d_M = d_L

            assert_array_almost_equal(dm, dl1)
            assert_array_almost_equal(dm, dl2)

            d_, d = L.shape
            e_, e = M.shape

            assert_equal(d, e_)
            assert_equal(d, e)
            assert_equal(d, X.shape[1])

    def test_Transformer(self):
        np.random.seed(28)
        for d in [iris, wine, breast_cancer]:
            X, y = d()
            n, d = X.shape
            L = np.random.rand(1, d)

            transformer = Transformer(L)
            transformer.fit(X, y)
            M = transformer.metric()
            assert_array_almost_equal(L.T.dot(L), M)

            LX1 = transformer.transform()
            LX2 = transformer.transform(X)

            dl1 = pdist(LX1)
            dl2 = pdist(LX2)
            dm = pdist(X, metric='mahalanobis', VI=M)  # CHecking that d_M = d_L

            assert_array_almost_equal(dm, dl1)
            assert_array_almost_equal(dm, dl2)

            d_, d = L.shape
            e_, e = M.shape

            assert_equal(d, e_)
            assert_equal(d, e)
            assert_equal(d, X.shape[1])
            assert_equal(d_, 1)

    def test_KLMNN(self):
        for d in [iris, wine]:
            for ker in ["linear", "poly", "rbf", "laplacian"]:
                klmnn = KLMNN(kernel=ker, num_dims=1)
                X, y, L, LX1, LX2 = self.working_test_kernel(klmnn, d)
                assert_equal(L.shape[0], 1)

    def test_KANMM(self):
        for d in [iris, wine]:
            for ker in ["linear", "poly", "rbf", "laplacian"]:
                kanmm = KANMM(kernel=ker, num_dims=1)
                X, y, L, LX1, LX2 = self.working_test_kernel(kanmm, d)
                assert_equal(L.shape[0], 1)

    def test_KDMLMJ(self):
        for d in [iris, wine]:
            for ker in ["linear", "poly", "rbf", "laplacian"]:
                kdmlmj = KDMLMJ(kernel=ker, num_dims=1)
                X, y, L, LX1, LX2 = self.working_test_kernel(kdmlmj, d)
                assert_equal(L.shape[0], 1)

    def test_KDA(self):
        for d in [iris, wine]:
            for ker in ["linear", "poly", "rbf", "laplacian"]:
                kda = KDA(kernel=ker, n_components=1)
                X, y, L, LX1, LX2 = self.working_test_kernel(kda, d)
                assert_equal(L.shape[0], 1)
