import toy_datasets
from test_utils import digits_data, draw_vector, iris_data

from dml import (NCA, LDA, PCA, LMNN, ANMM, NCMML, knn_plot, dml_plot, classifier_plot,
                 classifier_pairplots, knn_pairplots, dml_pairplots, dml_multiplot, NCMC_Classifier)

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from scipy.spatial import Voronoi, voronoi_plot_2d


seed = 28


class TestPlot:
    nplots_ = 1

    def newsave(self):
        TestPlot.nplots_ += 1
        plt.savefig("plots/testplot" + str(TestPlot.nplots_) + ".png")

    def test_plot1(self):
        np.random.seed(seed)
        X, y = toy_datasets.circular_toy_dataset(rads=[1, 2, 3], samples=[200, 200, 200],
                                                 noise=[0.4, 0.4, 0.4], seed=seed)
        knn_plot(X, y, k=3, cmap="gist_rainbow", figsize=(15, 8))
        self.newsave()
        plt.close()

    def test_plot2(self):
        np.random.seed(seed)
        X, y = toy_datasets.hiperplane_toy_dataset(seed=seed)
        knn_plot(X, y, k=3, cmap="gist_rainbow", figsize=(15, 8))
        self.newsave()
        plt.close()

    def test_plot3(self):
        np.random.seed(seed)
        X, y = toy_datasets.iris2d_toy_dataset()
        iris_labels = ['Setosa' if yi == 0 else 'Versicolor' if yi == 1 else 'Virginica' for yi in y]
        knn_plot(X, iris_labels, region_intensity=0.2, k=3,
                 label_colors=["red", "green", "blue"], figsize=(15, 8))
        self.newsave()
        plt.close()

    def test_plot4(self):
        np.random.seed(seed)
        X, y = toy_datasets.balls_toy_dataset(seed=seed)
        knn_plot(X, y, k=3, label_colors=['red', 'blue', 'green', 'orange', 'purple'], figsize=(15, 8))
        self.newsave()
        plt.close()

    def test_plot5(self):
        np.random.seed(seed)
        X = np.array([[3, 1], [4, 2], [3, 4], [5, 4], [-1, 1], [1, 2], [2, 2], [3, 3]])
        y = np.array(['OK', 'OK', 'OK', 'OK', 'ERR', 'ERR', 'ERR', 'ERR'])
        knn_plot(X, y, k=3, label_colors=["red", "blue"], figsize=(15, 8))
        self.newsave()
        plt.close()

    def test_plot6(self):
        np.random.seed(seed)
        X = np.array([[0, 0], [0, 1], [1.1, 0], [2, 0]])
        y = np.array(['RED', 'RED', 'BLUE', 'BLUE'])
        knn_plot(X, y, k=3, label_colors=["red", "blue"], figsize=(15, 8))
        self.newsave()
        plt.close()

    def test_plot7(self):
        np.random.seed(seed)
        X, y = digits_data()
        selected = np.where(np.isin(y, [0, 1, 3, 4, 6, 9]))[0]
        X, y = X[selected, :], y[selected]
        lda = LDA(num_dims=2)
        anmm = ANMM(num_dims=2, n_friends=1, n_enemies=1)
        algs = [lda, anmm]

        for alg in algs:
            alg.fit(X, y)
            Xn = alg.transform(X)
            if(Xn.shape[1] < 2):
                X2 = np.empty([Xn.shape[0], 2])
                for i in range(Xn.shape[0]):
                    X2[i, :] = [Xn[i, 0], 0.0]
                Xn = X2

            knn_plot(Xn, y, k=3, cmap="gist_rainbow", region_intensity=0.4, legend_plot_points=True,
                     figsize=(15, 8))
            self.newsave()
        plt.close()

    def test_plot8(self):
        np.random.seed(seed)
        svmc = SVC()
        X = np.array([[3, 1], [4, 2], [3, 4], [5, 4], [-1, 1], [1, 2], [2, 2], [3, 3]])
        y = np.array(['OK', 'OK', 'OK', 'OK', 'ERR', 'ERR', 'ERR', 'ERR'])

        classifier_plot(X, y, svmc, label_colors=['red', 'blue'], figsize=(15, 8))
        self.newsave()
        plt.close()

    def test_plot9(self):
        np.random.seed(seed)
        X, y = digits_data()
        selected = np.where(np.isin(y, [0, 1, 3, 4, 6, 9]))[0]
        X, y = X[selected, :], y[selected]

        lda = LDA(num_dims=2)
        anmm = ANMM(num_dims=2, n_friends=1, n_enemies=1)

        dml_multiplot(X, y, nrow=2, ncol=2, ks=[1, 1, 11, 11], dmls=[lda, anmm, lda, anmm],
                      title="Comparing DMLS",
                      subtitles=["k=1, LDA", "k=1, ANMM", "k=11, LDA", "k=11, ANMM"],
                      cmap="gist_rainbow", plot_points=True, figsize=(20, 16))
        self.newsave()
        plt.close()

    def test_plot10(self):
        np.random.seed(seed)
        X, y = digits_data()
        lda = LDA(num_dims=5)
        anmm = ANMM(num_dims=5)
        X = anmm.fit_transform(X, y)
        knn = KNeighborsClassifier()
        classifier_pairplots(X, y, knn, sections="zeros", cmap="gist_rainbow", figsize=(25, 25))
        self.newsave()
        classifier_pairplots(X, y, knn, sections="mean", cmap="gist_rainbow", figsize=(25, 25))
        self.newsave()

        knn_pairplots(X, y, k=3, dml=lda, cmap="gist_rainbow", figsize=(25, 25))
        self.newsave()
        plt.close()

    def test_plot11(self):
        np.random.seed(seed)
        X, y = iris_data()
        iris_labels = ['Setosa' if yi == 0 else 'Versicolor' if yi == 1 else 'Virginica' for yi in y]
        X = pd.DataFrame(X, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
        svmc = SVC()

        classifier_pairplots(X, iris_labels, svmc, cmap="gist_rainbow", figsize=(20, 20))
        self.newsave()
        classifier_pairplots(X, iris_labels, svmc, xattrs=['Sepal Length', 'Sepal Width'],
                             yattrs=['Petal Length', 'Petal Width'], cmap="gist_rainbow", figsize=(20, 20))
        self.newsave()
        plt.close()

    def test_plot12(self):
        np.random.seed(seed)
        X = np.array([[0.0, 0.1], [0.5, 0.1], [-0.5, -0.1], [1.0, 0.2], [-1.0, -0.1], [0.1, 1.0],
                      [-0.1, 1.0], [0.1, -1.0], [-0.1, -1.0]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2])

        lmnn = LMNN(max_iter=1000, learning_rate="adaptive", eta0=1.0, k=1, mu=0.5, tol=1e-15, prec=1e-15)
        dml_multiplot(X, y, 2, 2, ks=[1, 1, 1], dmls=[None, lmnn, lmnn], transforms=[True, True, False],
                      title="LMNN", subtitles=["Original", "Transformed", "Region LMNN+KNN"],
                      cmap="rainbow", figsize=(20, 20))
        self.newsave()
        knn_plot(X, y, k=1, cmap="gist_rainbow", figsize=(15, 8))
        self.newsave()
        knn_plot(X, y, k=1, cmap="gist_rainbow", figsize=(15, 8),
                 transformer=np.array([[0.0, 0.0], [0.0, 3.0]]), transform=False)
        self.newsave()
        plt.close()

    def test_plot13(self):
        np.random.seed(seed)
        X, y = iris_data()
        X = X[:, [0, 2]]
        dml = NCMML()
        clf = NearestCentroid()
        dml_plot(X, y, clf, cmap="gist_rainbow", figsize=(15, 8))
        self.newsave()
        dml_plot(X, y, dml=dml, clf=clf, cmap="gist_rainbow", figsize=(15, 8))
        self.newsave()
        dml_pairplots(X, y, dml=dml, clf=clf, cmap="gist_rainbow", figsize=(15, 8))
        self.newsave()
        plt.close()

    def test_plot14(self):
        np.random.seed(seed)

        L = np.array([[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]])

        X = [[i, 0.5 * np.random.random() - 0.25] for i in np.linspace(-1, 1, 50)]
        X = np.array(X)

        y = np.array([1 for i in np.linspace(-1, 1, 50)])

        f, ax = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(12, 12))
        f.tight_layout()
        LX = X.dot(L.T)
        ax[0, 0].set_xlim((-1, 1))
        ax[0, 0].set_ylim((-1, 1))
        ax[0, 0].scatter(LX[:, 0], LX[:, 1])

        pca = PCA(num_dims=2)
        pca.fit(LX, y)
        LL = pca.transformer()
        vv = pca.skpca_.explained_variance_
        draw_vector([0, 0], 3 * LL[:, 0] * vv[0], ax[0, 0], 'red')
        draw_vector([0, 0], 3 * LL[:, 1] * vv[1], ax[0, 0], 'green')

        ax[0, 1].set_xlim((-1, 1))
        ax[0, 1].set_ylim((-1, 1))
        LLX = pca.transform()
        ax[0, 1].scatter(LLX[:, 0], LLX[:, 1])

        pca2 = PCA(num_dims=1)
        pca2.fit(LX, y)
        LLX2 = pca2.transform()
        LLXy = [0 for i in range(LLX2.size)]
        ax[1, 0].set_xlim((-1, 1))
        ax[1, 0].set_ylim((-1, 1))
        ax[1, 0].scatter(LLX2[:, 0], LLXy)

        UWX = pca2.skpca_.inverse_transform(LLX2)
        ax[1, 1].set_xlim((-1, 1))
        ax[1, 1].set_ylim((-1, 1))
        ax[1, 1].scatter(UWX[:, 0], UWX[:, 1])
        ax[1, 1].scatter(LX[:, 0], LX[:, 1], c='lightblue')
        self.newsave()

        ff, ax = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 4))
        ff.tight_layout()
        ax[0].set_xlim((-1, 1))
        ax[0].set_ylim((-1, 1))
        ax[0].scatter(LX[:, 0], LX[:, 1])
        draw_vector([0, 0], 3 * LL[:, 0] * vv[0], ax[0], 'red')
        draw_vector([0, 0], 4 * LL[:, 1] * vv[1], ax[0], 'green')

        ax[1].set_xlim((-1, 1))
        ax[1].set_ylim((-1, 1))
        ax[1].scatter(UWX[:, 0], UWX[:, 1])
        ax[1].scatter(LX[:, 0], LX[:, 1], c='lightblue')
        self.newsave()
        plt.close()

    def test_plot15(self):
        np.random.seed(seed)
        L = np.array([[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]])
        X, y = toy_datasets.balls_toy_dataset(centers=[[0.0, 0.35], [0.0, -0.35]],
                                              rads=[0.3, 0.3], samples=[50, 50], noise=[0.0, 0.0])

        M = np.array([[3.0, 0], [0, 1]])
        aux = X.dot(M.T)
        LX = aux.dot(L.T)

        f, ax = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(12, 6))
        f.tight_layout()
        ax[0].set_xlim(-1, 1)
        ax[0].set_ylim(-1, 1)
        ax[1].set_ylim(-1, 1)
        ax[1].set_xlim(-6, 6)

        ax[0].scatter(LX[:, 0], LX[:, 1], c=y, cmap='rainbow')

        pca = PCA()
        lda = LDA()

        pca.fit(LX)
        lda.fit(LX, y)
        LLX = lda.transform()
        LL = lda.transformer()[0, :]
        LLp = pca.transformer()[0, :]
        LLXy = [0 for i in range(LLX.size)]
        ax[1].plot([-10, 10], [0, 0], c='green')
        sc = ax[1].scatter(LLX[:, 0], LLXy, c=y, cmap='rainbow')
        ax[0].plot([-LL[0], LL[0]], [-LL[1], LL[1]], c='green')
        ax[0].plot([-2 * LLp[0], 2 * LLp[0]], [-2 * LLp[1], 2 * LLp[1]], c='orange')
        handles = [plt.plot([], color=sc.get_cmap()(sc.norm(c)), ls="", marker="o",
                            mec="k")[0] for c in [-1, 1]]
        f.legend(handles, ['A', 'B'], loc="lower right")
        self.newsave()
        plt.close()

    def test_plot16(self):
        np.random.seed(seed)
        X, y = toy_datasets.balls_toy_dataset(centers=[[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
                                              rads=[0.3, 0.3, 0.3], samples=[50, 50, 50],
                                              noise=[0.1, 0.1, 0.1])
        y[y == 2] = 0
        y = y.astype(int)

        ncm = NearestCentroid()
        ncmc = NCMC_Classifier(centroids_num=[2, 1])
        dml_multiplot(X, y, nrow=1, ncol=2, clfs=[ncm, ncmc], cmap='rainbow',
                      subtitles=['NCM', 'NCMC'], figsize=(6, 3))
        self.newsave()
        plt.close()

    def test_plot17(self):
        np.random.seed(seed)
        svm = SVC(kernel='linear')
        X, y = toy_datasets.hiperplane_toy_dataset(ws=[[1, 1]], bs=[[0, 0]], nsamples=100, noise=0.0)
        y = y.astype(int)
        X[y == 1] += np.array([0.1, 0.1])
        X[y == 0] -= np.array([0.1, 0.1])
        classifier_plot(X, y, svm, cmap='rainbow', figsize=(6, 6))
        self.newsave()
        plt.close()

    def test_plot18(self):
        np.random.seed(seed)
        svm = SVC(kernel='linear')
        Xa = np.array([[1.8 * np.random.random() + -0.9, 0.0] for i in range(40)])
        Xb = np.array([[0.4 * np.random.random() + 1.1, 0.0] for i in range(20)])
        Xc = np.array([[0.4 * np.random.random() - 1.5, 0.0] for i in range(20)])
        X = np.concatenate([Xa, Xb, Xc], axis=0)
        y = np.empty(X.shape[0])
        y[np.abs(X[:, 0]) > 1] = 1
        y[np.abs(X[:, 0]) < 1] = -1
        y = y.astype(int)
        f, ax = plt.subplots(sharex='row', sharey='row', figsize=(6, 3))
        sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', s=20, edgecolor='k')
        handles = [plt.plot([], color=sc.get_cmap()(sc.norm(c)), ls="", marker="o", mec="k")[0] for c in [-1, 1]]
        ax.legend(handles, [-1, 1], loc="lower right")
        self.newsave()

        f1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.set_xlim([-1.6, 1.6])
        ax1.set_ylim([-2.0, 2.0])
        sc = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', s=20, edgecolor='k')
        handles = [plt.plot([], color=sc.get_cmap()(sc.norm(c)), ls="", marker="o", mec="k")[0] for c in [-1, 1]]
        ax1.legend(handles, [-1, 1], loc="lower right")

        X[:, 1] = X[:, 0] * X[:, 0]
        L = np.array([[1, 0], [0, 0]])
        svq = SVC(kernel='poly', degree=2)
        dml_multiplot(X, y, nrow=1, ncol=2, clfs=[svm, svq], transformers=[None, L], transforms=[False, True],
                      cmap='rainbow', figsize=(12, 6))
        self.newsave()

        f2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.set_xlim([-1.6, 1.6])
        ax2.set_ylim([-0.1, 2.5])
        ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', s=20, edgecolor='k')
        handles = [plt.plot([], color=sc.get_cmap()(sc.norm(c)), ls="", marker="o", mec="k")[0] for c in [-1, 1]]
        ax2.legend(handles, [-1, 1], loc="lower right")
        self.newsave()

        dml_plot(X, y, clf=svm, transform=False, cmap='rainbow', figsize=(4, 4), xrange=[-1.6, 1.6],
                 yrange=[-0.1, 2.5])
        self.newsave()
        dml_plot(X, y, clf=svq, transformer=L, transform=True, cmap='rainbow', xrange=[-1.6, 1.6],
                 yrange=[-2.0, 2.0], figsize=(4, 4))
        self.newsave()
        plt.close()

    def test_plot19(self):
        np.random.seed(seed)
        X, y = digits_data()
        selected = np.where(np.isin(y, [0, 1, 3, 4, 6, 9]))[0]
        X = X[selected, :]
        y = y[selected]

        lda = LDA(num_dims=2)

        knn_plot(X, y, k=3, dml=lda, cmap="gist_rainbow", figsize=(12, 8))
        self.newsave()
        plt.close()

    def test_plot20(self):
        np.random.seed(seed)
        Xa = np.array([[i, 0.0] for i in np.linspace(-10.0, 10.0, 40)])
        Xb = np.array([[i, 0.2] for i in np.linspace(-10.0, 10.0, 20)])
        Xc = np.array([[i, -0.2] for i in np.linspace(-10.0, 10.0, 20)])
        ya = ['A' for i in range(40)]
        yb = ['B' for i in range(20)]
        yc = ['C' for i in range(20)]
        X = np.concatenate([Xa, Xb, Xc], axis=0)
        y = np.concatenate([ya, yb, yc])

        nca = NCA()
        # rcParams['text.usetex'] = 'True' # Latex installed needed
        # rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        nca.fit(X, y)
        dml_multiplot(X, y, nrow=1, ncol=3, ks=[1, 1, 1], fitted=True, dmls=[None, nca, nca],
                      transforms=[False, False, True], cmap="gist_rainbow",
                      # subtitles=[r'$M=\begin{pmatrix}1 & 0 \\ 0 & 1 \end{pmatrix}$',
                      #            r'$M \approx \begin{pmatrix} 0 & -0.004 \\ -0.004 & 27.5 \end{pmatrix}$',
                      #            r'$L \approx \begin{pmatrix} -0.0001 & 0.073 \\ -0.0008 & 5.24 \end{pmatrix}$'],
                      figsize=(18, 6))
        self.newsave()
        plt.close()

    def test_plot21(self):
        np.random.seed(seed)

        X = [[i, 0.2 * np.random.random() - 0.1] for i in np.linspace(-1, 1, 50)]
        X = np.array(X)
        L = np.array([[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]])
        y = np.array([1 for i in np.linspace(-1, 1, 50)])

        f, ax = plt.subplots(nrows=1, ncols=3, sharex='col', sharey='row', figsize=(18, 6))

        LX = X.dot(L.T)
        ax[0].set_xlim((-1.2, 1.2))
        ax[0].set_ylim((-1.2, 1.2))
        ax[0].scatter(LX[:, 0], LX[:, 1])
        # ax[0].set_title(r'$L=\begin{pmatrix}1 & 0 \\ 0 & 1 \end{pmatrix}$')

        pca = PCA(num_dims=2)
        pca.fit(LX, y)

        ax[1].set_xlim((-1.2, 1.2))
        ax[1].set_ylim((-1.2, 1.2))
        # ax[1].set_title(r'$L=\begin{pmatrix}\sqrt{2}/2 & \sqrt{2}/2 \\ \sqrt{2}/2 & -\sqrt{2}/2 \end{pmatrix}$')
        LLX = pca.transform()
        ax[1].scatter(LLX[:, 0], LLX[:, 1])

        pca2 = PCA(num_dims=1)
        pca2.fit(LX, y)
        LLX2 = pca2.transform()
        LLXy = [0 for i in range(LLX2.size)]
        ax[2].set_xlim((-1.2, 1.2))
        ax[2].set_ylim((-1.2, 1.2))
        # ax[2].set_title(r'$L=\begin{pmatrix}\sqrt{2}/2 & \sqrt{2}/2\end{pmatrix}$')
        ax[2].scatter(LLX2[:, 0], LLXy)
        self.newsave()
        plt.close()

    def test_plot22(self):
        np.random.seed(seed)
        X, y = toy_datasets.circular_toy_dataset(rads=[1, 2], samples=[200, 200], noise=[0.0, 0.0])
        yy = y.astype(str)
        yy[40:200] = '?'
        yy[220:] = '?'
        knn1 = KNeighborsClassifier(1)
        knn1.fit(X[np.isin(yy, ['0', '1'])], yy[np.isin(yy, ['0', '1'])])
        knn2 = KNeighborsClassifier(1)
        knn2.fit(X, y)
        dml_multiplot(X, yy, clfs=[knn1, knn2], label_colors=['red', 'blue', 'lightgray'], fitted=True, figsize=(12, 6))
        self.newsave()
        plt.close()

    def test_plot23(self):
        np.random.seed(seed)
        X, y = iris_data()
        y = ['Setosa' if yi == 0 else 'Versicolor' if yi == 1 else 'Virginica' for yi in y]
        X = pd.DataFrame(X, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
        arr = ['Sepal Length', 'Sepal Width']
        X = X[arr]
        dml_multiplot(X, y, ks=[1, 30], cmap='rainbow', subtitles=[r'$k=1$', r'$k=30$'], figsize=(12, 6))
        self.newsave()
        plt.close()

    def test_plot24(self):
        np.random.seed(seed)
        X = [[np.random.random(), np.random.random()] for i in range(25)]
        vor = Voronoi(X)
        voronoi_plot_2d(vor)
        self.newsave()
        plt.close()
