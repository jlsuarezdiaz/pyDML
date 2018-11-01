from test_utils import iris
from dml import ANMM, NCA, tune
from numpy.testing import assert_equal
import numpy as np


class TestTune:

    def test_tune_ANMM(self):
        X, y = iris()
        dml_params = {}
        tune_args = {'num_dims': [1, 2, 3, 4],
                     'n_friends': [1, 3, 5],
                     'n_enemies': [1, 3, 5]}
        metrics = [3, 5, 'acum_eig']
        ntests = 1
        n_folds = 5
        n_reps = 2
        for key in tune_args:
            ntests *= len(tune_args[key])

        results, best, anmm_best, detailed = tune(ANMM, X, y, dml_params=dml_params, tune_args=tune_args,
                                                  metrics=metrics, n_folds=n_folds, n_reps=n_reps, verbose=True)

        assert_equal(results.shape[0], ntests)
        assert_equal(results.shape[1], len(metrics))
        assert_equal(len(best[0]), len(tune_args))
        assert_equal(best[1], np.max(results.iloc[:, 0]))
        assert_equal(anmm_best.__class__, ANMM)
        assert_equal(len(detailed), ntests)
        for key in detailed:
            assert_equal(detailed[key].shape[0], n_folds * n_reps + 2)  # nº evals + MEAN + AVG
            assert_equal(detailed[key].shape[1], len(metrics))

    def test_tune_NCA(self):
        X, y = iris()
        dml_params = {}
        tune_args = {'learning_rate': ['constant', 'adaptive'],
                     'eta0': [0.001, 0.01, 0.1]}
        metrics = [3, 5, 'final_expectance']
        ntests = 1
        n_folds = 5
        n_reps = 2
        for key in tune_args:
            ntests *= len(tune_args[key])

        results, best, nca_best, detailed = tune(NCA, X, y, dml_params=dml_params, tune_args=tune_args,
                                                 metrics=metrics, n_folds=n_folds, n_reps=n_reps, verbose=True)

        assert_equal(results.shape[0], ntests)
        assert_equal(results.shape[1], len(metrics))
        assert_equal(len(best[0]), len(tune_args))
        assert_equal(best[1], np.max(results.iloc[:, 0]))
        assert_equal(nca_best.__class__, NCA)
        assert_equal(len(detailed), ntests)
        for key in detailed:
            assert_equal(detailed[key].shape[0], n_folds * n_reps + 2)  # nº evals + MEAN + AVG
            assert_equal(detailed[key].shape[1], len(metrics))
