"""
Distance Metric Algorithm basis.

"""

# Authors: Juan Luis Suárez <jlsuarezd@gmail.com>
#
# License:

from numpy.linalg import cholesky
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_kernels
from .dml_utils import metric_to_linear


class DML_Algorithm(BaseEstimator, TransformerMixin):
    """
        Abstract class that defines a distance metric learning algorithm.
        Distance metric learning are implemented as subclasses of DML_Algorithm.
        A DML Algorithm can compute either a Mahalanobis metric matrix or an associated linear transformation.
        DML subclasses must override one of the following methods (metric or transformer), according to their computation way.
    """

    def __init__(self):
        raise NotImplementedError('Class DML_Algorithm is abstract and cannot be instantiated.')

    def metric(self):
        """Computes the Mahalanobis matrix from the transformation matrix.
        .. math:: M = L^T L

        Returns
        -------
        M : (d x d) matrix. M defines a metric whose distace is given by
        ..math:: d(x,y) = \\sqrt{(x-y)^TM(x-y)}.
        """
        if hasattr(self, 'M_'):
            return self.M_
        else:
            if hasattr(self, 'L_'):
                L = self.transformer()
                self.M_ = L.T.dot(L)
                return self.M_
            else:
                raise NameError("Metric was not defined. Algorithm was not fitted.")

    def transformer(self):
        """Computes a transformation matrix from the Mahalanobis matrix.
        ..math:: L = M^{1/2}

        Returns
        -------
        L : (d' x d) matrix, with d' <= d. It defines a projection. The distance can be calculated by
        ..math:: d(x,y) = \\|L(x-y)\\|_2.
        """

        if hasattr(self, 'L_'):
            return self.L_
        else:
            if hasattr(self, 'M_'):
                try:
                    L = cholesky(self.metric()).T
                    return L
                except:
                    L = metric_to_linear(self.metric())
                    return L
                self.L_ = L
                return L
            else:
                raise NameError("Transformer was not defined. Algorithm was not fitted.")

    def transform(self, X=None):
        """Applies the metric transformation.

        Parameters
        ----------
        X : (N x d) matrix, optional
            Data to transform. If not provided, the training data will be used.

        Returns
        -------
        transformed : (N x d') matrix
            Input data transformed to the metric space. The learned distance can be measured using
            the euclidean distance with the transformed data.
        """
        if X is None:
            X = self.X_
        else:
            X = check_array(X, accept_sparse=True)
        L = self.transformer()
        return X.dot(L.T)

    def metadata(self):
        """
        Obtains the algorithm metadata. Must be implemented in subclasses.

        Returns
        -------

        dict : A map from string to any.
        """
        return {}


class KernelDML_Algorithm(DML_Algorithm):
    """
        Abstract class that defines a kernel distance metric learning algorithm.
        Distance metric learning are implemented as subclasses of KernelDML_Algorithm.
        A Kernel DML Algorithm can compute a (d' x n) transformer that maps the high dimensional data using the kernel trick.
        Kernel DML subclasses must override the transformer method, providing the matrix A that performs the kernel trick, that is

        .. math:: Lx = A(K(x_1,x),\\dots,K(x_n,x)),

        where L is the high dimensional transformer and K is the kernel function.
    """

    def __init__(self):
            raise NotImplementedError('Class KernelDML_Algorithm is abstract and cannot be instantiated.')

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel_):
            params = self.kernel_params_ or {}
        else:
            params = {'gamma': self.gamma_,
                      'degree': self.degree_,
                      'coef0': self.coef0_}

        return pairwise_kernels(X, Y, metric=self.kernel_, filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel_ == "precomputed"

    def transform(self, X=None):
        """Applies the kernel transformation.

        Parameters
        ----------
        X : (N x d) matrix, optional
            Data to transform. If not supplied, the training data will be used.

        Returns
        -------
        transformed: (N x d') matrix.
            Input data transformed by the learned mapping.
        """
        if X is None:
            X = self.X_
        else:
            X = check_array(X, accept_sparse=True)

        L = self.transformer()
        K = self._get_kernel(X, self.X_)
        return K.dot(L.T)
