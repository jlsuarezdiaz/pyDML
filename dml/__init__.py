"""
The Distance Metric Learning module.
"""

from __future__ import absolute_import

from .nca import NCA
from .knn import kNN
from .lda import LDA
from .pca import PCA
from .lmnn import LMNN, KLMNN
from .lsi import LSI
from .anmm import ANMM, KANMM
from .itml import ITML
from .dmlmj import DMLMJ, KDMLMJ
from .ncmml import NCMML
from .ncmc import NCMC, NCMC_Classifier
from .kda import KDA
from .dml_eig import DML_eig
from .mcml import MCML
from .ldml import LDML
from .multidml_knn import MultiDML_kNN
from .dml_plot import *
from .base import Metric, Transformer, Euclidean
from .tune import *
