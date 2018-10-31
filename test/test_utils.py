from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def Xy_dataset(load_function):
    data = load_function()
    X = data['data']
    y = data['target']
    mms = MinMaxScaler()
    X = mms.fit_transform(X)
    return X, y


def iris():
    return Xy_dataset(load_iris)


def wine():
    return Xy_dataset(load_wine)


def breast_cancer():
    return Xy_dataset(load_breast_cancer)


def digits_data():
    data = load_digits()     # DIGITS
    X = data['data']
    y = data['target']

    return X, y


def draw_vector(v0, v1, ax=None, col='black'):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0, color=col)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
