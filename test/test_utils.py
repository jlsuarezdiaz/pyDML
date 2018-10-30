from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler


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
