import numpy
from sklearn.base import BaseEstimator, TransformerMixin

class DenseTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return X.todense()

    def fit(self, X, y=None):
        return self
