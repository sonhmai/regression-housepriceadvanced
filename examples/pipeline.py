"""
FunctionTransformer
- turns any func into a transformer
- works well for stateless transformations
"""
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
import numpy


class Log1pTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xlog = numpy.log1p(X)
        return Xlog


"""
Custom transformer - one hot encoding
"""


class DummyTransformer(TransformerMixin):
    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        Xdict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(Xdict)
        return self

    def transform(self, X):
        # convert back to dataframe
        Xdict = X.to_dict('records')


# series of imputers








