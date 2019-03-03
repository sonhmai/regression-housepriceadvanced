from sklearn.base import TransformerMixin


class RemovingFeatures(TransformerMixin):
    """
    removing feature missing to many values
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_removing = ['Alley', 'PoolQC', 'MiscFeature']
        X.drop(cols_removing, axis=1, inplace=True)
        return X


#



