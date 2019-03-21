from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


class GarageTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        impute num features with 0
        impute categorical features with NA
        """
        num_cols = ['GarageYrBlt', 'GarageCars', 'GarageArea']
        cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        X[num_cols] = (X[num_cols]).fillna(0)
        X[cat_cols] = (X[cat_cols]).fillna('NA')
        return X


class RemovingFeatures(TransformerMixin):
    """
    removing feature missing to many values
    """
    def __init__(self, cols_removing):
        self.cols_removing = cols_removing

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.cols_removing, axis=1)


class PoolQCTransformer(TransformerMixin):
    """
    transform PoolQC column
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        map_poolqc = {"PoolQC": {"Ex": 3, "Gd": 2, "Fa": 1, np.nan: 0}}
        return X.replace(map_poolqc)


class PoolAreaTransformer(TransformerMixin):
    """
    replace nan in PoolArea with 0
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = {"PoolArea": 0}
        return X.fillna(value=values)


class GrLivAreaTransformer(TransformerMixin):
    """
    perform log transformation of GrLivArea col because it is right-skewed in
    both train and test set
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['GrLivArea'] = np.log1p(X['GrLivArea'])  # log1p(x) more accurate for small x
        return X


class MSSubClassTransformer(TransformerMixin):
    """
    one-hot encoded this feature because it is categorical
    instead of recognize it as an int feature
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        col = 'MSSubClass'
        dummy = pd.get_dummies(X[col], prefix=col)
        drop_col = X.drop(col, axis=1)
        df = pd.concat([dummy, drop_col], axis=1)
        return df


class RemodAfterAdding(TransformerMixin):
    """
    add column RemodAfter = YearRemodAdd - YearBuilt
    YearBuilt and YearRemodAdd has no missing values
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['RemodAfter'] = X['YearRemodAdd'] - X['YearBuilt']
        return X

# cols to impute
# ['LotFrontage', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'Fence']

def piping(train):
    estimators = list()
    estimators.append(('a', RemovingFeatures()))  # remove features too many missing values

    pipe = Pipeline(estimators)
    df = pipe.transform(train)
    print(df.head())



