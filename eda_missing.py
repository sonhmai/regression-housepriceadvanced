import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from transformers import RemovingFeatures, piping
from data import get_data


class MissingProcessor():

    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.cols_drop = ['Alley', 'PoolQC', 'MiscFeature']
        self.get_cols_missing()

    def _get_cols_missing(self, df):
        # get list names of cols containing missing values
        assert isinstance(df, pd.DataFrame)
        col_index = df.columns[df.isnull().any()]
        cols = col_index.tolist()
        return cols

    def get_cols_missing(self):
        cols_missing_train = self._get_cols_missing(self.train)
        cols_missing_test = self._get_cols_missing(self.test)
        cols_train_not_test = [col for col in cols_missing_train if col not in cols_missing_test]
        cols_test_not_train = [col for col in cols_missing_test if col not in cols_missing_train]
        print("#missing cols in train", len(cols_missing_train))
        print("#missing cols in test", len(cols_missing_test))
        print("missing cols in train, not in test:", cols_train_not_test)
        print("missing cols in test, not in train:", cols_test_not_train)

    def get_cols_to_impute(self, cols):
        print(cols)
        cols_imputing = [col for col in cols if col not in self.cols_drop]
        print(cols_imputing)


if __name__ == '__main__':
    train, test = get_data()
    processor = MissingProcessor(train, test)

