import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from transformers import RemovingFeatures


def train_val():
    train = pd.read_csv('data/train.csv')
    print(train.head())

    return train


def impute(train):
    # list names of cols containing missing values
    assert isinstance(train, pd.DataFrame)
    col_index = train.columns[train.isnull().any()]
    cols = col_index.tolist()


def encode():
    pass


def normalize():
    pass


def piping():
    estimators = list()
    # estimators.append('train_val', FunctionTransformer(train_val))
    estimators.append(('a', RemovingFeatures()))
    pipe = Pipeline(estimators)
    train = train_val()
    df = pipe.transform(train)
    print(df.head())


if __name__ == '__main__':
    piping()
