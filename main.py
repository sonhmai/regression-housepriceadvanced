import pandas as pd
from data import get_data


def impute(train):
    # get list names of cols containing missing values
    assert isinstance(train, pd.DataFrame)
    col_index = train.columns[train.isnull().any()]
    cols = col_index.tolist()
    print(cols)
    cols_removed = ['Alley', 'PoolQC', 'MiscFeature']
    cols_imputing = [col for col in cols if col not in cols_removed]
    print(cols_imputing)


def encode():
    pass


def normalize():
    pass


if __name__ == '__main__':
    train, test = get_data()
    impute(train)