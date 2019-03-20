from sklearn.impute import SimpleImputer
import pandas as pd
from data import get_data


def impute_cols():
    train, test = get_data()
    imputer = SimpleImputer(strategy='most_frequent')
    cols = ['GarageCond', 'GarageType', 'LotFrontage']
    GarageCond = train[cols]
    train[cols] = imputer.fit_transform(GarageCond)
    for col in cols:
        print(train[col].value_counts(dropna=False), end='\n\n')


def impute_df():
    train, test = get_data()
    imputer = SimpleImputer(strategy='most_frequent')
    a = imputer.fit_transform(train)
    train[:] = imputer.fit_transform(train)

    print(type(a), end='\n---\n')
    print(a.shape, end='\n---\n')
    print(a[:5, :5])
    print(train.info())


if __name__ == '__main__':
    impute_df()


