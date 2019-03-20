import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    train = get_train()
    test = get_test()
    return train, test

def get_train():
    return pd.read_csv('data/train.csv')

def get_test():
    return pd.read_csv('data/test.csv')





