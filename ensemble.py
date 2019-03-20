from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from transformers import RemovingFeatures
from data import get_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import category_encoders as ce
import numpy as np
import warnings
from sklearn import svm
import xgboost as xgb
import pandas as pd
from submission import post_kaggle


def run_linear_model(X_train, X_val, y_train, y_val):
    cols_removing = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu']
    model = svm.SVR()
    encoder = ce.OneHotEncoder()
    pipe = Pipeline([
        ('dropping', RemovingFeatures(cols_removing)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder),
        ('scaler', StandardScaler()),
        ('reg', model),
    ])

    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    val_log = np.log(y_val)
    pred_log = np.log(val_pred)
    mse = mean_squared_error(val_log, pred_log)
    rmse = np.sqrt(mse)
    print("linear reg validation RMSE:", rmse)
    return pipe


def run_xgb(X_train, X_val, y_train, y_val):
    cols_removing = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu']
    model = xgb.XGBRegressor()
    encoder = ce.OneHotEncoder()
    pipe = Pipeline([
        ('dropping', RemovingFeatures(cols_removing)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder),
        ('scaler', StandardScaler()),
        ('reg', model),
    ])

    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    val_log = np.log(y_val)
    pred_log = np.log(val_pred)
    mse = mean_squared_error(val_log, pred_log)
    rmse = np.sqrt(mse)
    print("xgb validation RMSE:", rmse)
    return pipe


def run():
    train, test = get_data()
    X = train.iloc[:, 0:-1]
    y = train.iloc[:, -1]  # label SalePrice, last column
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    xgbreg = run_xgb(X_train, X_val, y_train, y_val)
    linreg = run_linear_model(X_train, X_val, y_train, y_val)
    submission = pd.DataFrame()
    submission['Id'] = test['Id']
    submission['LinPrice'] = linreg.predict(test)
    submission['XgbPrice'] = xgbreg.predict(test)
    submission['SalePrice'] = (submission['LinPrice'] + 2*submission['XgbPrice'])/3
    print(submission.head())
    df_submitting = submission[['Id', 'SalePrice']]
    path = 'output/submission.csv'
    df_submitting.to_csv(path, index=False)
    post_kaggle(path, "ensemble xgb and linear regression, no param tuning")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    run()
