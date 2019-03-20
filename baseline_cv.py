import transformers
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data import get_data
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import category_encoders as ce
import numpy as np
from submission import post_kaggle
from sklearn.preprocessing import StandardScaler
import warnings


def get_train_val(train):
    X = train.iloc[:, 0:-1]
    y = train.iloc[:, -1]  # label SalePrice, last column
    return train_test_split(X, y, test_size=0.3, random_state=42)


def run_all_steps(pipe, kaggle_msg):
    train, test = get_data()
    X_train, X_val, y_train, y_val = get_train_val(train)
    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    mse = mean_squared_error(np.log(y_val), np.log(val_pred))
    rmse = np.sqrt(mse)
    print("validation RMSE:", rmse)
    # predict test set
    test['SalePrice'] = pipe.predict(test)
    # submit to kaggle
    df_submitting = test[['Id', 'SalePrice']]
    path = 'output/submission.csv'
    df_submitting.to_csv(path, index=False)
    post_kaggle(path, kaggle_msg)


def run_baseline(encoder, kaggle_msg):
    cols_removing = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu']
    pipe = Pipeline([
        ('dropping', transformers.RemovingFeatures(cols_removing)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder),
        ('scaler', StandardScaler()),
        ('reg', XGBRegressor()),
    ])
    run_all_steps(pipe, kaggle_msg)


def run_baseline_cv(encoder, kaggle_msg):
    cols_removing = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu']
    pipe = Pipeline([
        ('dropping', transformers.RemovingFeatures(cols_removing)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder),
        ('scaler', StandardScaler()),
        ('reg', XGBRegressor()),
    ])
    run_all_steps_cv(pipe, kaggle_msg)


def run_baseline_add_poolqc(encoder, kaggle_msg):
    cols_removing = ['Alley', 'MiscFeature', 'Fence', 'FireplaceQu']
    pipe = Pipeline([
        ('dropping', transformers.RemovingFeatures(cols_removing)),
        ('PoolQC', transformers.PoolQCTransformer()),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder),
        ('scaler', StandardScaler()),
        ('reg', XGBRegressor()),
    ])
    run_all_steps_cv(pipe, kaggle_msg)


def run_all_steps_cv(pipe, kaggle_msg):
    train, test = get_data()
    X_train, X_val, y_train, y_val = get_train_val(train)
    param_grid = {
        'reg__n_estimators': [5, 7, 10, 15],
        'reg__learning_rate': [0.1, 0.4, 1],
        'reg__max_depth': [5, 7, 10],
    }
    # pipe.fit(X_train, y_train)
    searchcv = GridSearchCV(pipe, cv=4, param_grid=param_grid)
    searchcv.fit(X_train, y_train)
    pipe = searchcv.best_estimator_
    print("best params\n", searchcv.best_params_)
    print("\nbest scores", searchcv.best_score_)
    val_pred = pipe.predict(X_val)
    mse = mean_squared_error(np.log(y_val), np.log(val_pred))
    rmse = np.sqrt(mse)
    print("validation RMSE:", rmse)
    # predict test set
    test['SalePrice'] = pipe.predict(test)
    # submit to kaggle
    df_submitting = test[['Id', 'SalePrice']]
    path = 'output/submission.csv'
    df_submitting.to_csv(path, index=False)
    post_kaggle(path, kaggle_msg)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    run_baseline_cv(ce.OneHotEncoder(), kaggle_msg='xgb-cv naive,onehot,scaler')










