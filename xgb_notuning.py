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
from scipy import stats
import submission


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
    run_all_steps_baseline(pipe, kaggle_msg)


def run_baseline_pool(encoder):
    cols_removing = ['Id', 'Alley', 'MiscFeature', 'Fence', 'FireplaceQu']
    cols_time = ['MoSold', 'YrSold', 'YearBuilt', 'YearRemodAdd']
    cols_removing2 = cols_removing + cols_time
    pipe = Pipeline([
        ('dropping', transformers.RemovingFeatures(cols_removing2)),
        ('PoolQC', transformers.PoolQCTransformer()),
        ('PoolArea', transformers.PoolAreaTransformer()),
        # ('GrLivArea', transformers.GrLivAreaTransformer()),
        # ('AddingColRemodAfter', transformers.RemodAfterAdding()),
        # ('MSSubClass', transformers.MSSubClassTransformer()),
        ('garage', transformers.GarageTransformer()),
        ('remove garage yr build', transformers.RemovingFeatures(['GarageYrBlt'])),
        ('impute LotFrontage 0', transformers.LotFrontageTransformer()),
        ('impute MasVnr type and area', transformers.MasVnrTransformer()),
        ('impute Electrical - most frequent', transformers.ElectricalTransformer()),
        ('impute basement features', transformers.BasementTransformer()),
        ('encoder', encoder),
        ('scaler', StandardScaler()),
        ('reg', XGBRegressor()),
    ])
    kaggle_msg = 'xgb, removing outliers'
    run_all_steps_baseline(pipe, kaggle_msg)


def run_all_steps_baseline(pipe, kaggle_msg):
    train, test = get_data()

    # remove outliers
    filter_not_outlier = np.abs(stats.zscore(train['GrLivArea'])) < 2.7
    train = train[filter_not_outlier]

    X_train, X_val, y_train, y_val = get_train_val(train)
    param_grid = {
        'reg__n_estimators': 15,
        'reg__learning_rate': 0.4,
        'reg__max_depth': 5,
    }
    pipe.fit(X=X_train, y=y_train)
    val_pred = pipe.predict(X_val)
    mse = mean_squared_error(y_val, val_pred)
    rmse = np.sqrt(mse)
    print("RMSE without log:", rmse)
    mse = mean_squared_error(np.log(y_val), np.log(val_pred))
    rmse = np.sqrt(mse)
    print("validation RMSE:", rmse)
    submission.submit(test, pipe, 'xgb')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    encoder = ce.OneHotEncoder(handle_unknown='ignore')
    run_baseline_pool(encoder)









