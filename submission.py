import subprocess
import pprint as pp

COMPETITION = 'house-prices-advanced-regression-techniques'


def post_kaggle(path, message):
    command = f'kaggle competitions submit -c {COMPETITION} -f {path} -m "{message}"'
    bytes_output = subprocess.check_output(command, shell=True)
    pp.pprint(str(bytes_output, encoding='UTF-8'))


def submit(test, pipe, message):
    print("Limit for this competition is 5 submissions per day!")
    test['SalePrice'] = pipe.predict(test)
    df_submitting = test[['Id', 'SalePrice']]  # create df for submitting
    path = 'output/submission.csv'
    df_submitting.to_csv(path, index=False)  # create scv file for submitting
    post_kaggle(path, message)  # submit the csv file with missing



