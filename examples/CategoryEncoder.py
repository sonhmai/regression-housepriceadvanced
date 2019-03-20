import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


pd.options.display.float_format = '{:.2f}'.format

# make some data
df = pd.DataFrame({
    'color': ["a", "c", "a", "a", "b", "b"],
    'f2': ['cat1', 'cat2', 'cat1', 'cat4', 'cat3', 'cat4'],
    'outcome': [1, 2, 0, 0, 0, 1]})
# set up X and y
X = df.drop('outcome', axis=1)
y = df.drop('color', axis=1)
# checking
print(type(X))
print(type(y))
print(df.head())

# ordinal
print("\nOrdinal - Single Column 'color'")
ce_ord = ce.OrdinalEncoder(cols=['color'])
print(ce_ord.fit_transform(df))
print("\nOrdinal - Multicols")
ce_ord = ce.OrdinalEncoder()
print(ce_ord.fit_transform(df))


