# Installs
#%%capture
#!pip install category_encoders==2.0.0

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import category_encoders as ce
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


import category_encoders as ce
from joblib import dump

# Load the raw data
df = pd.read_csv('https://raw.githubusercontent.com/JimKing100/Multiple_Offers/master/data/Marin-MulitOffers-2015-2019-Final.csv')


df.head()

# Map the area codes to city names
dict = {10:'Belvedere', 20:'Bolinas', 30:'Corte Madera', 40:'Dillon Beach',
        50:'Fairfax', 70:'Greenbrae', 80:'Inverness', 90:'Kentfield',
        100:'Larkspur', 110:'Marshall', 120:'Mill Valley', 130:'Muir Beach',
        140:'Nicasio', 150:'Novato', 170:'Point Reyes', 180:'Ross',
        190:'San Anselmo', 200:'San Geronimo Valley', 210:'San Rafael',
        220:'Sausalito', 230:'Stinson Beach', 240:'Tiburon', 250:'Tomales'}

df = df.replace({'Area': dict})
df.head()

# Split into train and test - 2015-2018 in train, 2019 in test
low_cutoff = 2015
high_cutoff = 2019
train = df[(df['Year'] >= low_cutoff) & (df['Year'] < high_cutoff)]
test  = df[df['Year'] >= high_cutoff]
print(train.shape)
print(test.shape)

# Split train into train and validation - 2015-2017 in train and 2018 in validation
cutoff = 2018
temp=train.copy()
train = temp[temp['Year'] < cutoff]
val  = temp[temp['Year'] >= cutoff]
print(train.shape, val.shape, test.shape)

# Encode and fit a XGBoost model
target = 'Selling Price'

features = train.columns.drop(target)
X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
y_test = test[target]

pipeline = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True),
    XGBRegressor(n_estimators=200, n_jobs=-1)
)

pipeline.fit(X_train, y_train)
dump(pipeline, 'pipeline.joblib')
y_pred = pipeline.predict(X_val)

# Print metrics for validation
val_mse = mean_squared_error(y_val, y_pred)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_pred)
val_r2 = r2_score(y_val, y_pred)
print('Validation Mean Absolute Error:', val_mae)
print('Validation R^2:', val_r2)
print('\n')

ty_pred = pipeline.predict(X_test)

# Print metrics for test
test_mse = mean_squared_error(y_test, ty_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, ty_pred)
test_r2 = r2_score(y_test, ty_pred)
print('Test Mean Absolute Error:', test_mae)
print('Test R^2:', test_r2)

# Add the prediction, difference and address to the final dataframe
final = test.copy()
final = final.reset_index()
final['Prediction'] = ty_pred
final['Difference'] = final['Prediction'] - final['Selling Price']
final['Prediction'] = final['Prediction'].astype(int)
final['Difference'] = final['Difference'].astype(int)
print(final.shape)
final.head()

# Calculate the metrics
final['Pred Percent'] = final['Difference']/final['Prediction']
pred_median_error = final['Pred Percent'].median()
pred_one_percent = (final['Pred Percent'][(final['Pred Percent'] >= -.01) &
                                          (final['Pred Percent'] <= .01)].count())/final.shape[0]

pred_five_percent = (final['Pred Percent'][(final['Pred Percent'] >= -.05) &
                                           (final['Pred Percent'] <= .05)].count())/final.shape[0]

pred_ten_percent = (final['Pred Percent'][(final['Pred Percent'] >= -.10) &
                                          (final['Pred Percent'] <= .10)].count())/final.shape[0]

pred_win_percent = (final['Pred Percent'][(final['Pred Percent'] >= 0)].count())/final.shape[0]

print('Median Error - %.4f%%' % (pred_median_error * 100))
print('Prediction Within 1 percent - %.4f%%' % (pred_one_percent * 100))
print('Prediction Within 5 percent - %.4f%%' % (pred_five_percent * 100))
print('Prediction Within 10 percent - %.4f%%' % (pred_ten_percent * 100))
print('Winning Prediction - %.4f%%' % (pred_win_percent * 100))

# Graph the prediction errors
import matplotlib.pyplot as plt
##%matplotlib inline

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(8,6))

plt.hist((final['Pred Percent'] * 100), bins=50)
plt.gca().set(title='Frequency Histogram of Prediction Errors', xlabel='Error Percentage', ylabel='Frequency');

