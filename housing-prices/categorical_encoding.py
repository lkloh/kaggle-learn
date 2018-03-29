import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor

'''
Filling empty values in with mean:
https://stackoverflow.com/questions/18689823/pandas-dataframe-replace-nan-values-with-average-of-columns
'''

data = pandas.read_csv('./data/melb_data.csv')

def get_mae(x, y):
  train_x, validate_x, train_y, validate_y = train_test_split(x, y,random_state = 0)
  model = RandomForestRegressor()
  model.fit(train_x, train_y)
  predicted_y = model.predict(validate_x)
  mae = mean_absolute_error(validate_y, predicted_y)
  return mae


# prediction target
y = data.Price

# predictors
predictors = [
  'Suburb',
  'Car',
  'Rooms',
  'Bathroom',
  'Landsize',
  'BuildingArea',
  'YearBuilt',
  'Lattitude',
  'Longtitude',
]
x = data[predictors]

x_encode = pandas.get_dummies(x)

print x_encode.head()

x_copy = x_encode.copy()

# fill in missing data
cols_with_missing_values = [col for col in x_copy.columns if x_copy[col].isnull().any()]
for col in cols_with_missing_values:
  x_copy[col + '_was_missing'] = x_copy[col].isnull()

# replace NaN
x_copy = x_copy.fillna(x_copy.mean())

mae = get_mae(x_copy, y)

print 'mae: %f' % mae


