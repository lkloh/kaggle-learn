import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor

data = pandas.read_csv('./data/melb_data.csv')

# prediction target
y = data.Price

# predictors
predictors = [
  'Rooms',
  'Bathroom',
  'Landsize',
  'BuildingArea',
  'YearBuilt',
  'Lattitude',
  'Longtitude',
]
x = data[predictors]

print '\nnumber of datapoints: %d' % len(y)

print '\nnumber of null y values'
print y.isnull().sum()
print '\nnumber of null x values'
print x.isnull().sum()


def drop_missing_values(x, y):
  cols_with_missing_values = [col for col in x.columns if x[col].isnull().any()]
  reduced_x = x.drop(cols_with_missing_values, axis=1)
  mae = get_mae(reduced_x, y)
  print '\ndrop missing values gives mae: %f' % mae


def basic_imputation(x, y):
  imputer = Imputer()
  imputed_x = imputer.fit_transform(x)
  mae = get_mae(imputed_x, y)
  print '\nSimple imputed data gives mae: %f' % mae


def advanced_imputation(x, y):
  x_copy = x.copy()

  cols_with_missing_values = [col for col in x_copy.columns if x_copy[col].isnull().any()]
  for col in cols_with_missing_values:
    x_copy[col + '_was_missing'] = x_copy[col].isnull()

  print x_copy.head()

  imputer = Imputer()
  imputed_x = imputer.fit_transform(x_copy)

  mae = get_mae(imputed_x, y)

  print '\nAdvanced imputed data gives mae: %f' % mae


def get_mae(x, y):
  train_x, validate_x, train_y, validate_y = train_test_split(x, y,random_state = 0)
  model = RandomForestRegressor()
  model.fit(train_x, train_y)
  predicted_y = model.predict(validate_x)
  mae = mean_absolute_error(validate_y, predicted_y)
  return mae


drop_missing_values(x, y)
basic_imputation(x, y)
advanced_imputation(x, y)

