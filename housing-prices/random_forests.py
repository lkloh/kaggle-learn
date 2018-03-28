import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pandas.read_csv('./data/train.csv')

# prediction target
y = data.SalePrice

# predictors
predictors = [
  'LotArea',
  'YearBuilt',
  '1stFlrSF',
  '2ndFlrSF',
  'FullBath',
  'BedroomAbvGr',
  'TotRmsAbvGrd',
]
x = data[predictors]

train_x, validate_x, train_y, validate_y = train_test_split(x, y,random_state = 0)
model = RandomForestRegressor()
model.fit(train_x, train_y)
predicted_y = model.predict(validate_x)
mae = mean_absolute_error(validate_y, predicted_y)
print mae

print 'MAE about 23500, less than when using RandomTree'

