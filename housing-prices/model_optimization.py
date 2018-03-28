import pandas
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

def get_mae(max_leaf_nodes, train_x, validate_x, train_y, validate_y):
  model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
  model.fit(train_x, train_y)
  predicted_prices = model.predict(validate_x)
  return mean_absolute_error(validate_y, predicted_prices)

for max_leaf_nodes in [5, 50, 500, 5000]:
  print 'mae for %d is %d' % (max_leaf_nodes, get_mae(max_leaf_nodes, train_x, validate_x, train_y, validate_y))

print 'best mae: 50 leafs'
