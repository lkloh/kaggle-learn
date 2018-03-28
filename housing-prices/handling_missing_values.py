import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
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

train_x, validate_x, train_y, validate_y = train_test_split(x, y,random_state = 0)
model = RandomForestRegressor()
model.fit(train_x, train_y)
predicted_y = model.predict(validate_x)
mae = mean_absolute_error(validate_y, predicted_y)
print mae

print 'MAE about 23500, less than when using RandomTree'

