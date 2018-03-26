import pandas
from sklearn.tree import DecisionTreeRegressor

data = pandas.read_csv('./data/melb_data.csv')
# prediction target
y = data.Price

# predictors
predictors = [
  'Bathroom',
  'BuildingArea',
  'Landsize',
  'Lattitude',
  'Longtitude',
  'Rooms',
  'YearBuilt',
]
x = data[predictors]
x = x.fillna(0)

# define model
model = DecisionTreeRegressor()

model.fit(x, y)

print("Making predictions for the following 5 houses:")
print(x.head())

print("\n Predictions")
print(model.predict(x.head()))

