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

num_houses_to_predict = 7

print("Making predictions for the following %d houses:" % num_houses_to_predict)
print(x.head(n=num_houses_to_predict))

print("\n Predictions")
print(model.predict(x.head(n=num_houses_to_predict)))

