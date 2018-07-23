# origin: https://www.dataquest.io/blog/machine-learning-tutorial/

import pandas as pd

listings = pd.read_csv('listings.csv')
print listings.shape
first_row = listings.iloc[0]
print first_row




