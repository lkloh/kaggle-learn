import pandas as pd

file_path = './data/melb_data.csv'
data = pd.read_csv(file_path)

# Prediction Target
y = data.Price
