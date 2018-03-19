import pandas as pd

file_path = './data/melb_data.csv'
data = pd.read_csv(file_path)

'''
1. Print a list of the columns
'''
col_list = data.columns.tolist()
print col_list

'''
2. From the list of columns, find a name of the column with the sales prices of the homes.
   Use the dot notation to extract this to a variable (as you saw above to create melbourne_price_data.)
'''
sale_prices = data.Price

'''
3. Use the head command to print out the top few lines of the variable you just created.
'''
print sale_prices.head()

'''
4. Pick any two variables and store them to a new DataFrame (as you saw above to create two_columns_of_data.)
'''
var_names = ['Bathroom', 'Rooms']
frame_two = data[var_names]

'''
5. Use the describe command with the DataFrame you just created to see summaries of those variables. 
'''
print frame_two.describe()

