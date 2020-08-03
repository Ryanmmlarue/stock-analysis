""" 
author: Ryan LaRue
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


# Store data from csv
df = pd.read_csv('individual_stocks_5yr\individual_stocks_5yr\AAPL_data.csv')

# Get close price
df = df[['close']]


# Create variable to predict the 'x' days out until the future
future_days = 25
# Create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['close']].shift(-future_days)


# Create the feature dataset (X), convert it to a numpy array and remove the
# last 'x' rows/days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
# Create the target dataset (Y), convert it to a numpy array and get all of
# the target values except the last 'x' days
Y = np.array(df['Prediction'])[:-future_days]

# Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Create the models
# Create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)

# Create the linear regression model
lr = LinearRegression().fit(x_train, y_train)

# Get the last 'x' rows of the feature dataset
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

# Show the model tree prediction
tree_prediction = tree.predict(x_future)
print(tree_prediction)
# Show the model linear regression prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction)

# Shows the actual close prices
actual_data = np.array(df['close'].tail(future_days))
print(actual_data)

actual_to_tree = ttest_ind(actual_data, tree_prediction)[0]
actual_to_lr = ttest_ind(actual_data, lr_prediction)[0]



def plot_prediction(df, X, predictions):
    """

    :param df: The Pandas Dataframe
    :param X: The Number of Days not including the predicted days
    :param predictions: The Predicted Y Values
    :return: None
    """
    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price USD')
    plt.plot(df['close'])
    plt.plot(valid[['close', 'Predictions']])
    plt.legend(['Orig', 'Val', 'Pred'])
    plt.show()

plot_prediction(df, X, lr_prediction)

plot_prediction(df, X, tree_prediction)