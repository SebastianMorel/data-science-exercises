# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the dataset
data = pd.read_csv('data.csv')

# Clean up the dataset to only include data from Caulfield North
df = data.loc[data['Suburb'] == 'Caulfield North']

# Extract count of bedrooms and price
df = df.loc[:, ['Bedroom2', 'Price']]

# Check missing values, as regression cannot handle this well
print('Missing values:')
print(df.isnull().sum())

# Plot the two lists against each other in a scatter plot
plt.title("Property prices vs. number of bedrooms in Caulfield North")
plt.xlabel("# of bedrooms")
plt.ylabel("Property price")
plt.ticklabel_format(style='plain')
plt.scatter(df.Bedroom2, df.Price)
plt.ylim(ymin=0)

# Extract variables
X = df.drop('Price', axis='columns')
y = df.Price

# Get data in right shape
X = pd.Series(X.stack().tolist())
y = pd.Series(y.tolist())

# We start by building our model
LR = 0.0003
m = 0
c = 0


def mean_squared_error(true_y, pred_y):
    """
    Calculates the Mean Squared Error.

    Parameters
    ----------
    true_y : list
        List of true y values.
    pred_y : list
        List of predicted y values.

    Returns
    -------
    MSE : float
        Mean Squared Error for the list.
    """
    # Assert lists are equal lengths
    assert len(true_y) == len(pred_y)

    # Transform to Numpy arrays for vector operations
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)

    # Calculate Mean Squared Error
    MSE = sum((true_y - pred_y) ** 2) * (1 / len(true_y))

    return MSE


# Gradient Descent with 1000 iterations
lowest_MSE = float('inf')  # Set value to highest possible value to start with
early_stopping_count = 3  # Used to stop Gradient Descent when value becomes 0
MSE_history = []  # Holds history of MSEs

for i in range(25_000_000):

    # Apply Gradient Descent
    y_pred = m * X + c
    derivate_m = (-2 / len(X)) * sum(X * (y - y_pred))
    derivate_c = (-2 / len(X)) * sum(y - y_pred)
    m = m - LR * derivate_m
    c = c - LR * derivate_c

    # Calculate Mean Squared Error
    MSE = mean_squared_error(y, y_pred)
    MSE_history.append(MSE)

    # Print details for every 100 iterations.
    if i % 100 == 0:
        print(f'Iteration: {i}. LR={LR}. m={m:.2f}. c={c:.2f}. MSE={MSE}')

    # Update lowest MSE
    if MSE < lowest_MSE:
        lowest_MSE = MSE
    else:
        early_stopping_count = early_stopping_count - 1

    # Stop iterations after to many tries
    if early_stopping_count <= 0:
        print(
            f'\nOptimized Gradient Descent at:\nIteration: {i}. LR={LR}. m={m:.2f}. c={c:.2f}. MSE={MSE}')
        break

# Plot the line according to our prediction
plt.plot(X, y_pred, '-r')
plt.show()

# Plot history of MSEs
plt.plot(MSE_history)
plt.title('History of Mean Squared Errors')
plt.xlabel("Itterations")
plt.ylabel("Mean Square Error")
plt.ticklabel_format(style='plain')
plt.ylim(ymin=0)
plt.show()

# Plot optimized Gradient Descent
print("Learning Rate: ", LR,
      "\nThe gradient (m) is: ", np.round(m, 2),
      "\nThe intercept (c) is: ", np.round(c, 2),
      '\nMSE: ', MSE)
