import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



# Load the diabetes dataset
X_test = np.load('X_test_regression1.npy')
X_train = np.load('X_train_regression1.npy') 
Y_train = np.load('y_train_regression1.npy')

regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_train, Y_pred[0:15]))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_train, Y_pred[0:15]))
