import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

X_test = np.load('X_test_regression1.npy')
X_train = np.load('X_train_regression1.npy')
Y_train = np.load('y_train_regression1.npy')

test_quant = 3

kf = KFold(n_splits=5, shuffle=True, random_state=0)



new_X_test = X_train[0:test_quant]
new_Y_test = Y_train[0:test_quant]

#regr = linear_model.LinearRegression()
regr = linear_model.Ridge(alpha=.5)

regr.fit(X_train[test_quant:15], Y_train[test_quant:15])

Y_pred = regr.predict(new_X_test)



# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(new_Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(new_Y_test, Y_pred))
