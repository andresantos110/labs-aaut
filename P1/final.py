#Utiliza Ridge Regression para calcular vetor y^.

import numpy as np
from sklearn.linear_model import Ridge

Xtest = np.load('X_test_regression1.npy')
Xtrain = np.load('X_train_regression1.npy') 
Ytrain = np.load('y_train_regression1.npy')

ridge = Ridge(alpha=1.78).fit(Xtrain, Ytrain)
Ytest = ridge.predict(Xtest)

np.save("Ytest", Ytest)