import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

X_test = np.load('X_test_regression1.npy')
X_train = np.load('X_train_regression1.npy') 
Y_train = np.load('y_train_regression1.npy')

SSE_linear = 0
SSE_ridge = 0
SSE_lasso = 0

for i in range(15):
    
    kf = KFold(n_splits=5, shuffle=True, random_state=i)
    N_splits = kf.get_n_splits(X_train)
        
    for train_index, test_index in kf.split(X_train):
        
        # Linear Regression
        linReg = linear_model.LinearRegression()
        linReg.fit(X_train, Y_train)
        Y_pred_linReg = linReg.predict(X_test)
        Y_pred_linReg_c = np.reshape(Y_pred_linReg, (int(100/N_splits),1))
        # temos vetor coluna com valores de Y, calcular SSE e somar a total, no fim calcular media
        
        # Ridge Regression
        ridReg = linear_model.Ridge()
        ridReg.fit(X_train, Y_train)
        Y_pred_ridReg = ridReg.predict(X_test)
        Y_pred_linReg_c = np.reshape(Y_pred_ridReg, (int(100/N_splits),1))
        # temos vetor coluna com valores de Y, calcular SSE e somar a total, no fim calcular media
        
        # Lasso Regression
        lasReg = linear_model.Lasso()
        lasReg.fit(X_train, Y_train) # not sure se falta aqui alguma coisa
        Y_pred_lasReg = lasReg.predict(X_test)
        Y_pred_lasReg_c = np.reshape(Y_pred_lasReg, (int(100/N_splits),1))
        # temos vetor coluna com valores de Y, calcular SSE e somar a total, no fim calcular media
        
        #TODO
        #calcular totais de SSE, somar todos e fazer media no fim. comparar SSE dos 3 modelos para definir o melhor
        
            

#para remover (?)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_train, Y_pred[0:15]))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_train, Y_pred[0:15]))
