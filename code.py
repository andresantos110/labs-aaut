import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import  KFold

Xtest = np.load('X_test_regression1.npy')
Xtrain = np.load('X_train_regression1.npy') 
Ytrain = np.load('y_train_regression1.npy')

SSE_linear = 0
SSE_ridge = 0
SSE_lasso = 0
SSE_fold_linear = []
SSE_fold_ridge = []
SSE_fold_lasso = []
SSE_all_linear = []
SSE_all_ridge = []
SSE_all_lasso = []

for i in range(15):
    
    kf = KFold(n_splits=5, shuffle=True, random_state=i)
    N_splits = kf.get_n_splits(Xtrain)
        
    for train_index, test_index in kf.split(Xtrain):
        
        X_train, X_test = Xtrain[train_index], Xtrain[test_index]
        Y_train, Y_test = Ytrain[train_index], Ytrain[test_index]
        
        # Linear Regression
        linReg = linear_model.LinearRegression()
        linReg.fit(X_train, Y_train)
        Y_pred_linReg = linReg.predict(X_test)
        Y_pred_linReg_c = np.reshape(Y_pred_linReg, (int(15/N_splits),1))
        SSE_linear = np.linalg.norm(Y_test-Y_pred_linReg_c)**2
        SSE_fold_linear.append(SSE_linear)
        # temos vetor coluna com valores de Y, calcular SSE e somar a total, no fim calcular media
        
        # Ridge Regression
        ridReg = linear_model.Ridge()
        ridReg.fit(X_train, Y_train)
        Y_pred_ridReg = ridReg.predict(X_test)
        Y_pred_ridReg_c = np.reshape(Y_pred_ridReg, (int(15/N_splits),1))
        SSE_ridge = np.linalg.norm(Y_test-Y_pred_ridReg_c)**2
        SSE_fold_ridge.append(SSE_ridge)
        # temos vetor coluna com valores de Y, calcular SSE e somar a total, no fim calcular media
        
        # Lasso Regression
        lasReg = linear_model.Lasso()
        lasReg.fit(X_train, Y_train) # not sure se falta aqui alguma coisa
        Y_pred_lasReg = lasReg.predict(X_test)
        Y_pred_lasReg_c = np.reshape(Y_pred_lasReg, (int(15/N_splits),1))
        SSE_lasso = np.linalg.norm(Y_test-Y_pred_lasReg_c)**2
        SSE_fold_lasso.append(SSE_lasso)
        # temos vetor coluna com valores de Y, calcular SSE e somar a total, no fim calcular media
        
        #TODO
        #calcular totais de SSE, somar todos e fazer media no fim. comparar SSE dos 3 modelos para definir o melhor
        # RESHAPE DESNECESSARIO?

    SSE_all_linear.append(np.mean(SSE_fold_linear))  
    SSE_all_ridge.append(np.mean(SSE_fold_ridge))
    SSE_all_lasso.append(np.mean(SSE_fold_lasso))       
    
print("SSE_linear = ", np.mean(SSE_all_linear))
print("SSE_ridge = ", np.mean(SSE_all_ridge))
print("SSE_lasso = ", np.mean(SSE_all_lasso))

#plots?
