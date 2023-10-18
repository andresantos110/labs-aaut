# CÃ³digo para testar os modelos linear, ridge e lasso
# Utilizar kfold para realizar crossvalidation utilizando os dados de treino
# Calcular SSE de cada prediction -> calcular mÃ©dia por fold -> calcular mÃ©dia final
# SSE mais baixo corresponde ao modelo que melhot se adapta aos dados

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
    
    kf = KFold(n_splits=15, shuffle=True, random_state=i)
    N_splits = kf.get_n_splits(Xtrain)
        
    for train_index, test_index in kf.split(Xtrain):
        
        X_train, X_test = Xtrain[train_index], Xtrain[test_index]
        Y_train, Y_test = Ytrain[train_index], Ytrain[test_index]
        
        # Linear Regression
        linReg = linear_model.LinearRegression()
        linReg.fit(X_train, Y_train)
        Y_pred_linReg = linReg.predict(X_test)
        SSE_linear = np.linalg.norm(Y_test-Y_pred_linReg)**2
        SSE_fold_linear.append(SSE_linear)
        
        # Ridge Regression
        ridge_cv = linear_model.RidgeCV().fit(X_train, Y_train)
        ridReg = linear_model.Ridge(alpha=ridge_cv.alpha_)
        ridReg.fit(X_train, Y_train)
        Y_pred_ridReg = ridReg.predict(X_test)
        SSE_ridge = np.linalg.norm(Y_test-Y_pred_ridReg)**2
        SSE_fold_ridge.append(SSE_ridge)
        
        # Lasso Regression
        lasso_cv = linear_model.LassoCV(random_state = 42, tol=1e-2).fit(X_train, Y_train.ravel())
        lasReg = linear_model.Lasso(alpha=lasso_cv.alpha_)
        lasReg.fit(X_train, Y_train)
        Y_pred_lasReg = lasReg.predict(X_test)
        Y_pred_lasReg_c = np.reshape(Y_pred_lasReg, (int(15/N_splits),1)) #necessario reshape para vetor Y_pred ficar com dimensao correta
        SSE_lasso = np.linalg.norm(Y_test-Y_pred_lasReg_c)**2
        SSE_fold_lasso.append(SSE_lasso)

    SSE_all_linear.append(np.mean(SSE_fold_linear))  
    SSE_all_ridge.append(np.mean(SSE_fold_ridge))
    SSE_all_lasso.append(np.mean(SSE_fold_lasso))
    SSE_fold_linear = []
    SSE_fold_ridge = []
    SSE_fold_lasso = []

       
    
print("SSE_linear = ", np.mean(SSE_all_linear))
print("SSE_ridge = ", np.mean(SSE_all_ridge))
print("SSE_lasso = ", np.mean(SSE_all_lasso))