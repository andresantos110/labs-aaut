import numpy as np
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

Xtest = np.load('X_test_regression2.npy')
Xtrain = np.load('X_train_regression2.npy') 
Ytrain = np.load('y_train_regression2.npy')

SSE_linear_0 = 0
SSE_ridge_0 = 0
SSE_lasso_0 = 0
SSE_fold_linear_0 = []
SSE_fold_ridge_0 = []
SSE_fold_lasso_0 = []

SSE_linear_1 = 0
SSE_ridge_1 = 0
SSE_lasso_1 = 0
SSE_fold_linear_1 = []
SSE_fold_ridge_1 = []
SSE_fold_lasso_1 = []

SSE_ridge_best_0 = 500
SSE_ridge_best_1 = 500
SSE_lasso_best_0 = 500
SSE_lasso_best_1 = 500

alphaBest_ridge_0 = 0
alphaBest_ridge_1 = 0
alphaBest_lasso_0 = 0
alphaBest_lasso_1 = 0


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(Xtrain)
Y_train_clusters = kmeans.predict(Xtrain)
Y_test_clusters = kmeans.predict(Xtest)

X_train_cluster_0 = Xtrain[Y_train_clusters == 0]
X_train_cluster_1 = Xtrain[Y_train_clusters == 1]

Y_train_cluster_0 = Ytrain[Y_train_clusters == 0]
Y_train_cluster_1 = Ytrain[Y_train_clusters == 1]

#está dividido em dois clusters, supostamente cada um para um modelo distinto.

for j in range(500):

    curr_alpha = 0.01 + 0.01*j

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X_train_cluster_0):
        
        
        X_train, X_test = X_train_cluster_0[train_index], X_train_cluster_0[test_index]
        Y_train, Y_test = Y_train_cluster_0[train_index], Y_train_cluster_0[test_index]
        
        # Linear Regression
        linReg = linear_model.LinearRegression()
        linReg.fit(X_train, Y_train)
        Y_pred_linReg = linReg.predict(X_test)
        SSE_linear_0 = np.linalg.norm(Y_test-Y_pred_linReg)**2
        SSE_fold_linear_0.append(SSE_linear_0)
        
        # Ridge Regression
        ridReg = linear_model.Ridge(alpha = curr_alpha)
        ridReg.fit(X_train, Y_train)
        Y_pred_ridReg = ridReg.predict(X_test)
        SSE_ridge_0 = np.linalg.norm(Y_test-Y_pred_ridReg)**2
        SSE_fold_ridge_0.append(SSE_ridge_0)
        
        # Lasso Regression
        lasReg = linear_model.Lasso(alpha = curr_alpha)
        lasReg.fit(X_train, Y_train)
        Y_pred_lasReg = lasReg.predict(X_test)
        Y_pred_lasReg_c = np.reshape(Y_pred_lasReg, (1,1)) #necessario reshape para vetor Y_pred ficar com dimensao (1,1)
        SSE_lasso_0 = np.linalg.norm(Y_test-Y_pred_lasReg_c)**2
        SSE_fold_lasso_0.append(SSE_lasso_0)
        
    print("SSE_linear_0 = ", np.mean(SSE_fold_linear_0))
    print("SSE_ridge_0 = ", np.mean(SSE_fold_ridge_0))
    print("SSE_lasso_0 = ", np.mean(SSE_fold_lasso_0))

    if(np.mean(SSE_fold_ridge_0) < SSE_ridge_best_0):
        SSE_ridge_best_0 = np.mean(SSE_fold_ridge_0)
        alphaBest_ridge_0 = curr_alpha
        
    if(np.mean(SSE_fold_lasso_0) < SSE_lasso_best_0):
        SSE_lasso_best_0 = np.mean(SSE_fold_lasso_0)
        alphaBest_lasso_0 = curr_alpha
        
    for train_index, test_index in loo.split(X_train_cluster_1):
        
        X_train, X_test = X_train_cluster_1[train_index], X_train_cluster_1[test_index]
        Y_train, Y_test = Y_train_cluster_1[train_index], Y_train_cluster_1[test_index]
        
        # Linear Regression
        linReg = linear_model.LinearRegression()
        linReg.fit(X_train, Y_train)
        Y_pred_linReg = linReg.predict(X_test)
        SSE_linear_1 = np.linalg.norm(Y_test-Y_pred_linReg)**2
        SSE_fold_linear_1.append(SSE_linear_1)
        
        # Ridge Regression
        ridReg = linear_model.Ridge(alpha = curr_alpha)
        ridReg.fit(X_train, Y_train)
        Y_pred_ridReg = ridReg.predict(X_test)
        SSE_ridge_1 = np.linalg.norm(Y_test-Y_pred_ridReg)**2
        SSE_fold_ridge_1.append(SSE_ridge_1)

        # Lasso Regression
        lasReg = linear_model.Lasso(alpha = curr_alpha)
        lasReg.fit(X_train, Y_train)
        Y_pred_lasReg = lasReg.predict(X_test)
        Y_pred_lasReg_c = np.reshape(Y_pred_lasReg, (1,1)) #necessario reshape para vetor Y_pred ficar com dimensao (1,1)
        SSE_lasso_1 = np.linalg.norm(Y_test-Y_pred_lasReg_c)**2
        SSE_fold_lasso_1.append(SSE_lasso_1)

    print("SSE_linear_1 = ", np.mean(SSE_fold_linear_1))
    print("SSE_ridge_1 = ", np.mean(SSE_fold_ridge_1))
    print("SSE_lasso_1 = ", np.mean(SSE_fold_lasso_1))
    
    if(np.mean(SSE_fold_ridge_1) < SSE_ridge_best_1):
        SSE_ridge_best_1 = np.mean(SSE_fold_ridge_1)
        alphaBest_ridge_1 = curr_alpha
        
    if(np.mean(SSE_fold_lasso_1) < SSE_lasso_best_1):
        SSE_lasso_best_1 = np.mean(SSE_fold_ridge_1)
        alphaBest_lasso_1 = curr_alpha
    
print("SSE_best_ridge_0 = ", SSE_ridge_best_0, "for alpha = ", alphaBest_ridge_0)
print("SSE_best_lasso_0 = ", SSE_lasso_best_0, "for alpha = ", alphaBest_lasso_0)
print("SSE_best_ridge_1 = ", SSE_ridge_best_1, "for alpha = ", alphaBest_ridge_1)
print("SSE_best_lasso_1 = ", SSE_lasso_best_1, "for alpha = ", alphaBest_lasso_0)

    
    
    
    
    
    
    