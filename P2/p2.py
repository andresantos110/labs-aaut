import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


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

##KMEANS
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(Xtrain)
Y_train_clusters = kmeans.predict(Xtrain)
Y_test_clusters = kmeans.predict(Xtest)

X_train_cluster_0 = Xtrain[Y_train_clusters == 0]
X_train_cluster_1 = Xtrain[Y_train_clusters == 1]

Y_train_cluster_0 = Ytrain[Y_train_clusters == 0]
Y_train_cluster_1 = Ytrain[Y_train_clusters == 1]


fig, axs = plt.subplots(2, 2)

axs[0,0].scatter(X_train_cluster_0[:, 0], Y_train_cluster_0)
axs[0,1].scatter(X_train_cluster_0[:, 1], Y_train_cluster_0)
axs[1,0].scatter(X_train_cluster_0[:, 2], Y_train_cluster_0)
axs[1,1].scatter(X_train_cluster_0[:, 3], Y_train_cluster_0)

axs[0,0].scatter(X_train_cluster_1[:, 0], Y_train_cluster_1)
axs[0,1].scatter(X_train_cluster_1[:, 1], Y_train_cluster_1)
axs[1,0].scatter(X_train_cluster_1[:, 2], Y_train_cluster_1)
axs[1,1].scatter(X_train_cluster_1[:, 3], Y_train_cluster_1)
plt.title("KMeans")
plt.show()

##GAUSSIAN
gaussian = GaussianMixture(n_components=2, random_state=42)
gaussian.fit(Xtrain)
Y_train_clusters = gaussian.predict(Xtrain)
Y_test_clusters = gaussian.predict(Xtest)

X_train_cluster_0 = Xtrain[Y_train_clusters == 0]
X_train_cluster_1 = Xtrain[Y_train_clusters == 1]

Y_train_cluster_0 = Ytrain[Y_train_clusters == 0]
Y_train_cluster_1 = Ytrain[Y_train_clusters == 1]


fig, axs = plt.subplots(2, 2)

axs[0,0].scatter(X_train_cluster_0[:, 0], Y_train_cluster_0)
axs[0,1].scatter(X_train_cluster_0[:, 1], Y_train_cluster_0)
axs[1,0].scatter(X_train_cluster_0[:, 2], Y_train_cluster_0)
axs[1,1].scatter(X_train_cluster_0[:, 3], Y_train_cluster_0)

axs[0,0].scatter(X_train_cluster_1[:, 0], Y_train_cluster_1)
axs[0,1].scatter(X_train_cluster_1[:, 1], Y_train_cluster_1)
axs[1,0].scatter(X_train_cluster_1[:, 2], Y_train_cluster_1)
axs[1,1].scatter(X_train_cluster_1[:, 3], Y_train_cluster_1)
plt.title("Gaussian")
plt.show()



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
    ridReg = linear_model.Ridge()
    ridReg.fit(X_train, Y_train)
    Y_pred_ridReg = ridReg.predict(X_test)
    SSE_ridge_0 = np.linalg.norm(Y_test-Y_pred_ridReg)**2
    SSE_fold_ridge_0.append(SSE_ridge_0)
    
    # Lasso Regression
    lasReg = linear_model.Lasso()
    lasReg.fit(X_train, Y_train)
    Y_pred_lasReg = lasReg.predict(X_test)
    Y_pred_lasReg_c = np.reshape(Y_pred_lasReg, (1,1)) #necessario reshape para vetor Y_pred ficar com dimensao (1,1)
    SSE_lasso_0 = np.linalg.norm(Y_test-Y_pred_lasReg_c)**2
    SSE_fold_lasso_0.append(SSE_lasso_0)
    
print("SSE_linear_0 = ", np.mean(SSE_fold_linear_0))
print("SSE_ridge_0 = ", np.mean(SSE_fold_ridge_0))
print("SSE_lasso_0 = ", np.mean(SSE_fold_lasso_0))

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
    ridReg = linear_model.Ridge()
    ridReg.fit(X_train, Y_train)
    Y_pred_ridReg = ridReg.predict(X_test)
    SSE_ridge_1 = np.linalg.norm(Y_test-Y_pred_ridReg)**2
    SSE_fold_ridge_1.append(SSE_ridge_1)
    
    # Lasso Regression
    lasReg = linear_model.Lasso()
    lasReg.fit(X_train, Y_train)
    Y_pred_lasReg = lasReg.predict(X_test)
    Y_pred_lasReg_c = np.reshape(Y_pred_lasReg, (1,1)) #necessario reshape para vetor Y_pred ficar com dimensao (1,1)
    SSE_lasso_1 = np.linalg.norm(Y_test-Y_pred_lasReg_c)**2
    SSE_fold_lasso_1.append(SSE_lasso_1)
    
print("SSE_linear_1 = ", np.mean(SSE_fold_linear_1))
print("SSE_ridge_1 = ", np.mean(SSE_fold_ridge_1))
print("SSE_lasso_1 = ", np.mean(SSE_fold_lasso_1))


    
    
    
    
    
    
    