import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

import sys

def plotter(X0, Y0, X1, Y1, mode, title):

    if(mode==0 or mode==2):
        fig, axs = plt.subplots(2, 2)

        axs[0,0].scatter(X0[:, 0], Y0)
        axs[0,1].scatter(X0[:, 1], Y0)
        axs[1,0].scatter(X0[:, 2], Y0)
        axs[1,1].scatter(X0[:, 3], Y0)

        axs[0,0].scatter(X1[:, 0], Y1)
        axs[0,1].scatter(X1[:, 1], Y1)
        axs[1,0].scatter(X1[:, 2], Y1)
        axs[1,1].scatter(X1[:, 3], Y1)

        plt.suptitle(title)
        plt.show()

    if(mode==1 or mode==2):

        plt.scatter(X0[:, 0], Y0, c='#1f77b4')
        plt.scatter(X0[:, 1], Y0, c='#1f77b4')
        plt.scatter(X0[:, 2], Y0, c='#1f77b4')
        plt.scatter(X0[:, 3], Y0, c='#1f77b4')

        plt.scatter(X1[:, 0], Y1, c='orange')
        plt.scatter(X1[:, 1], Y1, c='orange')
        plt.scatter(X1[:, 2], Y1, c='orange')
        plt.scatter(X1[:, 3], Y1, c='orange')

        
        plt.title(title)
        plt.show()



Xtest = np.load('X_test_regression2.npy')
Xtrain = np.load('X_train_regression2.npy') 
Ytrain = np.load('y_train_regression2.npy')

SSE_linear_0 = 0
SSE_ridge_0 = 0
SSE_lasso_0 = 0

SSE_linear_1 = 0
SSE_ridge_1 = 0
SSE_lasso_1 = 0

show_graphs=0 # defaults to no graphs

# input arguments: -f "1-KMeans/2-Gaussian" -g "0-no graphs/1-show graphs"
# -g is optional, default is 0

if(sys.argv[1] == "-f" and (sys.argv[2]=='1' or sys.argv[2]=='2' or sys.argv[2] == '3')):
    fit = int(sys.argv[2])
if(len(sys.argv) > 3):
    if(sys.argv[3] == "-g" and (sys.argv[4]=='0' or sys.argv[4]=='1')):
        show_graphs = int(sys.argv[4])

#fit = int(input("Use 1. Kmeans or 2. Gaussian Mixture?\n"))
print("fit: ", fit)
print("show_graphs: ", show_graphs)

if(fit == 1): ##KMEANS
    kmeans = KMeans(n_clusters=2, random_state=42, n_init = 10)
    kmeans.fit(Xtrain)
    Y_train_clusters_k = kmeans.predict(Xtrain)
    Y_test_clusters_k = kmeans.predict(Xtest)

    X_train_cluster_0_k = Xtrain[Y_train_clusters_k == 0]
    X_train_cluster_1_k = Xtrain[Y_train_clusters_k == 1]

    Y_train_cluster_0_k = Ytrain[Y_train_clusters_k == 0]
    Y_train_cluster_1_k = Ytrain[Y_train_clusters_k == 1]

    if(show_graphs==1):
        plotter(X_train_cluster_0_k, Y_train_cluster_0_k, X_train_cluster_1_k, Y_train_cluster_1_k, 2, "KMeans")

    X_train_cluster_0 = X_train_cluster_0_k
    Y_train_cluster_0 = Y_train_cluster_0_k
    X_train_cluster_1 = X_train_cluster_1_k
    Y_train_cluster_1 = Y_train_cluster_1_k

elif(fit == 2): ##GAUSSIAN
    gaussian = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gaussian.fit(Xtrain)
    Y_train_clusters_g = gaussian.predict(Xtrain)
    Y_test_clusters_g = gaussian.predict(Xtest)

    X_train_cluster_0_g = Xtrain[Y_train_clusters_g == 0]
    X_train_cluster_1_g = Xtrain[Y_train_clusters_g == 1]

    Y_train_cluster_0_g = Ytrain[Y_train_clusters_g == 0]
    Y_train_cluster_1_g = Ytrain[Y_train_clusters_g == 1]

    if(show_graphs==1):
        plotter(X_train_cluster_0_g, Y_train_cluster_0_g, X_train_cluster_1_g, Y_train_cluster_1_g, 2, "GaussianMixture")

    X_train_cluster_0 = X_train_cluster_0_g
    Y_train_cluster_0 = Y_train_cluster_0_g
    X_train_cluster_1 = X_train_cluster_1_g
    Y_train_cluster_1 = Y_train_cluster_1_g

elif(fit ==3): #LINEAR REGRESSION
    splitReg = linReg = linear_model.LinearRegression().fit(Xtrain, Ytrain)
    Y_pred_splitReg = splitReg.predict(Xtrain)

    residuals = Ytrain - Y_pred_splitReg
    mean_residuals = residuals.mean()
    std_residuals = residuals.std()
    
    k = 0.25
    threshold = mean_residuals + k * std_residuals
    
    Y_pred_splitReg = Y_pred_splitReg.flatten()
        
    X_train_cluster_0 = Xtrain[Y_pred_splitReg >= threshold]
    Y_train_cluster_0 = Ytrain[Y_pred_splitReg >= threshold]
    X_train_cluster_1 = Xtrain[Y_pred_splitReg < threshold]
    Y_train_cluster_1 = Ytrain[Y_pred_splitReg < threshold]

    if(show_graphs==1):
        plotter(X_train_cluster_0, Y_train_cluster_0, X_train_cluster_1, Y_train_cluster_1, 2, "Linear")

    
else:
    print("Invalid")
    exit

SSE_fold_linear_0 = np.zeros_like(X_train_cluster_0)
SSE_fold_ridge_0 = np.zeros_like(X_train_cluster_0)
SSE_fold_lasso_0 = np.zeros_like(X_train_cluster_0)
SSE_fold_linear_1 = np.zeros_like(X_train_cluster_1)
SSE_fold_ridge_1 = np.zeros_like(X_train_cluster_1)
SSE_fold_lasso_1 = np.zeros_like(X_train_cluster_1)

loo = LeaveOneOut()


for train_index, test_index in loo.split(X_train_cluster_0):
        
    X_train, X_test = X_train_cluster_0[train_index], X_train_cluster_0[test_index]
    Y_train, Y_test = Y_train_cluster_0[train_index], Y_train_cluster_0[test_index]
    
    # Linear Regression
    linReg = linear_model.LinearRegression()
    linReg.fit(X_train, Y_train)
    Y_pred_linReg = linReg.predict(X_test)
    SSE_linear_0 = np.linalg.norm(Y_test-Y_pred_linReg)**2
    #SSE_fold_linear_0.append(SSE_linear_0)
    SSE_fold_linear_0[test_index] = SSE_linear_0
    
    # Ridge Regression
    ridge_cv = linear_model.RidgeCV(cv=5).fit(X_train, Y_train)
    ridReg = linear_model.Ridge(alpha = ridge_cv.alpha_)
    ridReg.fit(X_train, Y_train)
    Y_pred_ridReg = ridReg.predict(X_test)
    SSE_ridge_0 = np.linalg.norm(Y_test-Y_pred_ridReg)**2
    #SSE_fold_ridge_0.append(SSE_ridge_0)
    SSE_fold_ridge_0[test_index] = SSE_ridge_0
    
    # Lasso Regression
    lasso_cv = linear_model.LassoCV(random_state = 42).fit(X_train, Y_train.ravel())
    lasReg = linear_model.Lasso(alpha=lasso_cv.alpha_)
    lasReg.fit(X_train, Y_train)
    Y_pred_lasReg = lasReg.predict(X_test)
    Y_pred_lasReg_c = np.reshape(Y_pred_lasReg, (1,1)) #necessario reshape
    SSE_lasso_0 = np.linalg.norm(Y_test-Y_pred_lasReg_c)**2
    #SSE_fold_lasso_0.append(SSE_lasso_0)
    SSE_fold_lasso_0[test_index] = SSE_lasso_0
    
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
    #SSE_fold_linear_1.append(SSE_linear_1)
    SSE_fold_linear_1[test_index] = SSE_linear_1
    
    # Ridge Regression
    ridge_cv = linear_model.RidgeCV().fit(X_train, Y_train)
    ridReg = linear_model.Ridge(alpha=ridge_cv.alpha_)
    ridReg.fit(X_train, Y_train)
    Y_pred_ridReg = ridReg.predict(X_test)
    SSE_ridge_1 = np.linalg.norm(Y_test-Y_pred_ridReg)**2
    #SSE_fold_ridge_1.append(SSE_ridge_1)
    SSE_fold_ridge_1[test_index] = SSE_ridge_1
    
    # Lasso Regression
    lasso_cv = linear_model.LassoCV(random_state = 42).fit(X_train, Y_train.ravel())
    lasReg = linear_model.Lasso(alpha=lasso_cv.alpha_)
    lasReg.fit(X_train, Y_train)
    Y_pred_lasReg = lasReg.predict(X_test)
    Y_pred_lasReg_c = np.reshape(Y_pred_lasReg, (1,1)) #necessario reshape
    SSE_lasso_1 = np.linalg.norm(Y_test-Y_pred_lasReg_c)**2
    #SSE_fold_lasso_1.append(SSE_lasso_1)
    SSE_fold_lasso_1[test_index] = SSE_lasso_1
    
print("SSE_linear_1 = ", np.mean(SSE_fold_linear_1))
print("SSE_ridge_1 = ", np.mean(SSE_fold_ridge_1))
print("SSE_lasso_1 = ", np.mean(SSE_fold_lasso_1))


if(np.mean(SSE_fold_linear_0) < np.mean(SSE_fold_ridge_0)):
    if(np.mean(SSE_fold_linear_0) < np.mean(SSE_fold_lasso_0)):
        flag1 = 0
    else:
        flag1 = 2
else:
    if(np.mean(SSE_fold_ridge_0) < np.mean(SSE_fold_lasso_0)):
        flag1 = 1
    else:
        flag1 = 2

if(np.mean(SSE_fold_linear_1) < np.mean(SSE_fold_ridge_1)):
    if(np.mean(SSE_fold_linear_1) < np.mean(SSE_fold_lasso_1)):
        flag2 = 0
    else:
        flag2 = 2
else:
    if(np.mean(SSE_fold_ridge_1) < np.mean(SSE_fold_lasso_1)):
        flag2 = 1
    else:
        flag2 = 2


if(flag1 == 0):
    linRegFinal = linear_model.LinearRegression()
    linRegFinal.fit(Xtrain, Ytrain)
    Y_pred1 = linRegFinal.predict(Xtest)
elif(flag1 == 1):
    ridge_cvFinal = linear_model.RidgeCV().fit(Xtrain, Ytrain)
    ridRegFinal = linear_model.Ridge(alpha=ridge_cvFinal.alpha_)
    ridRegFinal.fit(Xtrain, Ytrain)
    Y_pred1 = ridRegFinal.predict(Xtest)
elif(flag1 == 2):
    lasso_cvFinal = linear_model.LassoCV(random_state = 42).fit(Xtrain, Ytrain.ravel())
    lasRegFinal = linear_model.Lasso(alpha=lasso_cvFinal.alpha_)
    lasRegFinal.fit(Xtrain, Ytrain)
    Y_pred1 = lasRegFinal.predict(Xtest)
    Y_pred1 = np.reshape(Y_pred1, (1000,1))


if(flag2 == 0):
    linRegFinal = linear_model.LinearRegression()
    linReg.fit(Xtrain, Ytrain)
    Y_pred2 = linReg.predict(Xtest)
elif(flag2 == 1):
    ridge_cvFinal = linear_model.RidgeCV().fit(Xtrain, Ytrain)
    ridRegFinal = linear_model.Ridge(alpha=ridge_cvFinal.alpha_)
    ridRegFinal.fit(Xtrain, Ytrain)
    Y_pred2 = ridRegFinal.predict(Xtest)
elif(flag2 == 2):
    lasso_cvFinal = linear_model.LassoCV(random_state = 42).fit(Xtrain, Ytrain.ravel())
    lasRegFinal = linear_model.Lasso(alpha=lasso_cvFinal.alpha_)
    lasRegFinal.fit(Xtrain, Ytrain)
    Y_pred2 = lasRegFinal.predict(Xtest)
    Y_pred2 = np.reshape(Y_pred2, (1000,1))

final = np.hstack((Y_pred1, Y_pred2))
np.save("Y_test2", final)
    
    
    
    
