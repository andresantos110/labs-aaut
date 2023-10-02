import numpy as np
from sklearn.cluster import KMeans
from sklearn import linear_model

X_test = np.load('X_test_regression2.npy')
X_train = np.load('X_train_regression2.npy') 
Y_train = np.load('y_train_regression2.npy')

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
Y_train_clusters = kmeans.predict(X_train)
Y_test_clusters = kmeans.predict(X_test)

X_train_cluster_0 = X_train[Y_train_clusters == 0]
X_train_cluster_1 = X_train[Y_train_clusters == 1]

Y_train_cluster_0 = Y_train[Y_train_clusters == 0]
Y_train_cluster_1 = Y_train[Y_train_clusters == 1]

#está dividido em dois clusters, supostamente cada um para um modelo distinto.