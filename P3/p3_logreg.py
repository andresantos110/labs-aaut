import numpy as np
from sklearn.cluster import KMeans
import random
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import tensorflow as tf

def balanced_accuracy(y_true, y_pred):

    true_positives = tf.reduce_sum(y_true * y_pred)
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    false_positives = tf.reduce_sum((1 - y_true) * y_pred)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

    sensitivity = true_positives / (true_positives + false_negatives + 1e-7)
    specificity = true_negatives / (true_negatives + false_positives + 1e-7)

    balanced_acc = (sensitivity + specificity) / 2.0

    return balanced_acc

Xtest = np.load('Xtest_Classification1.npy')
Xtrain = np.load('Xtrain_Classification1.npy') 
Ytrain = np.load('ytrain_Classification1.npy')

X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Ytrain, test_size = 0.2)

#X_train = X_train.reshape(len(X_train), 2352)
#X_test = X_test.reshape(len(X_test), 2352)

X_train = X_train/255.0
X_test = X_test/255.0

X_train = X_train.reshape(len(X_train), 28, 28, 3)
X_test = X_test.reshape(len(X_test), 28, 28, 3)
Xtest = Xtest.reshape(len(Xtest), 28, 28, 3)

total_samples = len(Y_train)

count_zeros = np.count_nonzero(Y_train == 0)
count_ones = np.count_nonzero(Y_train == 1)

#DATA AUGMENTATION

row = np.asmatrix([[0,1]])

diff = count_zeros - count_ones
initdiff = diff
print("running data augmentation.")
while diff != 0:
    #print("Data Augmentation Progress:", "{:.2f}".format(((initdiff - diff) / initdiff ) * 100),"%")
    
    random_number = random.randint(0, total_samples-1)
    
    if(Y_train[random_number] == 1):
        operation = random.randint(1,5)
        if operation == 1:
            image = tf.image.flip_left_right(X_train[random_number])
        elif operation == 2:
            image = tf.image.flip_up_down(X_train[random_number])
        elif operation == 3:
            image = tf.image.rot90(X_train[random_number])
        elif operation == 4:
            image = tf.roll(X_train[random_number], shift=[-1, 1], axis=[0, 1])
        elif operation == 5:
            image = tf.roll(X_train[random_number], shift=[1, -1], axis=[0, 1])
        diff -= 1
        
        #print(operation)
        #plt.imshow(X_train[random_number])
        #plt.xlabel(operation)
        #plt.show()
        #plt.imshow(image)
        #plt.xlabel(operation)
        #plt.show()
        
        
        image_np = image.numpy()
        image_np = np.reshape(image_np, (1,28,28,3))
        
        X_train = np.append(X_train, image_np, axis = 0)
        Y_train = np.append(Y_train, 1)

X_train = X_train.reshape(len(X_train), 2352)
X_test = X_test.reshape(len(X_test), 2352)

logR = LogisticRegression(solver = "lbfgs", random_state=42, max_iter=2000).fit(X_train, Y_train)

Y_pred = logR.predict(X_test)

b_acc = balanced_accuracy(Y_test, Y_pred)
print("balanced accuracy: ", b_acc)
