import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


Xtest = np.load('Xtest_Classification1.npy')
Xtrain = np.load('Xtrain_Classification1.npy') 
Ytrain = np.load('ytrain_Classification1.npy')

X_train, X_test, Y_train, Y_test = train_test_split(Xtrain,Ytrain,test_size=0.2)

X_train = X_train.reshape(len(X_train), 28, 28, 3)
X_test = X_test.reshape(len(X_test), 28, 28, 3)
Xtest = Xtest.reshape(len(Xtest), 28, 28, 3)

X_train = X_train/255.0
X_test = X_test/255.0

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 2)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 2)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs = 100, batch_size = 100, verbose=0, validation_data = (X_test,Y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
