import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def balanced_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    true_positives = tf.reduce_sum(y_true * y_pred)
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    false_positives = tf.reduce_sum((1 - y_true) * y_pred)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

    sensitivity = true_positives / (true_positives + false_negatives + 1e-7)
    specificity = true_negatives / (true_negatives + false_positives + 1e-7)

    balanced_acc = (sensitivity + specificity) / 2.0

    return balanced_acc

#CHECK FOR GPU
gpus = tf.config.experimental.list_physical_devices('GPU')

if not gpus:
    print("No GPUs are available. Running on CPU.")
else:
    print("Available GPUs:")
    for gpu in gpus:
        print(gpu)
        
#LOAD FILES AND FORMAT ARRAYS
Xtest = np.load('Xtest_Classification1.npy')
Xtrain = np.load('Xtrain_Classification1.npy') 
Ytrain = np.load('ytrain_Classification1.npy')

X_train, X_test, Y_train, Y_test = train_test_split(Xtrain,Ytrain,test_size=0.2)

X_train = X_train.reshape(len(X_train), 28, 28, 3)
X_test = X_test.reshape(len(X_test), 28, 28, 3)
Xtest = Xtest.reshape(len(Xtest), 28, 28, 3)

X_train = X_train/255.0
X_test = X_test/255.0

total_samples = len(Y_train)

count_zeros = np.count_nonzero(Y_train == 0)
count_ones = np.count_nonzero(Y_train == 1)

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 2)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 2)

#DATA AUGMENTATION

row = np.asmatrix([[0,1]])

diff = count_zeros - count_ones
initdiff = diff
print("running data augmentation.")
while diff != 0:
    #print("Data Augmentation Progress:", "{:.2f}".format(((initdiff - diff) / initdiff ) * 100),"%")
    
    random_number = random.randint(0, total_samples-1)
    
    if(Y_train[random_number,1] == 1):
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
        
        image_np = image.numpy()
        image_np = np.reshape(image_np, (1,28,28,3))
        
        X_train = np.append(X_train, image_np, axis = 0)
        Y_train = np.append(Y_train, row, axis = 0)
        

#MODEL DEFINITION
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', balanced_accuracy])

model.summary()

early_stopping = EarlyStopping(monitor='balanced_accuracy', mode='max', patience=20)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_balanced_accuracy', mode='max', verbose=0, save_best_only=True)

epochs = 200

history = model.fit(X_train, Y_train, epochs = epochs, validation_data = (X_test,Y_test), callbacks = [early_stopping, model_checkpoint])

#PLOT ACCURACY
plt.plot(history.history['balanced_accuracy'], label='balanced_accuracy')
plt.plot(history.history['val_balanced_accuracy'], label = 'val_balanced_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

#EVALUATE MODEL
test_loss, test_acc, test_bal_acc = model.evaluate(X_test,  Y_test, verbose=2)


model.load_weights('best_model.h5')

Xtest = Xtest/255.0

Y_pred = model.predict(Xtest)
Y_pred = np.around(Y_pred)
Y_pred = Y_pred.argmax(axis=1)

np.save("Ytest_Classification1.npy", Y_pred)
