import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

def balanced_accuracy(y_true, y_pred):
    y_pred_binary = tf.round(y_pred)
    true_positives = tf.reduce_sum(y_true * y_pred_binary)
    actual_positives = tf.reduce_sum(y_true)
    predicted_positives = tf.reduce_sum(y_pred_binary)
    balanced_acc = (true_positives / actual_positives + predicted_positives) / 2
    return balanced_acc


#DEFINE FUNCTIONS FOR CALCULATING F1 SCORE
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

#Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 2)
#Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 2)


#DATA AUGMENTATION
row = np.asmatrix([[0,1]])

num_iterations = Y_train.shape[0]

for i in range(num_iterations):
    
    random_num = random.randint(1,3)
    
    if (Y_train[i,1] == 1):

        match random_num:
            case 1:
                img_tf = tf.image.flip_left_right(X_train[i])
            case 2:
                img_tf = tf.image.flip_up_down(X_train[i])
            case 3:
                img_tf = tf.image.rot90(X_train[i])
    
        img_tf_np = img_tf.numpy()
        img_tf_np = np.reshape(img_tf_np, (1,28,28,3))

        X_train = np.append(X_train, img_tf_np,axis=0)
        Y_train = np.append(Y_train, row, axis=0)
        
        
#MODEL DEFINITION
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', balanced_accuracy])

model.summary()

early_stopping = EarlyStopping(monitor='balanced_accuracy', mode='max', patience=20)

epochs = 50

history = model.fit(X_train, Y_train, epochs = epochs, validation_data = (X_test,Y_test), callbacks = [early_stopping])

#PLOT ACCURACY
plt.plot(history.history['balanced_accuracy_score'], label='balanced_accuracy_score')
plt.plot(history.history['val_balanced_accuracy_score'], label = 'val_balanced_accuracy_score')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

#EVALUATE MODEL
test_loss, test_acc, test_bal_acc = model.evaluate(X_test,  Y_test, verbose=2)
