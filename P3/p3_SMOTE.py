import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTEN
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN

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

#SMOTE
#smote = SMOTE(random_state=42)

#SMOTEENN
#over-sampling com SMOTE + under-sampling com ENN (edited nearest neighbours)
#pode ser viavel, mas é bastante instável (varia muito entre runs).

#smote = SMOTEN(sampling_strategy = 0.3, random_state=42) 

#smoteenn = SMOTEENN(random_state=42, smote=smote)

svmsmote = SVMSMOTE(sampling_strategy = 0.3, random_state = 42)

#borderlinesmote = BorderlineSMOTE(sampling_strategy = 0.3, random_state = 42)

#adasyn = ADASYN(sampling_strategy = 0.3, random_state = 42)

X_train, Y_train = svmsmote.fit_resample(X_train, Y_train)

X_train = X_train.reshape(len(X_train), 28, 28, 3)
X_test = X_test.reshape(len(X_test), 28, 28, 3)
Xtest = Xtest.reshape(len(Xtest), 28, 28, 3)

X_train = X_train/255.0
X_test = X_test/255.0

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 2)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 2)
        
#MODEL DEFINITION
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
#model.add(layers.Dropout(0.2)) #testar melhor dropout. dropout ou early stopping? ou ambos?
model.add(layers.Dense(2))

model.compile(optimizer='adam',
              #loss='binary_crossentropy',
              #loss='categorical_crossentropy'
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy',f1_m,precision_m, recall_m])

model.summary()

early_stopping = EarlyStopping(monitor='f1_m', mode='max', patience=3)

epochs = 50

history = model.fit(X_train, Y_train, epochs = epochs, validation_data = (X_test,Y_test), callbacks = [early_stopping])

#PLOT ACCURACY
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
#plt.annotate('max val', xy=(2, 1), xycoords='data', xytext=(0.01, .99))
plt.show()

#EVALUATE MODEL
test_loss, test_acc, test_f1, test_precision, test_recall = model.evaluate(X_test,  Y_test, verbose=2)
