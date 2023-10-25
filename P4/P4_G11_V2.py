import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

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
Xtest = np.load('Xtest_Classification2.npy')
Xtrain = np.load('Xtrain_Classification2.npy') 
Ytrain = np.load('ytrain_Classification2.npy')

X_train, X_test, Y_train, Y_test = train_test_split(Xtrain,Ytrain,test_size=0.2)

X_train = X_train.reshape(len(X_train), 28, 28, 3)
X_test = X_test.reshape(len(X_test), 28, 28, 3)
Xtest = Xtest.reshape(len(Xtest), 28, 28, 3)

X_train = X_train/255.0
X_test = X_test/255.0

total_samples = len(Y_train)

count_images = np.zeros(6, dtype=np.int64)

count_images[0] = np.count_nonzero(Y_train == 0)
count_images[1] = np.count_nonzero(Y_train == 1)
count_images[2] = np.count_nonzero(Y_train == 2)
count_images[3] = np.count_nonzero(Y_train == 3)
count_images[4] = np.count_nonzero(Y_train == 4)
count_images[5] = np.count_nonzero(Y_train == 5)

class_ini_index = np.zeros(6, dtype=np.int64)

class_ini_index[1] = count_images[0]
class_ini_index[2] = count_images[0] + count_images[1]
class_ini_index[3] = count_images[0:3].sum()
class_ini_index[4] = count_images[0:4].sum()
class_ini_index[5] = count_images[0:5].sum()

biggest_class = 0
diff = np.zeros(6)

for i in range(1, 5):
    if(count_images[i] > count_images[biggest_class]):
        biggest_class = i
        #class_ini_index[i] = int(count_images[0:i].sum())
for i in range(0, 5):
    diff[i] = count_images[biggest_class] - count_images[i]


print("Images per category:"
      "\n0: ", count_images[0], 
      "\n1: ", count_images[1], 
      "\n2: ", count_images[2], 
      "\n3: ", count_images[3], 
      "\n4: ", count_images[4], 
      "\n5: ", count_images[5],
      "\nbiggest category: ", biggest_class,
      "\nclass_ini_index = ", class_ini_index)

Y_sorted_indexes = np.argsort(Y_train)

X_train_sorted = X_train[Y_sorted_indexes]

#class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
#class_weights_normalized = class_weights / sum(class_weights)

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 6)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 6)

#DATA AUGMENTATION

row = np.zeros((6, 6))

np.fill_diagonal(row, 1)

print("row: ", row[i])
print("Y_train rand:", Y_train[234])

#row = np.asmatrix([[0, 1, 0, 0, 0, 0]])

# falta fazer: fazer uma copia de X_train e Y_train ordenados por classes para iterar facilmente no data augmentation,
# colocar as novas imagens num vetor y_temp, no fim dar shuffle e adicionar ao Y_train original

initdiff = np.copy(diff)

print("Running data augmentation.\n")

for i in range(0, 6):
    j = 0
    while diff[i] !=0:
    #while diff[i] > initdiff[i]*(9.8/10):   # para quando não se quer esperar 15 min 
        j += j
        if(j % 100 == 0 and diff[i] != initdiff[i]):
            print("Data Augmentation Progress:", "{:.2f}".format(((initdiff[i] - diff[i]) / initdiff[i] ) * 100),
                  "%",
                 " of class ", i,
                 " X size: ", X_train.shape,
                 " Y size: ", Y_train.shape,
                 end='\r')
        
        
        operation = random.randint(1,5)
        if operation == 1:
            image = tf.image.flip_left_right(X_train_sorted[j + class_ini_index[i]])
        elif operation == 2:
            image = tf.image.flip_up_down(X_train_sorted[j + class_ini_index[i]])
        elif operation == 3:
            image = tf.image.rot90(X_train_sorted[j + class_ini_index[i]])
        elif operation == 4:
            image = tf.roll(X_train_sorted[j + class_ini_index[i]], shift=[-1, 1], axis=[0, 1])
        elif operation == 5:
            image = tf.roll(X_train_sorted[j + class_ini_index[i]], shift=[1, -1], axis=[0, 1])
        diff[i] -= 1

        if(j>= count_images[i]):
            j = 0
        
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
        Y_train = np.append(Y_train, row[:,i].reshape(-1, 6), axis = 0)

#MODEL DEFINITION
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', balanced_accuracy])

model.summary()

early_stopping = EarlyStopping(monitor='balanced_accuracy', mode='max', patience=20)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_balanced_accuracy', mode='max', verbose=0, save_best_only=True)

epochs = 100

history = model.fit(X_train, Y_train, epochs = epochs, validation_data = (X_test,Y_test), 
                    #class_weight=class_weights,
                    callbacks = [early_stopping, model_checkpoint])

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
