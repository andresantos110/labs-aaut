import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K

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
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.3)

X_train = X_train.reshape(len(X_train), 28, 28, 3)
X_test = X_test.reshape(len(X_test), 28, 28, 3)
X_val = X_val.reshape(len(X_val), 28, 28, 3)
Xtest = Xtest.reshape(len(Xtest), 28, 28, 3)


X_train = X_train/255.0
X_val = X_val/255.0
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


#FOR USE WITHOUT DATA AUGMENTATION
#class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
#class_weights_dict = dict(enumerate(class_weights))
#class_weights_normalized = class_weights / sum(class_weights)

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 6)
Y_val = tf.keras.utils.to_categorical(Y_val, num_classes = 6)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 6)

#DATA AUGMENTATION

row = np.zeros((6, 6))

np.fill_diagonal(row, 1)

initdiff = np.copy(diff)

print("Running data augmentation.\n")

for i in range(0, 6):
    j = 0
    while j < count_images[i] and diff[i] > initdiff[i]*(0.9):
        j += j
        if(j % 100 == 0 and diff[i] != initdiff[i]):
            print("Data Augmentation Progress:", "{:.2f}".format(((initdiff[i] - diff[i]) / initdiff[i] ) * 100),
                  "%",
                 " of class ", i,
                 " X size: ", X_train.shape,
                 " Y size: ", Y_train.shape,
                 end='\r')
        
        
        image = tf.image.flip_left_right(X_train_sorted[j + class_ini_index[i]])
        image_np = image.numpy()
        image_np = np.reshape(image_np, (1,28,28,3))
        X_train = np.append(X_train, image_np, axis = 0)
        Y_train = np.append(Y_train, row[:,i].reshape(-1, 6), axis = 0)

        image = tf.image.flip_left_right(X_train_sorted[j + class_ini_index[i]])
        image = tf.image.rot90(image)
        image_np = image.numpy()
        image_np = np.reshape(image_np, (1,28,28,3))
        X_train = np.append(X_train, image_np, axis = 0)
        Y_train = np.append(Y_train, row[:,i].reshape(-1, 6), axis = 0)

        image = tf.image.flip_up_down(X_train_sorted[j + class_ini_index[i]])
        image_np = image.numpy()
        image_np = np.reshape(image_np, (1,28,28,3))
        X_train = np.append(X_train, image_np, axis = 0)
        Y_train = np.append(Y_train, row[:,i].reshape(-1, 6), axis = 0)

        image = tf.image.rot90(X_train_sorted[j + class_ini_index[i]])
        image_np = image.numpy()
        image_np = np.reshape(image_np, (1,28,28,3))
        X_train = np.append(X_train, image_np, axis = 0)
        Y_train = np.append(Y_train, row[:,i].reshape(-1, 6), axis = 0)

        image = tf.roll(X_train_sorted[j + class_ini_index[i]], shift=[-1, 1], axis=[0, 1])
        image_np = image.numpy()
        image_np = np.reshape(image_np, (1,28,28,3))
        X_train = np.append(X_train, image_np, axis = 0)
        Y_train = np.append(Y_train, row[:,i].reshape(-1, 6), axis = 0)

        image = tf.roll(X_train_sorted[j + class_ini_index[i]], shift=[1, -1], axis=[0, 1])
        image_np = image.numpy()
        image_np = np.reshape(image_np, (1,28,28,3))
        X_train = np.append(X_train, image_np, axis = 0)
        Y_train = np.append(Y_train, row[:,i].reshape(-1, 6), axis = 0)


        diff[i] -= 6

        
        #print(operation)
        #plt.imshow(X_train[random_number])
        #plt.xlabel(operation)
        #plt.show()
        #plt.imshow(image)
        #plt.xlabel(operation)
        #plt.show()
    
#WEIGHTS FOR USE WITH DATA AUGMENTATION
#class_indices = np.argmax(Y_train, axis=1)
#class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
#class_weights_dict = dict(enumerate(class_weights))
        

#MODEL DEFINITION
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              #loss=tf.keras.losses.CategoricalFocalCrossentropy(),
              metrics=['accuracy', balanced_accuracy])

model.summary()

early_stopping = EarlyStopping(monitor='balanced_accuracy', mode='max', patience=20)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_balanced_accuracy', mode='max', verbose=0, save_best_only=True)

epochs = 200

history = model.fit(X_train, Y_train, epochs = epochs, validation_data = (X_val,Y_val), 
                    #class_weight=class_weights_dict,
                    callbacks = [early_stopping, model_checkpoint])

#PLOT ACCURACY
plt.plot(history.history['balanced_accuracy'], label='balanced_accuracy')
plt.plot(history.history['val_balanced_accuracy'], label = 'val_balanced_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

Y_pred = model.predict(X_test)

print("Test balanced accuracy: ", balanced_accuracy(Y_test, Y_pred))

#EVALUATE MODEL
test_loss, test_acc, test_bal_acc = model.evaluate(X_test,  Y_test, verbose=2)



model.load_weights('best_model.h5')

#print(tf.math.confusion_matrix(Y_pred1, Y_test))

Xtest = Xtest/255.0

Ypred = model.predict(Xtest)
Ypred = Ypred.argmax(axis=1)

np.save("Ytest_Classification2.npy", Ypred)
