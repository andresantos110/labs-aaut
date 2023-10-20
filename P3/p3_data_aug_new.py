import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import balanced_accuracy_score

tf.compat.v1.enable_eager_execution()

# =============================================================================
# def balanced_accuracy(y_true, y_pred):
#     y_pred_binary = tf.round(y_pred)
#     true_positives = tf.reduce_sum(y_true * y_pred_binary)
#     actual_positives = tf.reduce_sum(y_true)
#     predicted_positives = tf.reduce_sum(y_pred_binary)
#     balanced_acc = (true_positives / actual_positives + predicted_positives) / 2
#     return balanced_acc
# =============================================================================


def balanced_accuracy(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
      
    balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)
    return balanced_acc
    
    
# =============================================================================
# def data_augmentation(image, label):
#     print("running data aug.")
#     random_num = random.randint(1,4)
#     print(random_num)
#     if random_num == 1:
#         image = tf.image.random_flip_left_right(image)
#     elif random_num == 2:
#         image = tf.image.random_flip_up_down(image)
#     elif random_num == 3:
#         image = tf.image.rot90(image)
#     elif random_num == 4:
#         image = tf.roll(image, shift=[1, 1], axis=[0, 1])
#     
#     #image = tf.roll(image, shift=[1, 1], axis=[0, 1])
#     #image = tf.image.rot90(image)
#     #image = tf.image.random_flip_up_down(image)
#     #image = tf.image.random_flip_left_right(image)                                       
#     return image, label
# =============================================================================

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

row = np.asmatrix([[0,1]])

#DATA AUGMENTATION mau?
# =============================================================================
# dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
# 
# majority_dataset = dataset.filter(lambda feature, label: label == 0)
# minority_dataset = dataset.filter(lambda feature, label: label == 1)
# 
# num_minority_samples = len(list(minority_dataset))
# num_majority_samples = len(list(majority_dataset))
# 
# minority_dataset = minority_dataset.repeat(count=num_majority_samples // num_minority_samples)
# 
# minority_dataset = minority_dataset.map(data_augmentation)
# 
# augmented_dataset = tf.data.experimental.sample_from_datasets([majority_dataset, minority_dataset])
# 
# #augmented_dataset = augmented_dataset.map(data_augmentation)
# 
# X_train = np.array([data[0] for data in augmented_dataset])
# Y_train = np.array([data[1] for data in augmented_dataset])
# =============================================================================

#DATA AUGMENTATION

diff = count_zeros - count_ones
initdiff = diff
print("running data augmentation.")
while diff != 0:
    #print("Data Augmentation Progress:", "{:.2f}".format(((initdiff - diff) / initdiff ) * 100),"%")
    
    random_number = random.randint(0, total_samples-1)
    
    if(Y_train[random_number,1] == 1):
        operation = random.randint(1,4)
        #print(operation)
        if operation == 1:
            image = tf.image.random_flip_left_right(X_train[random_number])
        elif operation == 2:
            image = tf.image.random_flip_up_down(X_train[random_number])
        elif operation == 3:
            image = tf.image.rot90(X_train[random_number])
        elif operation == 4:
            image = tf.roll(X_train[random_number], shift=[-1, 1], axis=[0, 1])
        diff -= 1
        
        #image2 = tf.io.encode_png(X_train[random_number], compression=-1, name=None)
        #plt.imshow(X_train[random_number])
        #plt.show()
        #plt.imshow(image)
        #plt.show()
        
        
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
#model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
#model.add(layers.Dense(32, activation='relu',kernel_regularizer=l2(0.1)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', balanced_accuracy])

model.summary()

early_stopping = EarlyStopping(monitor='balanced_accuracy', mode='max', patience=20)

epochs = 200

#class_0_weight = total_samples / (2 * 0.84 * total_samples)
#class_1_weight = total_samples / (2 * 0.16 * total_samples)

#class_weights = {0: class_0_weight, 1: class_1_weight}

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
