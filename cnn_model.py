### IMPORT LIBRARIES ###

'''
Tensorflow-gpu 2.0 can be used in Google Colab

!pip install tensorflow-gpu==2.0.0.alpha0

'''

import numpy as np
import datetime
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

### LOAD DATASET AND CHECK THE SHAPE OF THE TRAINING AND TESTING DATA ###

from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X_train.shape

X_test.shape

y_train.shape

y_test.shape

### DATA VISUALIZATION ###

i = 3000
plt.imshow(X_train[i])
print(y_train[i])

W_grid = 4
L_grid = 4

fig, axes = plt.subplots(L_grid, W_grid, figsize = (15, 15))
axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training) # pick a random number
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')
    
plt.subplots_adjust(hspace = 0.4)

n_training

### DATA PREPARATION ###

X_train

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

number_cat = 10

y_train

from tensorflow import keras
y_train = keras.utils.to_categorical(y_train, number_cat)

y_train

y_test = keras.utils.to_categorical(y_test, number_cat)

y_test

X_train = X_train/255
X_test = X_test/255

X_train

X_train.shape

Input_shape = X_train.shape[1:]

Input_shape

y_train.shape

### TRAIN THE MODEL ###

def cifar10_cnn_model():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1024, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(1024, activation = 'relu'))

    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    model.summary()

    model.compile(optimizer = tf.keras.optimizers.RMSprop(0.0001, decay = 1e-6), loss ='categorical_crossentropy', metrics =['accuracy'])

    return model

model = cifar10_cnn_model()

epochs = 100

model.fit(X_train, y_train, batch_size = 512, epochs = epochs)

model.save('model.h5')

### EVALUATE THE MODEL ###

evaluation = model.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(evaluation[1]))

predicted_classes = model.predict_classes(X_test) 
predicted_classes

y_test

y_test = y_test.argmax(1)

y_test

L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(predicted_classes, y_test)
cm
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True)