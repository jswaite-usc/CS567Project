# -*- coding: utf-8 -*-
"""Base+dropout+comparison.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1q8K1-SC4YX42AXo5MH1Xjl7ih0BItyXz
"""

#Test if have GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

import sys
print(sys.executable)
#We will make a basic CNN model to classify CIFAR-10#
#We will use Keras from Tensorflow to build the model#
import ssl

print(ssl.get_default_verify_paths())



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json

#show the tensorflow version
print(tf.__version__)

#load the CIFAR-10 dataset & split into train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Normalize pixel values from 1-255 to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

#show the shape of the dataset
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#shuffle train & test data

# Assuming x_train and y_train are your data
#indices = np.arange(x_train.shape[0])
#np.random.shuffle(indices)

# # Let's say we want to select 10000 random samples
# x_train_subset = x_train[indices[:10000]]
# y_train_subset = y_train[indices[:10000]]

#Build model
modelA_1 = Sequential()
# Block 1
modelA_1.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelA_1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelA_1.add(MaxPooling2D(pool_size=(2, 2)))
modelA_1.add(Dropout(0.01))
# Block 2
modelA_1.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelA_1.add(MaxPooling2D(pool_size=(2, 2)))
# Block 3
modelA_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_1.add(MaxPooling2D(pool_size=(2, 2)))

modelA_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_1.add(MaxPooling2D(pool_size=(2, 2)))

modelA_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_1.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelA_1.add(Flatten())
modelA_1.add(Dense(256, activation='relu'))
# Block 5
modelA_1.add(Dense(10, activation='softmax'))

#Build model
modelB_1 = Sequential()
# Block 1
modelB_1.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelB_1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelB_1.add(MaxPooling2D(pool_size=(2, 2)))
modelB_1.add(Dropout(0.1))
# Block 2
modelB_1.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelB_1.add(MaxPooling2D(pool_size=(2, 2)))
# Block 3
modelB_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_1.add(MaxPooling2D(pool_size=(2, 2)))

modelB_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_1.add(MaxPooling2D(pool_size=(2, 2)))

modelB_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_1.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelB_1.add(Flatten())
modelB_1.add(Dense(256, activation='relu'))
# Block 5
modelB_1.add(Dense(10, activation='softmax'))

#Build model
modelC_1 = Sequential()
# Block 1
modelC_1.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelC_1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelC_1.add(MaxPooling2D(pool_size=(2, 2)))
modelC_1.add(Dropout(0.2))
# Block 2
modelC_1.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelC_1.add(MaxPooling2D(pool_size=(2, 2)))
# Block 3
modelC_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_1.add(MaxPooling2D(pool_size=(2, 2)))

modelC_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_1.add(MaxPooling2D(pool_size=(2, 2)))

modelC_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_1.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelC_1.add(Flatten())
modelC_1.add(Dense(256, activation='relu'))
# Block 5
modelC_1.add(Dense(10, activation='softmax'))

#Build model
modelD_1 = Sequential()
# Block 1
modelD_1.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelD_1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelD_1.add(MaxPooling2D(pool_size=(2, 2)))
modelD_1.add(Dropout(0.5))
# Block 2
modelD_1.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelD_1.add(MaxPooling2D(pool_size=(2, 2)))
# Block 3
modelD_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_1.add(MaxPooling2D(pool_size=(2, 2)))

modelD_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_1.add(MaxPooling2D(pool_size=(2, 2)))

modelD_1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_1.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelD_1.add(Flatten())
modelD_1.add(Dense(256, activation='relu'))
# Block 5
modelD_1.add(Dense(10, activation='softmax'))

#Build model
modelA_2 = Sequential()
# Block 1
modelA_2.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelA_2.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelA_2.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelA_2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelA_2.add(MaxPooling2D(pool_size=(2, 2)))
modelA_2.add(Dropout(0.01))
# Block 3
modelA_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_2.add(MaxPooling2D(pool_size=(2, 2)))

modelA_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_2.add(MaxPooling2D(pool_size=(2, 2)))

modelA_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_2.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelA_2.add(Flatten())
modelA_2.add(Dense(256, activation='relu'))
# Block 5
modelA_2.add(Dense(10, activation='softmax'))

#Build model
modelB_2 = Sequential()
# Block 1
modelB_2.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelB_2.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelB_2.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelB_2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelB_2.add(MaxPooling2D(pool_size=(2, 2)))
modelB_2.add(Dropout(0.1))
# Block 3
modelB_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_2.add(MaxPooling2D(pool_size=(2, 2)))

modelB_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_2.add(MaxPooling2D(pool_size=(2, 2)))

modelB_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_2.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelB_2.add(Flatten())
modelB_2.add(Dense(256, activation='relu'))
# Block 5
modelB_2.add(Dense(10, activation='softmax'))

#Build model
modelC_2 = Sequential()
# Block 1
modelC_2.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelC_2.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelC_2.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelC_2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelC_2.add(MaxPooling2D(pool_size=(2, 2)))
modelC_2.add(Dropout(0.2))
# Block 3
modelC_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_2.add(MaxPooling2D(pool_size=(2, 2)))

modelC_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_2.add(MaxPooling2D(pool_size=(2, 2)))

modelC_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_2.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelC_2.add(Flatten())
modelC_2.add(Dense(256, activation='relu'))
# Block 5
modelC_2.add(Dense(10, activation='softmax'))

#Build model
modelD_2 = Sequential()
# Block 1
modelD_2.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelD_2.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelD_2.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelD_2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelD_2.add(MaxPooling2D(pool_size=(2, 2)))
modelD_2.add(Dropout(0.5))
# Block 3
modelD_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_2.add(MaxPooling2D(pool_size=(2, 2)))

modelD_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_2.add(MaxPooling2D(pool_size=(2, 2)))

modelD_2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_2.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelD_2.add(Flatten())
modelD_2.add(Dense(256, activation='relu'))
# Block 5
modelD_2.add(Dense(10, activation='softmax'))

#Build model
modelA_3 = Sequential()
# Block 1
modelA_3.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelA_3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelA_3.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelA_3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelA_3.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelA_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_3.add(MaxPooling2D(pool_size=(2, 2)))
modelA_3.add(Dropout(0.01))

modelA_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_3.add(MaxPooling2D(pool_size=(2, 2)))

modelA_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_3.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelA_3.add(Flatten())
modelA_3.add(Dense(256, activation='relu'))
# Block 5
modelA_3.add(Dense(10, activation='softmax'))

#Build model
modelB_3 = Sequential()
# Block 1
modelB_3.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelB_3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelB_3.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelB_3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelB_3.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelB_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_3.add(MaxPooling2D(pool_size=(2, 2)))
modelB_3.add(Dropout(0.1))

modelB_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_3.add(MaxPooling2D(pool_size=(2, 2)))

modelB_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_3.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelB_3.add(Flatten())
modelB_3.add(Dense(256, activation='relu'))
# Block 5
modelB_3.add(Dense(10, activation='softmax'))

#Build model
modelC_3 = Sequential()
# Block 1
modelC_3.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelC_3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelC_3.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelC_3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelC_3.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelC_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_3.add(MaxPooling2D(pool_size=(2, 2)))
modelC_3.add(Dropout(0.2))

modelC_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_3.add(MaxPooling2D(pool_size=(2, 2)))

modelC_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_3.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelC_3.add(Flatten())
modelC_3.add(Dense(256, activation='relu'))
# Block 5
modelC_3.add(Dense(10, activation='softmax'))

#Build model
modelD_3 = Sequential()
# Block 1
modelD_3.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelD_3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelD_3.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelD_3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelD_3.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelD_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_3.add(MaxPooling2D(pool_size=(2, 2)))
modelD_3.add(Dropout(0.5))

modelD_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_3.add(MaxPooling2D(pool_size=(2, 2)))

modelD_3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_3.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelD_3.add(Flatten())
modelD_3.add(Dense(256, activation='relu'))
# Block 5
modelD_3.add(Dense(10, activation='softmax'))

#Build model
modelA_4 = Sequential()
# Block 1
modelA_4.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelA_4.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelA_4.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelA_4.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelA_4.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelA_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_4.add(MaxPooling2D(pool_size=(2, 2)))

modelA_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_4.add(MaxPooling2D(pool_size=(2, 2)))
modelA_4.add(Dropout(0.01))

modelA_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_4.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelA_4.add(Flatten())
modelA_4.add(Dense(256, activation='relu'))
# Block 5
modelA_4.add(Dense(10, activation='softmax'))

#Build model
modelB_4 = Sequential()
# Block 1
modelB_4.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelB_4.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelB_4.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelB_4.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelB_4.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelB_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_4.add(MaxPooling2D(pool_size=(2, 2)))

modelB_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_4.add(MaxPooling2D(pool_size=(2, 2)))
modelB_4.add(Dropout(0.1))

modelB_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_4.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelB_4.add(Flatten())
modelB_4.add(Dense(256, activation='relu'))
# Block 5
modelB_4.add(Dense(10, activation='softmax'))

#Build model
modelC_4 = Sequential()
# Block 1
modelC_4.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelC_4.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelC_4.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelC_4.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelC_4.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelC_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_4.add(MaxPooling2D(pool_size=(2, 2)))

modelC_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_4.add(MaxPooling2D(pool_size=(2, 2)))
modelC_4.add(Dropout(0.2))

modelC_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_4.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelC_4.add(Flatten())
modelC_4.add(Dense(256, activation='relu'))
# Block 5
modelC_4.add(Dense(10, activation='softmax'))

#Build model
modelD_4 = Sequential()
# Block 1
modelD_4.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelD_4.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelD_4.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelD_4.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelD_4.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelD_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_4.add(MaxPooling2D(pool_size=(2, 2)))

modelD_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_4.add(MaxPooling2D(pool_size=(2, 2)))
modelD_4.add(Dropout(0.5))

modelD_4.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_4.add(MaxPooling2D(pool_size=(2, 2)))
# Block 4
modelD_4.add(Flatten())
modelD_4.add(Dense(256, activation='relu'))
# Block 5
modelD_4.add(Dense(10, activation='softmax'))

#Build model
modelA_5 = Sequential()
# Block 1
modelA_5.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelA_5.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelA_5.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelA_5.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelA_5.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelA_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_5.add(MaxPooling2D(pool_size=(2, 2)))

modelA_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_5.add(MaxPooling2D(pool_size=(2, 2)))

modelA_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelA_5.add(MaxPooling2D(pool_size=(2, 2)))
modelA_5.add(Dropout(0.01))
# Block 4
modelA_5.add(Flatten())
modelA_5.add(Dense(256, activation='relu'))
# Block 5
modelA_5.add(Dense(10, activation='softmax'))

#Build model
modelB_5 = Sequential()
# Block 1
modelB_5.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelB_5.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelB_5.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelB_5.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelB_5.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelB_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_5.add(MaxPooling2D(pool_size=(2, 2)))

modelB_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_5.add(MaxPooling2D(pool_size=(2, 2)))

modelB_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelB_5.add(MaxPooling2D(pool_size=(2, 2)))
modelB_5.add(Dropout(0.1))
# Block 4
modelB_5.add(Flatten())
modelB_5.add(Dense(256, activation='relu'))
# Block 5
modelB_5.add(Dense(10, activation='softmax'))

#Build model
modelC_5 = Sequential()
# Block 1
modelC_5.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelC_5.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelC_5.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelC_5.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelC_5.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelC_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_5.add(MaxPooling2D(pool_size=(2, 2)))

modelC_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_5.add(MaxPooling2D(pool_size=(2, 2)))

modelC_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelC_5.add(MaxPooling2D(pool_size=(2, 2)))
modelC_5.add(Dropout(0.2))
# Block 4
modelC_5.add(Flatten())
modelC_5.add(Dense(256, activation='relu'))
# Block 5
modelC_5.add(Dense(10, activation='softmax'))

#Build model
modelD_5 = Sequential()
# Block 1
modelD_5.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
modelD_5.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelD_5.add(MaxPooling2D(pool_size=(2, 2)))
# Block 2
modelD_5.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelD_5.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
modelD_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_5.add(MaxPooling2D(pool_size=(2, 2)))

modelD_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_5.add(MaxPooling2D(pool_size=(2, 2)))

modelD_5.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
modelD_5.add(MaxPooling2D(pool_size=(2, 2)))
modelD_5.add(Dropout(0.5))
# Block 4
modelD_5.add(Flatten())
modelD_5.add(Dense(256, activation='relu'))
# Block 5
modelD_5.add(Dense(10, activation='softmax'))

MODELS = [modelA_1,
          modelB_1,
          modelC_1,
          modelD_1,
          modelA_2,
          modelB_2,
          modelC_2,
          modelD_2,
          modelA_3,
          modelB_3,
          modelC_3,
          modelD_3,
          modelA_4,
          modelB_4,
          modelC_4,
          modelD_4,
          modelA_5,
          modelB_5,
          modelC_5,
          modelD_5]

NAMES = ["A_1",
         "B_1",
         "C_1",
         "D_1",
         "A_2",
         "B_2",
         "C_2",
         "D_2",
         "A_3",
         "B_3",
         "C_3",
         "D_3",
         "A_4",
         "B_4",
         "C_4",
         "D_4",
         "A_5",
         "B_5",
         "C_5",
         "D_5"]

def call_on_model(model, names, fcn):
    return fcn(model)

def call_on_models(models, names, fcn):
    output = []
    for i in range(len(models)):
        output.append(fcn(models[i], names[i]))

    return output

def compile_and_train(model, name):
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'./model_{name}.model.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test), callbacks=[model_checkpoint_callback])
    history_dict = history.history
    json.dump(history_dict, open(f"./history_{name}.json", 'w'))

outputs = call_on_models(MODELS, NAMES, compile_and_train)

