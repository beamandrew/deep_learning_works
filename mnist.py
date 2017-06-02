from __future__ import division, print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import *
from keras.optimizers import SGD
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

def get_cnn(n_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

'''
First we'll do an analysis of 0s vs 1s
'''

## Load the data ##
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## Set the number of training samples to use ##
train_sizes = [10,20,40,60,80]
batch_size = 1

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims (X_test, axis=3)

# Get training data for 0 and 1
inds = np.where((y_train == 0) | (y_train == 1))[0]

X_train_small = X_train[inds]
X_train_small = X_train_small[:train_size]
y_train_small = y_train[inds]
y_train_small = y_train_small[:train_size]

# Get test data for 0 and 1 only
inds_test = np.where((y_test == 0) | (y_test == 1))
X_test_01 = X_test[inds_test]
y_test_01 = y_test[inds_test]

# split into validation and test sets
test_start_ind = int(np.floor((len(X_test_01)/2)))
X_test_small = X_test_01[:test_start_ind]
X_final_test = X_test_01[test_start_ind:]

y_test_small = y_test_01[:test_start_ind]
y_final_test = y_test_01[test_start_ind:]

# one hot labels
one_hot_train = to_categorical(y_train_small, 2)
one_hot_test = to_categorical(y_test_small, 2)
one_hot_final_test = to_categorical(y_final_test, 2)

n_classes = one_hot_train.shape[1]

folds = 5
evals = np.zeros((len(train_sizes*folds),3))
index = 0
for train_size in train_sizes:
    for i in range(5):
        fold_inds = np.random.choice(inds,train_size)
        X_train_fold = X_train[fold_inds]
        y_train_fold = y_train[fold_inds]
        one_hot_fold = to_categorical(y_train_fold, 2)
        model = get_cnn(n_classes)
        model.fit(X_train_fold, one_hot_fold, nb_epoch=200,
                        validation_data=[X_test_small, one_hot_test])
        score = (model.evaluate(X_final_test, one_hot_final_test, batch_size=batch_size))[1]
        evals[index,0] = train_size
        evals[index,1] = i
        evals[index,2] = score
        index += 1



