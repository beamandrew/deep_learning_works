from __future__ import division, print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import *
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from sklearn.linear_model import LogisticRegression


def get_cnn(n_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_mlp(n_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu',input_shape=(784,)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_mlp_leek(n_classes):
    model = Sequential()
    model.add(Dense(160, activation='tanh',input_shape=(784,)))
    model.add(Dense(160, activation='tanh'))
    model.add(Dense(160, activation='tanh'))
    model.add(Dense(160, activation='tanh'))
    model.add(Dense(160, activation='tanh'))
    model.add(Dense(n_classes, activation='softmax'))
    ## Attempts to match h2o documentation as closely as possible
    ## https://cran.r-project.org/web/packages/h2o/h2o.pdf
    opt = SGD(lr=0.005, momentum=0.0,decay=0.99, nesterov=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_leekassso_predictors(x,y,n_preds):
    eps = 1e-6
    ## This really makes me appreciate dplyr!
    x_0 = x[np.where(y==0)]
    x_1 = x[np.where(y==1)]
    x0_bar = x_0.mean(axis=0)
    x1_bar = x_1.mean(axis=0)
    x0_sigma = x_0.std(axis=0)
    x1_sigma = x_1.std(axis=0)
    sigma_pool = np.sqrt( (x0_sigma**2)/len(x_0) + (x1_sigma**2)/len(x_1) ) + eps
    all_t = np.abs((x0_bar - x1_bar)/sigma_pool)
    top_10 = np.argsort(-all_t)[0:n_preds]
    return top_10

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
        model.fit(X_train_fold, one_hot_fold, nb_epoch=200)
        score = (model.evaluate(X_final_test, one_hot_final_test, batch_size=128))[1]
        evals[index,0] = train_size
        evals[index,1] = i
        evals[index,2] = score
        index += 1

np.savetxt('cnn.csv',evals)

X_test_flat = X_test_small.reshape((len(X_test_small),784))
X_final_test_flat = X_final_test.reshape((len(X_final_test),784))

folds = 5
evals = np.zeros((len(train_sizes*folds),3))
index = 0
for train_size in train_sizes:
    for i in range(5):
        fold_inds = np.random.choice(inds,train_size)
        X_train_fold = X_train[fold_inds].reshape(train_size,784)
        y_train_fold = y_train[fold_inds]
        one_hot_fold = to_categorical(y_train_fold, 2)
        model = get_mlp(n_classes)
        model.fit(X_train_fold, one_hot_fold, nb_epoch=200,
                        validation_data=[X_test_flat, one_hot_test])
        score = (model.evaluate(X_final_test_flat, one_hot_final_test, batch_size=batch_size))[1]
        evals[index,0] = train_size
        evals[index,1] = i
        evals[index,2] = score
        index += 1

np.savetxt('mlp.csv',evals)


folds = 5
evals = np.zeros((len(train_sizes*folds),3))
index = 0
for train_size in train_sizes:
    for i in range(5):
        fold_inds = np.random.choice(inds,train_size)
        X_train_fold = X_train[fold_inds].reshape(train_size,784)
        y_train_fold = y_train[fold_inds]
        one_hot_fold = to_categorical(y_train_fold, 2)
        model = get_mlp_leek(n_classes)
        model.fit(X_train_fold, one_hot_fold, nb_epoch=20,batch_size=1)
        score = (model.evaluate(X_final_test_flat, one_hot_final_test, batch_size=256))[1]
        evals[index,0] = train_size
        evals[index,1] = i
        evals[index,2] = score
        index += 1

np.savetxt('mlp_leek.csv',evals)

folds = 5
evals = np.zeros((len(train_sizes*folds),3))
index = 0
for train_size in train_sizes:
    for i in range(5):
        fold_inds = np.random.choice(inds,train_size)
        X_train_fold = X_train[fold_inds].reshape(train_size,784)
        y_train_fold = y_train[fold_inds]
        leekasso_preds = get_leekassso_predictors(X_train_fold,y_train_fold,10)
        X_leek = X_train_fold[:,leekasso_preds]
        model = LogisticRegression(C=1e6) ## no L2 penalty
        model.fit(X_leek, y_train_fold)
        score = model.score(X_final_test_flat[:,leekasso_preds],y_final_test)
        evals[index,0] = train_size
        evals[index,1] = i
        evals[index,2] = score
        index += 1

np.savetxt('leekasso.csv',evals)
