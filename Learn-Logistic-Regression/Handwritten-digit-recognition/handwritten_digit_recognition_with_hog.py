# -*- coding: utf-8 -*-
"""Handwritten-digit-recognition-with-HOG.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Kg0dmR7i550DbZk-iSpkJiqW9erHpeNo
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.feature import hog
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

dataset = keras.datasets.mnist

(X_trainfull, Y_trainfull), (X_testfull, Y_testfull) = dataset.load_data()

print(X_trainfull.shape[0],'X train samples')
print(X_testfull.shape[0],'X test samples')
print(Y_trainfull.shape[0], 'Y train samples')
print(Y_testfull.shape[0],'Y test samples')

pd.Series(Y_trainfull).value_counts()

pd.Series(Y_testfull).value_counts()

X_train = X_trainfull[:6000]
Y_train = Y_trainfull[:6000]
X_test = X_testfull[:1000]
Y_test = Y_testfull[:1000]

# x_train
X_train_feature = []
for i in range(len(X_train)):
    feature = hog(X_train[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_train_feature.append(feature)
X_train_feature = np.array(X_train_feature,dtype = np.float32)

# x_test
X_test_feature = []
for i in range(len(X_test)):
    feature = hog(X_test[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_test_feature.append(feature)
X_test_feature = np.array(X_test_feature,dtype=np.float32)

#Training model with HOG
model = LogisticRegression(random_state= 0, solver = 'newton-cg')
model.fit(X_train_feature, Y_train)
Y_pre = model.predict(X_test_feature)
print(accuracy_score(Y_test, Y_pre))
plot_confusion_matrix(model, X_test_feature, Y_test)
plt.show()

print(classification_report(Y_test,Y_pre))

# Reducing dimensions
nsample, nx, ny = X_test.shape
X_test = X_test.reshape(nsample,nx*ny)
nsample, nx, ny = X_train.shape
X_train = X_train.reshape(nsample,nx*ny)

# Training model
model = LogisticRegression(random_state= 0, solver = 'newton-cg')
model.fit(X_train, Y_train)
Y_pre = model.predict(X_test)
print(accuracy_score(Y_test, Y_pre))
plot_confusion_matrix(model, X_test, Y_test)
plt.show()

print(classification_report(Y_test,Y_pre))