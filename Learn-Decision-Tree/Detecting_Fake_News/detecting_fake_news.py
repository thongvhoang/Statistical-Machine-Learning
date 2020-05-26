# -*- coding: utf-8 -*-
"""Detecting_Fake_News.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XMz5MqYm9xwdH74gWVCb-SdtuRtCGxZE
"""

!pip install --upgrade gensim

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from matplotlib import pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

Dataset = pd.read_csv("/content/drive/My Drive/Colab Notebooks/news.csv")
Dataset

Training_Dataset = Dataset[0:(int)(0.8*6335)]
Training_Dataset

Test_Set = Dataset[(int)(0.8*6335):]
Test_Set

# X_train = Training_Dataset.iloc[:,[1,2]].values
# X_test = Test_Set.iloc[:,[1,2]].values
# create the transform
vectorizer = CountVectorizer()
# tokenize 
X_train = vectorizer.fit_transform(Training_Dataset['text']+Training_Dataset['title'])
# X_test = vectorizer.transform(X_test.ravel())
X_test = vectorizer.transform(Test_Set['text']+Test_Set['title'])

print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()   
Training_Dataset['label']= label_encoder.fit_transform(Training_Dataset['label']) 
Y_train = Training_Dataset.iloc[:,3].values

Test_Set['label'] = label_encoder.fit_transform(Test_Set['label'])
Y_test = Test_Set.iloc[:,3].values

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Training model
from sklearn.tree import DecisionTreeClassifier

Decision_Tree_Classifier = DecisionTreeClassifier(criterion = "entropy")
Decision_Tree_Classifier.fit(X_train, Y_train)

from sklearn.linear_model import LogisticRegression
Logistic_Regression_Classifier = LogisticRegression(random_state= 0)
Logistic_Regression_Classifier.fit(X_train, Y_train)

# Evaluating training and testing model
from sklearn.metrics import confusion_matrix, r2_score, plot_confusion_matrix

cm = confusion_matrix(Y_train, Decision_Tree_Classifier.predict(X_train))
plot_confusion_matrix(Decision_Tree_Classifier, X_train, Y_train)
plt.title('Confusion Matrix of Train Set - Decision Tree')
print("Confusion matrix of Decision Tree Classifier's training model: ",cm)

cm = confusion_matrix(Y_test, Decision_Tree_Classifier.predict(X_test))
print("Confusion matrix of Decision Tree Classifier's testing model: ",cm)
plot_confusion_matrix(Decision_Tree_Classifier, X_test, Y_test)
plt.title('Confusion Matrix of Test Set - Decision Tree')

cm = confusion_matrix(Y_train, Logistic_Regression_Classifier.predict(X_train))
plot_confusion_matrix(Logistic_Regression_Classifier, X_train, Y_train)
plt.title('Confusion Matrix of Train Set - Logistic Regression')

print("Confusion matrix of Logistic Regression Classifier's training model: ",cm)

cm = confusion_matrix(Y_test, Logistic_Regression_Classifier.predict(X_test))

print("Confusion matrix of Logistic Regression Classifier's testing model: ",cm)
plot_confusion_matrix(Logistic_Regression_Classifier, X_test, Y_test)
plt.title('Confusion Matrix of Test Set - Logistic Regression')

accuracy_score = Decision_Tree_Classifier.score(X_test, Y_test)
print("Accuracy of Decision Tree Classifier's train model: ", Decision_Tree_Classifier.score(X_train,Y_train))
print("Accuracy of Decision Tree Classifier's test model: ",accuracy_score)

accuracy_score = Logistic_Regression_Classifier.score(X_test, Y_test)
print("Accuracy of Logistic Regression Classifier's train model: ", Logistic_Regression_Classifier.score(X_train,Y_train))
print("Accuracy of Logistic Regression Classifier's test model: ",accuracy_score)