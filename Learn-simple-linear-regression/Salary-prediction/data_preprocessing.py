import pandas as pd 
import numpy as np 
# Read dataset
dataset = pd.read_csv("/home/tintin/study/Data-science/Statistical-machine-learning/Learn-Simple-Linear-Regression/Salary_Data.csv")
# Split dataset to input X and outcome Y
X = np.array(dataset.iloc[:, 0].values).reshape(-1,1)
Y = np.array(dataset.iloc[:,1].values)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = 0.8, random_state = 0)

import matplotlib.pyplot as plt
#Visualize training data
plt.scatter(X_train, Y_train, color = "red")
plt.title("Salary vs Experiment")
plt.xlabel("Experiment (years)")
plt.show()
plt.ylabel("Salary (dollars/year)")

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_train_pred = regressor.predict(X_train)
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, Y_train_pred, color = "blue")
plt.title("Salary vs Experiment (Training set)")
plt.title("Experiment (years)")
plt.ylabel("Salary (dollars/year)")
plt.show()

Y_test_pred = regressor.predict(X_test)
plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_test, Y_test_pred, color = "blue")
plt.scatter(X_test, Y_test_pred, color = "black")
plt.title("Salary vs Experiment (Testing set)")
plt.xlabel("Experiment (years)")
plt.ylabel("Salary (dollars/year)")
plt.show()

def compare(i_example):
    x = X_test[i_example:i_example+1]
    y = Y_test[i_example]
    y_pred = regressor.predict(x)
    print(x, y, y_pred)
for i in range(len(X_test)):
    compare(i)