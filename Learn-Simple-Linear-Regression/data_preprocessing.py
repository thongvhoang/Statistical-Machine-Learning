import pandas as pd 
import numpy as np 
# Read dataset
dataset = pd.read_csv("Salary_Data.csv")
# Split dataset to input X and outcome Y
X = np.array(dataset.iloc[:, 0].values).reshape(-1,1)
Y = np.array(dataset.iloc[:,1].values)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = 0.8, random_state = 0)
