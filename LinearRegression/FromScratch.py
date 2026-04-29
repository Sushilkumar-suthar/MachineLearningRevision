import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
from sklearn import datasets

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

class Linear():
    def __init__(self, learning_rate= 0.1):
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.W = np.array([])

    def fit(self, x_train, y_train):
        self.X_train = np.insert(X_train,0,1,axis=1)
        self.y_train = y_train

        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise Exception("Input velues X and traget variable Y shape does not match")

        self.W = np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train


    def predict(self,X_test):
        return (X_test @ self.W[1:]) + self.W[0]

model = Linear(learning_rate=0.01)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(r2_score(y_test,y_pred))