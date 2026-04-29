import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch import nn
# Load the dataset
from sklearn import datasets

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

X_t,y_t,X_tst,y_tst = torch.tensor(X_train, dtype=torch.float32),torch.tensor(y_train, dtype=torch.float32),torch.tensor(X_test, dtype=torch.float32),torch.tensor(y_test, dtype=torch.float32)

class LinearRegressionTorch(nn.Module):
    def __init__(self,epoch=10,learning_rate=0.01) -> None:
        super().__init__()
        self.epoch = epoch
        self.lr = learning_rate

        # Layer 1
        self.fc1 = nn.Linear(10,1)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.fc1.parameters(),lr=self.lr)

    def forward(self, X_train, y_train):
        for i in range(self.epoch):
            y_pred = self.fc1(X_train)
            loss = self.loss_fn(y_pred,y_train)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f"Epoch:{i+1} Loss:{loss}")

y_pred = model.predict(X_test)

print(r2_score(y_test,y_pred))