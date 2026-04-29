import torch
from torch import nn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

centers = [[1,1],[-1,-1]]
X,y = make_blobs(n_samples=1000, centers=centers, random_state=2,cluster_std=0.8)

plt.scatter(X[:,0],X[:,1],c=y)
plt.plot([-3,3],[3,-3],"r")
plt.title("Dummy Sample Data")
plt.show()

class LogisticRegression(nn.Module):
    def __init__(self,learning_rate,epoch=10,thresold = 0.5) -> None:
        super().__init__()
        self.lr= learning_rate
        self.epoch = epoch
        self.thresold = thresold
        self.fc1 = nn.Linear(2,1)

        # Loss Function
        self.loss_fn = torch.nn.BCELoss()

        # Optimizer
        self.optimizer = torch.optim.SGD(self.fc1.parameters(), lr = self.lr)

    def forward(self,X,y):
        for i in range(self.epoch):
            z = self.fc1(X)
            y_pred = torch.sigmoid(z)

            # Calulate the loss
            loss = self.loss_fn(y_pred,y.reshape(y_pred.shape))

            # Optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
    def predict(self,x):
        with torch.inference_mode():
            y_pred = self.fc1(x)

        return (y_pred>self.thresold).int()

X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)
model = LogisticRegression(learning_rate=0.01, epoch=5000,thresold=0.5)
model(X_train,y_train)

y_pred = model.predict(X_train)
print("Accuracy:",accuracy_score(y_true=y_train,y_pred=y_pred)) # Approx 0.96
