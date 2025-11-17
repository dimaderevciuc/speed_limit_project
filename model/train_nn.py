import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SpeedLimitNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5,10),
            nn.ReLU(),
            nn.Linear(10,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self, x):
        return self.layers(x)

def train_model():

    data = pd.read_csv("../data/readings.csv", sep=";")
    data = data.dropna()

    X = np.random.rand(500, 5) * [100, 300, 70, 120, 1000000]
    y = (X[:, 0] * 0.2 + X[:, 1] * 0.01 - X[:, 2] * 0.3 + np.random.randn(500) * 5) / 10

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = SpeedLimitNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("Training neural network...")
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    print(f"Training done! Final loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "../model/speed_limit_nn.pth")
    print("Model saved to model/speed_limit_nn.pth")

if __name__ == "__main__":
    train_model()