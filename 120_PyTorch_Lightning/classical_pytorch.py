#%% packages
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from DataPrep import X, y
import seaborn as sns

#%% Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.1
BATCH_SIZE = 512

#%% Dataset class
class AnxietyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#%% DataLoader
dataset = AnxietyDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#%% Model class
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        

    def forward(self, x):   
        x = self.linear(x)
        return x
    
#%% Model instance
model = LinearRegression(X.shape[1], 1)

#%% Loss function
loss_fun = torch.nn.MSELoss()

#%% Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#%% 
loss_list = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i, (X_batch, y_batch) in enumerate(dataloader):
        # get batch

        # forward pass
        y_predict = model(X_batch)

        # calculate loss
        loss = loss_fun(y_predict, y_batch.reshape(-1, 1))

        # backward pass
        loss.backward()

        # update weights and biases
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()   

        # Store loss for plotting
        epoch_loss += loss.item()
    
    # Print loss for this epoch
    print(f"Epoch {epoch}, Loss: {epoch_loss}")
    loss_list.append(epoch_loss)
#%% plot loss
sns.lineplot(x=range(EPOCHS), y=loss_list)