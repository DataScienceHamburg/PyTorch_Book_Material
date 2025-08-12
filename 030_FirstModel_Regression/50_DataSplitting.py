#%% packages
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from DataPrep import X, y
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import seaborn as sns

#%% Hyperparameters
EPOCHS = 20
LEARNING_RATE = 0.1
BATCH_SIZE = 512

#%% split data
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

#%% scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

#%% Dataset class
class AnxietyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#%% DataLoader
train_dataset = AnxietyDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True)

val_dataset = AnxietyDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, 
                             batch_size=BATCH_SIZE, 
                             shuffle=False)

#%% Model class
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        

    def forward(self, x):   
        x = self.linear(x)
        return x
    
#%% Model instance
model = LinearRegression(input_size=train_dataset.X.shape[1], 
                        output_size=1)

#%% Loss function
loss_fun = torch.nn.MSELoss()

#%% Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


#%% 
loss_train_list, loss_val_list = [], []
for epoch in range(EPOCHS):
    epoch_loss_train = 0
    epoch_loss_val = 0
    for i, (X_train_batch, y_train_batch) in enumerate(train_dataloader):
        # get batch

        # forward pass
        y_pred_train = model(X_train_batch)

        # calculate loss
        loss_train = loss_fun(y_pred_train, y_train_batch.reshape(-1, 1)).mean()

        # backward pass
        loss_train.backward()

        # update weights and biases
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()   

        # Store loss for plotting
        epoch_loss_train += loss_train.item()

    # evaluate on test set
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_dataloader:
            y_pred_val = model(X_val_batch)
            loss_val = loss_fun(y_pred_val, y_val_batch.reshape(-1, 1)).mean()
            epoch_loss_val += loss_val.item()
    # Store the losses for plotting
    loss_train_list.append(epoch_loss_train / len(train_dataloader))
    loss_val_list.append(epoch_loss_val / len(val_dataloader))
    
    # Print loss for this epoch
    print(f"Epoch {epoch}, Train Loss: {epoch_loss_train}, Test Loss: {loss_val.item()}")
#%% plot loss
# Convert to numpy arrays
import matplotlib.pyplot as plt
loss_train_arr = np.array(loss_train_list)
loss_val_arr = np.array(loss_val_list)

# Train loss: scale independently
train_min = loss_train_arr.min()
train_max = loss_train_arr.max()
train_range = train_max - train_min if train_max > train_min else 1
loss_train_scaled = (loss_train_arr - train_min) / train_range

# Val loss: scale independently
val_min = loss_val_arr.min()
val_max = loss_val_arr.max()
val_range = val_max - val_min if val_max > val_min else 1
loss_val_scaled = (loss_val_arr - val_min) / val_range

sns.lineplot(x=range(EPOCHS), y=loss_train_scaled, color='blue', label='Train')
sns.lineplot(x=range(EPOCHS), y=loss_val_scaled, color='red', label='Validation')
plt.title('Losses over Epochs: Train (blue) vs. Validation (red)')
plt.xlabel('Epoch [-]')
plt.ylabel('Loss [-]')
plt.legend()
plt.show()


# %%
