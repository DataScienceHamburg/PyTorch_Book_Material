
#%% Packages
# data preparation
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# modeling
import torch
from torch.utils.data import Dataset, DataLoader

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

#%% Hyper Parameter
BATCH_SIZE = 4
seq_len = 10

# %% Data Import
flights = sns.load_dataset("flights")
print(f'Number of Entries: {len(flights)}')
flights.head(2)

# %% Visualization
sns.lineplot(x=range(len(flights)), y='passengers', data=flights)
plt.title('Number of Flight Passengers per Month')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.show()

# %% Convert year and month to datetime
flights["year_month"] = pd.to_datetime(flights["year"].astype(str) + "-" + flights["month"].astype(str))

#%% Scale the data
scaler = MinMaxScaler()
Xy = flights.passengers.values.astype(np.float32)
Xy_scaled = scaler.fit_transform(Xy.reshape(-1, 1))

# %% Data Restructuring
X_restruct = np.array([Xy_scaled[i:i+seq_len] for i in range(len(Xy_scaled) - seq_len)])
y_restruct = np.array([Xy_scaled[i+seq_len] for i in range(len(Xy_scaled) - seq_len)])
print(f'X_restruct shape: {X_restruct.shape}')
print(f'y_restruct shape: {y_restruct.shape}')
#%% train/test split
last_n_months = 12
clip_point = len(X_restruct) - last_n_months
X_train = X_restruct[:clip_point]
X_test = X_restruct[clip_point:]
y_train = y_restruct[:clip_point]
y_test = y_restruct[clip_point:]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
#%% Visualisierung des Train-/Test-Splits
plt.figure(figsize=(10, 6))
sns.lineplot(data=flights, x='year_month', y='passengers', label='Total Data')
# Hier musst du die Indizes an die Zeitstempel anpassen
split_date = flights['year_month'].iloc[clip_point + seq_len]
plt.axvline(x=split_date, color='red', linestyle='--', label='Train/Test Split')
plt.title('Number of Flight Passengers per Month')
plt.legend()
plt.show()
# %% Dataset
class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].squeeze(-1), self.y[idx]
#%% Dataloader
train_loader = DataLoader(FlightDataset(X_train, y_train), batch_size=BATCH_SIZE)
test_loader = DataLoader(FlightDataset(X_test, y_test), batch_size=len(y_test))

