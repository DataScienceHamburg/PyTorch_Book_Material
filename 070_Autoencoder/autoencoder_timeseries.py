#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#%% Generate Synthetic Time Series Data with an Anomaly
np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)
baseline = np.sin(0.1 * time) + np.random.normal(0, 0.01, n_samples)
anomaly_start = 150
baseline[anomaly_start:anomaly_start+20] += 1.5  # Introduce a clear anomaly

time_series = baseline.reshape(-1, 1)

# visualize
plt.plot(time_series)
plt.show()

# %% Scale the data
scaler = MinMaxScaler()
scaled_time_series = scaler.fit_transform(time_series)

time_series_tensor = torch.tensor(scaled_time_series, dtype=torch.float32)
# %% Autoencoder Model
class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(TimeSeriesAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_size = 1  
encoding_dim = 5
model = TimeSeriesAutoencoder(input_size, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% Train the Autoencoder
EPOCHS = 1000
for epoch in range(EPOCHS):
    X = time_series_tensor
    y = model(X)
    loss = criterion(y, X)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

#%% Anomaly Detection
model.eval()
with torch.no_grad():
    reconstructed = model(time_series_tensor)
    reconstruction_error = torch.mean((reconstructed - time_series_tensor)**2, dim=1)

# Set a threshold for anomaly detection (you might need to tune this)
threshold = 0.0001
anomalies = reconstruction_error > threshold
anomaly_indices = torch.where(anomalies)[0].numpy()

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(time, scaled_time_series, label='Original Time Series')
plt.plot(time, reconstructed.numpy(), label='Reconstructed Time Series')
plt.scatter(time[anomaly_indices], scaled_time_series[anomaly_indices], color='red', label='Anomalies')
plt.title('Time Series Anomaly Detection using Autoencoder')
plt.xlabel('Time Step')
plt.ylabel('Value (Scaled)')
plt.legend()
plt.grid(True)
plt.show()
# %%
