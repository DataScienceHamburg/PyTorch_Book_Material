#%% packages
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import torch
from sklearn.metrics import r2_score
from DataPrep import X, y


#%% convert to tensor
X_tensor = torch.from_numpy(X.astype(np.float32))
y_tensor = torch.from_numpy(y.astype(np.float32))  # Ensure y is float32

#%% Hyperparameters
EPOCHS = 100
LEARNING_RATE = 0.01
BATCH_SIZE = 512
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
criterion = torch.nn.MSELoss()

#%% Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


#%% 
loss_list = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i in range(0, len(X_tensor), BATCH_SIZE):
        # get batch
        X_batch = X_tensor[i:i+BATCH_SIZE]
        y_batch = y_tensor[i:i+BATCH_SIZE].unsqueeze(1)

        # forward pass
        y_predict = model(X_batch)

        # calculate loss
        loss = criterion(y_predict, y_batch)

        # backward pass
        loss.backward()

        # update weights and biases
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()   

        # Store loss for plotting
        epoch_loss += loss.item()
    
    # Print loss for this epoch
    print(f"Epoch {epoch}, Loss: {epoch_loss/len(X_tensor):.4f}")
    loss_list.append(epoch_loss)
#%% plot loss
sns.lineplot(x=range(EPOCHS), y=loss_list)

#%% check results
print(f"Weights: {model.linear.weight.detach().numpy().flatten()}, Bias: {model.linear.bias.detach().numpy().flatten()}")
# %%
#%% predict
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()


# Use scatter_kws to set marker size smaller for better visibility
sns.regplot(x=y_pred, y=y, color='red', scatter_kws={'s': 10, 'color': 'blue', 'alpha': 0.1})

# %% calculate correlation coefficient
r2 = r2_score(y_pred, y)
print(f"R-squared: {r2}")
#%%
