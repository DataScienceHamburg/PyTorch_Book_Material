#%% packages
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from DataPrep import X, y

#%% Hyperparameters
EPOCHS = 100
LEARNING_RATE = 0.1

#%% convert to tensor
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y.astype(np.float32))  # Ensure y is float32

#%% Model class
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        

    def forward(self, x):   
        x = self.linear(x)
        return x
    
#%% Model instance
model = LinearRegression(input_size=X.shape[1], 
                         output_size=1)


#%% Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

#%% 
loss_list = []
for epoch in range(EPOCHS):
    # Forward pass
    y_predict = model(X_tensor)
    
    # Calculate loss (MSE)
    loss = loss_fn(y_predict.squeeze(), y_tensor)
    
    # Backward pass
    loss.backward()
    
    # Update weights and biases
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()
    
    # Store loss for plotting
    loss_list.append(loss.item())
    
    # Print loss for this epoch
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
#%% plot loss
sns.lineplot(x=range(EPOCHS), y=loss_list)
plt.title('Loss over Epochs')
plt.xlabel('Epoch [-]')
plt.ylabel('Loss [-]')

#%% check results
print(f"Weights: {model.linear.weight.detach().numpy().flatten()}, Bias: {model.linear.bias.detach().numpy().flatten()}")
# %%
#%% predict
with torch.no_grad():
    y_pred = model(X_tensor).detach().numpy().flatten()


# Use scatter_kws to set marker size smaller for better visibility
sns.regplot(x=y_pred, y=y, color='red', scatter_kws={'s': 10, 'color': 'blue', 'alpha': 0.1})

# %% calculate correlation coefficient
r2 = r2_score(y_pred, y)
print(f"R-squared: {r2}")
#%%
