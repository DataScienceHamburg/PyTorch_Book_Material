#%% packages
from DataPrep import X, y
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#%% convert to tensor
X_tensor = torch.from_numpy(X.astype(np.float32))
y_tensor = torch.from_numpy(y.astype(np.float32))  # Ensure y is float32

#%% training
# Initialize weights with smaller values to prevent exploding gradients
w = torch.zeros(X.shape[1], 1, requires_grad=True, dtype=torch.float32)
b = torch.zeros(1, requires_grad=True, dtype=torch.float32)
print(f"w shape: {w.shape}, b shape: {b.shape}")


EPOCHS = 100
LEARNING_RATE = 0.01  # Reduced learning rate for more stable training

#%% 
loss_list = []
for epoch in range(EPOCHS):
    # Forward pass
    y_predict = torch.matmul(X_tensor, w) + b
    
    # Calculate loss (MSE)
    loss = torch.nn.functional.mse_loss(y_predict, y_tensor)
    
    # Backward pass
    loss.backward()
    
    # Update weights and biases
    with torch.no_grad():
        w -= LEARNING_RATE * w.grad
        b -= LEARNING_RATE * b.grad
        # Zero gradients after using them
        w.grad.zero_()
        b.grad.zero_()
    
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
print(f"Weights: {w.detach().numpy().flatten()}, \n"
      f"Bias: {b.item()}")
# %%
with torch.no_grad():
    y_pred = (torch.matmul(X_tensor, w) + b).detach().numpy().flatten()
# %%
sns.regplot(x=y_pred, y=y, color='red', 
            scatter_kws={'s': 10, 
                         'color': 'blue', 
                         'alpha': 0.1})
plt.title('Predicted Anxiety Level vs Actual Anxiety Level')
plt.xlabel('Predicted Anxiety Level [-]')
plt.ylabel('Actual Anxiety Level [-]')

#%% Calculate R-squared
r2 = r2_score(y_true=y, 
              y_pred=y_pred)
print(f"R-squared: {r2:.2f}")

