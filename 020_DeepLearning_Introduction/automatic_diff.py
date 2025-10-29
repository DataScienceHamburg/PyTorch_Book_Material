# %% Import packages
import torch
import matplotlib.pyplot as plt

# %% 1. Initialize data and model
# We want to approximate the linear function y = 3x + 2.
# X_true is our input tensor (x).
X_true = torch.arange(1, 10, dtype=torch.float32).unsqueeze(1)
# y_true is our target tensor (y).
y_true = 3 * X_true + 2

#%% visualize data
plt.scatter(X_true, y_true)
plt.plot(X_true, 3 * X_true + 2, 'r-', label='True regression line')
plt.xlabel('Independent variable (X_true)')
plt.ylabel('Dependent variable (y_true)')
plt.legend()
plt.show()
#%%
# Our learnable parameters that we want to optimize.
# requires_grad=True is key here!
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"Weights 'w' (randomly initialized): {w.item():.4f}")
print(f"Bias 'b' (randomly initialized): {b.item():.4f}")

# Define a learning rate, which determines the size of our steps.
LEARNING_RATE = 0.01
EPOCHS = 1000

# %% 2. The training process in a loop
# We train the model for 1000 epochs (iterations).
w_list = []
b_list = []
loss_list = []
for epoch in range(EPOCHS):
    
    # 3. Forward pass: Calculate the prediction
    y_pred = w * X_true + b
    
    # 4. Calculate loss
    # We use the Mean Squared Error (MSE), which measures the squared
    # distance between the prediction and the target value.
    loss = torch.mean((y_pred - y_true)**2)
    
    # 5. Backward pass: Compute gradients
    # This is the crucial step that computes the gradients for w and b.
    loss.backward()
    
    # 6. Parameter update
    # We update the parameters in the opposite direction of the gradient
    # to minimize the loss. Here we use the context 'with torch.no_grad()'
    # to ensure these updates are not part of the computation graph.
    with torch.no_grad():
        w -= LEARNING_RATE * w.grad
        b -= LEARNING_RATE * b.grad
        w_list.append(w.item())
        b_list.append(b.item())
        loss_list.append(loss.item())
    # 7. Reset gradients
    # After the update we need to manually set the gradients to zero,
    # otherwise they would accumulate in the next iteration.
    w.grad.zero_()
    b.grad.zero_()
    
    # f) Output progress (optional)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

#%% plot w, b, and loss
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

ax1.plot(w_list, label='w')
ax1.set_title('Weight')
ax1.set_xlabel('Epoch [-]')
ax1.set_ylabel('Weight w [-]')
ax1.legend()

ax2.plot(b_list, label='b')
ax2.set_title('Bias')
ax2.set_xlabel('Epoch [-]')
ax2.set_ylabel('Bias b [-]')
ax2.legend()

ax3.plot(loss_list, label='loss')
ax3.set_title('Loss')
ax3.set_xlabel('Epoch [-]')
ax3.set_ylabel('Loss [-]')
ax3.set_ylim(0, 1)  # Set vertical range from 0 to 1
ax3.legend()

plt.tight_layout()
plt.show()

# %% 3. Output final result
print("\n--- 3. Final result after training ---")
print(f"Final weights 'w': {w.item():.4f}")
print(f"Final bias 'b': {b.item():.4f}")
print(f"Expected values: w=3, b=2")

