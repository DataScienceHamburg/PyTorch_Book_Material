#%% packages
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
def derive_mean_and_sd(x):
    extracted_mean = x.mean().item()
    extracted_sd = x.std().item()
    return f"extracted_mean: {extracted_mean:.2f}, extracted_sd: {extracted_sd:.2f}"


#%% Settings
TIMESTEPS = 100
EPOCHS = 200
BATCH_SIZE = 128
DATA_SIZE = 1024*8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% find out how many GPUs are available
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Make some 1D training data (just numbers around +2)
data = torch.randn(DATA_SIZE, 1) * 0.5 + 2
# Linearly increasing noise
noise_levels = torch.linspace(0.0001, 0.02, TIMESTEPS)
noise_strength = torch.sqrt(noise_levels).unsqueeze(1)

#%% copy everything to device
data = data.to(device)
noise_strength = noise_strength.to(device)



# Tiny MLP that takes (noisy x, timestep) and predicts the noise
model = nn.Sequential(
    nn.Linear(2, 32), 
    nn.ReLU(), 
    nn.Linear(32, 1)
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

#%% Training
for epoch in range(EPOCHS):
    for X_test in data.split(BATCH_SIZE):
        # extract batch size
        batch_size = X_test.size(0)
        # sample random timestep
        t = torch.randint(0, TIMESTEPS, (batch_size, 1))
        # sample noise
        noise = torch.randn_like(X_test)
        # normalize timestep
        t_norm = t.float() / TIMESTEPS
        t_norm = t_norm.to(device)
        # add noise to data
        noisy_x = X_test + noise_strength[t.squeeze()] * noise
        # Ensure dimensions match before concatenation
        t_norm = t_norm.reshape(batch_size, 1)  # Ensure t_norm is [batch_size, 1]
        
        noisy_x = noisy_x.reshape(batch_size, 1)  # Ensure noisy_x is [batch_size, 1]
        # concatenate noisy_x and t_norm
        X_batch = torch.cat([noisy_x, t_norm], dim=1)
        # predict noise
        y_pred = model(X_batch)
        # calculate loss
        loss = loss_fn(y_pred, noise)
        # backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

# --- Sampling (generation) ---
#%% Start from random noise
X_test = torch.randn(DATA_SIZE, 1) - 10 
sns.histplot(X_test.numpy(), bins=100, binrange=(-15, 5))
plt.title("Start: Pure Noise")
plt.show()
X_test = X_test.to(device)

# Denoise step by step using the model
for t in reversed(range(TIMESTEPS)):
    batch_size = X_test.size(0)
    t_val = torch.full((batch_size, 1), t) / TIMESTEPS
    t_val = t_val.to(device)
    X_batch = torch.cat([X_test, t_val], dim=1)
    y_pred = model(X_batch.to(device))
    X_test = X_test - noise_strength[t] * y_pred

    # Show a few steps
    if t % 10 == 0 or t == 0:
        sns.histplot(X_test.detach().cpu().numpy(), 
                     bins=100, binrange=(-15, 5))
        plt.title(f"Step {t}")
        plt.show()
        print(derive_mean_and_sd(X_test))

#%% Show original data
sns.histplot(data.detach().cpu().numpy(), bins=50, binrange=(-15, 5))
plt.title("Real Data")

print(derive_mean_and_sd(data))

