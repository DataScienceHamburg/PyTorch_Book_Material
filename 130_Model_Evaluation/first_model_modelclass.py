#%% package
import torch
import kagglehub
import os 
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/anxiety_model')

# Download latest version
path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset")

print("Path to dataset files:", path)
#%% data import
file_name = "enhanced_anxiety_dataset.csv"
complete_file_path = os.path.join(path, file_name)
anxiety = pd.read_csv(complete_file_path)
# %%
anxiety
# %%
anxiety.columns

#%%
anxiety.info()

#%%
anxiety['Occupation']
# %% One-Hot-Encoding von kategorischen Features
anxiety_dummies = pd.get_dummies(anxiety, drop_first=True, dtype=int)
anxiety_dummies

# %%
anxiety_dummies.columns

#%%
anxiety_dummies.shape

#%%
sns.regplot(data=anxiety_dummies, x='Sleep Hours', y='Anxiety Level (1-10)')

#%% correlation matrix
corr_vals = anxiety[['Age', 'Sleep Hours', 'Stress Level (1-10)', 'Anxiety Level (1-10)']].corr()
sns.heatmap(corr_vals)
# %% separate independent / dependent features
X = np.array(anxiety_dummies.drop(columns=['Anxiety Level (1-10)']), dtype=np.float32)
y = np.array(anxiety_dummies[['Anxiety Level (1-10)']], dtype=np.float32)
print(f"X shape: {X.shape}, y shape: {y.shape}")

#%% data scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %% convert to tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

#%% training
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LinearRegression, self).__init__()
        self.linear_in = torch.nn.Linear(input_size, hidden_size)
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.linear_out(x)
        return x
#%%
HIDDEN_SIZE = 50
model = LinearRegression(input_size=X.shape[1], output_size=y.shape[1], hidden_size=HIDDEN_SIZE)

#%% training loop hyperparameter
EPOCHS = 500
LEARNING_RATE = 0.01
BATCH_SIZE = 128
#%% Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#%% training loop
losses_epoch = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for i in range(0, X.shape[0], BATCH_SIZE):
        # forward pass
        y_pred = model(X_tensor[i: i+BATCH_SIZE])

        # loss calculation
        loss = torch.nn.functional.mse_loss(y_pred, y_tensor[i:i+BATCH_SIZE])

        # backward pass
        loss.backward()

        # weight update
        optimizer.step()
        
        # zero gradients
        optimizer.zero_grad()

        # update loss epoch
        loss_epoch += loss.item()

    # store losses
    print(f"Epoch: {epoch}, current loss: {loss_epoch}")
    losses_epoch.append(loss_epoch)
    writer.add_scalar('loss', loss_epoch, epoch)

# %% visualise losses
sns.lineplot(x = list(range(EPOCHS)), y = losses_epoch)

# %% predict
with torch.no_grad():
    y_pred = model(X_tensor).detach().numpy().flatten()
y_pred.shape
# %% calc correlation coefficient
from sklearn.metrics import r2_score
r2 = r2_score(y_pred=y_pred, y_true=y)
print(f"R-squared: {r2}")
#%%
# %%
