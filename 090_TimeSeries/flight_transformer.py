
#%% Packages
# modeling
import numpy as np
import torch
from torch import nn
# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import train_loader, test_loader

#%% Hyper Parameter
EPOCHS = 100
BATCH_SIZE = 4
SEQ_LEN = 10
NHEAD = 4
D_MODEL = 48
NUM_LAYERS = 2
HIDDEN_SIZE = 50

#%% Model
class FlightModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(FlightModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc1(x[:, -1, :])
        return x

# %% Model, Loss and Optimizer
# Instanziieren des Modells mit den definierten Hyperparametern
model = FlightModel(
    input_size=1,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    output_size=1
)

loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

#%% Train
loss_train = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for j, train_batch in enumerate(train_loader):
        X_batch, y_batch = train_batch
       
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fun(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    loss_train.append(loss_epoch/len(train_loader))
    
    print(f"Epoch: {epoch}, Loss: {loss.data}")


#%% Loss Plot
sns.lineplot(x=range(EPOCHS), y=loss_train)
plt.xlabel('Epoche [-]')
plt.ylabel('Verlust [-]')
plt.title('Verlust')

# %% Create Predictions
X_test_torch, y_test_torch = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(X_test_torch)
y_act = y_test_torch.numpy().squeeze()
x_act = range(y_act.shape[0])
sns.lineplot(x=x_act, y=y_act, label = 'Actual',color='black')
sns.lineplot(x=x_act, y=y_pred.squeeze(), label = 'Predicted',color='red')



# %% correlation plot
sns.scatterplot(x=y_act, y=y_pred.squeeze(), label = 'Predicted',color='red', alpha=0.5)
# Add diagonal line with slope 1 and same range as data
plt.plot([0.5, 1.], [0.5, 1.], 'k--', alpha=0.5, label='Perfect Prediction')
plt.legend()
plt.title('Vorhersage vs. tatsächlicher Wert')
plt.xlabel('Tatsächlicher Wert')
plt.ylabel('Vorhergesagter Wert')
# %%
# %%
rmse = np.sqrt(np.mean((y_act - y_pred.squeeze().numpy())**2))
print(f"RMSE: {rmse:.2f}")

#%% calculate mape error
mape = np.mean(np.abs((y_act - y_pred.squeeze().numpy()) / y_act)) * 100
print(f"MAPE: {mape:.2f}%")
# %%
