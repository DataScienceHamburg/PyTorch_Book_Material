#%% packages
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from DataPrep import X, y
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

#%% Hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.1
BATCH_SIZE = 512

#%% DataModule - Pure Lightning Solution
class AnxietyDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size=512, val_split=0.2):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.batch_size = batch_size
        self.val_split = val_split
        
    def setup(self, stage=None):
        # Erstelle vollständiges Dataset direkt in Lightning
        full_dataset = torch.utils.data.TensorDataset(self.X, self.y)
        
        # Lightning's random_split für automatische Aufteilung
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

#%% Model class
class LinearRegression(pl.LightningModule):
    def __init__(self, input_size, output_size, learning_rate=0.1):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.learning_rate = learning_rate
        self.loss_fun = torch.nn.MSELoss()
        
        # Store training and validation losses for plotting
        self.training_losses = []
        self.validation_losses = []

    def forward(self, x):   
        return self.linear(x)
    
    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_predict = self(X_batch)
        loss = self.loss_fun(y_predict, y_batch.reshape(-1, 1))
        
        # Log loss for monitoring
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store loss for plotting
        self.training_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_predict = self(X_batch)
        loss = self.loss_fun(y_predict, y_batch.reshape(-1, 1))
        
        # Log validation loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Store validation loss for plotting (only at epoch end)
        if batch_idx == 0:  # Only store once per epoch
            self.validation_losses.append(loss.item())
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

#%% Setup data and model
data_module = AnxietyDataModule(X, y, batch_size=BATCH_SIZE)
model = LinearRegression(X.shape[1], 1, learning_rate=LEARNING_RATE)

#%% Setup callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkpoints',
    filename='anxiety_model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min',
    verbose=True
)

#%% Setup trainer
trainer = Trainer(
    max_epochs=EPOCHS,
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=10,
    accelerator='auto',
    devices='auto'
)

#%% Train the model
trainer.fit(model, data_module)

#%% Plot training and validation loss
plt.figure(figsize=(10, 6))

# Trainings-Verlust plotten (pro Batch)
sns.lineplot(x=range(len(model.training_losses)), y=model.training_losses, label='Trainings-Verlust (pro Batch)')

# Validierungs-Verlust plotten (pro Epoche)
if len(model.validation_losses) > 0:
    # X-Koordinaten für Validierungs-Verlust berechnen (jede Epoche)
    val_x = np.linspace(0, len(model.training_losses)-1, len(model.validation_losses))
    sns.lineplot(x=val_x, y=model.validation_losses, label='Validierungs-Verlust (pro Epoche)', marker='o')

plt.title('Trainings- und Validierungs-Verlust über Zeit')
plt.xlabel('Trainingsschritte') 
plt.ylabel('Verlust')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%% Load best model and make predictions
best_model = LinearRegression.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    input_size=X.shape[1], 
    output_size=1
)

# %%
