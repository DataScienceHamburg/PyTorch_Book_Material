
#%% packages
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%% hyperparameters
BATCH_SIZE = 32
EPOCHS = 12
LR = 0.001
LATENT_DIMS = 16
VAL_SPLIT = 0.1
OUT_DIR = "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% transformations
my_transforms = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    # Normalize to [-1, 1] so a Tanh decoder output matches input scale (1 channel)
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = MNIST(root=os.path.join("data", "mnist"), train=True, download=True, transform=my_transforms)

#%% check the pixel range of the dataset
x, y = dataset[0]           # transforms applied here
print(x.min().item(), x.max().item())  # should be ~ -1.0 to 1.0

#%% Split into train/val for stable visualization and basic validation loss
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset=dataset, lengths=[train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#%% get number of observations for train and val
print(f"Number of observations for train: {len(train_dataset)}")
print(f"Number of observations for val: {len(val_dataset)}")

#%% VAE model classes
class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=6, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(6*26*26, 128)
        self.relu2 = nn.ReLU()
        
        # Separate layers for mean and log variance
        self.fc_mean = nn.Linear(128, LATENT_DIMS)
        self.fc_logvar = nn.Linear(128, LATENT_DIMS)

    def forward(self, x): 
        x = self.conv1(x)  # in: (BS, 1, 28, 28), out: (BS, 6, 26, 26)
        x = self.relu(x)
        x = self.flatten(x)  # out: (BS, 6*26*26)
        x = self.fc(x)  # out: (BS, 128)
        x = self.relu2(x)
        
        # Get mean and log variance
        mu = self.fc_mean(x)  # out: (BS, LATENT_DIMS)
        logvar = self.fc_logvar(x)  # out: (BS, LATENT_DIMS)
        
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIMS, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 6*26*26)
        self.relu2 = nn.ReLU()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(6, 26, 26))
        self.conv1 = nn.ConvTranspose2d(6, 1, 3)
        self.tanh = nn.Tanh()

    def forward(self, x): 
        x = self.fc(x)  # in: (BS, LATENT_DIMS), out: (BS, 128)
        x = self.relu(x)
        x = self.fc2(x)  # out: (BS, 6*26*26)
        x = self.relu2(x)
        x = self.unflatten(x)  # out: (BS, 6, 26, 26)
        x = self.conv1(x)  # out: (BS, 1, 28, 28)
        x = self.tanh(x)  # match input scale [-1, 1]
        return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def generate(self, num_samples=1):
        """Generate new samples by sampling from prior N(0,1)"""
        z = torch.randn(num_samples, LATENT_DIMS).to(DEVICE)
        return self.decoder(z)

#%% utility functions
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)	

def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    # Input in [-1, 1] -> [0, 1]
    return (img_tensor + 1.0) / 2.0

def vae_loss_function(recon_x, x, mu, logvar):
    """VAE loss = reconstruction loss + KL divergence"""
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    
    # KL divergence loss: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss

def save_reconstruction_grid(model: nn.Module, images: torch.Tensor, epoch: int, out_dir: str) -> None:
    model.eval()
    with torch.no_grad():
        images = images.to(DEVICE)
        reconstructed, _, _ = model(images)
        # Prepare a grid with originals (top) and reconstructions (bottom)
        original = denormalize(images).cpu()
        reconstructed_image = denormalize(reconstructed).cpu()
        grid = vutils.make_grid(torch.cat([original, reconstructed_image], dim=0), nrow=images.size(0))
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0).squeeze())
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"reconstruction_epoch_{epoch:03d}.png"))
        plt.close()

def save_generated_samples(model: nn.Module, epoch: int, out_dir: str, num_samples=16) -> None:
    """Generate and save new samples from the VAE"""
    model.eval()
    with torch.no_grad():
        generated = model.generate(num_samples)
        generated_images = denormalize(generated).cpu()
        grid = vutils.make_grid(generated_images, nrow=4)
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0).squeeze())
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"generated_epoch_{epoch:03d}.png"))
        plt.close()

#%% Model, optimizer
model = VAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Fixed validation batch for consistent visualization
if len(val_loader) > 0:
    fixed_images, fixed_labels = next(iter(val_loader))
else:
    fixed_images, fixed_labels = next(iter(train_loader))
# Use a small grid (up to 8 images)
fixed_images = fixed_images[:8]

#%% Training loop
loss_train, loss_val = [], []
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for i, (X_batch, _) in enumerate(train_loader):
        X_batch = X_batch.to(DEVICE)
        
        # Forward pass
        recon_batch, mu, logvar = model(X_batch)
        
        # Calculate loss
        loss = vae_loss_function(recon_batch, X_batch, mu, logvar)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_epoch_loss = running_loss / len(train_loader)
    loss_train.append(train_epoch_loss)
    
    # Validation loss
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for X_val, _ in val_loader:
            X_val = X_val.to(DEVICE)
            recon_val, mu_val, logvar_val = model(X_val)
            v_loss = vae_loss_function(recon_val, X_val, mu_val, logvar_val)
            val_running_loss += v_loss.item()
    
    val_epoch_loss = val_running_loss / len(val_loader)
    loss_val.append(val_epoch_loss)
    
    print(f"Epoch {epoch:02d} | train_loss={train_epoch_loss:.4f} | val_loss={val_epoch_loss:.4f}")

    # Save reconstructions and generated samples every epoch
    save_reconstruction_grid(model, fixed_images, epoch, OUT_DIR)
    save_generated_samples(model, epoch, OUT_DIR)

#%% Visualize train, val loss
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, EPOCHS + 1), y=loss_train, label="train")
sns.lineplot(x=range(1, EPOCHS + 1), y=loss_val, label="val")
plt.title("VAE Training- and Validation Loss")
plt.xlabel("Epoch [-]")
plt.ylabel("Loss [-]")
plt.show()

#%% Generate new digits
model.eval()
with torch.no_grad():
    # Generate 16 new samples
    new_digits = model.generate(16)
    new_digits_denorm = denormalize(new_digits).cpu()
    
    # Create a grid and display
    grid = vutils.make_grid(new_digits_denorm, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.title("Generated Digits")
    plt.tight_layout()
    plt.show()
# %%
