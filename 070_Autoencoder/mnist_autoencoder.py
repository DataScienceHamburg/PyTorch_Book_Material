
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

#%% hyperparameters
BATCH_SIZE = 32
EPOCHS = 12
LR = 0.001
LATENT_DIMS = 16
VAL_SPLIT = 0.1
OUT_DIR = "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %% transformations
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

#%% model classes
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=6, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Linear(6*26*26, LATENT_DIMS)

    def forward(self, x): 
        x = self.conv1(x)  # in: (BS, 1, 28, 28), out: (BS, 6, 26, 26)
        x = self.relu(x)  # 
        x = self.flatten(x)  # out: (BS, 6*26*26)
        x = self.fully_connected(x)  # out: (BS, LATENT_DIMS)
        return x

# %% Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fully_connected = nn.Linear(LATENT_DIMS, 6*26*26)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(6, 26, 26))
        self.conv1 = nn.ConvTranspose2d(6, 1, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x): 
        x = self.fully_connected(x)  # in: (BS, LATENT_DIMS), out: (BS, 6*26*26)
        x = self.relu(x)
        x = self.unflatten(x)  # out: (BS, 6, 26, 26)
        x = self.conv1(x)  # out: (BS, 1, 28, 28)
        x = self.tanh(x)  # match input scale [-1, 1]
        return x

# %% Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
#%% utility functions
if not os.path.exists(OUT_DIR):
	os.makedirs(OUT_DIR, exist_ok=True)	

def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
	# Input in [-1, 1] -> [0, 1]
	return (img_tensor + 1.0) / 2.0

def save_reconstruction_grid(model: nn.Module, images: torch.Tensor, epoch: int, out_dir: str) -> None:
	model.eval()
	with torch.no_grad():
		images = images.to(DEVICE)
		reconstructed = model(images)
		# Prepare a grid with originals (top) and reconstructions (bottom)
		original = denormalize(images).cpu()
		reconstructed_image = denormalize(reconstructed).cpu()
		grid = vutils.make_grid(torch.cat([original, reconstructed_image], dim=0), nrow=images.size(0))
		plt.figure(figsize=(12, 6))
		plt.axis('off')
		plt.imshow(grid.permute(1, 2, 0).squeeze())
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}.png"))
		plt.close()

#%% Model, optimizer, loss
model = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fun = nn.MSELoss()

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
        y_pred = model(X_batch)
        loss = loss_fun(y_pred, X_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * X_batch.size(0)
    train_epoch_loss = running_loss / (len(train_loader.dataset) if len(train_loader) > 0 else 1)
    loss_train.append(train_epoch_loss)
    # Validation loss
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for X_val, _ in val_loader:
            X_val = X_val.to(DEVICE)
            y_val = model(X_val)
            v_loss = loss_fun(y_val, X_val)
            val_running_loss += v_loss.item() * X_val.size(0)
    val_epoch_loss = val_running_loss / (len(val_loader.dataset) if len(val_loader) > 0 else 1)
    loss_val.append(val_epoch_loss)
    print(f"Epoch {epoch:02d} | train_loss={train_epoch_loss:.4f} | val_loss={val_epoch_loss:.4f}")

    # Save reconstructions and latent space every epoch
    save_reconstruction_grid(model, fixed_images, epoch, OUT_DIR)
# %% visualise train, val loss
sns.lineplot(x=range(1, EPOCHS + 1), y=loss_train, label="train")
sns.lineplot(x=range(1, EPOCHS + 1), y=loss_val, label="val")
plt.title("Training- and Validation Loss")
plt.xlabel("Epoche [-]")
plt.ylabel("Loss [-]")
plt.show()
# %%
