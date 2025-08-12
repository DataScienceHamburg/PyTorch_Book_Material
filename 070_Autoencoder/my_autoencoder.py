#%% packages
import kagglehub
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Download latest version
path = kagglehub.dataset_download("sriramr/apples-bananas-oranges")

print("Path to dataset files:", path)
#%% hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
LATENT_DIMS = 256

# %% transformations
my_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

dataset = ImageFolder(root=path, transform=my_transforms)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

#%% model classes
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=6, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Linear(6*62*62, LATENT_DIMS)

    def forward(self, x): 
        x = self.conv1(x)  # in: (BS, 1, 64, 64), out: (BS, 6, 62, 62)
        x = self.relu(x)  # 
        x = self.flatten(x)  # out: (BS, 6*62*62)
        x = self.fully_connected(x)  # out: (BS, LATENT_DIMS)
        return x

# sample_tensor = torch.zeros((1, 1, 64, 64))
# encoder_model = Encoder()
# encoder_model(sample_tensor).shape

# %% Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fully_connected = nn.Linear(LATENT_DIMS, 6*62*62)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(6, 62, 62))
        self.conv1 = nn.ConvTranspose2d(6, 1, 3)
        self.relu = nn.ReLU()

    def forward(self, x): 
        x = self.fully_connected(x)  # in: (BS, LATENT_DIMS), out: (BS, 6*62*62)
        x = self.relu(x)
        x = self.unflatten(x)  # out: (BS, 6, 62, 62)
        x = self.conv1(x)  # out: (BS, 1, 64, 64)
        return x

# sample_tensor = torch.zeros((1, LATENT_DIMS))
# decoder_model = Decoder()
# decoder_model(sample_tensor).shape

# %%
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
sample_tensor = torch.zeros((1, 1, 64, 64))
model = Autoencoder()
model(sample_tensor).shape

#%% optimizer, loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fun = nn.MSELoss()

# %% training loop
for epoch in range(EPOCHS):
    loss_epoch = 0
    for i, (X_batch, y_batch) in enumerate(dataloader):
        # forward pass
        y_pred = model(X_batch)

        # loss
        loss = loss_fun(y_pred, X_batch)  # input shape == output shape
        loss_epoch += loss.item()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()
    print(f"Epoch: {epoch}, Loss: {loss_epoch}")
    #

# %% data reduction from 4096 -> 256

#%% 
images, labels = next(iter(dataloader))

#%%
import numpy as np
import matplotlib.pyplot as plt
def show_image(img):
    img = 0.5 * (img + 1)  # denormalizeA
    # img = img.clamp(0, 1) 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
# %% Original
import torchvision
show_image(torchvision.utils.make_grid(images))

#%% latent space
latent_image = model.encoder(images)
latent_image_reshaped = latent_image.view(-1, 1, 16, 16)
show_image(torchvision.utils.make_grid(latent_image_reshaped))

#%% reconstruction
reconstructed_image = model.decoder(latent_image)
reconstructed_image.shape
show_image(torchvision.utils.make_grid(reconstructed_image))

#%%

