#%% packages
import os
from PIL import Image
import kagglehub
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
#%% data import
path = kagglehub.dataset_download("koryakinp/fingers")

print("Path to dataset files:", path)
# %%
image_path_train = os.path.join(path, "train")
image_path_test = os.path.join(path, "test")

#%% Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = 32
LEARNING_RATE = 0.001

# %% Dataset
class FingersDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Load all images during initialization
        self.images = []
        self.labels = []
        for img_name in self.image_files:
            image = Image.open(os.path.join(image_path, img_name))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            self.images.append(image)
            self.labels.append(int(img_name[-6]))  

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        y = self.labels[idx]
        
        # Apply transforms
        X = self.transform(image)
        
        return X, y
    
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.Grayscale(num_output_channels=1),
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.Grayscale(num_output_channels=1),
])

# Create train and validation datasets
train_dataset = FingersDataset(image_path_train, transform=transform_train)
val_test_dataset = FingersDataset(image_path_test, transform=transform_test)

# Split validation dataset into validation and test
val_size = len(val_test_dataset)
val_split = val_size // 2
test_split = val_size - val_split

val_test_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [val_split, test_split])

# Test the splits
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_test_dataset)}")
print(f"Test set size: {len(test_dataset)}")

#%% sample from train_dataset
X_sample, y_sample = next(iter(train_dataset))
X_sample.shape, y_sample

#%% dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#%% model class
class FingersModel(nn.Module):
    def __init__(self, num_classes):
        super(FingersModel, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # Calculate input features for first FC layer
        self._to_linear = 64 * 8 * 8  # After 2 pooling layers: 32->16->8
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Activations and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2d = nn.Dropout2d(0.25)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # FC Layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)        
        return x

model = FingersModel(num_classes=6)
model

# test tensor
test_tensor = torch.randn(1, 1, 32, 32)
model(test_tensor).shape

#%% loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#%% training loop
losses_train, losses_val = [], []
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    epoch_loss = 0
    for X_train, y_train in train_dataloader:
        # Forward pass
        y_pred = model(X_train)
        y_pred_class = torch.argmax(y_pred, dim=1)
        # calculate loss
        loss_train = criterion(y_pred, y_train)  # CrossEntropyLoss expects raw logits and target as class indices
        # zero gradients
        optimizer.zero_grad()
        # backward pass
        loss_train.backward()
        # update weights
        optimizer.step()
        # update epoch loss
        epoch_loss += loss_train.item()
    losses_train.append(epoch_loss / len(train_dataloader))

    # Validation loop
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        epoch_loss = 0
        for X_val, y_val in val_dataloader:
            # forward pass
            y_pred = model(X_val)
            # calculate loss
            loss_val = criterion(y_pred, y_val)  # CrossEntropyLoss expects raw logits and target as class indices
            # update epoch loss
            epoch_loss += loss_val.item()
        losses_val.append(epoch_loss / len(val_dataloader))
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {losses_train[-1]:.4f}, Val Loss: {losses_val[-1]:.4f}")
# %%
sns.lineplot(x=range(len(losses_train)), y=losses_train, label='Training')
sns.lineplot(x=range(len(losses_val)), y=losses_val, label='Validierung')
plt.xlabel('Epoch [-]')
plt.ylabel('Verlust [-]')
plt.title('Verlustkurve für Trainings- und Validierungsdaten')
plt.legend()
plt.xticks(range(len(losses_train)))  # Set integer ticks
plt.show()

# %% evaluate test data
y_true = []
y_pred = []
model.eval()  # Set model to evaluation mode
with torch.no_grad():    
    for X_test, y_test in test_dataloader:
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy().tolist())
        y_true.extend(y_test.cpu().numpy().tolist())

#%%
cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='.0f')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Tatsächliche Klasse') 
plt.title('Finger-Modell:Konfusionsmatrix')
plt.show()

#%% accuracy
accuracy_score(y_true, y_pred)

#%% dummy classifier
y_pred_dummy = np.zeros(len(y_true))
accuracy_score(y_true, y_pred_dummy)
# %%
