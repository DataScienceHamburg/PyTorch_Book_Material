
#%% packages
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#%% load data path
with open('data_path.json', 'r') as f:
    data_path_json = json.load(f)
path_train = data_path_json["path_train"]
path_test = data_path_json["path_test"]


#%% check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
device
# %% transformations
train_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10),     
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

# Transformationen für Validierung und Test (ohne Augmentierung)
test_val_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# %% Hyperparameter
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

# %% dataset
train_dataset = torchvision.datasets.ImageFolder(root=path_train,                                    transform=train_transforms)
VALIDATION_SPLIT = 0.2
train_size = int((1 - VALIDATION_SPLIT) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
val_dataset.dataset.transform = test_val_transforms

test_dataset = torchvision.datasets.ImageFolder(root=path_test, transform=test_val_transforms)

# %% dataloader
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset, 
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset, 
                          batch_size=BATCH_SIZE,
                          shuffle=True)

#%% check sizes of datasets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# %% model
class ImageClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16*6*6, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)  # [BS, 6, 30, 30]
        x = self.relu(x)
        x = self.pool(x)  # [BS, 6, 15, 15]
        x = self.conv2(x)  #  [BS, 16, 13, 13]
        x = self.relu(x)
        x = self.pool(x)  # [BS, 16, 6, 6]
        x = self.flatten(x)  # [BS, 16*6*6]
        x = self.fc1(x) # [BS, 64]
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)  # x  # output [BS, 1]
        return x
    
model = ImageClassificationModel().to(device)
# dummy_input = torch.randn(1, 1, 32, 32)  # (BS, C, H, W)
# model(dummy_input).shape
# %% optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), 
                            lr = LEARNING_RATE)

loss_fun = nn.BCEWithLogitsLoss()
# %% training loop
train_losses, val_losses = [], []
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    running_train_loss = 0
    for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
        # move data to device
        X_train_batch = X_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)
        
        # zero gradients
        optimizer.zero_grad()

        # forward pass
        y_train_pred = model(X_train_batch)

        # loss calc
        loss = loss_fun(y_train_pred, y_train_batch.reshape(-1, 1).float())

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # extract losses
        running_train_loss += loss.item()
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch}: Train Loss: {avg_train_loss}")

    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch = X_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            y_val_pred = model(X_val_batch)
            val_loss = loss_fun(y_val_pred, y_val_batch.reshape(-1, 1).float())
            running_val_loss += val_loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # store best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch}: Validation Loss: {avg_val_loss}")

# %%
plt.figure(figsize=(10,6))
sns.lineplot(x=range(EPOCHS), y=train_losses, label='Trainingsverlust [-]')
sns.lineplot(x=range(EPOCHS), y=val_losses, label='Validierungsverlust [-]')
plt.xlabel('Epoche [-]')
plt.ylabel('Verlust [-]')
plt.title('Trainings und Validierungsverlust über Epochen')
plt.xticks(range(0, EPOCHS, 5))
plt.legend()


#%% load best model
model.load_state_dict(torch.load('best_model.pth'))

#%% test loop
y_test_true, y_test_pred = [], []
for i, (X_test_batch, y_test_batch) in enumerate(test_loader):
    # Move input to same device as model
    X_test_batch = X_test_batch.to(device)
    with torch.no_grad():
        y_test_pred_batch = model(X_test_batch)
        y_test_true.extend(y_test_batch.cpu().detach().numpy().tolist())
        y_test_pred.extend(y_test_pred_batch.cpu().detach().numpy().tolist())
# %%
# check the classes of the test set
test_dataset.classes

#%%
threshold = 0.5
y_test_pred_class_str = ['chihuahua' if float(i[0]) > threshold else 'muffin' for i in y_test_pred]
y_test_true_labels_str = ['chihuahua' if i == 1 else 'muffin' for i in y_test_true]

cm = confusion_matrix(y_pred=y_test_pred_class_str, y_true=y_test_true_labels_str)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", 
            xticklabels=['true_muffin', 'true_chihuahua'],
            yticklabels=['pred_muffin', 'pred_chihuahua'],
            cbar=False)
plt.title('Confusion Matrix')
# %%
accuracy = accuracy_score(y_test_pred_class_str, y_test_true_labels_str)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# %%
# %% dummy classifier
y_test_pred_class = [1 if float(i[0]) > threshold else 0 for i in y_test_pred]
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(y_test_true, y_test_pred_class)
dummy_clf.score(y_test_true, y_test_pred_class_str)
print(f"Dummy Classifier Accuracy: {dummy_clf.score(y_test_true, y_test_pred_class)*100:.2f}%")
# %%
print(classification_report(y_test_true, y_test_pred_class))


#%% get the dimensions of train_loader
X_train_batch, _ = next(iter(train_loader))
print(X_train_batch.shape)

# %%
# %% export the model as ONNX
dummy_input = torch.randn(1, 1, 32, 32)
torch.onnx.export(model=model.to("cpu"), 
                 args=dummy_input, 
                 f="bin_class_model.onnx", 
                 verbose=True,
                 opset_version=12)  # Use older opset version to avoid float4 type issues
# %%
