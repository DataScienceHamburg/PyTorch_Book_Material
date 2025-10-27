#%% packages
# data handling
from torch.utils.data import random_split, DataLoader
import os
import numpy as np
import kagglehub

# modeling
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from sklearn.dummy import DummyClassifier
# evaluation
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

#%% data import
# Download latest version
path = kagglehub.dataset_download("zlatan599/garbage-dataset-classification")

print("Path to dataset files:", path)
folder_path = os.path.join(path, "Garbage_Dataset_Classification", "images")

# %% Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Load the full dataset
full_dataset = torchvision.datasets.ImageFolder(
    root=folder_path,
    transform=transform
)

#%% Split into train, validation and test sets (60% train, 20% val, 20% test)
train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size]
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#%% get the class labels
class_labels = full_dataset.classes
OUTPUT_FEATURES = len(class_labels)
class_labels


#%% model
model = models.densenet121(pretrained = True)


#%% freeze ALL model layers
for params in model.parameters():
    params.requires_grad = False
# %% overwrite the classifier
model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(in_features=1024, out_features=OUTPUT_FEATURES)),
    ('Output', nn.Softmax(dim=1))
]))
model = model.to(DEVICE)

# %% Optimizer and Loss Function
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# %% train loop
losses_train, losses_val = [], []
for epoch in range(EPOCHS):
    loss_epoch_train, loss_epoch_val = 0, 0
    # Extract current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    for _, (X_batch, y_batch) in enumerate(train_loader):
        # zero gradients
        optimizer.zero_grad()
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        # forward pass
        y_batch_pred = model(X_batch)
        
        
        # loss calculation
        loss = loss_function(y_batch_pred, y_batch)
        loss_epoch_train += loss.item()
        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
    losses_train.append(loss_epoch_train)
    # validation loop
    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for _, (X_batch, y_batch) in enumerate(val_loader):
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_batch_pred = model(X_batch)
            loss = loss_function(y_batch_pred, y_batch)
            val_loss_epoch += loss.item()
    losses_val.append(val_loss_epoch)
    print(f"Epoch: {epoch}, Loss: {loss_epoch_train}, Val Loss: {val_loss_epoch}")

#%% losses
sns.lineplot(x=range(EPOCHS), y=losses_train)
sns.lineplot(x=range(EPOCHS), y=losses_val)
plt.xlabel('Epoche [-]')
plt.ylabel('Verlust [-]')
plt.title('Training und Validierungsverlust')

# %% Evaluate model
model.eval()
y_test_true = []
y_test_pred = []
for _, (X_batch, y_batch) in enumerate(test_loader):
    # forward pass
    with torch.no_grad():
        X_batch = X_batch.to(DEVICE)
        y_test_pred_batch = model(X_batch).detach().cpu().numpy()
    y_test_pred.extend(np.argmax(y_test_pred_batch, axis=1).tolist())
    y_test_true.extend(y_batch.numpy().tolist())
        
# %% confusion matrix and accuracy

cm = confusion_matrix(y_test_true, y_test_pred)
cm
#%%
cm_normalized = confusion_matrix(y_test_true, y_test_pred, normalize='true')*100
cm_normalized = cm_normalized - 2*np.triu(cm_normalized, 1) - 2*np.tril(cm_normalized, -1)
plt.title("Konfusionsmatrix")
sns.heatmap(cm_normalized, xticklabels=class_labels, yticklabels=class_labels, annot=cm, fmt='.0f', vmin=-100, vmax=100, cmap='PiYG', cbar_kws={'format':'%d%%'})
#%%
accuracy_score(y_test_true, y_test_pred)
# %% dummy classification
dummy_clf = DummyClassifier(strategy="most_frequent")
y_pred_dummy_all = []
y_true_dummy_all = []
for _, (X_batch, y_batch) in enumerate(test_loader):
    dummy_clf.fit(X_batch, y_batch)
    y_pred_dummy = dummy_clf.predict(X_batch)
    y_pred_dummy_all.extend(y_pred_dummy.tolist())
    y_true_dummy_all.extend(y_batch.numpy().tolist())

accuracy_score(y_true_dummy_all, y_pred_dummy_all)




# %%
