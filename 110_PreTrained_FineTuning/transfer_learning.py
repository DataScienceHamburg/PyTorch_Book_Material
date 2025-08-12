#%% packages
import torch
import torchvision
from torchvision import transforms, models
import seaborn as sns
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split, DataLoader
import os
import numpy as np
import torch.nn as nn
from collections import OrderedDict


# %% Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 100
EPOCHS = 10

#%% 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Load the full dataset
full_dataset = torchvision.datasets.ImageFolder(
    root="../data/tesla_sun_trafficlight",
    transform=transform
)

# Split into train and test sets (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


#%% model
model = models.densenet121(pretrained = True)
model


#%% freeze ALL model layers
for params in model.parameters():
    params.requires_grad = False
# %% overwrite the classifier
model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(in_features=1024, out_features=4)),
    ('Output', nn.Softmax(dim=1))
]))

# %% Optimizer and Loss Function

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# %% train loop
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    # Extract current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    for _, (X_batch, y_batch) in enumerate(train_loader):
        # zero gradients
        optimizer.zero_grad()
        
        # forward pass
        y_batch_pred = model(X_batch)
        
        
        # loss calculation
        loss = loss_function(y_batch_pred, y_batch)
        loss_epoch += loss.item()
        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
    losses.append(loss_epoch)
    print(f"Epoch: {epoch}, Loss: {loss_epoch}, Learning Rate: {current_lr}")

#%% losses
sns.lineplot(x=range(EPOCHS), y=losses)
# %% Evaluate model
y_test_true = []
y_test_pred = []
for _, (X_batch, y_batch) in enumerate(test_loader):
    # forward pass
    with torch.no_grad():
        y_test_pred_batch = model(X_batch).round().numpy()
        y_test_pred.extend(np.argmax(y_test_pred_batch, axis=1).tolist())
        y_test_true.extend(y_batch.numpy().tolist())
        
        
# %% confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test_true, y_test_pred)
cm
#%%
cm_normalized = confusion_matrix(y_test_true, y_test_pred, normalize='true')*100
cm_normalized = cm_normalized - 2*np.triu(cm_normalized, 1) - 2*np.tril(cm_normalized, -1)
labels = np.unique(y_test_true)
sns.heatmap(cm_normalized, xticklabels=labels, yticklabels=labels, annot=cm, fmt='.0f', vmin=-100, vmax=100, cmap='PiYG', cbar_kws={'format':'%d%%'})
#%%
accuracy_score(y_test_true, y_test_pred)
# %% dummy classification
from sklearn.dummy import DummyClassifier
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
