#%% packages
import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.dummy import DummyClassifier
import numpy as np


#%% data import
path = kagglehub.dataset_download("developerghost/intrusion-detection-logs-normal-bot-scan")

print("Path to dataset files:", path)

file_path = os.path.join(path, "Network_logs.csv")
df = pd.read_csv(file_path)
df.head()

#%% drop features that are not useful for the analysis
df = df.drop(columns=["Source_IP", "Destination_IP", "Intrusion"])

#%% treat categorical variables
df_cat = pd.get_dummies(df, columns=[ 'Request_Type', 'Protocol',  'User_Agent','Status'], drop_first=True, dtype=int)


#%% separate independent and dependent variables
X = df_cat.drop(columns=['Scan_Type']).astype(float)
y = pd.factorize(df_cat["Scan_Type"])[0].astype(float)  # Convert categorical to numeric labels
print(f"X shape: {X.shape}, y shape: {y.shape}")

#%% split data into training, validation and testing sets
# First split off test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# Split remaining data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")


#%% Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 50  
HIDDEN_SIZE = 64  
OUTPUT_SIZE = len(df_cat["Scan_Type"].unique())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Number of classes: {OUTPUT_SIZE}")

# Check class distribution
print("Class distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
for class_idx, count in zip(unique, counts):
    print(f"Class {class_idx}: {count} samples ({count/len(y_train)*100:.1f}%)")

#%% visualize class distribution
plt.figure()
# Convert counts to percentages for training data
percentages_train = (counts / len(y_train)) * 100

# Get validation data distribution
unique_val, counts_val = np.unique(y_val, return_counts=True)
percentages_val = (counts_val / len(y_val)) * 100

# Get test data distribution
unique_test, counts_test = np.unique(y_test, return_counts=True)
percentages_test = (counts_test / len(y_test)) * 100

# Plot distributions
width = 0.25
plt.bar(unique - width, percentages_train, width, label='Training')
plt.bar(unique, percentages_val, width, label='Validation') 
plt.bar(unique + width, percentages_test, width, label='Test')

plt.xlabel('Klasse [-]')
plt.ylabel('Anteil [%]')
plt.title('Klassenverteilung in Training-, Validierungs- und Testset')
plt.xticks([0, 1, 2])
plt.legend()

# %% create a custom dataset class
train_dataset = TensorDataset(torch.FloatTensor(X_train.values), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val.values), torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test.values), torch.LongTensor(y_test))

# %% create a data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% model with regularization
class MulticlassModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate=0.3):
        super(MulticlassModel, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear_hidden1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.linear_hidden2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.linear_out = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear_hidden1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear_hidden2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear_out(x)
        return x

input_size = X_train.shape[1]

model = MulticlassModel(input_size=input_size, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)

# %% optimizer and loss function with weight decay for regularization
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

#%% training loop
train_losses = []
val_losses = []
for epoch in range(EPOCHS):
    train_loss = 0
    val_loss = 0
    for X_train_batch, y_train_batch in train_loader:
        # move data to device
        X_train_batch, y_train_batch = X_train_batch.to(DEVICE), y_train_batch.to(DEVICE)
        # forward pass
        y_train_batch_pred = model(X_train_batch)
        # calculate loss
        loss = loss_fn(y_train_batch_pred, y_train_batch)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()
        # reset gradients
        optimizer.zero_grad()
        # update train loss
        train_loss += loss.item()
    # normalize and append train loss
    train_losses.append(train_loss / len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_losses[-1]:.4f}")
    for X_val_batch, y_val_batch in val_loader:
        # move data to device
        X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
        # forward pass
        y_val_batch_pred = model(X_val_batch)
        # calculate loss
        loss = loss_fn(y_val_batch_pred, y_val_batch)
        # update val loss
        val_loss += loss.item()
    # normalize and append val loss
    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_losses[-1]:.4f}")

# %% visualize training and validationloss
plt.figure()
sns.lineplot(x=list(range(EPOCHS)), y=train_losses, label='Trainingsverlust')
sns.lineplot(x=list(range(EPOCHS)), y=val_losses, label='Validierungsverlust')
plt.xticks(range(0, EPOCHS, 5))
plt.xlabel('Epoche [-]')
plt.ylabel('Verlust [-]')
plt.title('Training und Validierungsverlust über die Epochen')
plt.legend()

#%% create test predictions for multiclass classification
y_test_pred_proba, y_test_pred_class, y_test_true = [], [], []

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        X_test_batch, y_test_batch = X_test_batch.to(DEVICE), y_test_batch.to(DEVICE)
        y_test_batch_pred = model(X_test_batch)
        
        # Apply softmax to get probabilities
        y_test_batch_pred_proba = torch.softmax(y_test_batch_pred, dim=1)
        y_test_batch_pred_proba = y_test_batch_pred_proba.cpu().numpy()
        
        # Get predicted classes (argmax)
        y_test_batch_pred_class = torch.argmax(y_test_batch_pred, dim=1).cpu().numpy()
        y_test_batch = y_test_batch.cpu().numpy()
        
        # Store predictions and true labels
        y_test_pred_proba.extend(y_test_batch_pred_proba.tolist())
        y_test_pred_class.extend(y_test_batch_pred_class.tolist())
        y_test_true.extend(y_test_batch.tolist())

#%% create confusion matrix
cm = confusion_matrix(y_true=y_test_true, y_pred=y_test_pred_class)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Wahre Klasse')
plt.title('Konfusionsmatrix')

#%% accuracy score
accuracy_score(y_true=y_test_true, y_pred=y_test_pred_class)

#%% naive classifier and accuracy score
model_naive = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
y_test_pred_naive = model_naive.predict(X_test)
accuracy_score(y_true=y_test_true, y_pred=y_test_pred_naive)


# %%
