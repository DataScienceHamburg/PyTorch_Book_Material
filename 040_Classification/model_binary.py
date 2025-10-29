#%% packages
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

from data_prep_binary import X_train, X_test, y_train, y_test

#%% Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 20
HIDDEN_SIZE = 4
OUTPUT_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% create a custom dataset class
import torch
train_dataset = TensorDataset(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train.values))
test_dataset = TensorDataset(torch.FloatTensor(X_test.values), torch.FloatTensor(y_test.values))

# %% create a data loader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% create a model with regularization to reduce overconfidence
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate=0.2):
        super(BinaryClassificationModel, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Add dropout for regularization
        x = self.lin2(x)
        return x

input_size = X_train.shape[1]

model = BinaryClassificationModel(input_size=input_size, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)

# %% optimizer and loss function with weight decay for regularization
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

#%% training loop
train_losses = []
for epoch in range(EPOCHS):
    train_loss = 0
    for X_train_batch, y_train_batch in train_loader:
        # move data to device
        X_train_batch, y_train_batch = X_train_batch.to(DEVICE), y_train_batch.to(DEVICE)
        # forward pass
        y_train_batch_pred = model(X_train_batch)
        # calculate loss
        loss = loss_fn(y_train_batch_pred, y_train_batch.reshape(-1, 1))
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()
        # reset gradients
        optimizer.zero_grad()
        # update train loss
        train_loss += loss.item()
    # append train loss
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_losses[-1]:.4f}")

# %% visualize training loss
plt.figure()
sns.lineplot(x=list(range(EPOCHS)), y=train_losses)
plt.xticks(range(EPOCHS))
plt.xlabel('Epoch [-]')
plt.ylabel('Loss [-]')
plt.title('Training Loss and Epochs')

#%% create test predictions - FIXED: Use probabilities for ROC curve
y_test_pred_proba, y_test_pred_class, y_test_true = [], [], []
THRESHOLD = 0.5

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        X_test_batch, y_test_batch = X_test_batch.to(DEVICE), y_test_batch.to(DEVICE)
        y_test_batch_pred = model(X_test_batch)
        y_test_batch_pred = y_test_batch_pred.cpu().numpy()
        
        # Store probabilities for ROC curve
        y_test_pred_proba.extend(y_test_batch_pred.flatten().tolist())
        
        # Store class predictions for confusion matrix
        y_test_batch_pred_class = (y_test_batch_pred > THRESHOLD).astype(int)
        y_test_batch = y_test_batch.cpu().numpy()
        y_test_pred_class.extend(y_test_batch_pred_class.flatten().tolist())
        y_test_true.extend(y_test_batch.flatten().tolist())

#%% create confusion matrix
cm = confusion_matrix(y_true=y_test_true, y_pred=y_test_pred_class)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (binary classification)')

#%% accuracy score
accuracy_score(y_true=y_test_true, y_pred=y_test_pred_class)

#%% naive classifier and accuracy score
model_naive = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
y_test_pred_naive = model_naive.predict(X_test)
accuracy_score(y_true=y_test_true, y_pred=y_test_pred_naive)

#%% FIXED: roc curve using probabilities instead of class predictions
fpr, tpr, thresholds = roc_curve(y_true=y_test_true, y_score=y_test_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR [-]')
plt.ylabel('TPR [-]')
plt.title('ROC Curve (binary classification)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)


# %%
