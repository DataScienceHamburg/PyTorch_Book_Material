
#%% packages
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchmetrics.classification import Accuracy

#%% hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
config = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
}


#%% wandb
import wandb
# wandb.login()
#%% init wandb
run = wandb.init(project="fashion_mnist_model", notes="CNN for Fashion MNIST, reduced batch size", config=config)



#%% Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
validation_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

#%% Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE)

# Get cpu or gpu for training.
device = "cuda" if torch.cuda.is_available() else "cpu"


#%% Define the model.
class FashionMnistCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1x28x28 -> 32x14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x14x14 -> 64x7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits
#%% train
loss_fn = nn.CrossEntropyLoss()
metric_train = Accuracy(task="multiclass", num_classes=10).to(device)
metric_val = Accuracy(task="multiclass", num_classes=10).to(device)
model = FashionMnistCnn().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


#%% training loop
train_loss_list, val_loss_list = [], []
train_accuracy_list, val_accuracy_list = [], []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}\n")
    epoch_loss_train = 0
    epoch_loss_val = 0
    
    
    # Training loop integrated directly
    model.train()
    for batch, (X_train, y_train) in enumerate(train_dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        metric_train.update(y_pred, y_train)

        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # store loss and accuracy
        epoch_loss_train += loss.item()
    # compute & reset train accuracy per epoch
    epoch_accuracy_train = metric_train.compute().item()
    metric_train.reset()
    train_loss_list.append(epoch_loss_train)
    train_accuracy_list.append(epoch_accuracy_train)
    wandb.log({
        "train_loss": epoch_loss_train,
        "train_accuracy": epoch_accuracy_train,
    })
    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch, (X_val, y_val) in enumerate(validation_dataloader):
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_pred = model(X_val)
            loss = loss_fn(y_pred, y_val)
            metric_val.update(y_pred, y_val)
            epoch_loss_val += loss.item()
    # compute & reset val accuracy per epoch
    epoch_accuracy_val = metric_val.compute().item()
    metric_val.reset()
    val_loss_list.append(epoch_loss_val)
    val_accuracy_list.append(epoch_accuracy_val)
    wandb.log({
        "val_loss": epoch_loss_val,
        "val_accuracy": epoch_accuracy_val,
    })

#%% save data artifacts directly as files
torch.save(train_loss_list, "train_loss_list.pth")
wandb.save("train_loss_list.pth")

#%% 
torch.save(model.state_dict(), "model.pth")
model_artifact = wandb.Artifact("model_artifact", type="model")
model_artifact.add_file("model.pth")
wandb.log_artifact(model_artifact)

#%% save tables as artifacts
train_loss_table = wandb.Table(dataframe=pd.DataFrame({"train_loss": train_loss_list, "train_accuracy": train_accuracy_list}))
train_loss_artifact = wandb.Artifact("train_loss_artifact", type="table")
train_loss_artifact.add(train_loss_table, "train_loss_table")
wandb.log_artifact(train_loss_artifact)


#%% define a sweep
sweep_config = {
    "name": "fashion_mnist_model",
    "method": "grid",
    "parameters": {
        "epochs": {"values": [5, 10]},
        "batch_size": {"values": [32, 64]},
        "learning_rate": {"values": [0.001, 0.0001]},
    },
}

def sweep_run():
    run = wandb.init(project="fashion_mnist_model", notes="CNN for Fashion MNIST, reduced batch size", config=config)
    model = FashionMnistCnn().to(device)
    train_loss_list, val_loss_list = [], []
    train_accuracy_list, val_accuracy_list = [], []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n")
        epoch_loss_train = 0
        epoch_loss_val = 0
        
        
        # Training loop integrated directly
        model.train()
        for batch, (X_train, y_train) in enumerate(train_dataloader):
            X_train, y_train = X_train.to(device), y_train.to(device)

            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            metric_train.update(y_pred, y_train)

            # Backpropagation.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # store loss and accuracy
            epoch_loss_train += loss.item()
        # compute & reset train accuracy per epoch
        epoch_accuracy_train = metric_train.compute().item()
        metric_train.reset()
        train_loss_list.append(epoch_loss_train)
        train_accuracy_list.append(epoch_accuracy_train)
        wandb.log({
            "train_loss": epoch_loss_train,
            "train_accuracy": epoch_accuracy_train,
        })
        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch, (X_val, y_val) in enumerate(validation_dataloader):
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                loss = loss_fn(y_pred, y_val)
                metric_val.update(y_pred, y_val)
                epoch_loss_val += loss.item()
        # compute & reset val accuracy per epoch
        epoch_accuracy_val = metric_val.compute().item()
        metric_val.reset()
        val_loss_list.append(epoch_loss_val)
        val_accuracy_list.append(epoch_accuracy_val)
        wandb.log({
            "val_loss": epoch_loss_val,
            "val_accuracy": epoch_accuracy_val,
        })
    wandb.finish()

sweep_id = wandb.sweep(sweep_config, project="fashion_mnist_model")
wandb.agent(sweep_id, function=sweep_run)







#%% finish wandb
wandb.finish()
