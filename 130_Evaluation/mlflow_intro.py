
#%% packages
import torch
print(torch.cuda.is_available())
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
import mlflow

#%% set tracking uri and experiment name
# Use local file storage instead of remote server
mlflow.set_tracking_uri(uri="file:./mlruns")
mlflow.set_experiment(experiment_name="Fashion MNIST")

#%% hyperparameters (aligned with tensorboard_intro.py)
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001

# Note: Parameters will be logged inside the run context below
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
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE)
# Get cpu or gpu for training.


#%% Define the model (from tensorboard_intro.py).
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

#%%

loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(DEVICE)
model = FashionMnistCnn().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


#%% training loop with MLflow run context
with mlflow.start_run():
    # Log training parameters
    params = {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "Adam",
    }
    mlflow.log_params(params)
    
    train_loss_list, val_loss_list = [], []
    train_accuracy_list, val_accuracy_list = [], []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n")
        epoch_loss_train = 0
        epoch_loss_val = 0
        
        # Training loop integrated directly
        model.train()
        for batch, (X_train, y_train) in enumerate(train_dataloader):
            X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)

            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            metric_fn.update(y_pred, y_train)

            # Backpropagation.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # store loss and accuracy
            epoch_loss_train += loss.item()
        # compute & reset train accuracy per epoch
        epoch_accuracy_train = metric_fn.compute().item()
        metric_fn.reset()
        train_loss_list.append(epoch_loss_train)
        train_accuracy_list.append(epoch_accuracy_train)
        # log with mlflow
        mlflow.log_metric("train_loss", epoch_loss_train, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_accuracy_train, step=epoch)
        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch, (X_val, y_val) in enumerate(validation_dataloader):
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                y_pred = model(X_val)
                loss = loss_fn(y_pred, y_val)
                metric_fn.update(y_pred, y_val)
                epoch_loss_val += loss.item()
        # compute & reset val accuracy per epoch
        epoch_accuracy_val = metric_fn.compute().item()
        metric_fn.reset()
        val_loss_list.append(epoch_loss_val)
        val_accuracy_list.append(epoch_accuracy_val)
        mlflow.log_metric("val_loss", epoch_loss_val, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_accuracy_val, step=epoch)
    
    # Log model summary.
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(model, input_size=(1, 1, 28, 28))))
    mlflow.log_artifact("model_summary.txt")

    # Save the trained model to MLflow.
    model_info = mlflow.pytorch.log_model(model, "fashion_mnist_model")
    
#%% Note: Model registration requires a remote server
    # For local storage, you can skip registration
mlflow.register_model(
    model_uri=model_info.model_uri,
    name="fashion_mnist_model"    
)
print(f"Model saved to: {model_info.model_uri}")

#%% get all registered models
registered_models = mlflow.search_registered_models()
print(registered_models)

#%% load the first registered model
model_uri = registered_models[0].latest_versions[0].source
print(model_uri)

#%% Load model from local storage
loaded_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)
loaded_model.predict(np.random.randn(1, 1, 28, 28).astype(np.float32))


#%% To view your experiments locally, run:
# mlflow ui --host 127.0.0.1 --port 8080
# Then open http://localhost:5000 in your browser