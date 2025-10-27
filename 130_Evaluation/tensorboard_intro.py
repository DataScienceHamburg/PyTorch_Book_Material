
#%% packages
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torchmetrics.classification import Accuracy

#%% writer
# Use an explicit log directory to avoid pointing TensorBoard to the wrong run
writer = SummaryWriter(log_dir="runs/tensorboard_intro")

#%% hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

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


def log_images_to_tensorboard(dataloader, epoch, num_images=16):
    """Log a grid of FashionMNIST images to TensorBoard"""
    # Get a batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Take only the first num_images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Create a grid of images
    img_grid = make_grid(images, nrow=4, normalize=True, scale_each=True)
    
    # Log to TensorBoard
    writer.add_image(
        tag=f'FashionMNIST_Samples_Epoch_{epoch}',
        img_tensor=img_grid,
        global_step=epoch,
        dataformats="CHW",
    )
    
    # Also log individual images with their labels
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    for i in range(min(4, len(images))):  # Log first 4 individual images
        writer.add_image(
            f'Individual_Images/{class_names[labels[i]]}',
            images[i],
            global_step=epoch,
            dataformats="CHW",
        )



#%% train
loss_fn = nn.CrossEntropyLoss()
metric_train = Accuracy(task="multiclass", num_classes=10).to(device)
metric_val = Accuracy(task="multiclass", num_classes=10).to(device)
model = FashionMnistCnn().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Log model graph to TensorBoard
# Get a sample batch to trace the model
sample_batch = next(iter(train_dataloader))
sample_images = sample_batch[0][:1].to(device)  # Take just one image
writer.add_graph(model, sample_images)

train_loss_list, val_loss_list = [], []
train_accuracy_list, val_accuracy_list = [], []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}\n")
    epoch_loss_train = 0
    epoch_loss_val = 0
    
    # Log images to TensorBoard at the beginning of each epoch
    log_images_to_tensorboard(train_dataloader, epoch)
    
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
    writer.add_scalar("train_loss", epoch_loss_train, global_step=epoch)
    writer.add_scalar("train_accuracy", epoch_accuracy_train, global_step=epoch)

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
    writer.add_scalar("val_loss", epoch_loss_val, global_step=epoch)
    writer.add_scalar("val_accuracy", epoch_accuracy_val, global_step=epoch)

    # Ensure events are written to disk for TensorBoard to pick up
    writer.flush()

writer.close()

#%%
# To view TensorBoard with images and model graph:
# tensorboard --logdir=runs
