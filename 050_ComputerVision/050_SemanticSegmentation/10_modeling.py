#%% packages
import kagglehub
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer, AutoImageProcessor
from evaluate import load

#%% Download data
# source: https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation
dataset_name = "faizalkarim/flood-area-segmentation"
path = kagglehub.dataset_download(dataset_name)

print("Path to dataset files:", path)

#%% load the dataset
metadata = pd.read_csv(os.path.join(path, "metadata.csv"))
metadata.head(2)

#%% visualize sample image and segmentation mask
index = 4
path_images = os.path.join(path, "Image")
path_masks = os.path.join(path, "Mask")
img_path = os.path.join(path_images, metadata.loc[index, "Image"])
mask_path = os.path.join(path_masks, metadata.loc[index, "Mask"])
img = Image.open(img_path)
mask = Image.open(mask_path)
# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image
ax1.imshow(img)
ax1.set_title('Original-Image')
ax1.axis('off')

# Plot segmentation mask
ax2.imshow(mask)
ax2.set_title('Segmentation Mask') 
ax2.axis('off')

plt.tight_layout()
plt.show()

# %% dataset preparation
class FloodSegmentationDataset(Dataset):
    def __init__(self, metadata_df, img_dir, mask_dir):
        self.metadata = metadata_df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
            ])


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get image and mask paths
        img_name = self.metadata.iloc[idx]["Image"]
        mask_name = self.metadata.iloc[idx]["Mask"]
        
        # Load image and mask
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # import image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
        
        # Apply transformations
        image = self.transform(image)
        mask = self.transform(mask)  # Convert mask to tensor
        mask = mask.squeeze(0)  # Remove channel dimension to get [H,W]
        mask = (mask > 0.5).long()  # Convert to binary mask with long dtype  
        return image, mask


# Create dataset
dataset = FloodSegmentationDataset(
    metadata_df=metadata,
    img_dir=path_images,
    mask_dir=path_masks
)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

#%% Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
#%%
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# %% extract one image and one mask
for i, (image, mask) in enumerate(train_loader):
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    break

#%% id2label and label2id
id2label = {0: "background", 1: "flood"}
label2id = {v: k for k, v in id2label.items()}

#%% preprocess model
checkpoint = "nvidia/mit-b0"
# image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
# metric = load("mean_iou")
# num_labels = len(id2label)
# %%
# Define data collator function
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([i[0] for i in batch]),
        "labels": torch.stack([i[1] for i in batch])
    }
training_args = TrainingArguments(
    output_dir="flood-segmentation-128",
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_total_limit=3,
    logging_dir=None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn
)

trainer.train()
# %% evaluate the model performance
# Get a sample image from validation dataset
sample_idx = 1
sample_image, sample_mask = val_dataset[sample_idx]

# Convert tensors to numpy arrays and adjust dimensions for plotting
sample_image = sample_image.permute(1, 2, 0).numpy()
sample_mask = sample_mask.numpy()

# %% create a prediction for the sample image
mask_prediction = trainer.predict(val_dataset)
sample_mask_pred = mask_prediction[0][1][sample_idx]

# adapt mask to have value 1 if >0, 0 otherwise
sample_mask_pred_binary = (sample_mask_pred > 0)

# Create figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

# Plot original image
ax1.imshow(sample_image)
ax1.set_title('Example Validation Image')
ax1.axis('off')

# Plot ground truth mask
ax2.imshow(sample_mask)
ax2.set_title('Ground Truth Maske')
ax2.axis('off')

# Plot prediction mask
ax3.imshow(sample_mask_pred)
ax3.set_title('Predicted Mask')
ax3.axis('off')

plt.tight_layout()
plt.show()


# %%
