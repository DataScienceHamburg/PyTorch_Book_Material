#%% packages
from datasets import load_dataset
from torchvision.utils import draw_bounding_boxes
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from PIL import Image

#%% data import
ds = load_dataset("keremberke/aerial-sheep-object-detection")

#%% create datasets
ds_train = ds['train']
ds_val = ds['validation']
ds_test = ds['test']

#%% show dataset with true labels (bounding boxes)
img = ds_train['image'][0]
boxes = ds_train['objects'][0]['bbox']

# Convert PIL Image to tensor and ensure correct format
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # Change from HWC to CHW format
boxes_tensor = torch.tensor(boxes)

# Convert boxes from [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max] format
boxes_xyxy = torch.zeros_like(boxes_tensor)
boxes_xyxy[:, 0] = boxes_tensor[:, 0] - boxes_tensor[:, 2]/2 + 0 # x_min 
boxes_xyxy[:, 1] = boxes_tensor[:, 1] - boxes_tensor[:, 3]/2 + 10  # y_min 
boxes_xyxy[:, 2] = boxes_tensor[:, 0] + boxes_tensor[:, 2]/2 + 10 # x_max 
boxes_xyxy[:, 3] = boxes_tensor[:, 1] + boxes_tensor[:, 3]/2 + 10  # y_max 

img_with_boxes = draw_bounding_boxes(img_tensor, boxes_xyxy, colors='red', width=2)

#%% show dataset with bounding boxes

plt.figure(figsize=(10,10))
plt.imshow(img_with_boxes.permute(1,2,0))  # Convert from CHW to HWC format for matplotlib
plt.axis('off')
plt.show()
# %% number of images in train, val, test
print(f"Number of images in train: {len(ds_train)}")
print(f"Number of images in val: {len(ds_val)}")
print(f"Number of images in test: {len(ds_test)}")
# %% Define paths for saving images and labels
data_dir = '.'
img_train_dir = os.path.join(data_dir, 'images', 'train')
labels_train_dir = os.path.join(data_dir, 'labels', 'train')
img_val_dir = os.path.join(data_dir, 'images', 'val')
labels_val_dir = os.path.join(data_dir, 'labels', 'val')
img_test_dir = os.path.join(data_dir, 'images', 'test')
labels_test_dir = os.path.join(data_dir, 'labels', 'test')

# Create directories if they don't exist
os.makedirs(img_train_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)
os.makedirs(img_val_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)
os.makedirs(img_test_dir, exist_ok=True)
os.makedirs(labels_test_dir, exist_ok=True)
# %%
def process_dataset_split(dataset_split, img_dir, label_dir):
    for i, item in enumerate(dataset_split):
        img = item['image']
        boxes = item['objects']['bbox']
        
        img_filename = f"{i:05d}.jpg" # Padding required
        label_filename = f"{i:05d}.txt"

        # Save image
        img_path = os.path.join(img_dir, img_filename)
        img.save(img_path)

        # Process and save labels
        label_path = os.path.join(label_dir, label_filename)
        with open(label_path, 'w') as f:
            for bbox in boxes:
                # Get image dimensions
                img_width, img_height = img.size

                x_center, y_center, width, height = bbox

                # Normalize bounding box coordinates
                # Ensure float conversion for division
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                # For YOLO, class IDs start from 0. Since 'sheep' is the first and only class, its ID is 0.
                f.write(f"0 {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

print("Processing training data...")
process_dataset_split(ds_train, img_train_dir, labels_train_dir)
print("Processing validation data...")
process_dataset_split(ds_val, img_val_dir, labels_val_dir)
print("Processing test data...")
process_dataset_split(ds_test, img_test_dir, labels_test_dir)

print("\nDataset preparation complete.")
# %% create yaml file for yolo training
yaml_file_content = {
    'path': f'./{data_dir}', # Base path to your dataset
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test', # Add test set for evaluation after training
    'nc': 1,
    'names': ['sheep']
}

# Save the YAML file
yaml_path = os.path.join(data_dir, 'sheep_yolo.yaml')
with open(yaml_path, 'w') as file:
    yaml.dump(yaml_file_content, file, default_flow_style=False)

print(f"\nYAML configuration saved to: {yaml_path}")

