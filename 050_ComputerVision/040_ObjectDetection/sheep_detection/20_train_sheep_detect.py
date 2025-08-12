#%% packages
from ultralytics import YOLO
import os

#%% base-model loading
model = YOLO('yolov8n.pt')

#%% train
# Set MLflow tracking URI to local file system
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

# Train the model
yaml_path = 'sheep_yolo.yaml'
print("\nStarting YOLOv8 training...")
results = model.train(data=yaml_path, epochs=10, imgsz=320, batch=16) # Adjust epochs and batch as needed

print("\nTraining complete!")

# %%
import torch
torch.cuda.empty_cache()