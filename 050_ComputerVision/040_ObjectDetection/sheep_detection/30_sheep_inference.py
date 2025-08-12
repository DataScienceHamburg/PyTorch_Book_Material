#%% packages
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datasets import load_dataset

#%% inference model
try:
    best_model = "../../../runs/detect/train3/weights/best.pt"
    inference_model = YOLO(best_model)
except:
    print("Error loading best model")

# %% load test dataset
ds = load_dataset("keremberke/aerial-sheep-object-detection")
ds_test = ds['test']

#%% select a specific image
test_image = ds_test['image'][1]

#%% show test image
plt.imshow(test_image)
plt.axis('off')
plt.show()

#%% inference
results = inference_model(test_image)

# %% show results
plt.imshow(results[0].plot())
plt.axis('off')
plt.show()



