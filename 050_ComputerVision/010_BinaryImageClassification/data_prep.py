#%% dataset: Muffin vs. Chihuahua
# Source: https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification
# License: CC0: Public Domain
# Author: Samuel Cortinhas

#%% packages
import kagglehub
import json
import os

# Download latest version
path = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")

print("Path to dataset files:", path)

#%% path train and test
path_train = os.path.join(path, "train")
path_test = os.path.join(path, "test")



# %% store datapath reference in json file
with open('data_path.json', 'w') as f:
    json.dump({"path_train": path_train, "path_test": path_test}, f)



