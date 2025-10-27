#%% packages
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import kagglehub

#%% Download latest version
path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset")

print("Path to dataset files:", path)
#%% data import
full_path = os.path.join(path, "enhanced_anxiety_dataset.csv")
anxiety = pd.read_csv(full_path)


#%% check data
print(f"anxiety.columns: {anxiety.columns}")
print(f"anxiety.shape: {anxiety.shape}")

#%% df shape
anxiety.shape
#%% One-Hot Encoding
anxiety_dummies = pd.get_dummies(anxiety, drop_first=True, dtype=int)
anxiety_dummies.shape





#%% convert data to tensor
X = np.array(anxiety_dummies.drop(columns=['Anxiety Level (1-10)']), dtype=np.float32)
y = np.array(anxiety_dummies[['Anxiety Level (1-10)']], dtype=np.float32)
print(f"X shape: {X.shape}, y shape: {y.shape}")

#%% normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)