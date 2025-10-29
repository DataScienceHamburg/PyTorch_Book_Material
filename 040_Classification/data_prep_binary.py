#%% packages
import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#%% data import
path = kagglehub.dataset_download("developerghost/intrusion-detection-logs-normal-bot-scan")

print("Path to dataset files:", path)

file_path = os.path.join(path, "Network_logs.csv")
df = pd.read_csv(file_path)
df.shape

#%% show the features
df.columns


#%% see which features are categorical and which are numerical
df.describe()



#%% drop features that are not useful for the analysis
df = df.drop(columns=["Source_IP", "Destination_IP", "Scan_Type"])
#%% see which features are categorical and which are numerical
df.dtypes
#%% treat categorical variables
df_cat = pd.get_dummies(df, drop_first=True, dtype=int)

#%% check the number of unique values in each feature
df_cat.shape

#%% visualize the distribution of the target variable
sns.countplot(x="Intrusion", data=df_cat)
plt.title("Target Variable Imbalance")

# %% visualise the correlation of independent variables
# Create correlation matrix
corr_matrix = df_cat.corr()

# Create heatmap
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr_matrix), k=1).T
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', mask=mask)
plt.title('Correlation Matrix of Variables')
plt.tight_layout()


# %%

#%% separate independent and dependent variables
X = df_cat.drop(columns=["Intrusion"])
y = df_cat["Intrusion"]
print(f"X shape: {X.shape}, y shape: {y.shape}")

#%% split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")



# %%
