#%% packages
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub

#%% Download latest version
path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset")

print("Path to dataset files:", path)
#%% data import
b
#%% check data
print(f"anxiety.columns: {anxiety.columns}")
print(f"anxiety.shape: {anxiety.shape}")

#%% df shape
anxiety.shape
#%% One-Hot Encoding
anxiety_dummies = pd.get_dummies(anxiety, drop_first=True, dtype=int)
anxiety_dummies.shape


#%% some correlation
# sns.scatterplot(x='gestation', y='bwt', data=babies, color='blue')
sns.regplot(x='Sleep Hours', y='Anxiety Level (1-10)', data=anxiety_dummies, color='blue', line_kws={'color': 'red'})
# add a title
plt.title('Sleep Hours vs Anxiety Level')
# add x title
plt.xlabel('Sleep Hours')
# add y title
plt.ylabel('Anxiety Level')

#%% check correlation
# Select only numerical features for correlation analysis
numerical_features = anxiety.select_dtypes(include=['int64', 'float64'])
corr = numerical_features.corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Plot correlation heatmap
sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, mask=mask)
plt.title('Correlation Heatmap (Numerical Features Only)', fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, ha='right', fontsize=8)
plt.tight_layout()
plt.show()

#%% convert data to tensor
X = np.array(anxiety_dummies.drop(columns=['Anxiety Level (1-10)']), dtype=np.float32)
y = np.array(anxiety_dummies[['Anxiety Level (1-10)']], dtype=np.float32)
print(f"X shape: {X.shape}, y shape: {y.shape}")

#%% normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)