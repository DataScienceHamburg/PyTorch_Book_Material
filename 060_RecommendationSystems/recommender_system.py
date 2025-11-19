#%% packages
# data handling
import kagglehub
import pandas as pd
import numpy as np
import os
# modeling
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# visualizations
import matplotlib.pyplot as plt
import seaborn as sns

#%% Data Download and Load
path = kagglehub.dataset_download("tamber/steam-video-games")
print("Path to dataset files:", path)
file_name = "steam-200k.csv"
df = pd.read_csv(os.path.join(path, file_name), header=None, names=["user_id", "item_title","purchase","behavior", "value"])

#%% delete "value" column
df = df.drop(columns=["value"])
df.head(2)

#%% visualize the distribution of "behavior"
plt.hist(df["behavior"], bins=100, range=(0,100))
plt.show()

#%% create rating column on "behavior" with range 1-5
df["rating"] = df["behavior"].apply(lambda x: 1 if x < 10 else 2 if x < 20 else 3 if x < 40 else 4 if x < 60 else 5)

#%% visualize the group count of "rating"
df["rating"].value_counts()

#%% implement label encoder for "user_id" and "item_id"
# Convert user_id and item_id to 0-based indices
df["user_id"] = pd.factorize(df["user_id"])[0]
df["item_id"] = pd.factorize(df["item_title"])[0]

#%% check the first 2 rows of the dataset
df[["user_id", "item_id", "rating"]].head(2)

#%% create recommender systemdataset
class RecommenderSystemDataset(Dataset):
    def __init__(self, df):
        # Convert RandomSplit subset back to DataFrame if needed
        if isinstance(df, torch.utils.data.dataset.Subset):
            self.df = pd.DataFrame(df.dataset.iloc[df.indices])
        else:
            self.df = df
        self.user_ids = self.df["user_id"].unique()
        self.item_ids = self.df["item_id"].unique()
        self.ratings = self.df["rating"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = self.df.iloc[idx]["user_id"]
        item_id = self.df.iloc[idx]["item_id"]
        rating = self.df.iloc[idx]["rating"]
        return user_id, item_id, rating
    
#%% Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
BATCH_SIZE = 128

#%% create dataloader
# Split data into train and validation sets
train_size = int(0.8 * len(df))
val_size = len(df) - train_size
train_df, val_df = train_test_split(df, test_size=val_size/len(df), random_state=42)

#%% Create datasets
train_dataset = RecommenderSystemDataset(train_df)
val_dataset = RecommenderSystemDataset(val_df)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#%% model class
class RecommenderSystemModel(nn.Module):
    def __init__(self, num_users, num_items):
        super(RecommenderSystemModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, 32)
        self.item_embedding = nn.Embedding(num_items, 32)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, user_id, item_id):
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        x = self.fc(x)
        return x

#%% get number of users and items
num_users = df["user_id"].nunique()
num_items = df["item_id"].nunique()
print(f"Number of users: {num_users}")
print(f"Number of items: {num_items}")

#%% model instance
model = RecommenderSystemModel(num_users, num_items)
model.to(DEVICE)

#%% loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% training loop
for epoch in range(EPOCHS):
    model.train()
    for user_id, item_id, rating in train_dataloader:
        user_id = user_id.to(DEVICE)
        item_id = item_id.to(DEVICE)
        rating = rating.float().to(DEVICE)  # Convert rating to float
        optimizer.zero_grad()
        rating_pred = model(user_id, item_id)
        loss = loss_fn(rating_pred, rating.unsqueeze(1))  # Add dimension to match prediction shape
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

#%% testing loop
model.eval()
rating_true, rating_pred = [], []
with torch.no_grad():
    for user_id, item_id, rating in val_dataloader:
        user_id = user_id.to(DEVICE)
        item_id = item_id.to(DEVICE)
        rating_pred_batch = model(user_id, item_id)
        rating_pred.extend(rating_pred_batch.tolist())
        rating_true.extend(rating.tolist())

#%% visualize the rating_true and rating_pred
sns.regplot(x=rating_true, y=rating_pred)
plt.xlabel('Tatsächliche Bewertungen')
plt.ylabel('Vorhergesagte Bewertungen')
plt.title('Tatsächliche vs. Vorhergesagte Bewertungen')


#%% implement recall at k
def calculate_recall_at_k(model, val_dataloader, k=10, threshold=4.0, device='cpu'):
    """
    Calculate Recall@k for the validation set.
    
    Args:
        model: Trained recommender system model
        val_dataloader: Validation data loader
        k: Number of top recommendations to consider
        threshold: Rating threshold to consider an item as relevant (default: 4.0)
        device: Device to run computations on
    
    Returns:
        float: Average recall@k across all users
    """
    model.eval()
    user_recalls = []
    
    # Group validation data by user
    user_items = {}
    user_ratings = {}
    
    with torch.no_grad():
        for user_id, item_id, rating in val_dataloader:
            for i in range(len(user_id)):
                u = user_id[i].item()
                item = item_id[i].item()
                r = rating[i].item()
                
                if u not in user_items:
                    user_items[u] = []
                    user_ratings[u] = []
                
                user_items[u].append(item)
                user_ratings[u].append(r)
    
    # Calculate recall@k for each user
    for user in user_items.keys():
        if len(user_items[user]) < 2:  # Skip users with too few interactions
            continue
            
        # Get relevant items for this user (items with rating >= threshold)
        relevant_items = set([item for item, rating in zip(user_items[user], user_ratings[user]) 
                            if rating >= threshold])
        
        if len(relevant_items) == 0:
            continue
        
        # Get predictions for all items for this user
        user_tensor = torch.full((num_items,), user, dtype=torch.long, device=device)
        item_tensor = torch.arange(num_items, device=device)
        
        predictions = model(user_tensor, item_tensor).squeeze()
        
        # Get top-k recommendations
        _, top_k_indices = torch.topk(predictions, k=min(k, len(predictions)))
        recommended_items = set(top_k_indices.cpu().numpy())
        
        # Calculate recall@k for this user
        relevant_recommended = len(relevant_items.intersection(recommended_items))
        recall = relevant_recommended / len(relevant_items)
        user_recalls.append(recall)
    
    # Return average recall@k
    return np.mean(user_recalls) if user_recalls else 0.0

def calculate_precision_at_k(model, val_dataloader, k=10, threshold=4.0, device='cpu'):
    """
    Calculate Precision@k for the validation set.
    
    Args:
        model: Trained recommender system model
        val_dataloader: Validation data loader
        k: Number of top recommendations to consider
        threshold: Rating threshold to consider an item as relevant (default: 4.0)
        device: Device to run computations on
    
    Returns:
        float: Average precision@k across all users
    """
    model.eval()
    user_precisions = []
    
    # Group validation data by user
    user_items = {}
    user_ratings = {}
    
    with torch.no_grad():
        for user_id, item_id, rating in val_dataloader:
            for i in range(len(user_id)):
                u = user_id[i].item()
                item = item_id[i].item()
                r = rating[i].item()
                
                if u not in user_items:
                    user_items[u] = []
                    user_ratings[u] = []
                
                user_items[u].append(item)
                user_ratings[u].append(r)
    
    # Calculate precision@k for each user
    for user in user_items.keys():
        if len(user_items[user]) < 2:  # Skip users with too few interactions
            continue
            
        # Get relevant items for this user (items with rating >= threshold)
        relevant_items = set([item for item, rating in zip(user_items[user], user_ratings[user]) 
                            if rating >= threshold])
        
        if len(relevant_items) == 0:
            continue
        
        # Get predictions for all items for this user
        user_tensor = torch.full((num_items,), user, dtype=torch.long, device=device)
        item_tensor = torch.arange(num_items, device=device)
        
        predictions = model(user_tensor, item_tensor).squeeze()
        
        # Get top-k recommendations
        _, top_k_indices = torch.topk(predictions, k=min(k, len(predictions)))
        recommended_items = set(top_k_indices.cpu().numpy())
        
        # Calculate precision@k for this user
        relevant_recommended = len(relevant_items.intersection(recommended_items))
        precision = relevant_recommended / len(recommended_items) if len(recommended_items) > 0 else 0.0
        user_precisions.append(precision)
    
    # Return average precision@k
    return np.mean(user_precisions) if user_precisions else 0.0

#%% Calculate and display recall@k and precision@k metrics
print("\n" + "="*50)
print("RECOMMENDATION SYSTEM EVALUATION")
print("="*50)

# Calculate metrics for different k values
k_values = [5, 10, 20, 50]
threshold = 4.0  # Consider items with rating >= 4 as relevant

for k in k_values:
    recall_k = calculate_recall_at_k(model, val_dataloader, k=k, threshold=threshold, device=DEVICE)
    precision_k = calculate_precision_at_k(model, val_dataloader, k=k, threshold=threshold, device=DEVICE)
    
    print(f"Recall@{k}: {recall_k:.4f}")
    print(f"Precision@{k}: {precision_k:.4f}")
    print("-" * 30)

#%% Visualize recall@k and precision@k for different k values
threshold = 4.0
k_values = [5, 10, 20, 50, 100]
recalls = []
precisions = []

for k in k_values:
    recall_k = calculate_recall_at_k(model, val_dataloader, k=k, threshold=threshold, device=DEVICE)
    precision_k = calculate_precision_at_k(model, val_dataloader, k=k, threshold=threshold, device=DEVICE)
    recalls.append(recall_k)
    precisions.append(precision_k)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Recall@k plot
ax1.plot(k_values, recalls, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('k (Number of Recommendations)')
ax1.set_ylabel('Recall@k')
ax1.set_title('Recall@k vs k')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Precision@k plot
ax2.plot(k_values, precisions, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('k (Number of Recommendations)')
ax2.set_ylabel('Precision@k')
ax2.set_title('Precision@k vs k')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.show()

#%% Example: Get top-k recommendations for a specific user
def get_top_k_recommendations(model, user_id, k=10, device='cpu'):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.full((num_items,), user_id, dtype=torch.long, device=device)
        item_tensor = torch.arange(num_items, device=device)
        
        predictions = model(user_tensor, item_tensor).squeeze()
        
        # Get top-k recommendations
        _, top_k_indices = torch.topk(predictions, k=min(k, len(predictions)))
        return top_k_indices.cpu().numpy().tolist()

#%% Example usage
# Get example user and their recommendations
example_user_id = 2  # Change this to any user ID you want to test
top_10_recommendations = get_top_k_recommendations(model, example_user_id, k=10, device=DEVICE)

# Get the games this user has rated highly (rating >= 4)
user_liked_items = df[
    (df['user_id'] == example_user_id) & 
    (df['rating'] >= 4)
]['item_title'].tolist()

print(f"\nUser {example_user_id}'s highly rated games:")
for item_title in user_liked_items:
    game_name = df[df['item_title'] == item_title]['item_title'].iloc[0]
    print(f"- Game ID: {game_name}")

print(f"\nTop 10 recommended games for user {example_user_id}:")
for item_id in top_10_recommendations:
    game_name = df[df['item_id'] == item_id]['item_title'].iloc[0] 
    print(f"- Game ID: {game_name}")


# %%
