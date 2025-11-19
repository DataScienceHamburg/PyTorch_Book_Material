#%% packages
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
import pytorch_lightning as pl

#%%
class RecommenderModel(pl.LightningModule):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_id, item_id):
        user_vector = self.user_embedding(user_id)
        item_vector = self.item_embedding(item_id)
        return (user_vector * item_vector).sum(1)

    def training_step(self, batch, batch_idx):
        user_id, item_id, rating = batch
        prediction = self(user_id, item_id)
        loss = nn.functional.mse_loss(prediction, rating)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.01)


class MovieLensDataset(Dataset):
    def __init__(self, dataframe=None):
        self.dataframe = dataframe or self.load_movielens_data()
        # Add 1 to account for 0-based indexing
        self.num_users = self.dataframe['user_id'].max() + 1
        self.num_items = self.dataframe['item_id'].max() + 1
    
    @staticmethod
    def load_movielens_data(path_u='data/ratings.csv'):
        df = pd.read_csv(path_u)
        # Rename columns to match expected names
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        # Convert IDs to 0-based indexing
        df['user_id'] -= 1
        df['item_id'] -= 1
        return df[['user_id', 'item_id', 'rating']]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_id = self.dataframe.iloc[idx, 0]
        item_id = self.dataframe.iloc[idx, 1]
        rating = self.dataframe.iloc[idx, 2]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(item_id, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
        )


#%%
dataset = MovieLensDataset()
train_data = DataLoader(dataset, batch_size=512, shuffle=True)
#%%
model = RecommenderModel(dataset.num_users, dataset.num_items, embedding_dim=20)
trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(model, train_data)


# %% get a sample of the data
user_id, item_id = 0, 2

# %% make a prediction
prediction = model(torch.tensor([user_id]), torch.tensor([item_id]))
prediction
# %% extract the related movie titles
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
ratings_w_movies = ratings.merge(movies, on='movieId', how='left')

# %% get movie titles from prediction
movie_titles = movies.iloc[prediction.argmax().item()]['title']
movie_titles
# %% other movies the user has rated
user_rated_movies = ratings_w_movies[ratings_w_movies['userId'] == user_id].sort_values(by='rating', ascending=False)
user_rated_movies
# %%
# %%
