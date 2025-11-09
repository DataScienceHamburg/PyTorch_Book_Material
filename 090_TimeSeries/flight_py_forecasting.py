#%% packages
import numpy as np
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import seaborn as sns
import matplotlib.pyplot as plt

#%% data loading
flights = sns.load_dataset("flights")

# Print data info for debugging
print(f"Form of the original dataset: {flights.shape}")
print(f"Date range: {flights['year'].min()} - {flights['year'].max()}")
print(f"Number of months: {len(flights)}")

# Create time index starting from 0
# This is a crucial step for pytoch-forecasting
flights["time_idx"] = np.arange(len(flights), dtype=int)

# Add additional time features
# We treat the original 'month' column as a known categorical feature
flights["month"] = flights["month"].astype("category")
# We treat year as a category and ensure all years are known
flights["year"] = flights["year"].astype(str).astype("category")
flights["series"] = "flights" # Single group identifier

# Ensure passengers column is float for proper tensor operations
flights["passengers"] = flights["passengers"].astype(float)

#%% dataset setup
MAX_ENCODER_LENGTH = 12
MAX_PREDICTION_LENGTH = 12

# We define the training cutoff to create a validation split
# Use the last 24 months for validation to ensure we have enough data for sequences
TRAINING_CUTOFF = len(flights) - 12

print(f"Training split point: {TRAINING_CUTOFF}")
print(f"Training data points: {TRAINING_CUTOFF}")
print(f"Validation data points: {len(flights) - TRAINING_CUTOFF}")


#%% Split data properly
train_data = flights.iloc[:TRAINING_CUTOFF].copy()
val_data = flights.iloc[TRAINING_CUTOFF:].copy()

print(f"Form of the training data: {train_data.shape}")
print(f"Form of the validation data: {val_data.shape}")
print(f"Time index range of the training data: {train_data['time_idx'].min()} - {train_data['time_idx'].max()}")
print(f"Time index range of the validation data: {val_data['time_idx'].min()} - {val_data['time_idx'].max()}")

#%% visualise train and val data
plt.figure(figsize=(10, 6))
sns.lineplot(x="time_idx", y="passengers", data=train_data, label='train')
sns.lineplot(x="time_idx", y="passengers", data=val_data, label='val')
plt.title('Training and Validation Data')
plt.xlabel('Time Index')
plt.ylabel('Passengers')
plt.show()

#%%
# First, create a TimeSeriesDataSet from the entire dataset to ensure all categories
# are known to the model from the beginning.
full_dataset = TimeSeriesDataSet(
    flights,
    time_idx="time_idx",
    target="passengers",
    group_ids=["series"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    time_varying_unknown_reals=["passengers"],
    time_varying_known_categoricals=["month"],
    static_categoricals=["year"],
)

#%%
# Create training dataset using from_dataset to inherit all categorical encodings
training = TimeSeriesDataSet.from_dataset(
    dataset=full_dataset,
    data=train_data,
    min_prediction_length=1,
    min_encoder_length=MAX_ENCODER_LENGTH//2
)
#%% Create validation dataset using from_dataset to ensure proper categorical encoding
validation = TimeSeriesDataSet.from_dataset(
    dataset=full_dataset, 
    data=val_data,
    min_prediction_length=1,
    min_encoder_length=MAX_ENCODER_LENGTH//2
)
#%% Hyperparameter
BATCH_SIZE = 4
LEARNING_RATE = 0.01
HIDDEN_SIZE = 4
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 4



#%%
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE)

#%% model and trainer
# The model is automatically configured from the dataset
model = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTENTION_HEAD_SIZE,
    dropout=DROPOUT,
    hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
    optimizer="adam",
    learning_rate=LEARNING_RATE,
)

#%%
trainer = pl.Trainer(
    max_epochs=10, 
    accelerator="auto",
    enable_checkpointing=False,
    logger=False,
)

# Fit the model
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

#%% make and plot predictions
predictions = model.predict(val_dataloader, return_x=False).cpu().numpy()

# Get actual values more safely
actual_values = []
for batch in val_dataloader:
    batch_actual = batch[1][0].numpy().flatten()
    # Filter out padding values (usually 0 or NaN)
    batch_actual = batch_actual[batch_actual != 0]
    if len(batch_actual) > 0:
        actual_values.extend(batch_actual)

actual_values = np.array(actual_values)

#%% Calculate and print evaluation metrics
# Flatten arrays for metric calculation
actual_flat = actual_values.flatten()
predicted_flat = predictions[:, 0]  # Use median prediction, removed extra dimension

# Ensure arrays have same length by truncating to shorter length
min_length = min(len(actual_flat), len(predicted_flat))
actual_flat = actual_flat[:min_length]
predicted_flat = predicted_flat[:min_length]

print(f"Form of the actual values: {actual_flat.shape}")
print(f"Form of the predicted values: {predicted_flat.shape}")

# Time series comparison
plt.figure(figsize=(10, 6))
x_act = range(len(actual_flat))
y_act = actual_flat
y_pred = predicted_flat
sns.lineplot(x=x_act, y=y_act, label='actual', color='black')
sns.lineplot(x=x_act, y=y_pred.squeeze(), label='predicted', color='red')
plt.ylabel('Passenger Numbers [-]')
plt.xlabel('Month [-]')
plt.title('Prediction vs. Actual Value')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#%%