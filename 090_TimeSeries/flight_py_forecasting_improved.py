#%%
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, RMSE

#%% data loading and preprocessing
flights = sns.load_dataset("flights")

# Create time index starting from 0
flights["time_idx"] = range(len(flights))

# Enhanced feature engineering
flights["month"] = flights["month"].astype("category")
flights["year"] = flights["year"].astype(str).astype("category")
flights["series"] = "flights"

# Add trend feature (time index as a real feature)
flights["trend"] = flights["time_idx"].astype(float)

# Add seasonal features
flights["month_num"] = pd.to_datetime(flights["year"].astype(str) + "-" + flights["month"].astype(str)).dt.month
flights["quarter"] = pd.to_datetime(flights["year"].astype(str) + "-" + flights["month"].astype(str)).dt.quarter.astype("category")

# Add lag features for better temporal modeling
flights["passengers_lag1"] = flights["passengers"].shift(1)
flights["passengers_lag12"] = flights["passengers"].shift(12)  # Year-over-year lag

# Add rolling statistics
flights["passengers_rolling_mean_3"] = flights["passengers"].rolling(window=3, min_periods=1).mean()
flights["passengers_rolling_mean_12"] = flights["passengers"].rolling(window=12, min_periods=1).mean()

# Fill NaN values in lag features
flights = flights.fillna(method='bfill')

# Ensure passengers column is float
flights["passengers"] = flights["passengers"].astype(float)

#%% Improved dataset setup
max_encoder_length = 24  # Increased from 6 to capture more temporal patterns
max_prediction_length = 12  # Increased from 6 for longer forecasts

# Training cutoff with more data for validation
training_cutoff = flights["time_idx"].max() - max_prediction_length

# Ensure time_idx is int
flights["time_idx"] = flights["time_idx"].astype(int)

#%%
# Create full dataset with enhanced features
full_dataset = TimeSeriesDataSet(
    flights,
    time_idx="time_idx",
    target="passengers",
    group_ids=["series"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    # Unknown features (target and derived features)
    time_varying_unknown_reals=[
        "passengers", "passengers_lag1", "passengers_lag12",
        "passengers_rolling_mean_3", "passengers_rolling_mean_12"
    ],
    # Known features (we know these in advance)
    time_varying_known_reals=["trend"],
    time_varying_known_categoricals=["month", "quarter"],
    static_categoricals=["year"],
    static_reals=[],
    target_normalizer=GroupNormalizer(groups=["series"]),
)

#%% Create training and validation datasets
train_data = flights[flights.time_idx <= training_cutoff]
val_data = flights[flights.time_idx > training_cutoff]

training = TimeSeriesDataSet(
    train_data,
    time_idx="time_idx",
    target="passengers",
    group_ids=["series"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=[
        "passengers", "passengers_lag1", "passengers_lag12",
        "passengers_rolling_mean_3", "passengers_rolling_mean_12"
    ],
    time_varying_known_reals=["trend"],
    time_varying_known_categoricals=["month", "quarter"],
    static_categoricals=["year"],
    target_normalizer=GroupNormalizer(groups=["series"]),
)

validation = TimeSeriesDataSet.from_dataset(
    dataset=full_dataset, 
    data=val_data,
    min_prediction_length=1,
    min_encoder_length=max_encoder_length//2
)

#%%
# Improved batch size and dataloader setup
batch_size = 32  # Reduced for better stability
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

#%% Improved model configuration
model = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=64,  # Increased from 8 to 64
    attention_head_size=4,  # Increased from 2 to 4
    dropout=0.2,  # Increased dropout for regularization
    hidden_continuous_size=32,  # Increased from 8 to 32
    loss=QuantileLoss(),  # Explicit loss function
    optimizer="adam",
    learning_rate=0.001,
    reduce_on_plateau_patience=4,  # Learning rate scheduling
)

#%% Improved trainer configuration
trainer = pl.Trainer(
    max_epochs=50,  # Increased from 10 to 50
    accelerator="cpu",
    enable_checkpointing=True,
    logger=False,
    callbacks=[
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        ),
        pl.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            factor=0.5,
            mode="min"
        )
    ]
)

# Fit the model
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

#%% Make predictions
predictions = model.predict(val_dataloader, return_x=False).cpu().numpy()
actual_values = np.concatenate([batch[1][0].numpy() for batch in val_dataloader], axis=0)

#%% Calculate and print evaluation metrics
actual_flat = actual_values.flatten()
predicted_flat = predictions[:, 0]  # Use median prediction

# Ensure arrays have same length
min_length = min(len(actual_flat), len(predicted_flat))
actual_flat = actual_flat[:min_length]
predicted_flat = predicted_flat[:min_length]

mae = mean_absolute_error(actual_flat, predicted_flat)
rmse = np.sqrt(mean_squared_error(actual_flat, predicted_flat))
mape = np.mean(np.abs((actual_flat - predicted_flat) / actual_flat)) * 100

print(f"Improved Model Validation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

#%% Enhanced visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Time series comparison
ax1.plot(actual_flat, label='Actual', marker='o', alpha=0.7, linewidth=2)
ax1.plot(predicted_flat, label='Predicted', marker='s', alpha=0.7, linewidth=2)
ax1.set_title('Flight Passenger Predictions vs Actual Values (Improved Model)')
ax1.set_ylabel('Number of Passengers')
ax1.set_xlabel('Time Steps')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Residuals plot
residuals = actual_flat - predicted_flat
ax2.scatter(predicted_flat, residuals, alpha=0.6)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_title('Residuals Plot')
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('Residuals')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% Print detailed results
print(f"\nDetailed Prediction Results (Improved Model):")
print(f"Number of validation samples: {len(actual_flat)}")
print(f"Actual values range: {actual_flat.min():.0f} - {actual_flat.max():.0f}")
print(f"Predicted values range: {predicted_flat.min():.0f} - {predicted_flat.max():.0f}")
print(f"Average actual value: {actual_flat.mean():.2f}")
print(f"Average predicted value: {predicted_flat.mean():.2f}")
print(f"Standard deviation of actual: {actual_flat.std():.2f}")
print(f"Standard deviation of predicted: {predicted_flat.std():.2f}")

#%% Feature importance analysis
feature_importance = model.interpret_output(predictions)
print(f"\nFeature importance analysis available in feature_importance variable")

#%%
