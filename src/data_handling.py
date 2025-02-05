import os.path
import sys

import pandas as pd
import torch
from numpy import dtype
from torch import nn, optim

from src.autoencoder import DenoisingAutoencoder, train_autoencoder, impute_missing
from src.model import IncomePredictor

# Loading data
df = pd.read_csv('./data/customer_analysis.csv', sep='\t')

# Missing values in data
missing_values = df.isnull().sum()
missing_percentage = missing_values / len(df) * 100

# ------------------------------
# Preparing the dataframe for NN
# ------------------------------
df = pd.get_dummies(df, columns=["Marital_Status"], dtype=float)
df = pd.get_dummies(df, columns=["Education"], dtype=float)

# Convert date column to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# Extract features
df['Dt_Customer_Year'] = df['Dt_Customer'].dt.year
df['Dt_Customer_month'] = df['Dt_Customer'].dt.month
df['Dt_Customer_Day'] = df['Dt_Customer'].dt.day

df = df.drop(['Dt_Customer'], axis=1)

# ------------------------------
# Imputing missing values in data using denoising autoencoder
# ------------------------------

# Select only complete rows for training (no missing values)
complete_df = df.dropna()

# Convert to a PyTorch tensor of type float32
train_data = torch.tensor(complete_df.values, dtype=torch.float32)

# Set the model parameters
input_dim = train_data.shape[1]
hidden_dim = 30  # Adjust this based on your needs

# Initialize the autoencoder
model = DenoisingAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)

import os

model_path = "denoising_autoencoder.pth"
if not os.path.exists(model_path):
    print("Training the autoencoder...")
    train_autoencoder(model, train_data, epochs=1000, noise_factor=0.2, lr=0.001)
    torch.save(model.state_dict(), model_path)
else:
    print("Loading the pretrained autoencoder...")
    model.load_state_dict(torch.load(model_path))

# Indexes of rows with missing values
missing_indexes = df[df.isna().any(axis=1)].index

for i in missing_indexes:
    row_with_missing = df.iloc[0]  # Replace with the appropriate row index

    # Create a boolean mask: True for missing values
    missing_mask = row_with_missing.isna().values

    # Fill missing values with the column mean
    filled_values = row_with_missing.fillna(df.mean())
    row_tensor = torch.tensor(filled_values.values, dtype=torch.float32)

    # Convert missing_mask to a tensor of booleans
    missing_mask_tensor = torch.tensor(missing_mask)

    # Impute the missing values
    imputed_row = impute_missing(model, row_tensor, missing_mask_tensor, iterations=10).numpy()

    df.loc[i] = imputed_row
    print("Imputed row:", imputed_row)



