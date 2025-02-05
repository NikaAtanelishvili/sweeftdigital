import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


# ------------------------------
# Define the Denoising Autoencoder
# ------------------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed


# ------------------------------
# Utility: Add Noise (Simulate Missingness)
# ------------------------------
def add_noise(x, noise_factor=0.2):
    """
    Randomly zeros out elements in the tensor to simulate missing values.
    """
    noisy_x = x.clone()
    # Create a random mask: True means "to be corrupted"
    mask = torch.rand_like(x) < noise_factor
    noisy_x[mask] = 0.0  # Applying the Mask (We could use mean or modian)
    return noisy_x


# ------------------------------
# Training Routine for the Autoencoder
# ------------------------------
def train_autoencoder(model, train_data, epochs=1000, noise_factor=0.2, lr=0.001):
    """
    Train the autoencoder using complete rows from your DataFrame.
    :param model: instance of DenoisingAutoencoder. # კლასად შეცვალე!!
    :param train_data: Tensor of shape (num_samples, num_features).
    :param epochs: Number of training epochs.
    :param noise_factor: Probability of zeroing out each element.
    :param lr: Learning rate.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # MSE measures the difference between the reconstructed output and the original (clean) input.

    for epoch in range(epochs):
        model.train()
        # Simulate missing data by adding noise
        noisy_data = add_noise(train_data, noise_factor)
        optimizer.zero_grad()
        outputs = model(noisy_data)
        loss = criterion(outputs, train_data)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


# ------------------------------
# Imputation Routine
# ------------------------------
def impute_missing(model, row, missing_mask, iterations=10):
    """
    Iteratively impute missing values in a row using the trained autoencoder.
    :param model: Trained autoencoder.
    :param row: 1D tensor of shape (input_dim,) representing a single row,
                where missing entries are pre-filled (e.g., with column means).
    :param missing_mask: Boolean tensor of same shape as row, with True for positions that were missing.
    :param iterations: Number of iterations for iterative refinement.
    :return: 1D tensor with imputed values in the positions specified by missing_mask.
    """
    row_imputed = row.clone()
    for _ in range(iterations):
        model.eval()
        with torch.no_grad():
            output = model(row_imputed.unsqueeze(0)).squeeze(0)
        # Update only the originally missing entries
        row_imputed[missing_mask] = output[missing_mask]
    return row_imputed


