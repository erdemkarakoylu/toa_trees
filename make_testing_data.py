from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

data_path = Path.cwd().parent / 'data/02_extracted/pcc_sims/subset'
x_path = data_path / 'df_nwa_rrs.pqt'
y_path = data_path / 'df_nwa_phyto.pqt'
assert x_path.exists()
assert y_path.exists()
test_path = Path.cwd() / 'tests' / 'test_data'
assert test_path.exists()
dX = pd.read_parquet(x_path)
dY = pd.read_parquet(y_path)

# Gather statistics
dX_stats = dX.describe()
dY_stats = dY.describe()

# Create a lower-dimensional latent space (e.g., representing principal components)
n_samples = 1000
n_features = dX.shape[1]
n_outputs = dY.shape[1]
n_latent_features = 5  # Adjust this based on how many PCs you expect to be important
latent_features = np.random.randn(n_samples, n_latent_features)

# Coefficients relating latent features to outputs
latent_coefficients = np.random.randn(n_latent_features, n_outputs) # n_outputs is used here

# Simulate highly correlated features from the latent space
feature_loadings = np.random.randn(n_features, n_latent_features)  # Loadings for each feature
features = latent_features @ feature_loadings.T + np.random.randn(n_samples, n_features) * 0.1  # Add noise

# Create synthetic datasets (now using latent features)
synthetic_targets = []
for i, col in enumerate(dY.columns):
    y_mean = dY_stats[col]['mean']
    y_std = dY_stats[col]['std']
    # Generate targets based on latent features and coefficients 
    y = latent_features @ latent_coefficients[:, i] + np.random.randn(n_samples) * y_std + y_mean  
    synthetic_targets.append(y)

# Combine into DataFrames
dX_synthetic = pd.DataFrame(features, columns=dX.columns)
dY_synthetic = pd.DataFrame(np.column_stack(synthetic_targets), columns=dY.columns)

# Add some noise for extra realism, if needed
noise_level = 0.05  
dX_synthetic += np.random.randn(n_samples, n_features) * noise_level

dX_synthetic.to_parquet(test_path/'dx_synthetic.pqt')
dY_synthetic.to_parquet(test_path/'dy_synthetic.pqt')