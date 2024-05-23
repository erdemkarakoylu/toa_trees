import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path


    df_phy = pd.read_parquet(
    preprocessed_data_path / 'df_nwa_phyto.pqt'
    )
    df_rrs = pd.read_parquet(
        preprocessed_data_path / 'df_nwa_rrs.pqt'
    )
    def load_data(self):
        """Loads and preprocesses data."""
        # Load data from your source (e.g., CSV files)
        dX = pd.read_csv(self.data_path + "/features.csv")
        dY = pd.read_csv(self.data_path + "/concentrations.csv")

        # Data preprocessing (handle missing values, type conversion, etc.)
        dX = dX.astype(float)  # Ensure features are numerical
        dY = dY.astype(float)  # Ensure targets are numerical

        return dX, dY
    
    
