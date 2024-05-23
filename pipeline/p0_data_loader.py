import pandas as pd
from logger_utils import logger


class DataLoader:
    def __init__(
            self, data_path, rrs_file='df_nwa_rrs.pqt',
            phy_file='df_nwa_phyto.pqt'):
        preprocessed_data_path = data_path / '02_extracted' / 'pcc_sims/subset'
        self.data_path = preprocessed_data_path
        self.X_name = rrs_file
        self.Y_name = phy_file
        logger.debug(f"Data directory set to {self.data_path.as_posix()}")
        logger.debug(f'Rrs file used: {self.X_name}')
        logger.debug(f'Phytoplankton file use {self.Y_name}')

    def load_data(self):
        """Loads and preprocesses data."""
        # Load data from your source (e.g., CSV files)
        try:
            dX = pd.read_parquet(self.data_path / self.X_name)
            dY = pd.read_parquet(self.data_path / self.Y_name)
            # Data preprocessing (handle missing values, type conversion, etc.)
            dX = dX.astype(float)  # Ensure features are numerical
            dY = dY.astype(float)  # Ensure targets are numerical
            return dX, dY
                     
        except FileNotFoundError:
            logger.error(f'Data files not found')
        
    
    
