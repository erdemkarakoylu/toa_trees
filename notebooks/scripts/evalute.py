from loguru import logger
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
import numpy as np

def model_eval (dY_true, dY_pred):
    scores_dict = dict()
    for col in dY_true.columns:  # Iterate over each output column
        mse = mean_squared_error(
            dY_true[col], dY_pred[col])  
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(
            dY_true[col], dY_pred[col])
        r2 = r2_score(
            dY_true[col], dY_pred[col])
        mape = mean_absolute_percentage_error(
           dY_true[col], dY_pred[col] 
        )
        mae_to_dev_ratio = mae / dY_true[col].std()
        logger.info(f"\nMetrics for {col}:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R-squared: {r2:.4f}")
        logger.info(f"  MAE/StDev_true {mae_to_dev_ratio:.3f}")
        scores_dict[col] = dict(mse=mse, rmse=rmse, mae=mae, r2=r2, mape=mape, mae_2_true_std_ratio=mae_to_dev_ratio)
    return scores_dict