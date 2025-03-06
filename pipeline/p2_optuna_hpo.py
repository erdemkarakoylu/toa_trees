import mlflow
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost

def objective(trial, X, y):
    params= dict(
        objective='reg:squarederror',
        learning_rate=trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        n_estimators=trial.suggest_int('n_estiamtors', 50, 500),
        subsample=trial.suggest_uniform('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        gamma=trial.suggest_loguniform('gamma', 1e-8, 1.0)
    )