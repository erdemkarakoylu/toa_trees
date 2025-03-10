import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def load_and_preprocess_data(data_file, output_vars):
    """
    Loads Rrs data from a CSV file and preprocesses it.

    Parameters
    ----------
    data_file : str
        Path to the CSV file containing Rrs data.
    output_vars : list of str
        Names of the output variables (columns) in the data.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
        Training and testing sets for features (X) and target variables (y).
    """
    data = pd.read_csv(data_file) 

    X = data.drop(output_vars, axis=1)
    y = data[output_vars]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def create_pipelines(y):
    """
    Creates pipelines for each output variable.

    Parameters
    ----------
    y : pandas DataFrame
        DataFrame containing the target variables.

    Returns
    -------
    pipelines : dict
        A dictionary where keys are output variable names and values are corresponding pipelines.
    """
    pipelines = {}
    for output_var in y.columns:
        pca = PCA(n_components=0.95) 
        estimator = XGBRegressor()
        selector = RFE(estimator, n_features_to_select=50, step=10)
        first_level_model = XGBRegressor()
        second_level_model = XGBRegressor()

        pipe = Pipeline(steps=[
            ('pca', pca),
            ('selector', selector),
            ('first_level_model', first_level_model),
            ('second_level_model', second_level_model)
        ])
        pipelines[output_var] = pipe
    return pipelines

def train_pipelines(pipelines, X_train, y_train):
    """
    Trains the pipelines for each output variable.

    Parameters
    ----------
    pipelines : dict
        A dictionary where keys are output variable names and values are corresponding pipelines.
    X_train : numpy array
        Training set for features.
    y_train : numpy array
        Training set for target variables.
    """
    for output_var in y_train.columns:
        pipe = pipelines[output_var]
        y_train_mean = y_train.mean(axis=1)
        pipe.named_steps['first_level_model'].fit(X_train, y_train_mean)
        y_train_residual = y_train[output_var] - pipe.named_steps['first_level_model'].predict(X_train)
        pipe.fit(X_train, y_train_residual)

def save_pipelines(pipelines):
    """
    Saves the trained pipelines to disk.

    Parameters
    ----------
    pipelines : dict
        A dictionary where keys are output variable names and values are corresponding pipelines.
    """
    for output_var in pipelines.keys():
        joblib.dump(pipelines[output_var], f'pipeline_{output_var}.joblib')

def load_and_predict(output_var, X_test):
    """
    Loads a saved pipeline and makes predictions on new data.

    Parameters
    ----------
    output_var : str
        Name of the output variable for which to load the pipeline.
    X_test : numpy array
        Testing set for features.

    Returns
    -------
    y_pred : numpy array
        Predicted values for the specified output variable.
    """
    loaded_pipeline = joblib.load(f'pipeline_{output_var}.joblib')
    y_pred = loaded_pipeline.predict(X_test)
    return y_pred

def evaluate(y_test, y_pred, output_var):
    """
    Evaluates the model's performance using Mean Squared Error.

    Parameters
    ----------
    y_test : numpy array
        Testing set for target variables
    y_pred : numpy array
        Predicted values.
    output_var : str
        Name of the output variable being evaluated

    Prints
    -------
    MSE for the specified output variable
    """
    mse = mean_squared_error(y_test[output_var], y_pred) 
    print(f"MSE for {output_var}: {mse}")

# Main execution
if __name__ == "__main__":
    data_file = "my_rrs_data.csv" 
    output_vars = ['output_var1', 'output_var2', ...] 

    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_file, output_vars)
    pipelines = create_pipelines(y_train)
    train_pipelines(pipelines, X_train, y_train)
    save_pipelines(pipelines)

    # Example prediction and evaluation for one output variable
    output_var = 'output_var1'
    y_pred = load_and_predict(output_var, X_test)
    evaluate(y_test, y_pred, output_var)