from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluator:
    def evaluate(self, y_true, y_pred):
        """Evaluates model performance."""
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # ... (add more metrics or visualizations as needed)
        return mse, r2
