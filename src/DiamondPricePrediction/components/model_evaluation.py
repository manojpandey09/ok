import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from src.DiamondPricePrediction.utils.utils import load_object

class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))  # RMSE
        mae = mean_absolute_error(actual, pred)           # MAE
        r2 = r2_score(actual, pred)                       # R2 score
        return rmse, mae, r2

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            # Split test data into features and target
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            # Load trained model
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            # MLflow tracking URI (change to your own DagsHub repo link if needed)
            mlflow.set_registry_uri("https://dagshub.com/sunny.savita/fsdsmendtoend.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(f"MLflow Tracking Store Type: {tracking_url_type_store}")

            with mlflow.start_run():
                predicted_qualities = model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)

                # Log metrics to MLflow
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log model to MLflow
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise e
