import pandas as pd
import numpy as np
import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception
from dataclasses import dataclass
from src.DiamondPricePrediction.utils.utils import save_object, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }
            
            # Evaluate all models
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(f'Model Report : {model_report}')
            print("\nModel Report:", model_report)

            # Get best model score
            best_model_score = max(model_report.values())

            # Get best model name
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            # Train best model
            best_model.fit(X_train, y_train)

            print(f"\nBest Model Found: {best_model_name} with R2 Score: {best_model_score}")
            logging.info(f'Best Model Found: {best_model_name} with R2 Score: {best_model_score}')

            # Save trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise customexception(e, sys)
