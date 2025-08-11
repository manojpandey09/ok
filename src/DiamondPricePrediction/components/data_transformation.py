import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception
import pickle

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Creating data transformation pipeline")

            # Example numeric & categorical columns (change according to your dataset)
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_cols = ['cut', 'color', 'clarity']

            # Numeric pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ])

            logging.info("Data transformation pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data for transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "price"
            drop_columns = [target_column, "id"] if "id" in train_df.columns else [target_column]

            X_train = train_df.drop(columns=drop_columns, axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=drop_columns, axis=1)
            y_test = test_df[target_column]

            preprocessor = self.get_data_transformation_object()

            logging.info("Fitting and transforming training data")
            X_train_transformed = preprocessor.fit_transform(X_train)

            logging.info("Transforming test data")
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            # Returning 3 values (train_arr, test_arr, preprocessor_path)
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise customexception(e, sys)
