from src.DiamondPricePrediction.components.data_ingestion import DataIngestion
from src.DiamondPricePrediction.components.data_transformation import DataTransformation
from src.DiamondPricePrediction.components.model_trainer import ModelTrainer
from src.DiamondPricePrediction.exception import customexception
import sys

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            print("\n[STEP 1] Starting Data Ingestion...")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            print(f"[OK] Data Ingestion Completed.\nTrain Path: {train_data_path}\nTest Path: {test_data_path}")
            return train_data_path, test_data_path
        except Exception as e:
            print("[ERROR] Data Ingestion failed!")
            raise customexception(e, sys)

    def start_data_transformation(self, train_data_path, test_data_path):
        try:
            print("\n[STEP 2] Starting Data Transformation...")
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            print("[OK] Data Transformation Completed.")
            return train_arr, test_arr
        except Exception as e:
            print("[ERROR] Data Transformation failed!")
            raise customexception(e, sys)

    def start_model_training(self, train_arr, test_arr):
        try:
            print("\n[STEP 3] Starting Model Training...")
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer(train_arr, test_arr)
            print("[OK] Model Training Completed.")
        except Exception as e:
            print("[ERROR] Model Training failed!")
            raise customexception(e, sys)

    def start_training(self):
        train_data_path, test_data_path = self.start_data_ingestion()
        train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
        self.start_model_training(train_arr, test_arr)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.start_training()
