import pandas as pd
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(">>>>> Data ingestion started <<<<<")
        try:
            # Read source CSV
            source_path = Path("notebooks/data/gemstone.csv")
            logging.info(f"Reading dataset from: {source_path}")
            print(f"Reading dataset from: {source_path}")

            data = pd.read_csv(source_path)
            logging.info(f"Dataset shape: {data.shape}")
            print(f"Dataset loaded. Shape: {data.shape}")

            # Create artifacts folder if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")
            print(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Train-test split
            logging.info("Performing train-test split (25% test size)...")
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("Train-test split completed.")

            # Save train & test data
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            print(f"Train data saved at: {self.ingestion_config.train_data_path}")
            print(f"Test data saved at: {self.ingestion_config.test_data_path}")

            logging.info(">>>>> Data ingestion completed successfully <<<<<")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occurred during data ingestion stage.")
            raise customexception(e, sys)
