import numpy as np
import pandas as pd

class DataTransformation:
    def __init__(self):
        pass

    def initialize_data_transformation(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Dummy example: convert to numpy (in real you’ll do scaling etc.)
        train_arr = train_df.values
        test_arr = test_df.values

        return train_arr, test_arr
