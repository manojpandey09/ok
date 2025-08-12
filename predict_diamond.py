import pickle
import pandas as pd

# ==== Load Preprocessor ====
with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# ==== Load Model ====
with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

# ==== Example Input ====
# Ye input tum CSV se ya user input se bhi le sakte ho
data = {
    "carat": [1.0],
    "cut": ["Ideal"],
    "color": ["E"],
    "clarity": ["SI2"],
    "depth": [61.5],
    "table": [55.0],
    "x": [3.95],
    "y": [3.98],
    "z": [2.43]
}

df = pd.DataFrame(data)

# ==== Transform data ====
data_transformed = preprocessor.transform(df)

# ==== Predict ====
prediction = model.predict(data_transformed)

print(f"Predicted Diamond Price: {prediction[0]}")
