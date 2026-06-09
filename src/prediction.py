import pandas as pd
import joblib

print("========================================")
print("SALES PREDICTION")
print("========================================")

# Load trained model
model = joblib.load("output/sales_model.pkl")

print("Model Loaded Successfully!")

# Load feature engineered dataset
df = pd.read_csv("output/feature_engineered.csv")

# Keep only numeric columns
numeric_df = df.select_dtypes(
    include=["int64", "float64", "bool"]
)

# Remove target column
X = numeric_df.drop("Amount", axis=1)

# Take first 5 records for prediction
sample_data = X.head(5)

# Predict
predictions = model.predict(sample_data)

# Display results
results = pd.DataFrame({
    "Predicted Sales": predictions
})

print("\nPrediction Results:")
print(results)

# Save predictions
results.to_csv(
    "output/future_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("File: output/future_predictions.csv")

print("\n========================================")
print("PREDICTION COMPLETED")
print("========================================")    