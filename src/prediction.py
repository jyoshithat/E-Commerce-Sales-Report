import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

print("========================================")
print("      SALES DEMAND PREDICTION")
print("========================================")

# Load trained model
print("\nLoading trained model...")
model = joblib.load("output/sales_model.pkl")
print("✓ Model Loaded Successfully!")

# Load feature engineered dataset
print("\nLoading feature engineered dataset...")
df = pd.read_csv("output/feature_engineered.csv")
print(f"✓ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Keep only numeric columns
numeric_df = df.select_dtypes(
    include=["int64", "float64", "bool"]
)

print(f"✓ Numeric Features Found: {numeric_df.shape[1]}")

# Check target column
if "Amount" not in numeric_df.columns:
    raise ValueError("Target column 'Amount' not found!")

# Separate Features and Target
X = numeric_df.drop("Amount", axis=1)
y = numeric_df["Amount"]

print(f"✓ Features Used: {X.shape[1]}")
print(f"✓ Records for Prediction: {len(X)}")

# Generate Predictions
print("\nGenerating predictions...")
predictions = model.predict(X)

# Performance Metrics
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

# Create Results DataFrame
results = pd.DataFrame({
    "Actual_Sales": y,
    "Predicted_Sales": predictions,
    "Difference": y - predictions
})

# Save Results
results.to_csv(
    "output/future_predictions.csv",
    index=False
)

# Display Summary
print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"MAE Score : {mae:.2f}")
print(f"R² Score  : {r2:.4f}")

print("\n=========================================")
print("SAMPLE PREDICTIONS")
print("=========================================")

print(results.head(10))

print("\n========================================")
print("FILES GENERATED")
print("========================================")

print("✓ output/future_predictions.csv")

print("\n========================================")
print("PREDICTION COMPLETED SUCCESSFULLY")
print("========================================")