import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("========================================")
print("MODEL TRAINING")
print("========================================")

# Load dataset
df = pd.read_csv("output/feature_engineered.csv")

print("Dataset Shape:", df.shape)

# Keep only numeric columns
numeric_df = df.select_dtypes(
    include=["int64", "float64", "bool"]
)

print("Numeric Shape:", numeric_df.shape)

# Target Variable
y = numeric_df["Amount"]

# Features
X = numeric_df.drop("Amount", axis=1)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# Random Forest Model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Model...")
model.fit(X_train, y_train)

# Save Model
joblib.dump(
    model,
    "output/sales_model.pkl"
)

print("Model saved successfully!")

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== RESULTS ==========")
print("MAE:", round(mae, 2))
print("R² Score:", round(r2, 4))

# Save Predictions
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

results.to_csv(
    "output/predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("File: output/predictions.csv")

print("\n========================================")
print("MODEL TRAINING COMPLETED")
print("========================================")
