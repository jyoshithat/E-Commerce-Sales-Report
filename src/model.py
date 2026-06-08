import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("output/feature_engineered.csv")

print("Dataset Shape:", df.shape)

# Keep only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64', 'bool'])

print("Numeric Shape:", numeric_df.shape)

# Target variable
y = numeric_df["Amount"]

# Features
X = numeric_df.drop("Amount", axis=1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# Random Forest Model
model = RandomForestRegressor(
    n_estimators=50,   # reduced for faster training
    random_state=42,
    n_jobs=-1
)

print("Training Model...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== RESULTS =====")
print("MAE:", round(mae, 2))
print("R² Score:", round(r2, 4))

# Save predictions
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

results.to_csv(
    "output/predictions.csv",
    index=False
)

print("\nPredictions saved to output/predictions.csv")
print("Model Training Completed!")