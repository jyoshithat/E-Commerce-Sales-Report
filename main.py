import os

print("=" * 50)
print("SALES & DEMAND FORECASTING PROJECT")
print("=" * 50)

# Step 1 - Data Cleaning
print("\nStep 1: Data Cleaning")
os.system("python src/clean.py")

# Step 2 - Data Merging
print("\nStep 2: Data Merging")
os.system("python src/merge.py")

# Step 3 - Analytics
print("\nStep 3: Analytics")
os.system("python src/analytics.py")

# Step 4 - Feature Engineering
print("\nStep 4: Feature Engineering")
os.system("python src/feature_engineering.py")

# Step 5 - Model Training
print("\nStep 5: Model Training")
os.system("python src/model.py")

# Step 6 - Prediction
print("\nStep 6: Prediction")
os.system("python src/prediction.py")

print("\n" + "=" * 50)
print("PROJECT EXECUTION COMPLETED")
print("=" * 50)
