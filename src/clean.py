import pandas as pd
import os

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

print("=" * 50)
print("CLEANING AMAZON SALE REPORT")
print("=" * 50)

# AMAZON DATASET
amazon = pd.read_csv(
    "data/Amazon Sale Report.csv",
    low_memory=False
)

print("Original Shape:", amazon.shape)

# Remove unnecessary columns
amazon.drop(
    columns=[
        'Unnamed: 22',
        'fulfilled-by',
        'promotion-ids'
    ],
    inplace=True,
    errors='ignore'
)

# Fill Amount with median
amazon['Amount'] = amazon['Amount'].fillna(
    amazon['Amount'].median()
)

# Fill currency with mode
amazon['currency'] = amazon['currency'].fillna(
    amazon['currency'].mode()[0]
)

# Remove rows with missing address details
amazon.dropna(
    subset=[
        'ship-city',
        'ship-state',
        'ship-postal-code',
        'ship-country'
    ],
    inplace=True
)

# Convert Date column
amazon['Date'] = pd.to_datetime(
    amazon['Date'],
    errors='coerce'
)

# Remove invalid dates
amazon.dropna(subset=['Date'], inplace=True)

# Standardize state names
amazon['ship-state'] = (
    amazon['ship-state']
    .astype(str)
    .str.upper()
    .str.strip()
)

print("Cleaned Shape:", amazon.shape)

amazon.to_csv(
    "output/amazon_cleaned.csv",
    index=False
)

print("Amazon Dataset Cleaned Successfully!\n")


print("=" * 50)
print("CLEANING INTERNATIONAL SALE REPORT")
print("=" * 50)

# INTERNATIONAL DATASET
international = pd.read_csv(
    "data/International sale Report.csv",
    low_memory=False
)

print("Original Shape:", international.shape)

# Remove index column
international.drop(
    columns=['index'],
    inplace=True,
    errors='ignore'
)

# Remove rows with missing values
international.dropna(inplace=True)

# Convert DATE column
international['DATE'] = pd.to_datetime(
    international['DATE'],
    errors='coerce'
)

# Remove invalid dates
international.dropna(subset=['DATE'], inplace=True)

print("Cleaned Shape:", international.shape)

international.to_csv(
    "output/international_cleaned.csv",
    index=False
)

print("International Dataset Cleaned Successfully!\n")


print("=" * 50)
print("CLEANING SALE REPORT")
print("=" * 50)

# SALE REPORT DATASET
sale = pd.read_csv(
    "data/Sale Report.csv",
    low_memory=False
)

print("Original Shape:", sale.shape)

# Remove index column
sale.drop(
    columns=['index'],
    inplace=True,
    errors='ignore'
)

# Remove rows with missing values
sale.dropna(inplace=True)

print("Cleaned Shape:", sale.shape)

sale.to_csv(
    "output/sale_cleaned.csv",
    index=False
)

print("Sale Report Dataset Cleaned Successfully!\n")


print("=" * 50)
print("ALL DATASETS CLEANED SUCCESSFULLY")
print("=" * 50)

print("\nSaved Files:")
print("1. output/amazon_cleaned.csv")
print("2. output/international_cleaned.csv")
print("3. output/sale_cleaned.csv")