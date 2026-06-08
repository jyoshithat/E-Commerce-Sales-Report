import pandas as pd

# Load cleaned datasets
amazon = pd.read_csv("output/amazon_cleaned.csv")
international = pd.read_csv("output/international_cleaned.csv")
sale = pd.read_csv("output/sale_cleaned.csv")

print("Amazon Shape:", amazon.shape)
print("International Shape:", international.shape)
print("Sale Shape:", sale.shape)

# Convert SKU columns to string
amazon['SKU'] = amazon['SKU'].astype(str)
international['SKU'] = international['SKU'].astype(str)
sale['SKU Code'] = sale['SKU Code'].astype(str)

# Merge Amazon + Sale Report
merged1 = pd.merge(
    amazon,
    sale,
    left_on='SKU',
    right_on='SKU Code',
    how='left'
)

print("\nAmazon + Sale Report Shape:")
print(merged1.shape)

# Merge with International Report
merged2 = pd.merge(
    merged1,
    international,
    on='SKU',
    how='left'
)

print("\nFinal Merged Shape:")
print(merged2.shape)

# Save merged dataset
merged2.to_csv(
    "output/merged_data.csv",
    index=False
)

print("\nMerged dataset saved successfully!")