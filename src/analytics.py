import pandas as pd

df = pd.read_csv("output/merged_data.csv")

print("=" * 50)
print("ANALYTICS")
print("=" * 50)

# Total Sales
print("\nTotal Sales Amount:")
print(df['Amount'].sum())

# Average Sales
print("\nAverage Sales:")
print(df['Amount'].mean())

# Top Categories
print("\nTop Categories:")
print(df['Category_x'].value_counts().head(10))

# Top States
print("\nTop States:")
print(df['ship-state'].value_counts().head(10))

# Order Status
print("\nOrder Status:")
print(df['Status'].value_counts())

# Date Conversion
df['Date'] = pd.to_datetime(df['Date'])

# Monthly Sales
df['Month'] = df['Date'].dt.month

monthly_sales = df.groupby('Month')['Amount'].sum()

print("\nMonthly Sales:")
print(monthly_sales)

# Top Products
print("\nTop Products:")
print(df['SKU'].value_counts().head(10))

print("\nAnalytics Completed Successfully!")