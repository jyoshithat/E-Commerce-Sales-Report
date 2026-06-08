import pandas as pd

df = pd.read_csv("output/merged_data.csv")

print("Original Shape:", df.shape)

# Date Features
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.day_name()

# Convert categorical columns
df = pd.get_dummies(
    df,
    columns=[
        'Category_x',
        'ship-state',
        'Fulfilment'
    ],
    drop_first=True
)

print("New Shape:", df.shape)

df.to_csv(
    "output/feature_engineered.csv",
    index=False
)

print("Feature Engineering Completed!")
