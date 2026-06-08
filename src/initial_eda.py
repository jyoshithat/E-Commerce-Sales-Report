import pandas as pd

files = [
    "data/Amazon Sale Report.csv",
    "data/International sale Report.csv",
    "data/Sale Report.csv"
]

for file in files:
    print("\n" + "="*50)
    print(file)
    print("="*50)

    df = pd.read_csv(file, low_memory=False)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())