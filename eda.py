import pandas as pd

df = pd.read_csv("Data/sales_cleaned_CA1.csv")

print("DATASET SHAPE:")
print(df.shape)

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATE RANGE:")
print(df["date"].min(), "to", df["date"].max())

print("\nTOTAL SALES:")
print(df["sales"].sum())

print("\nTOP 10 PRODUCTS BY SALES:")
print(
    df.groupby("item_id")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nSALES BY CATEGORY:")
print(
    df.groupby("cat_id")["sales"]
    .sum()
    .sort_values(ascending=False)
)