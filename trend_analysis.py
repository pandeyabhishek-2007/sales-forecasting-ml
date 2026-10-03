import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("Data/sales_cleaned_CA1.csv")

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# Select product
product_id = "FOODS_3_090"

product_df = df[df["item_id"] == product_id].copy()

# Monthly total sales
monthly_trend = (
    product_df
    .set_index("date")["sales"]
    .resample("ME")
    .sum()
)

print("TREND ANALYSIS")
print("==============")
print("Product:", product_id)

print("\nMONTHLY SALES:")
print(monthly_trend)