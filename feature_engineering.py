import pandas as pd

# Load cleaned dataset
df = pd.read_csv("Data/sales_cleaned_CA1.csv")

# Select one product
product = "FOODS_3_090"

product_df = df[df["item_id"] == product].copy()

# Convert date to datetime
product_df["date"] = pd.to_datetime(product_df["date"])

# Sort by date
product_df = product_df.sort_values("date")

# Create lag features
product_df["sales_lag_1"] = product_df["sales"].shift(1)
product_df["sales_lag_7"] = product_df["sales"].shift(7)
product_df["sales_lag_14"] = product_df["sales"].shift(14)
product_df["sales_lag_28"] = product_df["sales"].shift(28)

# Create rolling averages
# shift(1) prevents using today's sales to predict today's sales
product_df["sales_rolling_7"] = (
    product_df["sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

product_df["sales_rolling_30"] = (
    product_df["sales"]
    .shift(1)
    .rolling(30)
    .mean()
)

# Remove rows containing NaN values
product_df = product_df.dropna()

# Display result
print("FEATURE ENGINEERING")
print("===================")

print("\nProduct:", product)

print("\nNew Features:")
print([
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_7",
    "sales_rolling_30"
])

print("\nDataset Shape:")
print(product_df.shape)

print("\nSample Data:")
print(
    product_df[
        [
            "date",
            "sales",
            "sales_lag_1",
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_28",
            "sales_rolling_7",
            "sales_rolling_30"
        ]
    ].head()
)

# Save engineered dataset
product_df.to_csv(
    "Data/sales_features.csv",
    index=False
)

print("\nFeature-engineered dataset saved successfully.")