import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned sales data
df = pd.read_csv("Data/sales_cleaned_CA1.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

print("SEASONALITY ANALYSIS")
print("====================")
print("Dataset shape:", df.shape)
print(df.head())
# Select our product
product_id = "FOODS_3_090"

product_df = df[df["item_id"] == product_id].copy()

print("\nSELECTED PRODUCT")
print("================")
print("Product:", product_id)
print("Rows:", len(product_df))
print(product_df[["date", "item_id", "sales", "weekday", "month", "year"]].head())
# Average sales by day of week
weekday_sales = (
    product_df.groupby("weekday")["sales"]
    .mean()
    .reindex([
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ])
)

print("\nAVERAGE SALES BY DAY OF WEEK")
print("============================")
print(weekday_sales)
# Plot weekly seasonality
plt.figure(figsize=(10, 5))

plt.bar(weekday_sales.index, weekday_sales.values)

plt.title(f"Average Sales by Day of Week - {product_id}")
plt.xlabel("Day of Week")
plt.ylabel("Average Units Sold")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# Average sales by month
monthly_sales = (
    product_df.groupby("month")["sales"]
    .mean()
)

print("\nAVERAGE SALES BY MONTH")
print("======================")
print(monthly_sales)
# Plot monthly seasonality
plt.figure(figsize=(10, 5))

plt.bar(monthly_sales.index, monthly_sales.values)

plt.title(f"Average Monthly Sales - {product_id}")
plt.xlabel("Month")
plt.ylabel("Average Units Sold")
plt.xticks(range(1, 13))

plt.tight_layout()
plt.show()