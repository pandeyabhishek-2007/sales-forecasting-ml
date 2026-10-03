import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("Data/sales_cleaned_CA1.csv")

# Select product
product_id = "FOODS_3_090"

product_data = df[df["item_id"] == product_id].copy()

# Convert date
product_data["date"] = pd.to_datetime(product_data["date"])

# Daily sales
daily_sales = product_data.groupby("date")["sales"].sum()

print("PRODUCT:", product_id)
print("TOTAL SALES:", daily_sales.sum())
print("AVERAGE DAILY SALES:", daily_sales.mean())
print("MAX DAILY SALES:", daily_sales.max())
print("MIN DAILY SALES:", daily_sales.min())

# Plot
plt.figure(figsize=(12, 5))
plt.plot(daily_sales.index, daily_sales.values)

plt.title(f"Daily Sales of {product_id}")
plt.xlabel("Date")
plt.ylabel("Units Sold")

plt.tight_layout()
plt.show()