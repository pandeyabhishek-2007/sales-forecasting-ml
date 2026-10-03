import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/sales_cleaned_CA1.csv")

df["date"] = pd.to_datetime(df["date"])

# 1. Daily sales trend
daily_sales = df.groupby("date")["sales"].sum()

plt.figure(figsize=(12, 5))
plt.plot(daily_sales)
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.show()

# 2. Monthly sales trend
monthly_sales = df.groupby(df["date"].dt.to_period("M"))["sales"].sum()

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales.index.astype(str), monthly_sales.values)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Units Sold")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 3. Sales by category
category_sales = df.groupby("cat_id")["sales"].sum()

plt.figure(figsize=(7, 5))
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.show()