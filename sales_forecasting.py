import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("Data/sales_cleaned_CA1.csv")

# Select product
product_id = "FOODS_3_090"

product_data = df[df["item_id"] == product_id].copy()

# Convert date
product_data["date"] = pd.to_datetime(product_data["date"])

# Aggregate daily sales
daily_sales = (
    product_data
    .groupby("date")["sales"]
    .sum()
    .reset_index()
)

# Create time index
daily_sales["day_number"] = np.arange(len(daily_sales))

# Features and target
X = daily_sales[["day_number"]]
y = daily_sales["sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
daily_sales["predicted_sales"] = model.predict(X)

# Print results
print("PRODUCT:", product_id)
print("SLOPE:", model.coef_[0])
print("INTERCEPT:", model.intercept_)

# Predict next 30 days
future_days = np.arange(
    len(daily_sales),
    len(daily_sales) + 30
).reshape(-1, 1)

future_predictions = model.predict(future_days)

print("\nNEXT 30 DAYS PREDICTED SALES:")
print(future_predictions)

# Plot actual vs predicted
plt.figure(figsize=(12, 5))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"],
    label="Actual Sales"
)

plt.plot(
    daily_sales["date"],
    daily_sales["predicted_sales"],
    label="Predicted Trend"
)

plt.title(f"Sales Prediction - {product_id}")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()

plt.tight_layout()
plt.show()