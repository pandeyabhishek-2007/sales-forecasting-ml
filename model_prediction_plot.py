import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("Data/sales_cleaned_CA1.csv")

product_id = "FOODS_3_090"

product_data = df[df["item_id"] == product_id].copy()
product_data["date"] = pd.to_datetime(product_data["date"])

# Daily sales
daily_sales = (
    product_data
    .groupby("date")["sales"]
    .sum()
    .reset_index()
    .sort_values("date")
)

# Date features
daily_sales["day_of_week"] = daily_sales["date"].dt.dayofweek
daily_sales["month"] = daily_sales["date"].dt.month
daily_sales["year"] = daily_sales["date"].dt.year

# Lag features
daily_sales["lag_1"] = daily_sales["sales"].shift(1)
daily_sales["lag_7"] = daily_sales["sales"].shift(7)
daily_sales["lag_14"] = daily_sales["sales"].shift(14)
daily_sales["lag_28"] = daily_sales["sales"].shift(28)

daily_sales["rolling_7"] = (
    daily_sales["sales"].shift(1).rolling(7).mean()
)

daily_sales["rolling_28"] = (
    daily_sales["sales"].shift(1).rolling(28).mean()
)

daily_sales = daily_sales.dropna()

features = [
    "day_of_week",
    "month",
    "year",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_28"
]

X = daily_sales[features]
y = daily_sales["sales"]

# Time-based split
split = int(len(daily_sales) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

dates_test = daily_sales["date"].iloc[split:]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Plot
plt.figure(figsize=(12, 5))

plt.plot(
    dates_test,
    y_test,
    label="Actual Sales"
)

plt.plot(
    dates_test,
    predictions,
    label="Predicted Sales"
)

plt.title(f"Actual vs Predicted Sales - {product_id}")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()

plt.tight_layout()
plt.show()