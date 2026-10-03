import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


# Load feature-engineered dataset
df = pd.read_csv("Data/sales_features.csv")

df["date"] = pd.to_datetime(df["date"])

# Select product
product = "FOODS_3_090"

df = df[df["item_id"] == product].copy()
df = df.sort_values("date")


# Features
features = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_7",
    "sales_rolling_30"
]


# Remove missing feature rows
df = df.dropna(subset=features)


# Train model on all available historical data
X = df[features]
y = df["sales"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)


# Get recent sales history
sales_history = df["sales"].tolist()

future_predictions = []

# Forecast next 30 days
for day in range(30):

    lag_1 = sales_history[-1]
    lag_7 = sales_history[-7]
    lag_14 = sales_history[-14]
    lag_28 = sales_history[-28]

    rolling_7 = np.mean(sales_history[-7:])
    rolling_30 = np.mean(sales_history[-30:])

    future_features = pd.DataFrame([[
        lag_1,
        lag_7,
        lag_14,
        lag_28,
        rolling_7,
        rolling_30
    ]], columns=features)

    prediction = model.predict(future_features)[0]

    future_predictions.append(prediction)

    # Add prediction to history
    sales_history.append(prediction)


# Print forecast
print("30-DAY SALES FORECAST")
print("=====================")
print()
print("Product:", product)
print()

for i, prediction in enumerate(future_predictions, start=1):
    print(f"Day {i}: {prediction:.2f} units")


# Plot historical + forecast
last_90_days = df.tail(90)

future_dates = pd.date_range(
    start=df["date"].max() + pd.Timedelta(days=1),
    periods=30
)

plt.figure(figsize=(14, 6))

plt.plot(
    last_90_days["date"],
    last_90_days["sales"],
    label="Historical Sales"
)

plt.plot(
    future_dates,
    future_predictions,
    label="30-Day Forecast"
)

plt.title(f"30-Day Sales Forecast - {product}")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()

plt.tight_layout()
plt.show()
