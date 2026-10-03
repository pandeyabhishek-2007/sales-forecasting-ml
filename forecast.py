import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Data/sales_cleaned_CA1.csv")

product = "FOODS_3_090"

product_df = df[df["item_id"] == product].copy()

# Features and target
X = product_df[["year", "month", "wm_yr_wk"]]
y = product_df["sales"]

# Train model on all available data
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Last known values
last_year = product_df["year"].iloc[-1]
last_month = product_df["month"].iloc[-1]
last_week = product_df["wm_yr_wk"].iloc[-1]

# Create future feature values
future_data = []

for i in range(1, 31):

    week = last_week + i

    year = last_year
    month = last_month

    if week > 52:
        week = week - 52
        year += 1

    if week % 4 == 0:
        month += 1

    if month > 12:
        month = 1

    future_data.append([year, month, week])

future_df = pd.DataFrame(
    future_data,
    columns=["year", "month", "wm_yr_wk"]
)

# Predict future sales
predictions = model.predict(future_df)

print("30-DAY SALES FORECAST")
print("=====================")
print("Product:", product)

for i, prediction in enumerate(predictions, start=1):
    print(f"Day {i}: {prediction:.2f} units")