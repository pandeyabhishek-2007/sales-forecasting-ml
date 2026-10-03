import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned data
df = pd.read_csv("Data/sales_cleaned_CA1.csv")

# Select product
product_id = "FOODS_3_090"

product_data = df[df["item_id"] == product_id].copy()

# Convert date
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

# Rolling averages
daily_sales["rolling_7"] = daily_sales["sales"].shift(1).rolling(7).mean()
daily_sales["rolling_28"] = daily_sales["sales"].shift(1).rolling(28).mean()

# Remove rows with missing feature values
daily_sales = daily_sales.dropna()

# Features
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

# Time-based train/test split
split = int(len(daily_sales) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("RANDOM FOREST MODEL")
print("-------------------")
print("Product:", product_id)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nMODEL PERFORMANCE")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 SCORE:", r2)