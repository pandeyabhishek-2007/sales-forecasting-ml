import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load feature-engineered data
df = pd.read_csv("Data/sales_features.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Select product
product = "FOODS_3_090"
df = df[df["item_id"] == product].copy()

# Features created during feature engineering
features = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_7",
    "sales_rolling_30"
]

# Remove rows where lag/rolling values are unavailable
df = df.dropna(subset=features)

# Sort chronologically
df = df.sort_values("date")

# Input and target
X = df[features]
y = df["sales"]

# Time-based train/test split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("FEATURE-ENGINEERED RANDOM FOREST")
print("===============================")
print()
print("Product:", product)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print()
print("MODEL PERFORMANCE")
print("-----------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 SCORE:", r2)