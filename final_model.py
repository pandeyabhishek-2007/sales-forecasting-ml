import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# FINAL MODEL
# =========================

PRODUCT_ID = "FOODS_3_090"

print("FINAL OPTIMIZED RANDOM FOREST")
print("=============================")
print(f"Product: {PRODUCT_ID}")


# Load feature-engineered data
df = pd.read_csv("Data/sales_features.csv")

df["date"] = pd.to_datetime(df["date"])

# Select product
product_df = df[df["item_id"] == PRODUCT_ID].copy()

product_df = product_df.sort_values("date")


# Features selected during feature engineering
features = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_7",
    "sales_rolling_30"
]

target = "sales"

# Remove rows containing NaN created by lag/rolling features
product_df = product_df.dropna(subset=features + [target])


X = product_df[features]
y = product_df[target]


# =========================
# TIME-BASED TRAIN/TEST SPLIT
# =========================

split_index = int(len(product_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# =========================
# OPTIMIZED RANDOM FOREST
# =========================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

print("\nTraining final model...")

model.fit(X_train, y_train)


# =========================
# MODEL EVALUATION
# =========================

predictions = model.predict(X_test)
# Save actual vs predicted values
test_results = pd.DataFrame({
    "date": product_df.iloc[split_index:]["date"].values,
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

test_results.to_csv(
    "Data/test_predictions.csv",
    index=False
)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)


print("\nFINAL MODEL PERFORMANCE")
print("=======================")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "final_sales_model.pkl")

print("\nFinal model saved successfully!")