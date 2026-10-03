import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# MODEL VALIDATION
# =========================================================

print("MODEL VALIDATION")
print("================")


# ---------------------------------------------------------
# 1. Load feature-engineered dataset
# ---------------------------------------------------------

df = pd.read_csv("Data/sales_features.csv")

df["date"] = pd.to_datetime(df["date"])

# Select our product
PRODUCT_ID = "FOODS_3_090"

product_df = df[df["item_id"] == PRODUCT_ID].copy()

# Sort chronologically
product_df = product_df.sort_values("date").reset_index(drop=True)


print(f"\nProduct: {PRODUCT_ID}")
print(f"Total rows: {len(product_df)}")


# ---------------------------------------------------------
# 2. Define features and target
# ---------------------------------------------------------

target = "sales"

features = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_7",
    "sales_rolling_30"
]

# Remove rows containing missing values
product_df = product_df.dropna(
    subset=features + [target]
).reset_index(drop=True)

X = product_df[features]
y = product_df[target]


# ---------------------------------------------------------
# 3. Time-Series Cross Validation
# ---------------------------------------------------------

tscv = TimeSeriesSplit(n_splits=5)

mae_scores = []
rmse_scores = []
r2_scores = []


print("\n5-FOLD TIME-SERIES VALIDATION")
print("==============================")


for fold, (train_index, test_index) in enumerate(
    tscv.split(X), start=1
):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Create model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    mae_scores.append(mae)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    print(f"\nFold {fold}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")


# ---------------------------------------------------------
# 4. Average performance
# ---------------------------------------------------------

print("\n\nVALIDATION SUMMARY")
print("==================")

print(f"Average MAE:  {np.mean(mae_scores):.2f}")
print(f"Average RMSE: {np.mean(rmse_scores):.2f}")
print(f"Average R2:   {np.mean(r2_scores):.4f}")

print("\nStandard Deviation")
print(f"MAE Std:  {np.std(mae_scores):.2f}")
print(f"RMSE Std: {np.std(rmse_scores):.2f}")
print(f"R2 Std:   {np.std(r2_scores):.4f}")