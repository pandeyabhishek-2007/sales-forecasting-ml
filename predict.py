import pandas as pd
import joblib


# =========================
# LOAD MODEL
# =========================

model = joblib.load("final_sales_model.pkl")

PRODUCT_ID = "FOODS_3_090"

print("SALES PREDICTION")
print("================")
print(f"Product: {PRODUCT_ID}")


# =========================
# LOAD FEATURE DATA
# =========================

df = pd.read_csv("Data/sales_features.csv")

df["date"] = pd.to_datetime(df["date"])

product_df = df[df["item_id"] == PRODUCT_ID].copy()
product_df = product_df.sort_values("date")


# =========================
# FEATURES
# =========================

features = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_7",
    "sales_rolling_30"
]


# Remove rows with missing feature values
product_df = product_df.dropna(subset=features)


# Get the most recent available row
latest_data = product_df.iloc[-1:]

X_latest = latest_data[features]


# =========================
# PREDICTION
# =========================

prediction = model.predict(X_latest)[0]


print("\nLATEST DATA")
print("-----------")
print(f"Date: {latest_data['date'].iloc[0].date()}")
print(f"Actual sales: {latest_data['sales'].iloc[0]}")

print("\nPREDICTION")
print("----------")
print(f"Predicted sales: {prediction:.2f} units")


print("\nPrediction completed successfully!")