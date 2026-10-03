import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# RANDOM FOREST HYPERPARAMETER TUNING
# =========================================================

print("RANDOM FOREST HYPERPARAMETER TUNING")
print("===================================")


# ---------------------------------------------------------
# 1. Load feature-engineered dataset
# ---------------------------------------------------------

df = pd.read_csv("Data/sales_features.csv")

df["date"] = pd.to_datetime(df["date"])

PRODUCT_ID = "FOODS_3_090"

product_df = df[df["item_id"] == PRODUCT_ID].copy()

product_df = product_df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------
# 2. Features and target
# ---------------------------------------------------------

features = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_7",
    "sales_rolling_30"
]

target = "sales"

product_df = product_df.dropna(
    subset=features + [target]
).reset_index(drop=True)

X = product_df[features]
y = product_df[target]


print(f"\nProduct: {PRODUCT_ID}")
print(f"Rows: {len(product_df)}")


# ---------------------------------------------------------
# 3. Time-series cross validation
# ---------------------------------------------------------

tscv = TimeSeriesSplit(n_splits=5)


# ---------------------------------------------------------
# 4. Random Forest
# ---------------------------------------------------------

model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------------------------
# 5. Hyperparameter grid
# ---------------------------------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# ---------------------------------------------------------
# 6. Grid Search
# ---------------------------------------------------------

print("\nSearching for best parameters...")
print("Please wait...\n")


grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)


# ---------------------------------------------------------
# 7. Best parameters
# ---------------------------------------------------------

print("\nBEST PARAMETERS")
print("================")

print(grid_search.best_params_)

print(
    f"\nBest CV MAE: "
    f"{-grid_search.best_score_:.2f}"
)


# ---------------------------------------------------------
# 8. Train final tuned model
# ---------------------------------------------------------

best_model = grid_search.best_estimator_

best_model.fit(X, y)


# ---------------------------------------------------------
# 9. Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\nFEATURE IMPORTANCE")
print("==================")

print(importance.to_string(index=False))


print("\nTUNING COMPLETED SUCCESSFULLY.")