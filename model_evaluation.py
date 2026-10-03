import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
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

# Create time feature
daily_sales["day_number"] = np.arange(len(daily_sales))

X = daily_sales[["day_number"]]
y = daily_sales["sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Evaluation metrics
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

print("MODEL EVALUATION")
print("----------------")
print("Product:", product_id)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 SCORE:", r2)