import pandas as pd

# Load data
sales = pd.read_csv("Data/sales_train_evaluation.csv")
calendar = pd.read_csv("Data/calendar.csv")

# Select one store
store_data = sales[sales["store_id"] == "CA_1"].copy()

# Sales columns
sales_columns = [col for col in sales.columns if col.startswith("d_")]

# Convert wide format to long format
sales_long = store_data.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id"],
    value_vars=sales_columns,
    var_name="d",
    value_name="sales"
)

# Connect day IDs with actual dates
calendar_small = calendar[["d", "date", "wm_yr_wk", "weekday", "month", "year"]]

sales_long = sales_long.merge(
    calendar_small,
    on="d",
    how="left"
)

# Convert date
sales_long["date"] = pd.to_datetime(sales_long["date"])

# Save cleaned dataset
sales_long.to_csv("Data/sales_cleaned_CA1.csv", index=False)

# Check result
print("Shape:", sales_long.shape)
print("\nColumns:")
print(sales_long.columns.tolist())

print("\nFirst 5 rows:")
print(sales_long.head())