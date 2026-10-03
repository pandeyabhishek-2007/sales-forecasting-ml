import pandas as pd

df = pd.read_csv("Data/sales_cleaned_CA1.csv")

# Total sales for each product
product_sales = (
    df.groupby(["item_id", "cat_id"])["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("TOP 10 PRODUCTS:")
print(product_sales.head(10))

# Select the highest-selling product
best_product = product_sales.index[0]

print("\nSELECTED PRODUCT:")
print("Product:", best_product[0])
print("Category:", best_product[1])
print("Total Sales:", product_sales.iloc[0])