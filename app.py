import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)


# =========================
# LOAD DATA AND MODEL
# =========================

PRODUCT_ID = "FOODS_3_090"

df = pd.read_csv("Data/sales_features.csv")
model = joblib.load("final_sales_model.pkl")

df["date"] = pd.to_datetime(df["date"])

product_df = df[df["item_id"] == PRODUCT_ID].copy()
product_df = product_df.sort_values("date")


# =========================
# TITLE
# =========================

st.title("📈 Sales Forecasting Dashboard")

st.markdown(
    "Machine Learning based sales analysis and forecasting"
)

st.divider()


# =========================
# PRODUCT INFORMATION
# =========================

st.subheader("Product Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Product", PRODUCT_ID)

with col2:
    st.metric("Historical Records", len(product_df))

with col3:
    st.metric(
        "Latest Dataset Date",
        product_df["date"].max().strftime("%d %b %Y")
    )


# =========================
# HISTORICAL SALES
# =========================

st.subheader("📊 Historical Sales")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    product_df["date"],
    product_df["sales"],
    label="Historical Sales"
)

ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.set_title(f"Historical Sales - {PRODUCT_ID}")

ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)


# =========================
# MODEL PERFORMANCE
# =========================

st.subheader("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "17.39")

with col2:
    st.metric("RMSE", "25.02")

with col3:
    st.metric("R² Score", "0.4386")


# =========================
# FEATURE IMPORTANCE
# =========================

st.subheader("🔍 Feature Importance")

features = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_rolling_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_30"
]

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")

st.pyplot(fig)


# =========================
# RECENT SALES
# =========================

st.subheader("📋 Recent Sales")

recent = product_df[
    ["date", "item_id", "sales"]
].tail(10).sort_values(
    "date",
    ascending=False
)

st.dataframe(
    recent,
    use_container_width=True
)


st.divider()

st.caption(
    "Forecasts are based on the historical dataset available through "
    f"{product_df['date'].max().strftime('%d %B %Y')}."
)

# ============================
# SEASONALITY ANALYSIS
# ============================

st.subheader("📅 Seasonality Analysis")

# Average sales by day of week
weekday_sales = (
    product_df.groupby("weekday")["sales"]
    .mean()
    .reindex([
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ])
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(
    weekday_sales.index,
    weekday_sales.values
)

ax.set_xlabel("Day of Week")
ax.set_ylabel("Average Units Sold")
ax.set_title(f"Average Sales by Day of Week - {PRODUCT_ID}")
ax.tick_params(axis="x", rotation=45)

st.pyplot(fig)


# Average sales by month
monthly_sales = (
    product_df.groupby("month")["sales"]
    .mean()
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(
    monthly_sales.index,
    monthly_sales.values
)

ax.set_xlabel("Month")
ax.set_ylabel("Average Units Sold")
ax.set_title(f"Average Monthly Sales - {PRODUCT_ID}")
ax.set_xticks(range(1, 13))

st.pyplot(fig)

# =========================
# NEXT DAY SALES PREDICTION
# =========================

st.divider()

st.subheader("🔮 Next Day Sales Prediction")

# Sort data by date
product_df = product_df.sort_values("date").reset_index(drop=True)

# Latest available date
latest_date = product_df["date"].max()

# Next day
next_date = latest_date + pd.Timedelta(days=1)

# Create prediction features
feature_values = {
    "sales_lag_1": product_df["sales"].iloc[-1],
    "sales_lag_7": product_df["sales"].iloc[-7],
    "sales_rolling_7": product_df["sales"].iloc[-7:].mean(),
    "sales_lag_14": product_df["sales"].iloc[-14],
    "sales_lag_28": product_df["sales"].iloc[-28],
    "sales_rolling_30": product_df["sales"].iloc[-30:].mean()
}

prediction_features = pd.DataFrame([feature_values])

# Arrange features in the exact order used during model training
prediction_features = prediction_features[model.feature_names_in_]

# Make prediction
next_prediction = model.predict(prediction_features)[0]

# Display prediction
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Forecast Date",
        next_date.strftime("%d %B %Y")
    )

with col2:
    st.metric(
        "Predicted Sales",
        f"{next_prediction:.2f} units"
    )

st.info(
    f"Based on historical data available through "
    f"{latest_date.strftime('%d %B %Y')}, "
    f"the model forecasts approximately "
    f"{next_prediction:.2f} units for "
    f"{next_date.strftime('%d %B %Y')}."
)