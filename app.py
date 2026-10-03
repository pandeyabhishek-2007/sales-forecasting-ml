import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================
# CUSTOM UI DESIGN
# =========================

st.markdown("""
<style>

    /* =========================
       MAIN BODY
       ========================= */

    .stApp {
        background-color: #eef2f7;
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background-color: #172033;
        border-right: 1px solid #26334d;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 25px;
    }


    /* =========================
       MAIN HEADINGS
       ========================= */

    h1 {
        color: #172033;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #172033;
        font-weight: 700;
    }

    h3 {
        color: #26334d;
        font-weight: 650;
    }


    /* =========================
       METRIC CARDS
       ========================= */

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dce3ed;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(23, 32, 51, 0.06);
        min-height: 105px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600;
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        color: #172033 !important;
        font-weight: 700;
        font-size: 30px;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background-color: #2563eb;
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 20px;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
        transform: translateY(-1px);
    }


    /* =========================
       SELECTBOX
       ========================= */

    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border: 1px solid #d4dce8;
        background-color: #ffffff;
    }


    /* =========================
       DATAFRAME
       ========================= */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #dce3ed;
    }


    /* =========================
       INFO / SUCCESS BOX
       ========================= */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* =========================
       DIVIDERS
       ========================= */

    hr {
        border-color: #d7dee9;
    }


    /* =========================
       SIDEBAR RADIO
       ========================= */

    div[role="radiogroup"] {
        gap: 8px;
    }

    div[role="radiogroup"] label {
        background-color: transparent;
        border-radius: 8px;
        padding: 8px 10px;
        transition: 0.2s;
    }

    div[role="radiogroup"] label:hover {
        background-color: #26334d;
    }


    /* =========================
       SELECTED SIDEBAR ITEM
       ========================= */

    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #26334d;
        border-radius: 8px;
    }


    /* =========================
       GENERAL TEXT
       ========================= */

    p {
        color: #334155;
    }


    /* =========================
       SMALL SCREEN
       ========================= */

    @media (max-width: 900px) {

        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }

        div[data-testid="stMetric"] {
            padding: 12px;
        }

    }

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR NAVIGATION
# =========================

st.sidebar.title("📈 Sales Forecasting")

st.sidebar.caption("Machine Learning Dashboard")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Forecast",
        "Model Performance",
        "Data Analysis"
    ]
)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

df = pd.read_csv("Data/sales_features.csv")

model = joblib.load("final_sales_model.pkl")

df["date"] = pd.to_datetime(df["date"])

PRODUCT_ID = "FOODS_3_090"

product_df = df[
    df["item_id"] == PRODUCT_ID
].copy()

product_df = product_df.sort_values(
    "date"
).reset_index(drop=True)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("📈 Sales Forecasting Dashboard")

    st.markdown(
        "### Machine Learning based Sales Analysis & Forecasting"
    )

    st.write(
        "Analyze historical sales, understand sales patterns, "
        "and forecast future demand using a Random Forest model."
    )

    st.divider()

   # -----------------------------------------------------
# PRODUCT INFORMATION
# -----------------------------------------------------

st.subheader("📦 Product Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Product",
        PRODUCT_ID
    )

with col2:
    st.metric(
        "Historical Records",
        f"{len(product_df):,}"
    )

with col3:
    st.metric(
        "Average Daily Sales",
        f"{product_df['sales'].mean():.2f}"
    )

with col4:
    st.metric(
        "Total Sales",
        f"{product_df['sales'].sum():,.0f}"
    )

st.divider()

    # -----------------------------------------------------
    # HISTORICAL SALES
    # -----------------------------------------------------

st.subheader("📊 Historical Sales")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
        product_df["date"],
        product_df["sales"],
        label="Historical Sales"
    )

ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.set_title(
        f"Historical Sales - {PRODUCT_ID}"
    )

ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

st.info(
        "Use the sidebar to explore forecasts, model performance, "
        "and detailed sales analysis."
    )


# =========================================================
# FORECAST
# =========================================================

if page == "Forecast":

    st.title("🔮 Sales Forecasting")

    st.write(
        "Use the trained Random Forest model to forecast future "
        "sales for the selected number of days."
    )

    st.divider()

    # -----------------------------------------------------
    # FORECAST HORIZON
    # -----------------------------------------------------

    forecast_days = st.selectbox(
        "Forecast Horizon",
        [1, 3, 7, 14, 30]
    )

    if st.button(
        "🚀 Generate Forecast",
        use_container_width=True
    ):

        # Copy historical data
        forecast_df = product_df.copy()

        predictions = []

        # -------------------------------------------------
        # RECURSIVE FORECASTING
        # -------------------------------------------------

        for i in range(forecast_days):

            feature_values = {

                "sales_lag_1":
                    forecast_df["sales"].iloc[-1],

                "sales_lag_7":
                    forecast_df["sales"].iloc[-7],

                "sales_lag_14":
                    forecast_df["sales"].iloc[-14],

                "sales_lag_28":
                    forecast_df["sales"].iloc[-28],

                "sales_rolling_7":
                    forecast_df["sales"].iloc[-7:].mean(),

                "sales_rolling_30":
                    forecast_df["sales"].iloc[-30:].mean()
            }

            prediction_features = pd.DataFrame(
                [feature_values]
            )

            # Make sure feature order is exactly
            # the same as during model training
            prediction_features = prediction_features[
                model.feature_names_in_
            ]

            # Model prediction
            prediction = model.predict(
                prediction_features
            )[0]

            # Forecast date
            forecast_date = (
                forecast_df["date"].iloc[-1]
                + pd.Timedelta(days=1)
            )

            predictions.append({
                "Date": forecast_date,
                "Predicted Sales": round(
                    prediction,
                    2
                )
            })

            # -------------------------------------------------
            # ADD PREDICTION BACK
            # -------------------------------------------------
            # This allows multi-day forecasting.
            # Tomorrow's prediction becomes part of
            # the features used for the next day.

            new_row = forecast_df.iloc[-1].copy()

            new_row["date"] = forecast_date
            new_row["sales"] = prediction

            forecast_df = pd.concat(
                [
                    forecast_df,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

        # -----------------------------------------------------
        # FORECAST RESULTS
        # -----------------------------------------------------

        forecast_results = pd.DataFrame(
            predictions
        )

        st.success(
            f"Forecast generated for the next "
            f"{forecast_days} day(s)."
        )

        # -----------------------------------------------------
        # FORECAST SUMMARY
        # -----------------------------------------------------

        total_forecast = (
            forecast_results["Predicted Sales"].sum()
        )

        average_forecast = (
            forecast_results["Predicted Sales"].mean()
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Forecasted Sales",
                f"{total_forecast:.2f} units"
            )

        with col2:

            st.metric(
                "Average Daily Forecast",
                f"{average_forecast:.2f} units"
            )

        st.divider()

        # -----------------------------------------------------
        # FORECAST TABLE
        # -----------------------------------------------------

        st.subheader("📋 Forecast Results")

        st.dataframe(
            forecast_results,
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------------------
        # FORECAST CHART
        # -----------------------------------------------------

        st.subheader("📈 Forecast Visualization")

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        # Recent historical sales
        recent_history = product_df.tail(14)

        ax.plot(
            recent_history["date"],
            recent_history["sales"],
            label="Historical Sales"
        )

        # Forecast
        ax.plot(
            forecast_results["Date"],
            forecast_results["Predicted Sales"],
            marker="o",
            label="Forecast"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")

        ax.set_title(
            f"Sales Forecast - Next "
            f"{forecast_days} Day(s)"
        )

        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.caption(
            "Forecasts are generated recursively: "
            "each predicted day is used as input for the "
            "following day's prediction."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Evaluation of the trained Random Forest model "
        "on previously unseen test data."
    )

    st.divider()

    # -----------------------------------------------------
    # LOAD TEST PREDICTIONS
    # -----------------------------------------------------

    test_results = pd.read_csv(
        "Data/test_predictions.csv"
    )

    test_results["date"] = pd.to_datetime(
        test_results["date"]
    )

    actual = test_results["Actual Sales"]
    predicted = test_results["Predicted Sales"]

    # Calculate metrics directly
    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = mean_squared_error(
        actual,
        predicted
    ) ** 0.5

    r2 = r2_score(
        actual,
        predicted
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    st.subheader("📊 Test Set Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

    with col2:

        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

    with col3:

        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    st.divider()

    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.subheader(
        "📈 Actual vs Predicted Sales"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        test_results["date"],
        test_results["Actual Sales"],
        label="Actual Sales"
    )

    ax.plot(
        test_results["date"],
        test_results["Predicted Sales"],
        label="Predicted Sales"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")

    ax.set_title(
        "Actual vs Predicted Sales — Test Set"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.divider()

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.subheader(
        "🔍 Feature Importance"
    )

    features = model.feature_names_in_

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    st.pyplot(fig)

    st.info(
        "Feature importance shows which historical-sales "
        "features contributed most to the Random Forest's "
        "predictions."
    )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "Data Analysis":

    st.title("📊 Sales Data Analysis")

    st.write(
        "Explore historical sales patterns, seasonality, "
        "and recent product performance."
    )

    st.divider()

    # -----------------------------------------------------
    # RECENT SALES
    # -----------------------------------------------------

    st.subheader("📋 Recent Sales")

    recent = product_df[
        ["date", "item_id", "sales"]
    ].tail(10).sort_values(
        "date",
        ascending=False
    )

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # WEEKDAY SEASONALITY
    # -----------------------------------------------------

    st.subheader(
        "📅 Average Sales by Day of Week"
    )

    weekday_sales = (
        product_df
        .groupby("weekday")["sales"]
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

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.bar(
        weekday_sales.index,
        weekday_sales.values
    )

    ax.set_xlabel(
        "Day of Week"
    )

    ax.set_ylabel(
        "Average Units Sold"
    )

    ax.set_title(
        f"Average Sales by Day of Week - {PRODUCT_ID}"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    st.pyplot(fig)

    # -----------------------------------------------------
    # MONTHLY SEASONALITY
    # -----------------------------------------------------

    st.subheader(
        "📅 Average Sales by Month"
    )

    monthly_sales = (
        product_df
        .groupby("month")["sales"]
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.bar(
        monthly_sales.index,
        monthly_sales.values
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Average Units Sold"
    )

    ax.set_title(
        f"Average Monthly Sales - {PRODUCT_ID}"
    )

    ax.set_xticks(
        range(1, 13)
    )

    st.pyplot(fig)

    st.divider()

    st.caption(
        "Dataset available through "
        f"{product_df['date'].max().strftime('%d %B %Y')}."
    )