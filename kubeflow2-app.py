import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Electrification Forecast Dashboard",
    layout="wide"
)

# ---------------------------
# TITLE & DESCRIPTION
# ---------------------------
st.title("Electrification Forecasting Dashboard")
st.markdown(
    """
This dashboard analyzes historical electricity access data and predicts future
electrification needs using time-series models.

**Project:** Kubeflow – AI-Driven Electrification Planning  
**Audience:** Policymakers, researchers, energy planners
"""
)

# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    return df

uploaded_file = st.sidebar.file_uploader(
    "Upload electrification dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

df = load_data(uploaded_file)

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("Controls")

countries = sorted(df["country"].unique())
selected_country = st.sidebar.selectbox("Select Country", countries)

access_type = st.sidebar.selectbox(
    "Electricity Access Type",
    ["electricity_access", "rural_access", "urban_access"]
)

forecast_years = st.sidebar.slider(
    "Forecast Horizon (Years)",
    min_value=3,
    max_value=15,
    value=5
)

# ---------------------------
# FILTER DATA
# ---------------------------
country_df = df[df["country"] == selected_country].sort_values("year")

# ---------------------------
# SECTION: EDA
# ---------------------------
st.header("Historical Electrification Trends")

fig, ax = plt.subplots()
ax.plot(
    country_df["year"],
    country_df[access_type],
    marker="o"
)
ax.set_xlabel("Year")
ax.set_ylabel("Access (%)")
ax.set_title(f"{selected_country} – {access_type.replace('_', ' ').title()}")
st.pyplot(fig)

st.markdown(
    """
**Interpretation:**  
This plot shows how electricity access has evolved over time.
Upward trends indicate infrastructure expansion, while flat regions
suggest stagnation or policy constraints.
"""
)

# ---------------------------
# SECTION: HEAT MAP
# ---------------------------
st.header(" Electrification Heat Map")

pivot = df.pivot_table(
    index="country",
    columns="year",
    values=access_type
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    pivot,
    cmap="YlGnBu",
    linewidths=0.2,
    ax=ax
)
ax.set_title("Electricity Access Heat Map")
st.pyplot(fig)

st.markdown(
    """
**Heat Map Insight:**  
Darker regions indicate higher electricity access. Persistent light areas
highlight regions requiring urgent electrification investment.
"""
)

# ---------------------------
# SECTION: FORECASTING
# ---------------------------
st.header(" Forecasting Future Electrification")

series = country_df.set_index("year")[access_type]

if len(series) < 6:
    st.error("Not enough data points for forecasting.")
    st.stop()

# Fit ARIMA model
model = ARIMA(series, order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=forecast_years)
forecast_year_index = range(
    series.index.max() + 1,
    series.index.max() + 1 + forecast_years
)

# ---------------------------
# PLOT FORECAST
# ---------------------------
fig, ax = plt.subplots()
ax.plot(series.index, series, label="Historical")
ax.plot(forecast_year_index, forecast, label="Forecast", linestyle="--")
ax.set_xlabel("Year")
ax.set_ylabel("Access (%)")
ax.set_title(f"{selected_country} – Electrification Forecast")
ax.legend()
st.pyplot(fig)

st.markdown(
    """
**Forecast Explanation:**  
The dashed line represents projected electricity access assuming historical
patterns continue. Forecast uncertainty increases further into the future,
so results should guide planning—not replace policy judgment.
"""
)

# ---------------------------
# SECTION: MODEL METRICS
# ---------------------------
st.header(" Model Summary")

st.text(model_fit.summary())

# ---------------------------
# SECTION: RECOMMENDATIONS
# ---------------------------
st.header("Planning Insights")

latest_value = series.iloc[-1]
future_value = forecast.iloc[-1]

if future_value < 90:
    st.warning(
        f" {selected_country} is projected to remain below universal access "
        f"({future_value:.1f}%). Accelerated investment is recommended."
    )
else:
    st.success(
        f" {selected_country} is on track to approach universal electricity access."
    )

st.markdown(
    """
**Suggested Actions:**
- Expand rural grid connections or mini-grids
- Strengthen long-term infrastructure financing
- Monitor population growth against access gains
"""
)

# ---------------------------
# FOOTER
# ---------------------------
st.caption(
    "Kubeflow Electrification Project | Streamlit Dashboard | AI-assisted planning"
)


