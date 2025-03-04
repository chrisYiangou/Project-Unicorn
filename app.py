import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

# Backend API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Streamlit UI
st.title("Google Trends Forecasting App 🚀")
st.write("Enter a keyword to see historical and predicted trends.")

# User Input
keyword = st.text_input("Enter Search Term", "Outsystems")
periods = st.slider("Forecast Days", 7, 90, 30)

if st.button("Get Forecast"):
    with st.spinner("Fetching data..."):
        # Call FastAPI backend
        response = requests.get(f"{API_URL}/forecast/{keyword}?periods={periods}")
        data = response.json()

        if "error" in data:
            st.error(data["error"])
        else:
            # Convert historical and forecast data to DataFrames
            df_hist = pd.DataFrame(data["historical_data"])
            df_forecast = pd.DataFrame(data["forecast_data"])

            # Plot Historical Data
            st.subheader("📈 Historical Trends")
            st.write(
                f"The chart below shows the search interest for **'{keyword}'** over the past 3 months. "
                "Higher values indicate more search activity. This data is pulled directly from Google Trends."
            )

            fig_hist = px.line(
                df_hist, x="ds", y="y", title=f"Past Trend for '{keyword}'"
            )
            st.plotly_chart(fig_hist)

            # Plot Forecast Data
            st.subheader("🔮 Forecasted Trends")
            st.write(
                f"The following chart predicts the search interest for **'{keyword}'** over the next {periods} days "
                "using a machine learning model (Prophet). The solid line represents the forecasted trend, "
                "while the dotted lines indicate the upper and lower confidence intervals."
            )
            fig_forecast = px.line(
                df_forecast, x="ds", y="yhat", title=f"Predicted Trend for '{keyword}'"
            )
            fig_forecast.add_scatter(
                x=df_forecast["ds"],
                y=df_forecast["yhat_lower"],
                mode="lines",
                name="Lower Bound",
                line=dict(dash="dot"),
            )
            fig_forecast.add_scatter(
                x=df_forecast["ds"],
                y=df_forecast["yhat_upper"],
                mode="lines",
                name="Upper Bound",
                line=dict(dash="dot"),
            )
            st.plotly_chart(fig_forecast)
