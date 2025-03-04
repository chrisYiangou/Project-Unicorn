from fastapi import FastAPI
from pytrends.request import TrendReq
import pandas as pd
from prophet import Prophet

app = FastAPI()
pytrends = TrendReq(hl='en-US', tz=360)

@app.get("/forecast/{keyword}")
def forecast_trends(keyword: str, periods: int = 30):
    try:
        # Fetch Google Trends data (last 3 months)
        pytrends.build_payload([keyword], timeframe='today 3-m')
        data = pytrends.interest_over_time()
        
        if data.empty:
            return {"error": "No data found. Try another keyword."}

        # Prepare data for Prophet
        df = data.reset_index()[['date', keyword]]
        df.rename(columns={'date': 'ds', keyword: 'y'}, inplace=True)

        # Train Prophet model
        model = Prophet()
        model.fit(df)

        # Create future dates
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Convert results to JSON
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient="records")
        
        return {
            "keyword": keyword,
            "historical_data": df.to_dict(orient="records"),
            "forecast_data": forecast_data
        }

    except Exception as e:
        return {"error": str(e)}