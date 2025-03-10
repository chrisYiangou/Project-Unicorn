import time
import json
from fastapi import FastAPI
from pytrends.request import TrendReq
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

app = FastAPI()
pytrends = TrendReq(hl="en-US", tz=360)

# Cache to store previous requests (prevents duplicate calls)
cache = {}

@app.get("/forecast/{keyword}")
def forecast_trends(keyword: str, periods: int = 30):
    try:
        current_time = datetime.now()
        
        # Check cache to avoid duplicate requests
        if keyword in cache:
            cached_data = cache[keyword]
            if (current_time - cached_data["timestamp"]).seconds < 600:  # 10-minute cache
                return cached_data["response"]

        # Implement rate-limiting with exponential backoff
        retries = 3
        delay = 5  # Initial delay in seconds

        for attempt in range(retries):
            try:
                time.sleep(delay)  # Throttle requests
                pytrends.build_payload([keyword], timeframe="today 3-m")
                data = pytrends.interest_over_time()

                if data.empty:
                    return {"error": "No data found. Try another keyword."}

                break  # Exit loop if successful

            except Exception as e:
                if attempt < retries - 1:
                    delay *= 2  # Double the delay for the next attempt
                    print(f"Retrying in {delay} seconds due to error: {str(e)}")
                else:
                    return {"error": f"Failed after multiple retries: {str(e)}"}

        # Prepare data for Prophet
        df = data.reset_index()[["date", keyword]]
        df.rename(columns={"date": "ds", keyword: "y"}, inplace=True)

        # Train Prophet model
        model = Prophet()
        model.fit(df)

        # Create future dates
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Convert results to JSON
        forecast_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")

        response = {
            "keyword": keyword,
            "historical_data": df.to_dict(orient="records"),
            "forecast_data": forecast_data,
        }

        # Store in cache to avoid excessive requests
        cache[keyword] = {"response": response, "timestamp": current_time}

        return response

    except Exception as e:
        return {"error": str(e)}