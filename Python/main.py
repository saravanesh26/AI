from fastapi import FastAPI
from opencage.geocoder import OpenCageGeocode
import openmeteo_requests
import requests_cache
from retry_requests import retry
from dotenv import load_dotenv
import os
import pandas as pd
import pickle
import numpy as np

# 🌾 Load ML model
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# 🔐 Load API key
load_dotenv()
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")

app = FastAPI()

# 🌦️ Open-Meteo setup
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# 📄 Load CSV once
df = pd.read_csv("district_avg.csv")

# 🔥 FIX: normalize ALL column names
df.columns = df.columns.str.lower().str.strip()

# now safely use lowercase
df["district"] = df["district"].str.lower().str.strip()


@app.get("/")
def home():
    return {"message": "Crop API running 🚀"}


# 🏡 LOCATION API
@app.get("/location")
def get_location(lat: float, lon: float):
    geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
    results = geocoder.reverse_geocode(lat, lon)

    components = results[0]['components']

    # 🔥 fallback logic
    district = (
        components.get("state_district")
        or components.get("county")
        or components.get("city")
        or components.get("town")
    )

    return {
        "village": components.get("village"),
        "city": components.get("city") or components.get("town"),
        "district": district,
        "sub_district": components.get("county"),
        "state": components.get("state"),
        "country": components.get("country")
    }


# 🌦️ WEATHER API
@app.get("/weather")
def get_weather(lat: float, lon: float):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "wind_speed_10m",
            "soil_temperature_6cm",
            "soil_moisture_3_to_9cm"
        ],
        "forecast_days": 1
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    i = -1  # latest values

    return {
        "temperature": round(float(hourly.Variables(0).ValuesAsNumpy()[i]), 2),
        "humidity": int(hourly.Variables(1).ValuesAsNumpy()[i]),
        "rain": float(hourly.Variables(2).ValuesAsNumpy()[i]),
        "wind_speed": round(float(hourly.Variables(3).ValuesAsNumpy()[i]), 2),
        "soil_temp": round(float(hourly.Variables(4).ValuesAsNumpy()[i]), 2),
        "soil_moisture": round(float(hourly.Variables(5).ValuesAsNumpy()[i]), 3)
    }

@app.get("/predict")
def predict_crop(
    N: float,
    P: float,
    K: float,
    pH: float,
    rainfall: float,
    temperature: float
):
    try:
        # 🧠 Prepare input
        data = np.array([[N, P, K, pH, rainfall, temperature]])

        # 🎯 Predict probabilities
        probs = model.predict_proba(data)[0]

        # 🏆 Top 3 crops
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_crops = le.inverse_transform(top3_idx)

        # 📊 Format response
        result = [
            {
                "crop": crop,
                "confidence": round(float(probs[idx]), 3)
            }
            for crop, idx in zip(top3_crops, top3_idx)
        ]

        return {
            "recommendations": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# 🌾 FULL DATA API
@app.get("/full-data")
def get_full_data(lat: float, lon: float):

    location = get_location(lat, lon)
    weather = get_weather(lat, lon)

    district = location.get("district")

    npk = None

    if district:
        district_clean = district.lower().strip()

        row = df[df["district"] == district_clean]

        if not row.empty:
            npk = {
                "N": float(row.iloc[0]["nitrogen value"]),
                "P": float(row.iloc[0]["phosphorous value"]),
                "K": float(row.iloc[0]["potassium value"]),
                "ph": float(row.iloc[0]["ph"])
            }
        else:
            npk = {
                "error": f"District '{district}' not found in dataset"
            }
    else:
        npk = {
            "error": "District not found from location API"
        }

    return {
        "location": location,
        "weather": weather,
        "npk": npk
    }