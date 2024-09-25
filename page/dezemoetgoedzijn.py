import requests
import pandas as pd
import time

# Steden en hun coördinaten
locations = {
    "Amsterdam": {"latitude": 52.37, "longitude": 4.90},
    "Brussel": {"latitude": 50.85, "longitude": 4.35},
    "Berlijn": {"latitude": 52.52, "longitude": 13.41}
}

# Functie om data op te halen met een retry-mechanisme
def get_weather_data(latitude, longitude, retries=5, backoff_factor=0.2):
    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "1950-01-01",
        "end_date": "2050-12-31",
        "models": "EC_Earth3P_HR",  # Example model
        "daily": "temperature_2m_max"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Verwerken van de dagelijkse data
                daily_data = {
                    "date": pd.to_datetime(data["daily"]["time"]),
                    "temperature_2m_max": data["daily"]["temperature_2m_max"]
                }
                return pd.DataFrame(daily_data)
            else:
                print(f"Error: Received status code {response.status_code}. Attempt {attempt + 1} of {retries}.")
        except requests.RequestException as e:
            print(f"Request failed: {e}. Attempt {attempt + 1} of {retries}.")
        
        # Wachten voor een retry
        time.sleep(backoff_factor * (2 ** attempt))

    # Als alle retries mislukken
    print(f"Failed to fetch data after {retries} attempts.")
    return pd.DataFrame()

# Ophalen van data voor elke stad
dfs = []
for city, coords in locations.items():
    df = get_weather_data(coords['latitude'], coords['longitude'])
    if not df.empty:
        df['city'] = city  # Voeg de stad toe aan de dataset
        dfs.append(df)

# Combineer alle datasets in één DataFrame
if dfs:
    combined_df = pd.concat(dfs)
    # Eerste paar rijen van de gecombineerde data bekijken
    print(combined_df.head())
else:
    print("Geen data opgehaald.")


import plotly.express as px

# Plot maken voor de temperatuurontwikkeling per stad
fig = px.line(
    combined_df, 
    x="date", 
    y="temperature_2m_max", 
    color="city",
    title="Maximale Temperatuur Ontwikkeling (1950-2050)",
    labels={
        "temperature_2m_max": "Maximale Temperatuur (°C)",
        "date": "Datum",
        "city": "Stad"
    }
)

# Interactieve grafiek tonen
fig.show()
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Voeg een kolom toe voor het jaar
combined_df['year'] = combined_df['date'].dt.year

# Bereid de data voor
X = combined_df[['year']]  # Features
y = combined_df['temperature_2m_max']  # Target variable

# Verdeel de data in trainings- en testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train het model
model = LinearRegression()
model.fit(X_train, y_train)

# Maak voorspellingen
y_pred = model.predict(X_test)

# Evalueer het model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')

# Toont de coefficients
print(f'Coefficient: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Functie om het regressiemodel te maken
def train_model(df):
    df['year'] = pd.to_datetime(df['date']).dt.year
    X = df[['year']]
    y = df['temperature_2m_max']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, mae, r2

# Titel van de app
st.title("Klimaatdata Analyse: Amsterdam, Brussel en Berlijn")

# Data inladen (simulatie van de opgehaalde data)
@st.cache
def load_data():
    # Hier zou je de echte data inladen
    data = pd.read_csv('climate_data.csv')  # Vervang met het juiste pad
    return data

df = load_data()

# Toon een voorbeeld van de data
st.subheader("Overzicht van de data")
st.write(df.head())

# Interactieve selectie van steden
st.subheader("Kies een stad om de temperatuurtrends te bekijken")
selected_city = st.selectbox("Selecteer een stad", df['city'].unique())

# Filter de data op de geselecteerde stad
filtered_data = df[df['city'] == selected_city]

# Visualisatie van temperatuurtrends
st.subheader(f"Maximale Temperatuur Ontwikkeling: {selected_city}")
fig = px.line(
    filtered_data,
    x='date',
    y='temperature_2m_max',
    title=f"Maximale Temperatuur Ontwikkeling in {selected_city} (1950-2050)",
    labels={"temperature_2m_max": "Maximale Temperatuur (°C)", "date": "Datum"}
)
st.plotly_chart(fig)

# Voorspellend model tonen
st.subheader("Voorspellend model voor toekomstige temperaturen")
model, mse, mae, r2 = train_model(filtered_data)

# Toon de modelresultaten
st.write(f"Mean Squared Error: {mse}")
st.write(f"Mean Absolute Error: {mae}")
st.write(f"R² Score: {r2}")

# Maak voorspellingen voor het geselecteerde jaar
st.subheader("Maak voorspellingen voor toekomstige jaren")
year_input = st.number_input("Voer een jaar in (bijv. 2030):", min_value=2023, max_value=2050, step=1)

if year_input:
    prediction = model.predict([[year_input]])
    st.write(f"Voorspelde maximale temperatuur in {selected_city} in {year_input}: {prediction[0]:.2f} °C")


