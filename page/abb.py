import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import matplotlib.pyplot as plt

# Pagina-instellingen voor Streamlit
st.set_page_config(page_title='Temperature Data for Three Cities')

# Titel en beschrijving van de applicatie
st.markdown('# Historical Temperature Data for Amsterdam, Brussels, and Berlin')
st.write('Deze applicatie toont de maximale dagelijkse temperatuur voor drie steden over meerdere jaren.')

# Coördinaten van de drie steden
locations = {
    "Amsterdam": {"latitude": 52.37, "longitude": 4.90},
    "Brussel": {"latitude": 50.85, "longitude": 4.35},
    "Berlijn": {"latitude": 52.52, "longitude": 13.41}
}

# Functie om data op te halen via de Open-Meteo API
def get_weather_data(city, latitude, longitude):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "1950-01-01",
        "end_date": "2024-01-01",
        "daily": "temperature_2m_max",
        "timezone": "Europe/Berlin"
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Haal dagelijkse temperatuurdata op
    dates = pd.to_datetime(data['daily']['time'])
    max_temps = data['daily']['temperature_2m_max']
    
    # Zet data in DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temperature_2m_max': max_temps
    })
    df['city'] = city
    return df

# Data ophalen voor elke stad
dfs = []
for city, coords in locations.items():
    df = get_weather_data(city, coords['latitude'], coords['longitude'])
    dfs.append(df)

# Combineer alle datasets in één DataFrame
combined_df = pd.concat(dfs)

# Filter optie om te kiezen tussen steden
city_selection = st.multiselect('Selecteer steden om te visualiseren', list(locations.keys()), default=list(locations.keys()))

# Filter data op basis van de geselecteerde steden
filtered_df = combined_df[combined_df['city'].isin(city_selection)]

# Plot de temperatuurdata
fig = go.Figure()

for city in city_selection:
    city_data = filtered_df[filtered_df['city'] == city]
    fig.add_trace(go.Scatter(
        x=city_data['date'],
        y=city_data['temperature_2m_max'],
        mode='lines',
        name=city
    ))

# Lay-out van de grafiek
fig.update_layout(
    title='Maximale Dagelijkse Temperatuur',
    xaxis_title='Datum',
    yaxis_title='Temperatuur (°C)',
    legend_title='Steden',
    template='plotly_white'
)

# Toon de grafiek in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Optie om de dataset te tonen
with st.expander('Bekijk ruwe data'):
    st.write(filtered_df)
