import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Fetch Real Weather Data for Munich (Open-Meteo API) ---
print("Downloading Munich weather data...")
url_weather = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 48.1371,
    "longitude": 11.5754,
    "start_date": "2019-01-01",
    "end_date": "2023-12-31",
    "hourly": ["surface_pressure"],
    "daily": ["precipitation_sum"],
    "timezone": "Europe/Berlin"
}

response_raw = requests.get(url_weather, params=params)
if response_raw.status_code != 200:
    print(f"Weather API failed: {response_raw.status_code}")
    sys.exit(1)

response = response_raw.json()

# Extract Daily Rain
dates = pd.to_datetime(response['daily']['time'])
precipitation = np.array(response['daily']['precipitation_sum'])

# Extract Hourly Pressure and calculate daily mean
hourly_times = pd.to_datetime(response['hourly']['time'])
hourly_pressure = np.array(response['hourly']['surface_pressure'])

df_hourly = pd.DataFrame({'time': hourly_times, 'pressure': hourly_pressure})
df_hourly.set_index('time', inplace=True)
daily_pressure = df_hourly.resample('D').mean()['pressure'].values

df_master = pd.DataFrame({'pressure': daily_pressure, 'precipitation': precipitation}, index=dates)

# ---  Fetch Real Global CO2 Data (NOAA Mauna Loa) ---
print("Downloading NOAA Mauna Loa CO2 data...")
url_co2 = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt"

try:
    df_co2 = pd.read_csv(url_co2, sep=r'\s+', comment='#',
                         names=['year', 'month', 'day', 'dec_year', 'co2'],
                         usecols=[0, 1, 2, 3, 4])
except Exception as e:
    print(f"Failed to fetch or parse NOAA data: {e}")
    sys.exit(1)

# Create a proper datetime index
df_co2['date'] = pd.to_datetime(df_co2[['year', 'month', 'day']])
df_co2.set_index('date', inplace=True)

# NOAA uses -999.99 for missing sensor days. Replace with NaN and interpolate.
df_co2['co2'] = df_co2['co2'].replace(-999.99, np.nan)
df_co2['co2'] = df_co2['co2'].interpolate(method='linear')

# Merge the CO2 data into our master dataframe based on exact dates
df_master = df_master.join(df_co2['co2'])

# If there are any missing days after the merge, forward-fill them to be safe
df_master['co2'] = df_master['co2'].ffill().bfill()

# --- Create the Presentation Plot ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Top: The Confounder (Real NOAA CO2)
ax1.plot(df_master.index, df_master['co2'], color='#E74C3C', linewidth=2.5)
ax1.set_title('Confounder ($U_t$): Real NOAA Mauna Loa Daily $CO_2$', fontsize=14, fontweight='bold')
ax1.set_ylabel('$CO_2$ (ppm)', fontsize=12)


# Middle: The Predictor (Real Pressure)
ax2.plot(df_master.index, df_master['pressure'], color='#4A90E2', linewidth=1)
ax2.set_title('Atmospheric Circulation ($X_t$): Real Munich Surface Pressure', fontsize=14, fontweight='bold')
ax2.set_ylabel('Pressure (hPa)', fontsize=12)

# Bottom: The Response (Real Rain)
ax3.plot(df_master.index, df_master['precipitation'], color='#2ECC71', linewidth=1)
ax3.set_title('Response ($Y_t$): Real Munich Daily Precipitation', fontsize=14, fontweight='bold')
ax3.set_ylabel('Rain (mm)', fontsize=12)
ax3.set_xlabel('Year', fontsize=12)

plt.tight_layout()
plt.savefig('munich_real_climate_data.pdf', format='pdf', bbox_inches='tight')