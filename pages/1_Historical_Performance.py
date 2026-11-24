import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="Historical Performance", layout="wide")
st.title("🏁 Past Race Performances & Environment")
st.markdown("Analyze historical driver results, track details, and weather conditions.")

# 1. LOAD COMBINED API & KAGGLE DATA
@st.cache_data
def load_historical_data():
    try:
        recent_data = pd.read_csv("for1sight_warehouse/recent_results.csv")
        return recent_data
    except FileNotFoundError:
        st.error("Missing recent_results.csv. Please run update_recent_results.py first.")
        return pd.DataFrame()

df = load_historical_data()

if df.empty:
    st.warning("No data found in recent_results.csv. Please run update_recent_results.py.")
    st.stop()

# 2. FETCH WEATHER FROM OPENF1 API
@st.cache_data
def get_weather_data(session_key):
    url = f"https://api.openf1.org/v1/weather?session_key={session_key}"
    try:
        response = requests.get(url).json()
        if response:
            weather_df = pd.DataFrame(response)
            avg_temp = weather_df['air_temperature'].mean()
            avg_track = weather_df['track_temperature'].mean()
            rainfall = "Yes" if weather_df['rainfall'].sum() > 0 else "No"
            return avg_temp, avg_track, rainfall
    except:
        pass
    return None, None, "Unknown"

# 3. DASHBOARD UI LAYOUT
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Filters")
    available_years = sorted(df['year'].unique(), reverse=True)
    selected_year = st.selectbox("Select Season", available_years)
    year_data = df[df['year'] == selected_year]
    
    unique_races = year_data['race_name'].unique()
    selected_race = st.selectbox("Select Grand Prix", unique_races)
    
    race_specific_data = year_data[year_data['race_name'] == selected_race]
    available_drivers = race_specific_data['driver_name'].unique()
    selected_driver = st.selectbox("Select Driver", available_drivers)

with col2:
    st.subheader(f"Results for {selected_driver} - {selected_race}")
    
    driver_stats = race_specific_data[race_specific_data['driver_name'] == selected_driver]
    
    if not driver_stats.empty:
        grid_pos = int(driver_stats['grid'].iloc[0])
        finish_pos = int(driver_stats['positionOrder'].iloc[0])
        points = float(driver_stats['points'].iloc[0])
        status = str(driver_stats['status'].iloc[0]) # Fixed: uses 'status' instead of 'statusId'
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(label="Starting Grid Position", value=grid_pos)
        
        pos_change = grid_pos - finish_pos
        m2.metric(label="Final Position", value=finish_pos, delta=pos_change)
        m3.metric(label="Points Scored", value=points)
        m4.metric(label="Status", value=status)
        
        st.markdown("---")
        st.write("**Top 10 Finishers & Points**")
        top_10 = race_specific_data[race_specific_data['positionOrder'] <= 10].sort_values('positionOrder')
        fig = px.bar(
            top_10, 
            x='driver_name', 
            y='points', 
            color='driver_name',
            title=f"Points Distribution: {selected_race}"
        )
        st.plotly_chart(fig, use_container_width=True)

# 4. WEATHER & TRACK ENVIRONMENT
st.markdown("---")
st.subheader("Track Environment (OpenF1 API)")

demo_session_key = 9158 
avg_temp, avg_track, rainfall = get_weather_data(demo_session_key)

w1, w2, w3 = st.columns(3)
w1.metric("Avg Air Temperature", f"{avg_temp:.1f} °C" if avg_temp else "N/A")
w2.metric("Avg Track Temperature", f"{avg_track:.1f} °C" if avg_track else "N/A")
w3.metric("Rainfall Detected", rainfall)