import fastf1
import pandas as pd
import os

print("Starting FastF1 Results Extraction...")

# Enable cache so we don't spam the servers
os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

years_to_fetch = [2024, 2025, 2026]
all_results = []

for year in years_to_fetch:
    print(f"\nFetching schedule for {year}...")
    schedule = fastf1.get_event_schedule(year)
    
    # Filter out pre-season testing (Round 0)
    races = schedule[schedule['RoundNumber'] > 0]
    
    for index, race in races.iterrows():
        round_num = race['RoundNumber']
        event_name = race['EventName']
        
        try:
            print(f" -> Extracting results for {year} {event_name}...")
            # Load the Race session ('R'). We set telemetry/weather to False to save RAM and time
            session = fastf1.get_session(year, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            
            # Extract the official race results dataframe
            results_df = session.results
            
            # Add identifying columns so we can filter it in the dashboard
            results_df['year'] = year
            results_df['race_name'] = f"{year} {event_name}"
            
            all_results.append(results_df)
            
        except Exception as e:
            # This handles future 2026 races that haven't happened yet
            print(f" -> Skipping {year} {event_name}: Data not available yet.")

# Combine all the races into one massive table
print("\nCombining data...")
final_recent_results = pd.concat(all_results, ignore_index=True)

# Select and rename only the columns we need for the dashboard
clean_results = final_recent_results[['year', 'race_name', 'FullName', 'GridPosition', 'Position', 'Points', 'Status']]
clean_results.columns = ['year', 'race_name', 'driver_name', 'grid', 'positionOrder', 'points', 'status']

# Save the filled data into your warehouse
clean_results.to_csv("for1sight_warehouse/recent_results.csv", index=False)
print("Success! Missing data from 2024-2026 has been saved to for1sight_warehouse/recent_results.csv")