import requests
import json
import os

print("Generating offline Driver Catalog for For1Sight Warehouse...")

# Fetch driver metadata once from OpenF1
url = "https://api.openf1.org/v1/drivers?session_key=latest"
try:
    response = requests.get(url, timeout=10).json()
    if not isinstance(response, list) or len(response) == 0:
        print("Latest session empty, trying fallback session 9158...")
        response = requests.get("https://api.openf1.org/v1/drivers?session_key=9158", timeout=10).json()
except Exception as e:
    print(f"Error fetching from API: {e}")
    response = []

# Process and standardize the drivers dataset
drivers_list = []
seen_numbers = set()

for d in response:
    num = d.get('driver_number')
    if num and num not in seen_numbers:
        seen_numbers.add(num)
        
        team = d.get('team_name', 'Unknown')
        color = d.get('team_colour', 'FFFFFF')
        
        # Apply custom team branding overrides
        if team in ["Kick Sauber", "Sauber"]:
            team = "Audi"
            color = "000000"
            
        driver_entry = {
            "driver_number": num,
            "first_name": d.get('first_name', ''),
            "last_name": d.get('last_name', ''),
            "name_acronym": d.get('name_acronym', 'F1'),
            "team_name": team,
            "team_colour": color,
            "country_code": d.get('country_code', 'F1')
        }
        drivers_list.append(driver_entry)

# Ensure target warehouse directory exists
os.makedirs("for1sight_warehouse", exist_ok=True)

# Save to warehouse catalog
catalog_path = "for1sight_warehouse/drivers_catalog.json"
with open(catalog_path, "w") as f:
    json.dump(drivers_list, f, indent=4)

print(f"Success! {len(drivers_list)} driver profiles saved to {catalog_path}")