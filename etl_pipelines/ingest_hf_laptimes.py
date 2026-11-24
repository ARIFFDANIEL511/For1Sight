import os
import json
import pandas as pd
from pathlib import Path

# Get the directory where the script is (etl_pipelines)
SCRIPT_DIR = Path(__file__).parent.resolve()

# Go UP one level to the main 'For1Sight Project' folder
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DATA_DIR = PROJECT_ROOT / "raw_hf_data"
WAREHOUSE_DIR = PROJECT_ROOT / "for1sight_warehouse"

print(f"🚀 Project Root: {PROJECT_ROOT}")
print(f"📂 Scanning Target: {RAW_DATA_DIR}")

WAREHOUSE_DIR.mkdir(exist_ok=True)

all_items = list(RAW_DATA_DIR.rglob("*"))
json_files = [f for f in all_items if f.is_file() and f.name.lower() == "laptimes.json"]

print(f"🎯 Found {len(json_files)} 'laptimes.json' files across 2024-2026. Processing...")

all_dfs = []
processed_count = 0
failed_count = 0

for file_path in json_files:
    try:
        rel_parts = file_path.relative_to(RAW_DATA_DIR).parts
        
        if len(rel_parts) < 5:
            continue
            
        year_str = rel_parts[0]
        event = rel_parts[1]
        session = rel_parts[2]
        driver = rel_parts[3]

        if not year_str.isdigit():
            continue

        year = int(year_str)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # PROPER DATAFRAME LOADING (No accidental transposing!)
        df = pd.DataFrame(data)

        if not df.empty:
            df['year'] = year
            df['event_name'] = event
            df['session_type'] = session
            df['driver_code'] = driver
            
            all_dfs.append(df)
            processed_count += 1
            
            if processed_count % 500 == 0:
                print(f"  ... successfully ingested {processed_count} / {len(json_files)} files ...")

    except Exception as e:
        failed_count += 1
        if failed_count <= 3:
            print(f"⚠️ Error parsing {file_path.name}: {e}")

if all_dfs:
    print(f"\nMerging {processed_count} driver session files into master DataFrame...")
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df = master_df.reset_index(drop=True)
    
    # Cast numeric columns cleanly
    numeric_cols = ['time', 'pos', 's1', 's2', 's3', 'lap', 'vfl', 'vi1', 'vi2', 'vst', 'life', 'stint']
    for col_name in numeric_cols:
        if col_name in master_df.columns:
            master_df[col_name] = pd.to_numeric(master_df[col_name], errors='coerce')
        
    # Cast object/mixed columns to string to resolve PyArrow schema conflicts
    for col in master_df.select_dtypes(include=['object']).columns:
        master_df[col] = master_df[col].astype(str)

    output_path = WAREHOUSE_DIR / "hf_laptimes.parquet"
    
    print(f"Compressing {len(master_df):,} total lap records to Big Data Parquet format...")
    master_df.to_parquet(
        output_path, 
        engine='pyarrow', 
        compression='snappy',
        partition_cols=['year', 'session_type']
    )
    
    print(f"\n✅ SUCCESS! Ingested {processed_count:,} driver sessions into Parquet.")
    print(f"📦 Data Warehouse file created at: {output_path.absolute()}")
else:
    print("\n❌ Failed to convert JSON files to DataFrame.")