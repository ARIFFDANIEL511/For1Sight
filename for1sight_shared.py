import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
import json
import os
import re
import unicodedata
import html
import streamlit.components.v1 as components
from pathlib import Path

# ==========================================
# 1. HELPER FUNCTIONS & ASSETS MAPPING
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "pages" else Path(__file__).resolve().parent
WAREHOUSE_DIR = PROJECT_ROOT / "for1sight_warehouse"
PARQUET_PATH = WAREHOUSE_DIR / "hf_laptimes.parquet"
# A default dark gray SVG silhouette placeholder for missing headshots
DEFAULT_HEADSHOT = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMDAgMjUwIiBmaWxsPSIjMzMzIj48Y2lyY2xlIGN4PSIxMDAiIGN5PSI3MCIgcj0iNTAiLz48cGF0aCBkPSJNMjAgMjUwaDE2MHYtNTBjMC00NC0zNi04MC04MC04MHMtODAgMzYtODAgODB2NTB6Ii8+PC9zdmc+"

def get_local_image_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as image_file:
            return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode()}"
    return ""

def clean_driver_filename(first_name, last_name, season):
    """Strips accents (Pérez -> perez), handles temporal folders, formats string."""
    full_name = f"{first_name}{last_name}".lower()
    full_name = unicodedata.normalize('NFKD', full_name).encode('ASCII', 'ignore').decode('utf-8')
    clean_name = re.sub(r'[^a-z0-9]', '', full_name)
    return f"assets/headshots/{season}/{clean_name}.png"

def get_team_logo(team_name):
    """Fuzzy matching to handle long API sponsor names (e.g., MoneyGram Haas)"""
    t = str(team_name).lower()
    if 'mclaren' in t: return "assets/logos/mclaren.png"
    if 'red bull' in t or 'rbr' in t: return "assets/logos/redbull.png"
    if 'ferrari' in t: return "assets/logos/ferrari.png"
    if 'mercedes' in t: return "assets/logos/mercedes.png"
    if 'aston' in t: return "assets/logos/astonmartin.png"
    if 'alpine' in t: return "assets/logos/alpine.png"
    if 'williams' in t: return "assets/logos/williams.png"
    if 'rb' in t or 'racing bulls' in t or 'alphatauri' in t: return "assets/logos/rb.png"
    if 'sauber' in t or 'alfa' in t or 'kick' in t or 'audi' in t: return "assets/logos/audi.png"
    if 'haas' in t: return "assets/logos/haas.png"
    if 'cadillac' in t: return "assets/logos/cadillac.png"
    return ""

def get_fallback_color(team_name):
    """Fallback hex colors if a historical team isn't in the 2026 catalog"""
    t = str(team_name).lower()
    if 'mclaren' in t: return "FF8000"
    if 'red bull' in t: return "3671C6"
    if 'ferrari' in t: return "E8002D"
    if 'mercedes' in t: return "27F4D2"
    if 'aston' in t: return "229971"
    if 'alpine' in t: return "0093CC"
    if 'williams' in t: return "64C4FF"
    if 'haas' in t: return "B6BABD"
    if 'sauber' in t or 'alfa' in t or 'kick' in t: return "52E252"
    if 'rb' in t or 'alphatauri' in t: return "6692FF"
    return "888888"

def get_chronological_events(df):
    if df.empty or 'event_name' not in df.columns: return []
    if 'lSD' in df.columns:
        temp_df = df.copy()
        temp_df['lSD_dt'] = pd.to_datetime(temp_df['lSD'], errors='coerce')
        event_dates = temp_df.groupby('event_name')['lSD_dt'].min()
        if not event_dates.isna().all(): return list(event_dates.sort_values().index)
    return list(pd.unique(df['event_name']))

# ==========================================
# 2. LOCAL DATA ENGINES
# ==========================================
@st.cache_data(show_spinner=False)
def _read_laps_cached(path_str, modified_ns):
    """Invalidate cached warehouse data whenever the Parquet file changes."""
    return pd.read_parquet(path_str, engine="pyarrow")

def load_parquet_laps(season):
    # DIAGNOSTIC 1: Check if the file still exists in the folder
    if not PARQUET_PATH.exists(): 
        st.sidebar.error(f"❌ FILE NOT FOUND: {PARQUET_PATH.absolute()}")
        st.sidebar.warning("Did the hf_laptimes.parquet file get moved or deleted?")
        return pd.DataFrame()
        
    try:
        df = _read_laps_cached(str(PARQUET_PATH), PARQUET_PATH.stat().st_mtime_ns).copy()
        
        # --- DIAGNOSTIC 2: The Data Inspector Sidebar ---
        with st.sidebar.expander("🛠️ Parquet Diagnostic Tool", expanded=False):
            st.write(f"**Total Rows in Parquet:** {len(df)}")
            st.write(f"**Available Columns:** {', '.join(df.columns)}")
            
            if 'year' in df.columns:
                unique_years = df['year'].dropna().unique()
                st.write(f"**Years Found:** {unique_years}")
            else:
                st.write("❌ 'year' column is MISSING from Parquet.")
                
            if 'session_type' in df.columns:
                sessions = df['session_type'].dropna().unique()
                st.write(f"**Sessions Found:** {sessions}")
            else:
                st.write("❌ 'session_type' column is MISSING.")
        # ------------------------------------------------
        
        if not df.empty and 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df[df['year'] == int(season)].copy()
            
        return df
    except Exception as e: 
        st.sidebar.error(f"🚨 Warehouse Read Error: {e}")
        return pd.DataFrame()
    
@st.cache_data
def load_warehouse_results():
    try: return pd.read_csv(WAREHOUSE_DIR / "recent_results.csv")
    except: return pd.DataFrame()

results_df = load_warehouse_results()

def calculate_driver_ratings(driver_last_name, season=2024):
    if results_df.empty: return {'Pace': 80, 'Qualifying': 80, 'Racecraft': 80, 'Consistency': 85, 'Experience': 75, 'Overall': 80}
    d_df = results_df[results_df['driver_name'].str.contains(driver_last_name, case=False, na=False)].copy()
    if 'year' in d_df.columns: d_df = d_df[d_df['year'] == int(season)]
    if len(d_df) == 0: return {'Pace': 74, 'Qualifying': 74, 'Racecraft': 75, 'Consistency': 80, 'Experience': 65, 'Overall': 74}
    
    total_races = len(d_df)
    avg_finish = d_df['positionOrder'].mean()
    valid_grid = d_df[d_df['grid'] > 0]
    avg_grid = valid_grid['grid'].mean() if len(valid_grid) > 0 else 10.0
    pos_delta = (d_df['grid'] - d_df['positionOrder']).mean()
    dnf_terms = ['Collision', 'Accident', 'Crash', 'Engine', 'Gearbox', 'Spun', 'Retire']
    dnf_count = d_df['status'].str.contains('|'.join(dnf_terms), case=False, na=False).sum()
    finish_rate = ((total_races - dnf_count) / total_races) * 100 if total_races > 0 else 85.0
    
    pace = max(55, min(99, int(100 - (avg_finish * 2.3))))
    qualifying = max(55, min(99, int(100 - (avg_grid * 2.3))))
    racecraft = max(55, min(99, int(75 + (pos_delta * 3.5))))
    consistency = max(55, min(99, int(finish_rate)))
    experience = max(55, min(99, int(65 + (total_races * 1.5))))
    overall = int(np.mean([pace, qualifying, racecraft, consistency, experience]))
    return {'Pace': pace, 'Qualifying': qualifying, 'Racecraft': racecraft, 'Consistency': consistency, 'Experience': experience, 'Overall': overall}

# ==========================================
# DRIVER BIOGRAPHY + SEASON FINISHING TREND
# ==========================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_driver_biography(season, given_name, family_name):
    """Jolpica season driver catalogue; avoids confusing historical and current teams."""
    try:
        response = requests.get(
            f"https://api.jolpi.ca/ergast/f1/{int(season)}/drivers.json",
            params={"limit": 100}, timeout=10,
        )
        response.raise_for_status()
        records = response.json().get("MRData", {}).get("DriverTable", {}).get("Drivers", [])
        for item in records:
            if (item.get("familyName", "").casefold() == family_name.casefold()
                and item.get("givenName", "").casefold() == given_name.casefold()):
                return item
        for item in records:
            if item.get("familyName", "").casefold() == family_name.casefold():
                return item
    except (requests.RequestException, ValueError, KeyError):
        pass
    return {}


def build_driver_finish_trend(season, family_name, team_color, teammate_family=None):
    """Official classified positions for the selected driver and their teammate."""
    races = fetch_api_results(season)
    fig = go.Figure()
    primary_rows = []
    for surname, name, color, is_primary in [
        (teammate_family, f"{teammate_family} · teammate", "#A0AAB8", False),
        (family_name, f"{family_name} · selected", team_color, True),
    ]:
        if not surname:
            continue
        rows = []
        for race in races:
            result = next((r for r in race.get("Results", []) if
                r.get("Driver", {}).get("familyName", "").casefold() == surname.casefold()), None)
            if not result:
                continue
            try:
                rnd, pos = int(race["round"]), int(result["position"])
            except (ValueError, TypeError, KeyError):
                continue
            if rnd < 1 or pos < 1:
                continue
            rows.append(dict(round=rnd, position=pos, event=race.get("raceName", ""),
                             points=float(result.get("points") or 0), status=result.get("status", "")))
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values("round").drop_duplicates("round")
        if is_primary:
            primary_rows = rows
        fig.add_trace(go.Scatter(
            x=df["round"], y=df["position"], mode="lines+markers", name=name,
            line=dict(color=color, width=3.6 if is_primary else 2, dash="solid" if is_primary else "dot"),
            opacity=1 if is_primary else 0.48,
            marker=dict(size=9 if is_primary else 6, color=color),
            customdata=df[["event", "points", "status"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>Round %{x}<br>Classified P%{y}<br>Points: %{customdata[1]:g}<br>Status: %{customdata[2]}<extra></extra>",
        ))
        if is_primary:
            podium = df[df["position"] <= 3]
            if not podium.empty:
                fig.add_trace(go.Scatter(
                    x=podium["round"], y=podium["position"], mode="markers",
                    name="Podium (selected)", marker=dict(color="#FFD166", size=12,
                    line=dict(color="#1A1D27", width=2)),
                    hovertemplate="Round %{x} · Podium P%{y}<extra></extra>",
                ))
    if not primary_rows:
        return None, 0
    fig.update_layout(
        title=dict(text="Race-by-Race Classified Position", font=dict(size=17)),
        height=370, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,24,33,0.35)",
        font=dict(color="#E8EDF5"), margin=dict(l=40,r=25,t=55,b=50),
        xaxis=dict(title="Grand Prix round", dtick=1, gridcolor="#303849"),
        yaxis=dict(title="Classified position · lower is better", autorange="reversed", dtick=2, gridcolor="#303849"),
        showlegend=True, legend=dict(orientation="h", y=1.13, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig, len(primary_rows)

# ==========================================
# ADDITIONAL INDIVIDUAL DRIVER VISUAL ANALYTICS
# ==========================================
def _driver_race_results(season, family_name):
    """Official race results; do not mistake classified position for a completed race."""
    records = []
    for race in fetch_api_results(season):
        for result in race.get("Results", []):
            if result.get("Driver", {}).get("familyName", "").casefold() != str(family_name).casefold():
                continue
            try:
                rnd = int(race["round"])
                finish = int(result["position"])
                grid = int(result.get("grid") or 0)
                points = float(result.get("points") or 0)
            except (ValueError, TypeError, KeyError):
                continue
            records.append({"round": rnd, "event": race.get("raceName", f"Round {rnd}"),
                            "finish": finish, "grid": grid, "points": points,
                            "status": result.get("status", "Unknown")})
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("round").drop_duplicates("round")


def _style_driver_figure(fig, title, team_color, height=355):
    fig.update_layout(title=dict(text=title, font=dict(size=17)), height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,24,33,0.35)",
                      font=dict(color="#E8EDF5"), margin=dict(l=40, r=25, t=60, b=55),
                      hoverlabel=dict(bgcolor="#1B2230", font_color="#FFFFFF"),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor="#303849", zeroline=False)
    fig.update_yaxes(gridcolor="#303849", zeroline=False)
    return fig


def build_points_per_race_chart(season, family_name, team_color):
    df = _driver_race_results(season, family_name)
    if df.empty:
        return None
    fig = go.Figure(go.Bar(x=df["round"], y=df["points"],
        marker=dict(color=["#FFD166" if p >= 25 else team_color for p in df["points"]]),
        customdata=df[["event", "finish", "status"]].to_numpy(),
        hovertemplate="<b>%{customdata[0]}</b><br>Round %{x}<br>Points: %{y:g}<br>Classified P%{customdata[1]}<br>%{customdata[2]}<extra></extra>"))
    _style_driver_figure(fig, "Championship Points by Grand Prix", team_color)
    fig.update_xaxes(title="Championship round", dtick=1)
    fig.update_yaxes(title="Points scored", rangemode="tozero")
    return fig


def build_grid_vs_finish_chart(season, family_name, team_color):
    df = _driver_race_results(season, family_name)
    if df.empty:
        return None, 0
    df = df[df["grid"] > 0].copy()  # Pit-lane/unknown starts cannot be compared reliably.
    if df.empty:
        return None, 0
    df["change"] = df["grid"] - df["finish"]
    fig = go.Figure(go.Bar(x=df["round"], y=df["change"],
        marker=dict(color=[team_color if v >= 0 else "#F36C79" for v in df["change"]]),
        customdata=df[["event", "grid", "finish", "status"]].to_numpy(),
        hovertemplate="<b>%{customdata[0]}</b><br>Grid P%{customdata[1]} → Classified P%{customdata[2]}<br>Change: %{y:+d}<br>%{customdata[3]}<extra></extra>"))
    fig.add_hline(y=0, line_color="#8993A3", line_width=1)
    _style_driver_figure(fig, "Grid-to-Classification Position Change", team_color)
    fig.update_xaxes(title="Championship round", dtick=1)
    fig.update_yaxes(title="Positions gained (+) / lost (−)", zeroline=True)
    return fig, len(df)


def _numeric_lap_seconds(series):
    """Accept numeric seconds or pandas-compatible duration strings."""
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna() & series.notna()
    if missing.any():
        numeric.loc[missing] = pd.to_timedelta(series.loc[missing].astype(str), errors="coerce").dt.total_seconds()
    return numeric


def _lap_data_for_consistency(season):
    df = load_parquet_laps(season)
    needed = {"driver_code", "event_name", "time"}
    if df.empty or not needed.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    if "session_type" in df.columns:
        # Restrict to race laps so practice/qualifying do not distort the distribution.
        session = df["session_type"].astype(str).str.lower().str.strip()
        df = df[session.isin(["race", "r"]) | session.str.contains(r"^race\b", regex=True)]
    if df.empty:
        return df
    df["lap_seconds"] = _numeric_lap_seconds(df["time"])
    df = df[df["lap_seconds"].between(35, 300)].copy()
    for flag in ("del", "iacc"):
        if flag in df.columns:
            bad = df[flag].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
            df = df[~bad]
    return df


def show_lap_consistency_chart(season, driver, team_drivers, team_color):
    df = _lap_data_for_consistency(season)
    if df.empty:
        st.info("Lap-time consistency requires race laps in the local Parquet warehouse with driver_code, event_name and time fields.")
        return
    driver_code = str(driver.get("name_acronym") or driver["last_name"][:3]).upper()
    peers = [d for d in team_drivers if d["last_name"] != driver["last_name"] and d.get("role", "main") == "main"]
    teammate = peers[0] if peers else None
    teammate_code = str(teammate.get("name_acronym") or teammate["last_name"][:3]).upper() if teammate else None
    available_codes = set(df["driver_code"].astype(str).str.upper())
    if driver_code not in available_codes:
        st.info(f"No valid race lap times were found for {driver_code} in the {season} warehouse.")
        return
    subset = df[df["driver_code"].astype(str).str.upper() == driver_code]
    events = get_chronological_events(subset)
    if not events:
        st.info("No race events were found for the selected driver.")
        return
    selected_event = st.selectbox("Choose Grand Prix", events, key=f"lap_consistency_{season}_{driver_code}")
    event_df = df[df["event_name"] == selected_event].copy()
    fig = go.Figure()
    for code, name, color in [(driver_code, f"{driver['first_name']} {driver['last_name']}", team_color),
                              (teammate_code, f"{teammate['first_name']} {teammate['last_name']}" if teammate else "", "#B7C4D7")]:
        if not code:
            continue
        laps = event_df[event_df["driver_code"].astype(str).str.upper() == code]["lap_seconds"].dropna()
        if len(laps) < 3:
            continue
        # Robustly remove pit/in-out and unusually slow laps without implying all are invalid.
        cutoff = laps.quantile(0.75) + 1.5 * (laps.quantile(0.75) - laps.quantile(0.25))
        laps = laps[laps <= cutoff]
        if len(laps) < 3:
            continue
        fig.add_trace(go.Box(y=laps, name=name, marker_color=color,
                             boxpoints="outliers", jitter=0.3,
                             hovertemplate="Lap time %{y:.3f} s<extra>%{fullData.name}</extra>"))
    if not fig.data:
        st.info("This Grand Prix has too few valid laps for a meaningful distribution.")
        return
    _style_driver_figure(fig, f"Race Lap-Time Consistency — {selected_event}", team_color, height=390)
    fig.update_yaxes(title="Lap time (seconds); lower is faster")
    st.plotly_chart(fig, width="stretch")
    st.caption("Box = middle 50% of recorded race lap times; centre line = median. Extreme slow laps are filtered using the IQR rule. Tyre compounds, weather, fuel load and safety-car conditions are not controlled; this is a descriptive comparison, not a pure pace ranking.")

# ==========================================
# 3. TEMPORAL ROSTER ENGINE (CACHE-ASIDE API)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_api_standings(season):
    url = f"https://api.jolpi.ca/ergast/f1/{season}/driverStandings.json"
    try:
        res = requests.get(url, timeout=10).json()
        return res['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
    except Exception:
        return []

def _results_cache_path(season):
    return WAREHOUSE_DIR / f"race_results_{int(season)}.json"


def _read_saved_results(season):
    """Read a complete season snapshot; reject partial/legacy snapshots."""
    path = _results_cache_path(season)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("complete") is True and isinstance(payload.get("races"), list):
            return payload["races"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def _download_all_race_results(season):
    """Follow Jolpica pagination; its limit counts driver result rows, not races."""
    base_url = f"https://api.jolpi.ca/ergast/f1/{int(season)}/results.json"
    offset = 0
    page_size = 100
    races_by_round = {}
    expected_total = None
    while True:
        response = requests.get(base_url, params={"limit": page_size, "offset": offset}, timeout=25)
        response.raise_for_status()
        mrdata = response.json()["MRData"]
        total = int(mrdata["total"])
        limit = int(mrdata["limit"])
        current_offset = int(mrdata["offset"])
        if limit <= 0 or current_offset != offset:
            raise ValueError("Unexpected pagination metadata from Jolpica")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise ValueError("Results changed during download; retry the refresh")
        for race in mrdata.get("RaceTable", {}).get("Races", []):
            key = int(race["round"])
            if key not in races_by_round:
                races_by_round[key] = {k: v for k, v in race.items() if k != "Results"}
                races_by_round[key]["Results"] = []
            races_by_round[key]["Results"].extend(race.get("Results", []))
        offset += limit
        if offset >= total:
            break
    # Result pagination can split a race across multiple pages.
    # Deduplicate drivers within each round without dropping race data.
    races = []
    for rnd in sorted(races_by_round):
        race = races_by_round[rnd]
        unique = {}
        for result in race["Results"]:
            driver = result.get("Driver", {})
            driver_id = driver.get("driverId") or driver.get("code")
            if driver_id:
                unique[driver_id] = result
        race["Results"] = list(unique.values())
        races.append(race)
    return races


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_api_results(season, refresh=False):
    """Offline-first complete season results; refresh=True forces an API update."""
    if not refresh:
        cached = _read_saved_results(season)
        if cached is not None:
            return cached
    try:
        races = _download_all_race_results(season)
        WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
        path = _results_cache_path(season)
        temp = path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump({"season": int(season), "complete": True,
                       "source": "Jolpica Ergast-compatible API",
                       "races": races}, f, ensure_ascii=False, indent=2)
        temp.replace(path)
        return races
    except (requests.RequestException, ValueError, KeyError, TypeError, OSError) as exc:
        cached = _read_saved_results(season)
        if cached is not None:
            return cached
        st.warning(f"Could not load season {season} race results: {exc}")
        return []


def get_base_driver_catalog():
    catalog_path = WAREHOUSE_DIR / "drivers_catalog.json"
    if catalog_path.exists():
        with open(catalog_path, "r") as f: 
            try:
                data = json.load(f)
                if data: return data
            except: pass
    try:
        res = requests.get("https://api.openf1.org/v1/drivers?session_key=latest", timeout=10).json()
        if isinstance(res, list) and len(res) > 0:
            with open(catalog_path, "w") as f: json.dump(res, f)
            return res
    except: pass
    return []

@st.cache_data
def get_season_roster(season):
    roster_path = WAREHOUSE_DIR / f"rosters_{season}.json"
    if roster_path.exists():
        try:
            with open(roster_path, "r") as f: 
                data = json.load(f)
                if data: return data
        except: pass

    with st.spinner(f"Loading {season} historical roster from Jolpica..."):
        results = fetch_api_results(season)
        
    if not results:
        return {}
    
    catalog = get_base_driver_catalog()
    driver_max_round = {}
    team_drivers_info = {}
    
    for race in results:
        round_no = int(race['round'])
        for r in race['Results']:
            team_id = r['Constructor']['name']
            d_id = r['Driver']['familyName']
            
            if team_id not in driver_max_round:
                driver_max_round[team_id] = {}
                team_drivers_info[team_id] = {}
            
            driver_max_round[team_id][d_id] = max(driver_max_round[team_id].get(d_id, 0), round_no)
            
            if d_id not in team_drivers_info[team_id]:
                cat_match = next((d for d in catalog if d['last_name'].lower() == d_id.lower()), None)
                team_drivers_info[team_id][d_id] = {
                    "first_name": r['Driver']['givenName'],
                    "last_name": d_id,
                    "driver_number": r['number'],
                    "name_acronym": r['Driver'].get('code', d_id[:3].upper()),
                    "team_name": team_id,
                    "team_colour": cat_match.get('team_colour', get_fallback_color(team_id)) if cat_match else get_fallback_color(team_id),
                    "headshot_url": cat_match.get('headshot_url', '') if cat_match else ""
                }
                
    teams_roster = {}
    for team, drivers in driver_max_round.items():
        sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)
        main_drivers = []
        ghost_drivers = []
        
        for i, (d_id, rnd) in enumerate(sorted_drivers):
            d_info = team_drivers_info[team][d_id]
            if i < 2:
                d_info['role'] = 'main'
                main_drivers.append(d_info)
            else:
                d_info['role'] = 'ghost'
                ghost_drivers.append(d_info)
        
        teams_roster[team] = main_drivers + ghost_drivers
        
    try:
        WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
        with open(roster_path, "w", encoding="utf-8") as f:
            json.dump(teams_roster, f)
    except OSError as exc:
        st.warning(f"Roster loaded, but could not save local cache: {exc}")
        
    return teams_roster

# ==========================================
# 4. CHART BUILDERS
# ==========================================
def get_driver_season_summary(driver_last_name, season):
    standings = fetch_api_standings(season)
    results = fetch_api_results(season)
    
    wdc, points, wins, podiums, poles, dnfs, gps = "N/A", 0, 0, 0, 0, 0, 0
    
    for s in standings:
        if driver_last_name.lower() in s['Driver']['familyName'].lower():
            wdc = f"P{s['position']}"
            pts = float(s['points'])
            points = int(pts) if pts.is_integer() else pts
            wins = int(s['wins'])
            break
            
    for race in results:
        for r in race['Results']:
            if driver_last_name.lower() in r['Driver']['familyName'].lower():
                gps += 1  # Count every race entered this season
                if int(r.get('position', 100)) <= 3:
                    podiums += 1
                if int(r.get('grid', 100)) == 1:
                    poles += 1
                status = r.get('status', '').lower()
                if 'finished' not in status and 'lap' not in status:
                    dnfs += 1
                    
    return {'wdc': wdc, 'points': points, 'wins': wins, 'podiums': podiums, 'poles': poles, 'dnfs': dnfs, 'gps': gps}

def calculate_h2h_stats(d1_name, d2_name, season):
    """Calculates H2H stats dynamically using the cached API results instead of the CSV."""
    results = fetch_api_results(season)
    if not results: return None
    
    q1, q2, r1, r2, pts1, pts2, pod1, pod2 = 0, 0, 0, 0, 0, 0, 0, 0
    
    for race in results:
        d1_res = next((r for r in race['Results'] if d1_name.lower() in r['Driver']['familyName'].lower()), None)
        d2_res = next((r for r in race['Results'] if d2_name.lower() in r['Driver']['familyName'].lower()), None)
        
        if d1_res and d2_res:
            g1, g2 = int(d1_res.get('grid', 100)), int(d2_res.get('grid', 100))
            if g1 < g2 and g1 != 0: q1 += 1
            elif g2 < g1 and g2 != 0: q2 += 1
            
            p1, p2 = int(d1_res.get('position', 100)), int(d2_res.get('position', 100))
            if p1 < p2: r1 += 1
            elif p2 < p1: r2 += 1
            
        if d1_res:
            pts1 += float(d1_res.get('points', 0))
            if int(d1_res.get('position', 100)) <= 3: pod1 += 1
        if d2_res:
            pts2 += float(d2_res.get('points', 0))
            if int(d2_res.get('position', 100)) <= 3: pod2 += 1
            
    return {'quali': (q1, q2), 'race': (r1, r2), 'points': (int(pts1), int(pts2)), 'podiums': (pod1, pod2)}

# [Keep the rest of the existing chart building functions below this point: get_top_tracks, build_dnf_pie_chart, etc.]
def get_top_tracks(driver_last_name):
    if results_df.empty: return []
    d_df = results_df[results_df['driver_name'].str.contains(driver_last_name, case=False, na=False)]
    if d_df.empty: return []
    race_col = 'name' if 'name' in d_df.columns else 'event_name' if 'event_name' in d_df.columns else None
    if not race_col: return []
    finishers = d_df[d_df['positionOrder'] < 20]
    track_stats = finishers.groupby(race_col).agg(avg_finish=('positionOrder', 'mean'), races=('positionOrder', 'count')).reset_index()
    track_stats = track_stats[track_stats['races'] >= 2]
    return track_stats.sort_values(['avg_finish', 'races'], ascending=[True, False]).head(3).to_dict('records')

def build_dnf_pie_chart(driver_last_name):
    if results_df.empty: return None
    d_df = results_df[results_df['driver_name'].str.contains(driver_last_name, case=False, na=False)]
    if d_df.empty: return None
    dnf_df = d_df[~d_df['status'].str.lower().str.contains('finished|lap', na=False)]
    if dnf_df.empty: return None
    def classify_dnf(status):
        s = str(status).lower()
        if any(w in s for w in ['crash', 'accident', 'collision']): return 'Crash / Collision'
        if any(w in s for w in ['engine', 'gearbox', 'hydraulics', 'electrical', 'power', 'brake']): return 'Mechanical Failure'
        return 'Other (Spin / Retire)'
    dnf_df['Category'] = dnf_df['status'].apply(classify_dnf)
    counts = dnf_df['Category'].value_counts().reset_index(); counts.columns = ['Category', 'Count']
    fig = px.pie(counts, names='Category', values='Count', color='Category', color_discrete_map={'Crash / Collision': '#FF4B4B', 'Mechanical Failure': '#FFA500', 'Other (Spin / Retire)': '#A0AAB8'})
    fig.update_layout(title=dict(text="💥 Historical DNF Breakdown", font=dict(color="#FFF", size=16)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#fff"), margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.2))
    return fig

def build_theoretical_lap_chart(season, driver_code, team_color):
    df = load_parquet_laps(season)
    if df.empty or 's1' not in df.columns: return None
    q_df = df[df['driver_code'].astype(str).str.upper() == str(driver_code).upper()].copy()
    if 'session_type' in q_df.columns: q_df = q_df[q_df['session_type'].astype(str).str.lower().str.contains('qualifying')]
    q_df[['time', 's1', 's2', 's3']] = q_df[['time', 's1', 's2', 's3']].apply(pd.to_numeric, errors='coerce')
    q_df = q_df.dropna(subset=['time', 's1', 's2', 's3'])
    if q_df.empty: return None
    best_lap_row = q_df.loc[q_df['time'].idxmin()]
    best_event = best_lap_row['event_name']
    event_df = q_df[q_df['event_name'] == best_event]
    min_s1, min_s2, min_s3 = event_df['s1'].min(), event_df['s2'].min(), event_df['s3'].min()
    fig = go.Figure()
    fig.add_trace(go.Bar(y=['Actual Fastest Lap', 'Theoretical Best Lap'], x=[best_lap_row['s1'], min_s1], name='Sector 1', orientation='h', marker=dict(color='#FF1801')))
    fig.add_trace(go.Bar(y=['Actual Fastest Lap', 'Theoretical Best Lap'], x=[best_lap_row['s2'], min_s2], name='Sector 2', orientation='h', marker=dict(color='#E10600')))
    fig.add_trace(go.Bar(y=['Actual Fastest Lap', 'Theoretical Best Lap'], x=[best_lap_row['s3'], min_s3], name='Sector 3', orientation='h', marker=dict(color='#A60000')))
    fig.update_layout(barmode='stack', title=dict(text=f"⏱️ Sector Perfect Stacking ({best_event})", font=dict(color="#FFF", size=16)), xaxis=dict(title="Time (Seconds)", gridcolor="#262B36"), yaxis=dict(tickfont=dict(color="#FFF", size=13)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,24,33,0.4)", font=dict(color="#ffffff"), height=250, margin=dict(l=10, r=20, t=50, b=40), legend=dict(orientation="h", y=1.15))
    return fig

def build_cumulative_points_chart(season, driver1_last, driver2_last, team_color, driver_colors_map):
    if results_df.empty or 'year' not in results_df.columns: return None
    season_df = results_df[results_df['year'] == int(season)].copy()
    if season_df.empty: return None
    season_df['points'] = pd.to_numeric(season_df['points'], errors='coerce').fillna(0)
    if 'round' in season_df.columns: season_df = season_df.sort_values(['driver_name', 'round'])
    season_df['cum_points'] = season_df.groupby('driver_name')['points'].cumsum()
    fig = go.Figure()
    all_drivers = season_df['driver_name'].unique()
    d1_match = [d for d in all_drivers if driver1_last.lower() in d.lower()]
    d2_match = [d for d in all_drivers if driver2_last.lower() in d.lower()]
    d1_full = d1_match[0] if d1_match else None
    d2_full = d2_match[0] if d2_match else None
    for driver in all_drivers:
        if driver not in [d1_full, d2_full]:
            d_data = season_df[season_df['driver_name'] == driver]
            x_vals = d_data['round'] if 'round' in d_data.columns else list(range(1, len(d_data)+1))
            d_color = "#666666"
            for last_name, color in driver_colors_map.items():
                if last_name.lower() in driver.lower(): d_color = color; break
            fig.add_trace(go.Scatter(x=x_vals, y=d_data['cum_points'], mode='lines', name=driver, line=dict(color=d_color, width=2.5), opacity=0.55, showlegend=False))
    if d1_full:
        d1_data = season_df[season_df['driver_name'] == d1_full]
        x_vals1 = d1_data['round'] if 'round' in d1_data.columns else list(range(1, len(d1_data)+1))
        fig.add_trace(go.Scatter(x=x_vals1, y=d1_data['cum_points'], mode='lines+markers', name=d1_full, line=dict(color=team_color, width=5), marker=dict(size=9, color=team_color, symbol='circle')))
    if d2_full:
        d2_data = season_df[season_df['driver_name'] == d2_full]
        x_vals2 = d2_data['round'] if 'round' in d2_data.columns else list(range(1, len(d2_data)+1))
        contrast_color = "#FFD700" if team_color.lower() in ['#00e5ff', '#000000'] else "#00F0FF"
        fig.add_trace(go.Scatter(x=x_vals2, y=d2_data['cum_points'], mode='lines+markers', name=d2_full, line=dict(color=contrast_color, width=4, dash='solid'), marker=dict(size=9, color=contrast_color, symbol='diamond')))
    fig.update_layout(title=dict(text=f"📈 {season} Cumulative Championship Points Progression", font=dict(size=18, color="#FFFFFF")), xaxis=dict(title="Grand Prix Round", gridcolor="#262B36", showgrid=True, dtick=1), yaxis=dict(title="Cumulative Points", gridcolor="#262B36", showgrid=True), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20, 24, 33, 0.4)", font=dict(color="#ffffff"), height=460, margin=dict(l=40, r=40, t=50, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
    return fig

def build_qualifying_delta_chart(season, d1_code, d2_code, team_color):
    df = load_parquet_laps(season)
    
    # Diagnostic checks
    if df.empty: 
        st.info(f"ℹ️ Qualifying telemetry has not been ingested into the Parquet warehouse yet for the {season} season.")
        return None
    if 'time' not in df.columns:
        st.warning(f"⚠️ Telemetry data found for {season}, but the 'time' metric is missing. The Parquet file might be corrupted.")
        return None
        
    q_df = df[df['session_type'].astype(str).str.lower().str.contains('qualifying', na=False)].copy() if 'session_type' in df.columns else df.copy()
    if q_df.empty: return None
    q_df['driver_code_upper'] = q_df['driver_code'].astype(str).str.upper().str.strip()
    d1_u, d2_u = str(d1_code).upper().strip(), str(d2_code).upper().strip()
    q_df = q_df[q_df['driver_code_upper'].isin([d1_u, d2_u])].copy()
    q_df['time'] = pd.to_numeric(q_df['time'], errors='coerce')
    q_df = q_df.dropna(subset=['time'])
    if q_df.empty: return None
    best_laps = q_df.groupby(['event_name', 'driver_code_upper'])['time'].min().reset_index()
    pivoted = best_laps.pivot(index='event_name', columns='driver_code_upper', values='time')
    if d1_u not in pivoted.columns or d2_u not in pivoted.columns: return None
    pivoted['delta'] = pivoted[d1_u] - pivoted[d2_u]
    pivoted = pivoted.dropna(subset=['delta'])
    chrono_order = get_chronological_events(q_df)
    ordered_events = [ev for ev in chrono_order if ev in pivoted.index]
    if not ordered_events: return None
    pivoted = pivoted.loc[ordered_events]
    events, deltas = list(pivoted.index), list(pivoted['delta'])
    colors, hover_texts = [], []
    contrast_color = "#FFD700" if team_color.lower() in ['#00e5ff', '#000000'] else "#00F0FF"
    for ev, d in zip(events, deltas):
        if d < 0:
            colors.append(team_color)
            hover_texts.append(f"<b>{ev}</b><br>{d1_u} faster by {abs(d):.3f}s")
        else:
            colors.append(contrast_color)
            hover_texts.append(f"<b>{ev}</b><br>{d2_u} faster by {abs(d):.3f}s")
    fig = go.Figure(go.Bar(y=events, x=deltas, orientation='h', marker=dict(color=colors, line=dict(color='#141821', width=1)), hovertext=hover_texts, hoverinfo='text'))
    max_abs = max(abs(min(deltas, default=0.5)), abs(max(deltas, default=0.5)), 0.4) + 0.1
    fig.update_layout(title=dict(text=f"⏱️ {season} Qualifying Pace Delta ({d1_u} vs {d2_u})", font=dict(size=18, color="#FFFFFF")), xaxis=dict(title=dict(text=f"← {d1_u} Faster  |  0.000s  |  {d2_u} Faster →", font=dict(size=13, color="#FFFFFF")), gridcolor="#262B36", showgrid=True, zeroline=True, zerolinecolor="#FFFFFF", zerolinewidth=2, range=[-max_abs, max_abs]), yaxis=dict(autorange="reversed", gridcolor="#262B36"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20, 24, 33, 0.4)", font=dict(color="#ffffff"), height=max(450, len(events) * 25), margin=dict(l=40, r=40, t=50, b=50), showlegend=False)
    return fig

def build_race_trajectory_chart(season, d1_code, d2_code, team_color):
    df = load_parquet_laps(season)
    
    if df.empty: 
        st.info(f"ℹ️ Race position telemetry has not been ingested into the Parquet warehouse yet for the {season} season.")
        return None
    if 'pos' not in df.columns or 'lap' not in df.columns:
        st.warning(f"⚠️ Telemetry data found for {season}, but the 'pos' or 'lap' metrics are missing. The Parquet file might be corrupted.")
        return None
        
    r_df = df[df['session_type'].astype(str).str.lower().str.contains('race', na=False)].copy() if 'session_type' in df.columns else df.copy()
    if r_df.empty: return None
    r_df['driver_code_upper'] = r_df['driver_code'].astype(str).str.upper().str.strip()
    d1_u, d2_u = str(d1_code).upper().strip(), str(d2_code).upper().strip()
    r_df = r_df[r_df['driver_code_upper'].isin([d1_u, d2_u])].copy()
    r_df['pos'], r_df['lap'] = pd.to_numeric(r_df['pos'], errors='coerce'), pd.to_numeric(r_df['lap'], errors='coerce')
    r_df = r_df.dropna(subset=['pos', 'lap'])
    if r_df.empty: return None
    fig = go.Figure()
    contrast_color = "#FFD700" if team_color.lower() in ['#00e5ff', '#000000'] else "#00F0FF"
    events = get_chronological_events(r_df)
    for ev in events:
        ev_df = r_df[r_df['event_name'] == ev]
        for d_code, accent_color in [(d1_u, team_color), (d2_u, contrast_color)]:
            drv_df = ev_df[ev_df['driver_code_upper'] == d_code].sort_values('lap')
            if len(drv_df) > 0:
                grid_pos, finish_pos = int(drv_df.iloc[0]['pos']), int(drv_df.iloc[-1]['pos'])
                pos_gained = grid_pos - finish_pos
                if pos_gained > 0: line_color, status_text = "#00E676", f"Gained +{pos_gained} Pos"
                elif pos_gained < 0: line_color, status_text = "#FF5252", f"Lost {abs(pos_gained)} Pos"
                else: line_color, status_text = "#A0AAB8", f"Held Position"
                fig.add_trace(go.Scatter(x=[ev, ev], y=[grid_pos, finish_pos], mode='lines+markers', line=dict(color=line_color, width=3.5), marker=dict(size=[8, 11], color=[accent_color, line_color], symbol=['circle', 'circle']), showlegend=False, hovertemplate=f"<b>{d_code} - {ev}</b><br>Start: P{grid_pos} | Finish: P{finish_pos}<br>{status_text}<extra></extra>"))
    fig.update_layout(title=dict(text=f"📉 {season} Race Trajectory (Grid → Finish)", font=dict(size=18, color="#FFFFFF")), xaxis=dict(title="Grand Prix Event", gridcolor="#262B36", tickangle=-45), yaxis=dict(title="Position", gridcolor="#262B36", autorange="reversed", tickmode="linear", tick0=1, dtick=2, range=[20.5, 0.5]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20, 24, 33, 0.4)", font=dict(color="#ffffff"), height=480, margin=dict(l=40, r=40, t=50, b=80), showlegend=False)
    return fig

# ==========================================
# WDC STANDINGS — INDIVIDUAL DRIVER SIDEBAR CHART
# ==========================================
def build_wdc_standings_chart(season, selected_driver, show_all=False):
    """Season driver standings with team colours and a strong selected-driver highlight."""
    standings = fetch_api_standings(season)
    if not standings:
        return None, 0, False

    records = []
    for entry in standings:
        person = entry.get("Driver", {})
        constructors = entry.get("Constructors") or []
        constructor = constructors[-1].get("name", "") if constructors else ""
        family = person.get("familyName", "Unknown")
        try:
            points = float(entry.get("points", 0))
            rank = int(entry.get("position", len(records) + 1))
        except (TypeError, ValueError):
            continue
        records.append({"name": family, "rank": rank, "points": points,
                        "color": "#" + get_fallback_color(constructor),
                        "constructor": constructor,
                        "selected": family.casefold() == str(selected_driver).casefold()})
    records.sort(key=lambda x: (x["rank"], -x["points"]))
    if not records:
        return None, 0, False

    total = len(records)
    visible = records if show_all else records[:10]
    selected_below = False
    if not show_all and not any(x["selected"] for x in visible):
        chosen = next((x for x in records if x["selected"]), None)
        if chosen:
            visible = visible + [chosen]
            selected_below = True

    max_points = max(x["points"] for x in records)
    axis_max = max(1, max_points * 1.10)
    fig = go.Figure()
    for item in reversed(visible):
        accent = item["color"]
        is_selected = item["selected"]
        # Other constructors keep their identity; the current driver gets a bright outline.
        label = f"{item['name'].upper()}  ·  {item['points']:g} PTS"
        fig.add_trace(go.Bar(
            x=[item["points"]], y=[f"P{item['rank']}  {item['name'][:3].upper()}"], orientation="h",
            width=0.72, marker=dict(color=accent, opacity=1 if is_selected else 0.72,
                                    line=dict(color="#FFFFFF" if is_selected else accent,
                                              width=2.6 if is_selected else 0)),
            text=[label], textposition="inside" if item["points"] >= axis_max * 0.23 else "outside",
            insidetextanchor="start",
            textfont=dict(color="#FFFFFF" if item["points"] >= axis_max * 0.23 else "#E8EDF5",
                          size=11 if is_selected else 10,
                          family="Arial Black" if is_selected else "Arial"),
            constraintext="none", cliponaxis=False,
            customdata=[[item["constructor"], item["rank"], item["name"]]],
            hovertemplate="<b>%{customdata[2]}</b><br>WDC P%{customdata[1]}<br>"
                          "Team: %{customdata[0]}<br>%{x:g} points<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        title=dict(text=f"{season} WORLD DRIVERS' CHAMPIONSHIP",
                   font=dict(size=13, color="#F2F5FA"), x=0.03),
        height=max(330, 86 + len(visible) * 31),
        margin=dict(l=44, r=16, t=58, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,24,33,0.35)",
        font=dict(color="#C8D2E0", size=11), bargap=0.20,
        xaxis=dict(range=[0, axis_max], visible=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, fixedrange=True,
                   categoryorder="array", categoryarray=[f"P{x['rank']}  {x['name'][:3].upper()}" for x in reversed(visible)]),
        hoverlabel=dict(bgcolor="#1B2230", font_color="#FFFFFF"),
        showlegend=False,
    )
    return fig, total, selected_below


# ==========================================
# SCROLLABLE WDC + OFFLINE-FIRST TYRE ANALYTICS
# ==========================================
def render_scrollable_wdc(season, selected_family):
    standings = fetch_api_standings(season)
    if not standings:
        st.info("WDC standings are unavailable for this season.")
        return
    bars = []
    max_points = max([float(x.get("points") or 0) for x in standings] + [1])
    for entry in sorted(standings, key=lambda x: int(x.get("position") or 999)):
        person = entry.get("Driver") or {}
        surname = str(person.get("familyName", "Unknown"))
        constructor = (entry.get("Constructors") or [{}])[-1].get("name", "Unknown")
        rank = int(entry.get("position") or 0)
        pts = float(entry.get("points") or 0)
        accent = "#" + get_fallback_color(constructor)
        selected = surname.casefold() == selected_family.casefold()
        safe_name, safe_team = html.escape(surname.upper()), html.escape(constructor)
        pct = max(1.5, 100 * pts / max_points)
        outline = "border:2px solid #fff;box-shadow:0 0 0 1px #fff8;" if selected else ""
        bars.append(f"<div class='entry' title='{safe_name} · {safe_team} · {pts:g} pts'><span class='rank'>P{rank}</span><div class='track' style='{outline}'><div class='fill' style='width:{pct:.2f}%;background:{accent};opacity:{1 if selected else .78}'></div><div class='barlabel'>{safe_name} <span>· {pts:g} PTS</span></div></div></div>")
    markup = f"""<!DOCTYPE html><html><head><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#111722;color:#f5f7fc;font:12px Arial,sans-serif}}
    .card{{border:1px solid #303848;border-radius:12px;padding:15px 12px;background:#111722}}
    h3{{font-size:12px;letter-spacing:.08em;margin:0 0 8px}}p{{color:#a4afc0;font-size:11px;margin:0 0 12px}}
    .scroll{{height:288px;overflow-y:auto;padding-right:7px;scrollbar-color:#657085 #151a23;scrollbar-width:thin}}
    .entry{{display:flex;gap:7px;align-items:center;margin-bottom:8px;height:24px}}
    .rank{{width:25px;flex-shrink:0;font-size:10px;color:#c3cfdf;text-align:right}}
    .track{{height:23px;flex:1;position:relative;background:#202734;border-radius:3px;overflow:hidden}}
    .fill{{height:100%;border-radius:2px;min-width:3px}}
    .barlabel{{position:absolute;inset:0;display:flex;align-items:center;gap:5px;padding-left:7px;color:white;font-size:10px;font-weight:800;text-shadow:0 1px 3px #000,0 0 3px #000;white-space:nowrap}}
    .barlabel span{{font-size:9px;font-weight:600}}.foot{{margin-top:8px;color:#9daabd;font-size:10px}}
    </style></head><body><div class='card'><h3>{season} WORLD DRIVERS' CHAMPIONSHIP</h3><p>Scroll to view all {len(bars)} drivers · selected driver outlined in white</p><div class='scroll'>{''.join(bars)}</div><div class='foot'>Official championship points · Jolpica API</div></div></body></html>"""
    components.html(markup, height=380, scrolling=False)


def _tyre_cache_file(season, session_key, driver_number):
    return WAREHOUSE_DIR / "tyre_stints" / f"stints_{season}_{session_key}_{driver_number}.json"


def _get_json_openf1(url, params):
    response = requests.get(url, params=params, timeout=18)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


@st.cache_data(ttl=3600, show_spinner=False)
def _get_race_sessions(season):
    path = WAREHOUSE_DIR / "tyre_stints" / f"sessions_{season}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    try:
        sessions = _get_json_openf1("https://api.openf1.org/v1/sessions", {"year": int(season), "session_type": "Race"})
        if sessions:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sessions), encoding="utf-8")
        return sessions
    except (requests.RequestException, ValueError, OSError):
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _get_driver_stints(season, session_key, driver_number):
    path = _tyre_cache_file(season, session_key, driver_number)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    try:
        stints = _get_json_openf1("https://api.openf1.org/v1/stints", {"session_key": session_key, "driver_number": driver_number})
        if stints:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(stints), encoding="utf-8")
        return stints
    except (requests.RequestException, ValueError, OSError):
        return []


def render_tyre_analysis(season, driver, team_color):
    """Actual OpenF1 stint records, cached locally; no invented compound win rates."""
    sessions = _get_race_sessions(season)
    if not sessions:
        st.info("Tyre information is not available yet. OpenF1 race sessions could not be loaded. Check your connection and try again.")
        return
    sessions = sorted(sessions, key=lambda x: str(x.get("date_start", "")))
    names = [f"{x.get('country_name', 'Race')} · {x.get('session_name', 'Race')}" for x in sessions]
    choice = st.selectbox("Grand Prix for tyre analysis", range(len(sessions)),
                          format_func=lambda i: names[i], index=len(sessions)-1,
                          key=f"tyre_event_{season}_{driver['last_name']}")
    session = sessions[choice]
    number = driver.get("driver_number")
    if not number:
        st.info("Driver number unavailable; cannot match tyre stints reliably.")
        return
    stints = _get_driver_stints(season, session.get("session_key"), int(number))
    records = []
    for x in stints:
        compound = str(x.get("compound") or "UNKNOWN").upper()
        try:
            first, last = int(x["lap_start"]), int(x["lap_end"])
        except (ValueError, TypeError, KeyError):
            continue
        laps = max(0, last-first+1)
        if laps > 0:
            records.append({"compound":compound, "laps":laps, "stint":x.get("stint_number", "—")})
    if not records:
        st.info("No tyre-stint data were returned for this driver and Grand Prix. Try another race; OpenF1 coverage varies.")
        return
    df = pd.DataFrame(records)
    colors = {"SOFT":"#EF4444", "MEDIUM":"#FACC15", "HARD":"#E2E8F0", "INTERMEDIATE":"#22C55E", "WET":"#38BDF8", "UNKNOWN":"#8794A8"}
    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        st.markdown("#### Tyre Compound Scorecard")
        total_laps = int(df["laps"].sum())
        total_stints = len(df)
        a,b,c = st.columns(3)
        a.metric("Recorded laps", total_laps)
        b.metric("Stints", total_stints)
        c.metric("Compounds", df["compound"].nunique())
        summary = df.groupby("compound")["laps"].agg(["sum","count","mean","max"]).reset_index().sort_values("sum",ascending=False)
        for _, row in summary.iterrows():
            compound = row["compound"]
            st.markdown(f"<div style='border-left:4px solid {colors.get(compound,'#8794A8')};padding:10px 13px;margin:8px 0;background:#171e29;border-radius:6px'><b>{html.escape(compound)}</b> · {int(row['sum'])} laps ({row['sum']/total_laps:.0%})<div style='color:#aeb9ca;font-size:.82rem'>{int(row['count'])} stint(s) · Avg {row['mean']:.1f} laps/stint · Longest {int(row['max'])} laps</div></div>",unsafe_allow_html=True)
    with right:
        st.markdown("#### Type of Tyre Usage")
        totals = df.groupby("compound",as_index=False)["laps"].sum().sort_values("laps",ascending=True)
        fig = go.Figure(go.Bar(x=totals["laps"],y=totals["compound"],orientation="h",
             marker_color=[colors.get(x,"#8794A8") for x in totals["compound"]],
             text=[f"{int(x)} laps" for x in totals["laps"]],textposition="auto",
             hovertemplate="%{y}: %{x} recorded laps<extra></extra>"))
        fig.update_layout(height=310,margin=dict(l=15,r=20,t=20,b=40),
             paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,24,33,.35)",
             font=dict(color="#e8edf5"),xaxis=dict(title="Laps on compound",gridcolor="#303849"),
             yaxis=dict(title=""),showlegend=False)
        st.plotly_chart(fig,width="stretch")
    st.caption("Source: OpenF1 race tyre stints, saved locally in for1sight_warehouse/tyre_stints. Counts represent recorded stint laps, not tyre-specific race wins or win rates; those cannot be inferred reliably from stint data alone.")


