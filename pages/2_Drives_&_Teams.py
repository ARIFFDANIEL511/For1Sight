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

st.set_page_config(page_title="Driver Statistics | For1Sight", layout="wide")

# ==========================================
# CUSTOM CSS FOR COMPACT METRICS & UI
# ==========================================
st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.25rem !important; font-weight: 800 !important; }
div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
/* Button spacing follows native Streamlit layout to avoid overlap. */
</style>
""", unsafe_allow_html=True)

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
    .scroll{{height:242px;overflow-y:auto;padding-right:7px;scrollbar-color:#657085 #151a23;scrollbar-width:thin}}
    .entry{{display:flex;gap:7px;align-items:center;margin-bottom:8px;height:24px}}
    .rank{{width:25px;flex-shrink:0;font-size:10px;color:#c3cfdf;text-align:right}}
    .track{{height:23px;flex:1;position:relative;background:#202734;border-radius:3px;overflow:hidden}}
    .fill{{height:100%;border-radius:2px;min-width:3px}}
    .barlabel{{position:absolute;inset:0;display:flex;align-items:center;gap:5px;padding-left:7px;color:white;font-size:10px;font-weight:800;text-shadow:0 1px 3px #000,0 0 3px #000;white-space:nowrap}}
    .barlabel span{{font-size:9px;font-weight:600}}.foot{{margin-top:8px;color:#9daabd;font-size:10px}}
    </style></head><body><div class='card'><h3>{season} WORLD DRIVERS' CHAMPIONSHIP</h3><p>Scroll to view all {len(bars)} drivers · selected driver outlined in white</p><div class='scroll'>{''.join(bars)}</div><div class='foot'>Official championship points · Jolpica API</div></div></body></html>"""
    components.html(markup, height=340, scrolling=False)


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


# ==========================================
# 4B. TEAM GRID & TEAM PAGE ENGINE (View B redesign + new View D)
# ==========================================
import io
import mimetypes
from difflib import SequenceMatcher
from itertools import accumulate
from statistics import mean

PERF_COLORS = {
    "win": "#F5C542", "podium": "#20C997", "points": "#4F8FC9",
    "nopts": "#3A4354", "dnf": "#F05A67", "absent": "rgba(255,255,255,.07)",
}


def _h(markup):
    """Collapse indented HTML to one line so Markdown never treats it as a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def _akey(value):
    norm = unicodedata.normalize("NFKD", str(value).lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", norm)


def _safe_hex(value, fallback="#888888"):
    raw = str(value or "").strip().lstrip("#")
    return f"#{raw}" if re.fullmatch(r"[0-9a-fA-F]{6}", raw) else fallback


def _lighten(hex_color, amount=0.45):
    c = _safe_hex(hex_color).lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(v + (255 - v) * amount) for v in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def team_short_name(team_name):
    t = str(team_name).lower()
    rules = [("ferrari", "Ferrari"), ("red bull", "Red Bull"), ("mclaren", "McLaren"),
             ("mercedes", "Mercedes"), ("aston", "Aston Martin"), ("alpine", "Alpine"),
             ("williams", "Williams"), ("haas", "Haas"), ("cadillac", "Cadillac"),
             ("audi", "Audi"), ("sauber", "Sauber"), ("alfa", "Alfa Romeo"),
             ("alphatauri", "AlphaTauri"), ("racing bulls", "Racing Bulls")]
    for needle, label in rules:
        if needle in t:
            return label
    if re.search(r"\brb\b", t):
        return "Racing Bulls"
    return " ".join(str(team_name).split()[:2]) or "Team"


def team_colour_for(team_name, drivers):
    raw = (drivers[0].get("team_colour") if drivers else "") or get_fallback_color(team_name)
    return _safe_hex(raw)


# ---------- images (cached, down-scaled so 11 team rows stay light) ----------
@st.cache_data(show_spinner=False)
def _image_data_uri(path_str, modified_ns, max_width, as_jpeg):
    path = Path(path_str)
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
            if im.width > max_width:
                im = im.resize((max_width, max(1, int(im.height * max_width / im.width))), Image.LANCZOS)
            buf = io.BytesIO()
            if as_jpeg:
                im.convert("RGB").save(buf, format="JPEG", quality=86, optimize=True)
                mime = "image/jpeg"
            else:
                im.convert("RGBA").save(buf, format="PNG", optimize=True)
                mime = "image/png"
        return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        try:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
        except OSError:
            return ""


def image_uri(path, max_width=360, as_jpeg=False):
    path = Path(path)
    if not path.is_file():
        return ""
    return _image_data_uri(str(path), path.stat().st_mtime_ns, max_width, as_jpeg)


_CAR_ALIASES = {
    "Ferrari": ["ferrari", "ferarri", "ferari"], "Red Bull": ["redbull", "rbr"],
    "McLaren": ["mclaren"], "Mercedes": ["mercedes"], "Aston Martin": ["astonmartin", "aston"],
    "Alpine": ["alpine"], "Williams": ["williams"], "Haas": ["haas"],
    "Racing Bulls": ["racingbulls", "rb", "alphatauri"], "AlphaTauri": ["alphatauri", "racingbulls", "rb"],
    "Sauber": ["sauber", "kicksauber", "audi"], "Audi": ["audi", "sauber", "kicksauber"],
    "Alfa Romeo": ["alfaromeo", "alfa", "sauber"], "Cadillac": ["cadillac"],
}


@st.cache_data(ttl=300, show_spinner=False)
def find_car_image(team_name, season):
    """Locate assets/cars/<season>/<team>.jpg or <season><team>car*.jpg (typo tolerant)."""
    car_dir = PROJECT_ROOT / "assets" / "cars"
    if not car_dir.is_dir():
        return ""
    short = team_short_name(team_name)
    aliases = [_akey(a) for a in _CAR_ALIASES.get(short, [short])]
    best = None
    for path in car_dir.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        key = _akey(path.stem)
        folders = [_akey(p) for p in path.relative_to(car_dir).parts[:-1]]
        year = next((f for f in folders if re.fullmatch(r"20\d\d", f)), None)
        lead = re.match(r"(20\d\d)", key)
        if not year and lead:
            year = lead.group(1)
        remainder = re.sub(r"^20\d\d", "", key)
        for word in ("carright", "carleft", "right", "left", "side", "view", "livery", "car"):
            remainder = remainder.replace(word, "")
        matched = any(
            (len(a) >= 4 and (a in key or SequenceMatcher(None, remainder, a).ratio() >= 0.82))
            or (len(a) < 4 and remainder == a)
            for a in aliases
        )
        if not matched or (year and year != str(int(season))):
            continue
        candidate = (0 if year else 1, str(path))
        best = candidate if best is None or candidate < best else best
    return best[1] if best else ""


# ---------- data: official constructor standings + per-team season stats ----------
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_constructor_standings(season, refresh=False):
    """Offline-first official constructor standings (includes sprint points)."""
    snapshot = WAREHOUSE_DIR / f"constructor_standings_{int(season)}.json"
    if snapshot.exists() and not refresh:
        try:
            rows = json.loads(snapshot.read_text(encoding="utf-8"))
            if rows:
                return rows
        except (OSError, ValueError):
            pass
    try:
        res = requests.get(f"https://api.jolpi.ca/ergast/f1/{int(season)}/constructorStandings.json", timeout=8)
        res.raise_for_status()
        rows = res.json()["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]
        WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return rows
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, OSError):
        return []


def _download_sprint_points(season):
    """Sprint classifications (Jolpica paginates by result rows)."""
    base_url = f"https://api.jolpi.ca/ergast/f1/{int(season)}/sprint.json"
    offset, rows = 0, []
    while True:
        response = requests.get(base_url, params={"limit": 100, "offset": offset}, timeout=25)
        response.raise_for_status()
        mrdata = response.json()["MRData"]
        total, limit = int(mrdata["total"]), int(mrdata["limit"])
        for race in mrdata.get("RaceTable", {}).get("Races", []):
            for res in race.get("SprintResults", []):
                rows.append({"round": int(race["round"]), "team": res["Constructor"]["name"],
                             "driver": res["Driver"]["familyName"], "points": float(res.get("points") or 0)})
        offset += limit
        if limit <= 0 or offset >= total:
            break
    return rows


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_sprint_points(season, refresh=False):
    """Offline-first sprint points so team totals match the official standings."""
    snapshot = WAREHOUSE_DIR / f"sprint_results_{int(season)}.json"
    if snapshot.exists() and not refresh:
        try:
            return json.loads(snapshot.read_text(encoding="utf-8"))["sprints"]
        except (OSError, ValueError, KeyError):
            pass
    try:
        rows = _download_sprint_points(season)
        WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps({"season": int(season), "sprints": rows}), encoding="utf-8")
        return rows
    except (requests.RequestException, ValueError, KeyError, TypeError, OSError):
        return []


def _is_dnf(status):
    s = str(status or "").lower()
    return not ("finished" in s or "lap" in s)


def compute_team_stats_from_races(races, sprints=None):
    """Pure function: Jolpica race list -> per-constructor season statistics."""
    races = sorted((r for r in races or [] if str(r.get("round", "")).isdigit()), key=lambda r: int(r["round"]))
    rounds = [{"round": int(r["round"]), "name": r.get("raceName", f"Round {r['round']}")} for r in races]
    index = {r["round"]: i for i, r in enumerate(rounds)}
    field = max((len(r.get("Results", []) or []) for r in races), default=0)
    teams = {}
    for race in races:
        rnd = int(race["round"])
        results = race.get("Results", []) or []
        for res in results:
            try:
                team = res["Constructor"]["name"]
                fam = res["Driver"]["familyName"]
            except (KeyError, TypeError):
                continue
            try:
                pos = int(res.get("position"))
            except (TypeError, ValueError):
                pos = None
            try:
                grid = int(res.get("grid") or 0)
            except (TypeError, ValueError):
                grid = 0
            try:
                pts = float(res.get("points") or 0)
            except (TypeError, ValueError):
                pts = 0.0
            status = res.get("status", "")
            dnf = _is_dnf(status)
            t = teams.setdefault(team, {
                "round_points": [0.0] * len(rounds), "wins": 0, "podiums": 0, "poles": 0,
                "entries": 0, "dnfs": 0, "dnf_list": [], "drivers": {}, "_g": [], "_f": [],
            })
            t["round_points"][index[rnd]] += pts
            d = t["drivers"].setdefault(fam, {
                "first": res["Driver"].get("givenName", ""), "code": res["Driver"].get("code") or fam[:3].upper(),
                "points": 0.0, "cells": {},
            })
            d["points"] += pts
            d["cells"][rnd] = {"pos": pos, "grid": grid, "status": status, "dnf": dnf, "points": pts}
            t["entries"] += 1
            if pos == 1:
                t["wins"] += 1
            if pos is not None and pos <= 3:
                t["podiums"] += 1
            if grid == 1:
                t["poles"] += 1
            if dnf:
                t["dnfs"] += 1
                t["dnf_list"].append({"round": rnd, "event": rounds[index[rnd]]["name"],
                                      "driver": fam, "status": status})
            elif pos is not None:
                t["_f"].append(pos)
                t["_g"].append(grid if grid > 0 else len(results))
    for s in sprints or []:
        t = teams.get(s.get("team"))
        idx = index.get(s.get("round"))
        if t is None or idx is None:
            continue
        t["round_points"][idx] += float(s.get("points") or 0)
        if s.get("driver") in t["drivers"]:
            t["drivers"][s["driver"]]["points"] += float(s.get("points") or 0)
    for t in teams.values():
        t["cum"] = list(accumulate(t["round_points"]))
        t["gp_points"] = t["cum"][-1] if t["cum"] else 0.0
        t["avg_grid"] = mean(t["_g"]) if t["_g"] else None
        t["avg_finish"] = mean(t["_f"]) if t["_f"] else None
        t["gain"] = (t["avg_grid"] - t["avg_finish"]) if t["_g"] else None
        t["finish_rate"] = 1 - t["dnfs"] / t["entries"] if t["entries"] else None
        del t["_g"], t["_f"]
    max_cum = max((t["gp_points"] for t in teams.values()), default=0.0)
    return {"rounds": rounds, "teams": teams, "max_cum": max_cum, "field": field or 20}


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_season_stats(season):
    return compute_team_stats_from_races(fetch_api_results(season), fetch_sprint_points(season))


def build_team_rank_info(stats, standings):
    """Rank + points per team. Official standings are used only if they cover every team."""
    official = {}
    for row in standings or []:
        try:
            official[row["Constructor"]["name"]] = {"rank": int(row["position"]), "points": float(row["points"])}
        except (KeyError, TypeError, ValueError):
            continue
    teams = stats["teams"]
    computed = sorted(teams, key=lambda n: (-teams[n]["gp_points"], -teams[n]["wins"], -teams[n]["podiums"], n))
    use_official = bool(official) and all(n in official for n in teams)
    info = {}
    for pos, name in enumerate(computed, start=1):
        if use_official:
            info[name] = {"rank": official[name]["rank"], "points": official[name]["points"], "source": "official"}
        else:
            info[name] = {"rank": pos, "points": teams[name]["gp_points"], "source": "results"}
    return info


def team_h2h(team_stat, last_a, last_b):
    """Teammate head-to-head over rounds where both main drivers started."""
    if not team_stat or last_a not in team_stat["drivers"] or last_b not in team_stat["drivers"]:
        return None
    ca, cb = team_stat["drivers"][last_a]["cells"], team_stat["drivers"][last_b]["cells"]
    q = [0, 0]
    r = [0, 0]
    for rnd in set(ca) & set(cb):
        a, b = ca[rnd], cb[rnd]
        if a["grid"] > 0 and b["grid"] > 0 and a["grid"] != b["grid"]:
            q[0 if a["grid"] < b["grid"] else 1] += 1
        if a["pos"] is not None and b["pos"] is not None and a["pos"] != b["pos"]:
            r[0 if a["pos"] < b["pos"] else 1] += 1
    return {"quali": tuple(q), "race": tuple(r)}


# ---------- HTML pieces ----------
def _cell_class(cell):
    if cell is None:
        return "absent"
    if cell["dnf"]:
        return "dnf"
    pos = cell["pos"]
    if pos == 1:
        return "win"
    if pos is not None and pos <= 3:
        return "podium"
    if cell["points"] > 0:
        return "points"
    return "nopts"


def strip_row_html(code, cells, rounds, height=14):
    parts = []
    for r in rounds:
        cell = cells.get(r["round"])
        kind = _cell_class(cell)
        if cell is None:
            tip = f"R{r['round']} {r['name']} · did not start for this team"
        else:
            result = "DNF" if cell["dnf"] else (f"P{cell['pos']}" if cell["pos"] else "—")
            tip = f"R{r['round']} {r['name']} · {result} · {cell['status']}"
        parts.append(f'<i title="{html.escape(tip, quote=True)}" style="background:{PERF_COLORS[kind]};height:{height}px"></i>')
    return f'<div class="tr-srow"><b>{html.escape(str(code))}</b><div class="tr-cells">{"".join(parts)}</div></div>'


def strip_legend_html():
    items = [("win", "Win"), ("podium", "Podium"), ("points", "Points"), ("nopts", "No points"), ("dnf", "DNF")]
    return '<div class="tr-legend">' + "".join(
        f'<span><i style="background:{PERF_COLORS[k]}"></i>{label}</span>' for k, label in items) + "</div>"


def _points_sparkline_uri(cum, leader_cum, max_cum, color, show_leader, H=78):
    W, PX, PY = 300, 5, 7
    n = len(cum)
    if n == 0 or max_cum <= 0:
        return ""

    def xy(i, v):
        x = PX + (W - 2 * PX) * (i / (n - 1) if n > 1 else 0.5)
        return x, PY + (H - 2 * PY) * (1 - v / max_cum)

    pts = [xy(i, v) for i, v in enumerate(cum)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pts[0][0]:.1f},{H - PY} {line} {pts[-1][0]:.1f},{H - PY}"
    leader = ""
    if show_leader and leader_cum and len(leader_cum) == n:
        lp = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(i, v) for i, v in enumerate(leader_cum)))
        leader = f'<polyline points="{lp}" fill="none" stroke="#6b778a" stroke-width="1.4" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>'
    ex, ey = pts[-1]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".45"/>'
        f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<line x1="0" y1="{H - PY}" x2="{W}" y2="{H - PY}" stroke="#3a4354" stroke-width="1"/>{leader}'
        f'<polygon points="{area}" fill="url(#g)"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
        f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.8" fill="{color}"/></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def points_gap_text(team_name, stats, rank_info):
    """'+241 on McLaren' for the leader, '−87 to leader' for everyone else."""
    info = rank_info.get(team_name)
    if not info:
        return ""
    ordered = sorted(rank_info.items(), key=lambda kv: kv[1]["rank"])
    if info["rank"] == 1 and len(ordered) > 1:
        nxt = ordered[1]
        return f"+{info['points'] - nxt[1]['points']:g} on {team_short_name(nxt[0])}"
    leader = ordered[0]
    return f"−{leader[1]['points'] - info['points']:g} to leader"


def _panel(title, right, body):
    return f'<div class="tr-panel"><div class="tr-ph"><span>{title}</span><span class="tr-pr">{right}</span></div>{body}</div>'


def build_team_panels_html(team_name, stats, rank_info, main_drivers, color):
    """Season snapshot: four soft panels, each with one headline number and one small visual."""
    team = stats["teams"].get(team_name)
    if not team or not team["entries"]:
        return ('<div class="tr-panels-wrap"><div class="tr-title">SEASON SNAPSHOT</div>'
                '<div class="tr-nodata">No race results for this team in this season yet.</div></div>')
    rounds = stats["rounds"]
    info = rank_info.get(team_name, {})
    last_names = [d["last_name"] for d in main_drivers[:2]]
    codes = [str(d.get("name_acronym") or d["last_name"][:3]).upper() for d in main_drivers[:2]]

    # 1 · Points race (one shared y-scale across teams)
    spark = _points_sparkline_uri(team["cum"], None, stats["max_cum"], color, show_leader=False)
    pts = info.get("points", team["gp_points"])
    p1 = _panel("POINTS RACE", "",
                f'<div class="tr-big">{pts:g}<small> pts</small></div>'
                + (f'<img class="tr-spark" src="{spark}" alt="">' if spark else "")
                + f'<div class="tr-sub">{html.escape(points_gap_text(team_name, stats, rank_info))}</div>')

    # 2 · Points split: one bar per driver, plus the teammate duel
    drv_pts = [team["drivers"].get(n, {}).get("points", 0.0) for n in last_names]
    top = max(drv_pts + [1.0])
    rows = ""
    for i, (code, value) in enumerate(zip(codes, drv_pts)):
        tone = color if i == 0 else _lighten(color)
        rows += (f'<div class="tr-brow"><b>{html.escape(code)}</b>'
                 f'<div class="tr-btrack"><div style="width:{value / top * 100:.1f}%;background:{tone}"></div></div>'
                 f'<span>{value:g}</span></div>')
    h2h = team_h2h(team, *last_names) if len(last_names) > 1 else None
    duel = (f'<div class="tr-sub">Qualifying {h2h["quali"][0]}–{h2h["quali"][1]} · Race {h2h["race"][0]}–{h2h["race"][1]}</div>'
            if h2h else "")
    p2 = _panel("TEAMMATE DUEL", "", rows + duel)

    # 3 · Grid → finish: a headline number instead of a ruler
    if team["gain"] is not None:
        gain = team["gain"]
        tone = "#20C997" if gain > 0.05 else ("#F05A67" if gain < -0.05 else "#9aa8ba")
        body = (f'<div class="tr-big" style="color:{tone}">{gain:+.1f}<small> places / race</small></div>'
                f'<div class="tr-sub">Starts P{team["avg_grid"]:.1f} → finishes P{team["avg_finish"]:.1f}<br>'
                f'<span>finishers only</span></div>')
    else:
        body = '<div class="tr-sub">No classified finishes yet.</div>'
    p3 = _panel("RACE-DAY GAIN", "", body)

    # 4 · Result strip
    strips = "".join(strip_row_html(code, team["drivers"].get(last, {}).get("cells", {}), rounds)
                     for last, code in zip(last_names, codes))
    rate = team["finish_rate"]
    p4 = _panel("RACE RESULTS", f'{rate:.0%} finish rate' if rate is not None else "",
                strips + strip_legend_html())

    return (f'<div class="tr-panels-wrap"><div class="tr-title">SEASON SNAPSHOT</div>'
            f'<div class="tr-panels">{p1}{p2}{p3}{p4}</div></div>')


def team_header_html(team_name, color, team_stat, info, ghost_names):
    """One calm header: big rank numeral, team name (once), plain-text stats, logo on the right."""
    logo_path = get_team_logo(team_name)
    logo = image_uri(PROJECT_ROOT / logo_path, max_width=200) if logo_path else ""
    logo_html = f'<img class="dc-logo" src="{html.escape(logo, quote=True)}" alt="">' if logo else ""
    rank = f'P{info["rank"]}' if info else "–"
    meta = ""
    if info and team_stat and team_stat["entries"]:
        meta = (f'{info["points"]:g} pts · {team_stat["wins"]} wins · {team_stat["podiums"]} podiums · '
                f'{team_stat["poles"]} poles')
    elif not info:
        meta = "No results yet"
    ghost = f'<div class="dc-ghost">Also drove: {html.escape(", ".join(ghost_names))}</div>' if ghost_names else ""
    return (f'<div class="dc-head" style="--tc:{color}"><div class="dc-rank">{rank}</div>'
            f'<div class="dc-id"><div class="dc-name">{html.escape(team_short_name(team_name))}</div>'
            f'<div class="dc-meta">{meta}</div>{ghost}</div>{logo_html}</div>')


def overview_head_html():
    return ('<div class="ov-head"><span>#</span><span>TEAM</span><span>DRIVERS</span>'
            '<span>POINTS</span><span>SEASON</span><span>LAST 5 RACES</span></div>')


def overview_row_html(team_name, season, stats, rank_info, main_drivers, color):
    team = stats["teams"].get(team_name)
    info = rank_info.get(team_name)
    logo_path = get_team_logo(team_name)
    logo = image_uri(PROJECT_ROOT / logo_path, max_width=120) if logo_path else ""
    logo_html = f'<img src="{html.escape(logo, quote=True)}" alt="">' if logo else ""
    sub = f'{team["wins"]} wins · {team["podiums"]} podiums' if team and team["entries"] else "No results yet"

    chips = ""
    for d in main_drivers[:2]:
        src = (image_uri(PROJECT_ROOT / clean_driver_filename(d["first_name"], d["last_name"], season), 120)
               or d.get("headshot_url") or DEFAULT_HEADSHOT)
        code = html.escape(str(d.get("name_acronym") or d["last_name"][:3]).upper())
        chips += (f'<span class="ov-chip"><img src="{html.escape(str(src), quote=True)}" alt="">'
                  f'<b>{code}</b></span>')

    pts_html, spark_html, form_html = '<span class="ov-dim">—</span>', "", ""
    if team and team["entries"] and info:
        gap = html.escape(points_gap_text(team_name, stats, rank_info))
        pts_html = f'<b>{info["points"]:g}</b><small>{gap}</small>'
        spark = _points_sparkline_uri(team["cum"], None, stats["max_cum"], color, show_leader=False, H=40)
        spark_html = f'<img src="{spark}" alt="">' if spark else ""
        peak = max((max(t["round_points"][-5:], default=0) for t in stats["teams"].values()), default=0) or 1
        last = list(zip(stats["rounds"], team["round_points"]))[-5:]
        form_html = "".join(
            f'<i title="R{r["round"]} {html.escape(r["name"], quote=True)} · {v:g} pts" '
            f'style="height:{max(4, v / peak * 34):.0f}px;background:{color}"></i>' for r, v in last)
    rank = f'P{info["rank"]}' if info else "–"
    return (f'<div class="ov-row" style="--tc:{color}"><span class="ov-rank">{rank}</span>'
            f'<span class="ov-team">{logo_html}<span><b>{html.escape(team_short_name(team_name))}</b><small>{sub}</small></span></span>'
            f'<span class="ov-drivers">{chips}</span><span class="ov-pts">{pts_html}</span>'
            f'<span class="ov-spark">{spark_html}</span><span class="ov-form">{form_html}</span></div>')


def render_team_card(team_name, season, teams, stats, rank_info):
    """Detailed team card (shared by the 'Detailed cards' layout and the expanded overview row)."""
    all_drivers = teams[team_name]
    main = [d for d in all_drivers if d.get("role", "main") == "main"] or all_drivers[:2]
    ghosts = [d for d in all_drivers if d.get("role") == "ghost"]
    if not main:
        return
    color = team_colour_for(team_name, main)
    team = stats["teams"].get(team_name)
    slug = _akey(team_name)
    st.markdown(
        f"<style>.st-key-teamrow_{season}_{slug}{{border-top:4px solid {color};"
        f"background:linear-gradient(115deg,{color}18 0%,rgba(17,22,31,0) 40%)}}"
        f".st-key-teamopen_{season}_{slug} button{{background:{color};border-color:{color};color:#0b0e14;font-weight:800}}"
        f".st-key-teamopen_{season}_{slug} button:hover{{filter:brightness(1.1);color:#0b0e14;border-color:{color}}}</style>",
        unsafe_allow_html=True)
    with st.container(border=True, key=f"teamrow_{season}_{slug}"):
        left, right = st.columns([5.2, 4.8], gap="large")
        with left:
            st.markdown(team_header_html(team_name, color, team, rank_info.get(team_name),
                                         [g["last_name"] for g in ghosts]), unsafe_allow_html=True)
            cols = st.columns(2, gap="small")
            for idx, driver in enumerate(main[:2]):
                with cols[idx]:
                    ghost = ghosts[0] if (idx == 1 and ghosts) else None
                    st.markdown(team_portrait_html(driver, season, color, ghost), unsafe_allow_html=True)
                    last = str(driver["last_name"])
                    if st.button(f"View {last} ›", key=f"view_{season}_{team_name}_{idx}", type="tertiary", width="stretch"):
                        st.session_state.selected_team = team_name
                        st.session_state.selected_driver = last
                        st.session_state.driver_origin = "grid"
                        st.rerun()
            st.markdown(team_car_html(team_name, season, color), unsafe_allow_html=True)
        with right:
            st.markdown(build_team_panels_html(team_name, stats, rank_info, main, color), unsafe_allow_html=True)
            if st.button(f"Open {team_short_name(team_name)} dashboard →", key=f"teamopen_{season}_{slug}", width="stretch"):
                st.session_state.selected_team = team_name
                st.session_state.selected_driver = None
                st.rerun()


def team_portrait_html(driver, season, color, ghost=None):
    src = (image_uri(PROJECT_ROOT / clean_driver_filename(driver["first_name"], driver["last_name"], season), 360)
           or driver.get("headshot_url") or DEFAULT_HEADSHOT)
    number = html.escape(str(driver.get("driver_number") or "—"))
    ghost_html = ""
    if ghost:
        gsrc = (image_uri(PROJECT_ROOT / clean_driver_filename(ghost["first_name"], ghost["last_name"], season), 240)
                or ghost.get("headshot_url") or DEFAULT_HEADSHOT)
        ghost_html = (f'<div class="tr-ghostpic" title="Replaced driver: {html.escape(str(ghost["first_name"]))} {html.escape(str(ghost["last_name"]))}">'
                      f'<img src="{html.escape(gsrc, quote=True)}" alt=""></div>')
    return (f'<div class="tr-portrait" style="--tc:{color}">{ghost_html}<span class="tr-num">{number}</span>'
            f'<img src="{html.escape(str(src), quote=True)}" alt="{html.escape(driver["last_name"])}">'
            f'<div class="tr-dname"><span>{html.escape(str(driver["first_name"]))}</span>'
            f'<b>{html.escape(driver["last_name"].upper())}</b></div></div>')


def _car_placeholder_uri(color):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 120">'
        f'<rect x="22" y="30" width="34" height="6" rx="2" fill="{color}"/><rect x="34" y="34" width="5" height="30" fill="#2a3140"/>'
        f'<path d="M40,80 L40,54 L72,48 L124,58 L190,42 L244,52 L300,64 L372,78 L372,86 L40,88 Z" fill="{color}" fill-opacity=".85"/>'
        '<ellipse cx="196" cy="40" rx="20" ry="8" fill="#0b0e14"/>'
        f'<rect x="352" y="90" width="44" height="5" rx="2" fill="{color}"/>'
        '<circle cx="96" cy="90" r="24" fill="#10141c" stroke="#3a4354" stroke-width="3"/><circle cx="96" cy="90" r="9" fill="#3a4354"/>'
        '<circle cx="304" cy="90" r="22" fill="#10141c" stroke="#3a4354" stroke-width="3"/><circle cx="304" cy="90" r="8" fill="#3a4354"/></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def team_car_html(team_name, season, color, large=False):
    path = find_car_image(team_name, season)
    uri = image_uri(path, max_width=1100 if large else 760, as_jpeg=path.lower().endswith((".jpg", ".jpeg"))) if path else ""
    cls = "tr-car large" if large else "tr-car"
    if uri:
        return f'<div class="{cls}"><img src="{html.escape(uri, quote=True)}" alt="{html.escape(str(team_name))} {season} car"></div>'
    return (f'<div class="{cls} missing"><img class="ph" src="{_car_placeholder_uri(color)}" alt="">'
            f'<div class="tr-sub">Car image not available for {season}</div></div>')


# ---------- team page (View D) charts ----------
def build_team_points_race_chart(stats, team_name, color):
    rounds = stats["rounds"]
    n = len(rounds)
    x, names = [r["round"] for r in rounds], [r["name"] for r in rounds]

    def end_label(label):
        return [""] * (n - 1) + [label]

    fig = go.Figure()
    rivals = sorted((kv for kv in stats["teams"].items() if kv[0] != team_name), key=lambda kv: -kv[1]["gp_points"])[:3]
    for (rival, data), tone in zip(rivals, ["#6C8EBF", "#C49A5A", "#9A86C4"]):
        short = team_short_name(rival)
        fig.add_trace(go.Scatter(
            x=x, y=data["cum"], mode="lines+text", text=end_label(short), textposition="middle right",
            textfont=dict(color=tone, size=12), cliponaxis=False, line=dict(color=tone, width=2), opacity=0.85,
            customdata=names, showlegend=False,
            hovertemplate="<b>%{customdata}</b><br>Round %{x}<br>%{y:g} points<extra>" + short + "</extra>"))
    short = team_short_name(team_name)
    fig.add_trace(go.Scatter(
        x=x, y=stats["teams"][team_name]["cum"], mode="lines+markers+text", text=end_label(short),
        textposition="middle right", textfont=dict(color=color, size=14), cliponaxis=False,
        line=dict(color=color, width=5), marker=dict(size=8), customdata=names, showlegend=False,
        hovertemplate="<b>%{customdata}</b><br>Round %{x}<br>%{y:g} points<extra>" + short + "</extra>"))
    _style_driver_figure(fig, "Constructor Points Race", color, height=430)
    fig.update_layout(showlegend=False, margin=dict(l=50, r=120, t=60, b=45))
    fig.update_xaxes(title="Grand Prix round", dtick=1, range=[0.5, n + 0.5])
    fig.update_yaxes(title="Cumulative points", rangemode="tozero")
    return fig


def build_team_race_by_race_chart(stats, team_name, main_drivers, color):
    rounds = stats["rounds"]
    n = len(rounds)
    team = stats["teams"][team_name]
    fig = go.Figure()
    for idx, drv in enumerate(main_drivers[:2]):
        data = team["drivers"].get(drv["last_name"])
        if not data:
            continue
        tone = color if idx == 0 else "#B7C4D7"
        rows = [(r["round"], r["name"], data["cells"][r["round"]]) for r in rounds if r["round"] in data["cells"]]
        ok = [(rd, nm, c) for rd, nm, c in rows if c["pos"] is not None and not c["dnf"]]
        bad = [(rd, nm, c) for rd, nm, c in rows if c["pos"] is not None and c["dnf"]]
        fig.add_trace(go.Scatter(
            x=[r[0] for r in ok], y=[r[2]["pos"] for r in ok], mode="lines+markers", name=drv["last_name"],
            line=dict(color=tone, width=3.5 if idx == 0 else 2.5), marker=dict(size=8, color=tone),
            customdata=[[r[1], r[2]["grid"], r[2]["status"]] for r in ok],
            hovertemplate="<b>%{customdata[0]}</b><br>Round %{x} · Finished P%{y}<br>Started P%{customdata[1]}<br>%{customdata[2]}<extra>" + drv["last_name"] + "</extra>"))
        if bad:
            fig.add_trace(go.Scatter(
                x=[r[0] for r in bad], y=[r[2]["pos"] for r in bad], mode="markers", name=f"{drv['last_name']} DNF",
                marker=dict(symbol="x", size=12, color=PERF_COLORS["dnf"], line=dict(width=2)), showlegend=False,
                customdata=[[r[1], r[2]["status"]] for r in bad],
                hovertemplate="<b>%{customdata[0]}</b><br>Round %{x} · Retired<br>%{customdata[1]}<extra>" + drv["last_name"] + "</extra>"))
    _style_driver_figure(fig, "Race-by-Race Classified Position", color, height=400)
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    fig.update_xaxes(title="Grand Prix round", dtick=1, range=[0.5, n + 0.5])
    fig.update_yaxes(title="Classified position · lower is better", autorange="reversed", dtick=2)
    return fig


def build_teammate_duel_chart(team_stat, main_drivers, color):
    if len(main_drivers) < 2:
        return None
    la, lb = main_drivers[0]["last_name"], main_drivers[1]["last_name"]
    h2h = team_h2h(team_stat, la, lb)
    da, db = team_stat["drivers"].get(la), team_stat["drivers"].get(lb)
    if not h2h or not da or not db:
        return None
    metrics = [
        ("Qualifying duels", h2h["quali"]), ("Race duels", h2h["race"]),
        ("Podiums", (sum(1 for c in da["cells"].values() if c["pos"] and c["pos"] <= 3),
                     sum(1 for c in db["cells"].values() if c["pos"] and c["pos"] <= 3))),
        ("Grand Prix points", (da["points"], db["points"])),
    ]
    metrics = [(n, v) for n, v in metrics if (v[0] + v[1]) > 0]
    if not metrics:
        return None
    fig = go.Figure()
    for idx, (name, tone) in enumerate([(la, color), (lb, "#B7C4D7")]):
        fig.add_trace(go.Bar(
            y=[m[0] for m in metrics], x=[m[1][idx] / (m[1][0] + m[1][1]) * 100 for m in metrics], orientation="h",
            name=name, marker_color=tone, text=[f"{m[1][idx]:g}" for m in metrics], textposition="inside",
            hovertemplate="%{y}: %{text}<extra>" + name + "</extra>"))
    _style_driver_figure(fig, f"Teammate Duel · {la} vs {lb}", color, height=330)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Share of head-to-head (%)", range=[0, 100])
    fig.update_yaxes(autorange="reversed")
    return fig


def compute_team_pace_gap(season, code_to_team):
    """Median clean race lap per team vs the fastest team, per Grand Prix (descriptive)."""
    df = _lap_data_for_consistency(season)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["code"] = df["driver_code"].astype(str).str.upper()
    df["team"] = df["code"].map(code_to_team)
    df = df.dropna(subset=["team"])
    if df.empty:
        return pd.DataFrame()
    order = get_chronological_events(df)
    rows = []
    for event, ev in df.groupby("event_name"):
        medians = {}
        for team, tg in ev.groupby("team"):
            kept = []
            for _, dg in tg.groupby("code"):
                s = dg["lap_seconds"].dropna()
                if len(s) < 10:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                kept.append(s[s <= q3 + 1.5 * (q3 - q1)])
            if kept:
                pooled = pd.concat(kept)
                if len(pooled) >= 15:
                    medians[team] = float(pooled.median())
        if len(medians) < 3:
            continue
        best = min(medians.values())
        ranked = sorted(medians, key=medians.get)
        for team, m in medians.items():
            rows.append({"event": event, "team": team, "gap_pct": (m / best - 1) * 100,
                         "rank": ranked.index(team) + 1, "order": order.index(event) if event in order else 999})
    return pd.DataFrame(rows).sort_values("order") if rows else pd.DataFrame()


def team_insight(team_name, stats, rank_info):
    team = stats["teams"].get(team_name)
    info = rank_info.get(team_name)
    if not team or not info or not team["entries"]:
        return ""
    short = team_short_name(team_name)
    ordered = sorted(rank_info.items(), key=lambda kv: kv[1]["rank"])
    if info["rank"] == 1 and len(ordered) > 1:
        text = (f"{short} lead the constructors' championship on {info['points']:g} points, "
                f"{info['points'] - ordered[1][1]['points']:g} clear of {team_short_name(ordered[1][0])}")
    else:
        text = (f"{short} sit P{info['rank']} on {info['points']:g} points, "
                f"{ordered[0][1]['points'] - info['points']:g} behind {team_short_name(ordered[0][0])}")
    if team["gain"] is not None:
        text += f", and gain {team['gain']:+.1f} places per race on average"
    return text + "."


def td_driver_html(driver, season, color, points):
    src = (image_uri(PROJECT_ROOT / clean_driver_filename(driver["first_name"], driver["last_name"], season), 200)
           or driver.get("headshot_url") or DEFAULT_HEADSHOT)
    pts = f"{points:g} pts for the team" if points is not None else ""
    return (f'<div class="td-driver" style="--tc:{color}"><img src="{html.escape(str(src), quote=True)}" alt="">'
            f'<div><div class="td-num">#{html.escape(str(driver.get("driver_number") or "—"))}</div>'
            f'<div class="td-name">{html.escape(str(driver["first_name"]))} <b>{html.escape(driver["last_name"].upper())}</b></div>'
            f'<div class="td-pts">{pts}</div></div></div>')


def render_team_page(team_name, teams, season):
    """VIEW D: compact hero (car + KPIs + drivers) with the analytics chart above the fold."""
    drivers_all = teams.get(team_name, [])
    main = [d for d in drivers_all if d.get("role", "main") == "main"] or drivers_all[:2]
    ghosts = [d for d in drivers_all if d.get("role") == "ghost"]
    color = team_colour_for(team_name, main or drivers_all)
    stats = get_team_season_stats(season)
    ranks = build_team_rank_info(stats, fetch_constructor_standings(season))
    team = stats["teams"].get(team_name)
    info = ranks.get(team_name)

    nav_l, nav_r = st.columns([4, 1])
    with nav_l:
        if st.button("← All teams", key="back_from_team"):
            st.session_state.selected_team = None
            st.session_state.selected_driver = None
            st.rerun()
    with nav_r:
        st.markdown(f"<div style='text-align:right;color:#AAB7C9;font-weight:800;padding-top:8px'>{season} SEASON</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='margin:0 0 .6rem'><div style='color:{color};font-size:.78rem;letter-spacing:.18em;font-weight:800'>TEAM INTELLIGENCE / {season}</div>"
        f"<div style='color:#F7F9FD;font-size:clamp(1.7rem,2.6vw,2.6rem);font-weight:900;line-height:1.2'>{html.escape(str(team_name).upper())}</div>"
        f"<div style='color:#AEB9C9;font-size:.95rem;margin-top:.2rem'>{html.escape(team_insight(team_name, stats, ranks))}</div></div>",
        unsafe_allow_html=True)

    hero_l, hero_r = st.columns([1.25, 1], gap="large")
    with hero_l:
        st.markdown(team_car_html(team_name, season, color, large=True), unsafe_allow_html=True)
    with hero_r:
        if team and team["entries"]:
            kpis = [("Position", f"P{info['rank']}" if info else "—"), ("Points", f"{info['points']:g}" if info else "—"),
                    ("Finish rate", f"{team['finish_rate']:.0%}"), ("Wins", team["wins"]),
                    ("Podiums", team["podiums"]), ("Poles", team["poles"])]
            cells = "".join(f'<div><b>{v}</b><span>{k}</span></div>' for k, v in kpis)
            st.markdown(f'<div class="td-kpis" style="--tc:{color}">{cells}</div>', unsafe_allow_html=True)
        else:
            st.info("No race results are available for this team in the selected season yet.")
        dcols = st.columns(max(len(main[:2]), 1), gap="small")
        for idx, drv in enumerate(main[:2]):
            with dcols[idx]:
                pts = team["drivers"].get(drv["last_name"], {}).get("points") if team else None
                st.markdown(td_driver_html(drv, season, color, pts), unsafe_allow_html=True)
                if st.button(f"View {drv['last_name']} ›", key=f"teampage_view_{season}_{team_name}_{idx}",
                             type="tertiary", width="stretch"):
                    st.session_state.selected_driver = drv["last_name"]
                    st.session_state.driver_origin = "team"
                    st.rerun()

    if not team or not team["entries"]:
        return
    options = ["🏆 Points race", "🏁 Race by race", "⚔️ Teammate duel", "⏱️ Pace gap", "🔧 Reliability"]
    view = st.segmented_control("Team analytics", options, default=options[0],
                                key=f"team_view_{season}", label_visibility="collapsed") or options[0]
    if view.startswith("🏆"):
        st.plotly_chart(build_team_points_race_chart(stats, team_name, color), width="stretch")
        st.caption("Cumulative Grand Prix and sprint points for the season.")
    elif view.startswith("🏁"):
        st.plotly_chart(build_team_race_by_race_chart(stats, team_name, main, color), width="stretch")
        st.caption("Classified finishing position for each main driver. Red crosses are retirements.")
    elif view.startswith("⚔️"):
        fig = build_teammate_duel_chart(team, main, color)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
            st.caption("Qualifying uses starting grid, race uses classified position, in rounds where both teammates started.")
        else:
            st.info("A teammate duel needs two main drivers with shared races this season.")
    elif view.startswith("⏱️"):
        code_to_team = {str(d.get("name_acronym") or d["last_name"][:3]).upper(): t for t, ds in teams.items() for d in ds}
        try:
            gap = compute_team_pace_gap(season, code_to_team)
        except Exception as exc:
            gap = pd.DataFrame()
            st.warning(f"Pace gap could not be computed: {exc}")
        mine = gap[gap["team"] == team_name] if not gap.empty else gap
        if mine.empty:
            st.info("Pace gap needs race laps for at least three teams in the local Parquet warehouse.")
        else:
            fig = go.Figure(go.Bar(
                x=mine["event"], y=mine["gap_pct"], marker_color=color, customdata=mine["rank"],
                hovertemplate="<b>%{x}</b><br>%{y:.2f}% slower than the fastest team<br>Pace rank P%{customdata}<extra></extra>"))
            _style_driver_figure(fig, "Race Pace Gap to the Fastest Team (median clean lap)", color, height=400)
            fig.update_yaxes(title="Gap (%) · 0 = fastest team", rangemode="tozero")
            fig.update_xaxes(tickangle=-40)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Median gap across {len(mine)} races: {mine['gap_pct'].median():.2f}%. Descriptive only: tyre compounds, fuel load, traffic and safety cars are not controlled.")
    else:
        st.markdown(strip_legend_html(), unsafe_allow_html=True)
        rows = "".join(strip_row_html(str(d.get("name_acronym") or d["last_name"][:3]).upper(),
                                       team["drivers"].get(d["last_name"], {}).get("cells", {}), stats["rounds"], height=26)
                       for d in main[:2])
        st.markdown(f'<div class="tr-bigstrip">{rows}</div>', unsafe_allow_html=True)
        if team["dnf_list"]:
            st.markdown(f"**{team['dnfs']} retirement(s) in {team['entries']} entries**")
            st.dataframe(pd.DataFrame(team["dnf_list"]).rename(columns={"round": "Round", "event": "Grand Prix", "driver": "Driver", "status": "Status"}),
                         hide_index=True, width="stretch")
        else:
            st.success("No retirements recorded for this team in the selected season.")

# ==========================================
# 5. SESSION STATE & GLOBAL ROUTING
# ==========================================
if 'global_season' not in st.session_state: st.session_state.global_season = 2024
if 'selected_team' not in st.session_state: st.session_state.selected_team = None
if 'selected_driver' not in st.session_state: st.session_state.selected_driver = None
if 'driver_origin' not in st.session_state: st.session_state.driver_origin = "grid"
if 'expanded_team' not in st.session_state: st.session_state.expanded_team = None

# Render a visible status BEFORE any potentially slow external API requests.

startup_status = st.empty()
startup_status.info(f"Loading {st.session_state.global_season} driver roster…")
try:
    teams = get_season_roster(st.session_state.global_season)
except Exception as exc:
    teams = {}
    st.error("Driver roster could not be loaded. See the error details below.")
    with st.expander("Technical error details", expanded=True):
        st.exception(exc)
startup_status.empty()

if st.session_state.selected_team not in teams:
    st.session_state.selected_team = None
    st.session_state.selected_driver = None

driver_colors_mapping = {d['last_name']: f"#{d['team_colour']}" for t_drivers in teams.values() for d in t_drivers}

# ==========================================
# FOR1SIGHT ENHANCEMENT: DYNAMIC TEAM THEME
# ==========================================
def theme_color_for_selection(team_name, roster):
    """Resolve selected constructor's accent; neutral racing red on the team grid."""
    if not team_name or team_name not in roster:
        return "#E10600"
    entries = roster.get(team_name) or []
    if not entries:
        return "#E10600"
    candidate = str(entries[0].get("team_colour", "")).strip().lstrip("#")
    return f"#{candidate}" if re.fullmatch(r"[0-9a-fA-F]{6}", candidate) else "#E10600"

ACTIVE_ACCENT = theme_color_for_selection(st.session_state.selected_team, teams)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;500;600;700;800;900&display=swap');
:root {{ --f1-accent: {ACTIVE_ACCENT}; --f1-surface: #151a24; --f1-stroke: #323948; }}
html, body, [class*='css'], .stApp, button, input {{ font-family: 'Titillium Web', sans-serif; }}
.stApp {{
  background: radial-gradient(ellipse at 88% 4%, {ACTIVE_ACCENT}24 0%, transparent 38%),
              linear-gradient(145deg, #080a0f 0%, #10151e 48%, #0b0d14 100%);
  background-attachment: fixed; color: #f3f5f9;
}}
[data-testid='stHeader'] {{ background: transparent !important; }}
[data-testid='stSidebar'] {{ background: #10141d !important; border-right: 1px solid #323948; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1600px; }}
h1, h2, h3 {{ font-family: 'Titillium Web', sans-serif !important; letter-spacing: .015em; }}
h1 {{ font-weight: 900 !important; }}
h2, h3 {{ font-weight: 800 !important; }}
hr {{ border-color: #333b4b !important; margin: 1.5rem 0 !important; }}
[data-testid='stVerticalBlockBorderWrapper'] > div,
[data-testid='stMetric'] {{ border-radius: 12px; }}
[data-testid='stMetric'] {{
 background: linear-gradient(150deg, #1c2230, #121721);
 border: 1px solid #303847; border-top: 2px solid {ACTIVE_ACCENT};
 padding: 14px 16px; min-height: 92px;
}}
[data-testid='stMetricValue'] {{ font-size: clamp(1.15rem, 1.7vw, 1.65rem) !important; font-weight: 900 !important; color: #fff; }}
[data-testid='stMetricLabel'] {{ font-size: .82rem !important; color: #bac4d3 !important; }}
.stButton > button {{
 border: 1px solid #465064 !important; border-radius: 9px !important;
 background: linear-gradient(150deg, #252d3a, #171d27) !important;
 color: #f8fafc !important; font-weight: 700 !important;
 transition: border-color .15s ease, transform .15s ease, box-shadow .15s ease;
}}
.stButton > button:hover {{ border-color: {ACTIVE_ACCENT} !important; color: white !important;
 box-shadow: 0 0 0 1px {ACTIVE_ACCENT}55 !important; transform: translateY(-1px); }}
.stButton > button:focus-visible {{ outline: 2px solid {ACTIVE_ACCENT} !important; outline-offset: 2px; }}
[data-baseweb='select'] > div {{ background-color: #1b2230 !important; border-color: #475064 !important; border-radius: 9px !important; }}
[data-testid='stExpander'] {{ background: #151b26; border: 1px solid #343d4b; border-radius: 10px; }}
[data-testid='stPlotlyChart'] {{ background: rgba(17,22,32,.48); border: 1px solid #2b3342; border-radius: 12px; padding: 8px; }}
.for1sight-eyebrow {{ color: {ACTIVE_ACCENT}; font-size: .77rem; letter-spacing: .18em; font-weight: 800; margin: 0 0 .25rem; }}
.for1sight-theme-line {{ height: 3px; width: 100%; margin: 0 0 1.2rem; border-radius: 3px;
 background: linear-gradient(90deg, {ACTIVE_ACCENT}, {ACTIVE_ACCENT}44 32%, transparent 80%); }}
.for1sight-caption {{ color: #aeb9c9; font-size: .86rem; }}
@media (max-width: 760px) {{
 .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 1.25rem; }}
 h1 {{ font-size: 1.7rem !important; }}
 [data-testid='stMetric'] {{ padding: 10px; }}
}}
</style>
<div class='for1sight-eyebrow'>FOR1SIGHT / DRIVER INTELLIGENCE · {st.session_state.global_season}</div>
<div class='for1sight-theme-line'></div>
""", unsafe_allow_html=True)

# ==========================================
# TEAM OVERVIEW / TEAM CARD / TEAM PAGE STYLES
# ==========================================
st.markdown("""
<style>
/* overview table */
.st-key-ovlist{gap:.5rem !important}
.ov-head,.ov-row{display:grid;grid-template-columns:48px minmax(150px,1.5fr) 1.25fr 1fr 1.2fr .9fr;align-items:center;column-gap:.9rem}
.ov-head{padding:0 .9rem;font-size:.62rem;font-weight:800;letter-spacing:.14em;color:#6f7f93}
.ov-row{padding:.5rem .9rem;border-radius:12px;background:rgba(255,255,255,.03);border-left:4px solid var(--tc);min-height:62px}
.ov-rank{font-size:1.15rem;font-weight:900;color:#fff}
.ov-team{display:flex;align-items:center;gap:.6rem;min-width:0}
.ov-team img{height:30px;width:30px;object-fit:contain;flex:0 0 auto}
.ov-team b{display:block;color:var(--tc);font-size:1rem;letter-spacing:.03em;text-transform:uppercase;line-height:1.1}
.ov-team small,.ov-pts small{display:block;font-size:.68rem;color:#8ea0b5}
.ov-drivers{display:flex;gap:.5rem;flex-wrap:wrap}
.ov-chip{display:inline-flex;align-items:center;gap:.35rem;background:rgba(255,255,255,.05);border-radius:999px;padding:.12rem .65rem .12rem .15rem}
.ov-chip img{width:30px;height:30px;border-radius:50%;object-fit:cover;object-position:top center;background:#1b2230}
.ov-chip b{font-size:.76rem;color:#e8edf5}
.ov-pts b{font-size:1.25rem;color:#fff;font-weight:900;line-height:1.1}
.ov-spark img{width:100%;height:auto;max-height:44px;display:block}
.ov-form{display:flex;align-items:flex-end;gap:4px;height:38px}
.ov-form i{width:12px;border-radius:3px 3px 0 0;opacity:.9}
.ov-dim{color:#6f7f93}

/* detailed card */
.dc-head{display:flex;align-items:center;gap:.9rem;margin:.1rem 0 .8rem}
.dc-rank{font-size:3rem;font-weight:900;font-style:italic;line-height:1;color:var(--tc)}
.dc-id{flex:1;min-width:0}
.dc-name{font-size:1.5rem;font-weight:900;letter-spacing:.04em;text-transform:uppercase;color:#f7f9fd;line-height:1.1}
.dc-meta{font-size:.82rem;color:#aeb9c9;margin-top:.15rem}
.dc-ghost{font-size:.7rem;color:#8ea0b5;margin-top:.15rem}
.dc-logo{height:46px;max-width:110px;object-fit:contain}
.tr-portrait{position:relative;height:190px;border-radius:12px;overflow:hidden;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;
  background:linear-gradient(180deg,transparent 0%,color-mix(in srgb,var(--tc) 14%,transparent) 100%)}
.tr-portrait>img{height:128px;max-width:100%;object-fit:contain;position:relative;z-index:2}
.tr-num{position:absolute;top:4px;left:10px;font-size:2.1rem;font-weight:900;font-style:italic;color:var(--tc);opacity:.9;z-index:1}
.tr-dname{position:relative;z-index:3;text-align:center;padding:.2rem 0 .4rem;line-height:1.05}
.tr-dname span{display:block;font-size:.72rem;color:#cdd6e3}
.tr-dname b{font-size:1rem;letter-spacing:.08em;color:#fff}
.tr-ghostpic{position:absolute;right:-6%;top:18%;width:48%;opacity:.5;filter:grayscale(1);z-index:0}
.tr-ghostpic img{width:100%;object-fit:contain}
.tr-car{margin-top:.3rem;padding:.2rem 0}
.tr-car img{width:100%;max-height:170px;object-fit:contain;display:block;mix-blend-mode:lighten}
.tr-car.large img{max-height:330px}
.tr-car.missing img.ph{opacity:.55;mix-blend-mode:normal}
.tr-car .tr-sub{text-align:center}

/* season snapshot */
.tr-title{font-size:.72rem;font-weight:900;letter-spacing:.2em;color:#8ea0b5;margin-bottom:.55rem}
.tr-panels{display:grid;grid-template-columns:1fr 1fr;gap:.6rem}
.tr-panel{background:rgba(255,255,255,.035);border-radius:12px;padding:.7rem .85rem;min-height:172px}
.tr-ph{display:flex;justify-content:space-between;gap:.4rem;font-size:.62rem;font-weight:900;letter-spacing:.13em;color:#8ea0b5;margin-bottom:.3rem}
.tr-pr{color:#c6d1e0;letter-spacing:.02em;font-weight:700;white-space:nowrap}
.tr-big{font-size:1.9rem;font-weight:900;color:#fff;line-height:1.05}
.tr-big small{font-size:.75rem;font-weight:700;color:#8ea0b5;margin-left:.2rem}
.tr-sub{font-size:.7rem;color:#8ea0b5;line-height:1.35;margin-top:.3rem}
.tr-sub span{opacity:.7}
.tr-spark{width:100%;height:auto;display:block;margin-top:.2rem}
.tr-brow{display:flex;align-items:center;gap:.5rem;margin:.5rem 0}
.tr-brow b{width:2.3rem;font-size:.72rem;color:#dce6f2}
.tr-btrack{flex:1;height:10px;border-radius:5px;background:#2a3140;overflow:hidden}
.tr-btrack div{height:100%;border-radius:5px}
.tr-brow span{width:2.6rem;text-align:right;font-size:.82rem;font-weight:800;color:#fff}
.tr-legend{display:flex;flex-wrap:wrap;gap:.2rem .7rem;font-size:.68rem;color:#c6d1e0;font-weight:700;margin-top:.4rem}
.tr-legend i{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:.3rem}
.tr-srow{display:flex;align-items:center;gap:.5rem;margin:.4rem 0}
.tr-srow>b{width:2.2rem;font-size:.72rem;color:#dce6f2;letter-spacing:.05em}
.tr-cells{display:flex;gap:2px;flex:1}
.tr-cells i{flex:1;border-radius:3px;min-width:3px}
.tr-bigstrip .tr-srow>b{width:3rem;font-size:.9rem}
.tr-nodata{border:1px dashed #3a4354;border-radius:12px;padding:2rem 1rem;text-align:center;color:#8ea0b5}

/* team page */
.td-kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem 1.2rem;margin:.2rem 0 1rem}
.td-kpis b{display:block;font-size:2rem;font-weight:900;color:#fff;line-height:1}
.td-kpis span{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#8ea0b5;font-weight:700}
.td-kpis div:first-child b{color:var(--tc)}
.td-driver{display:flex;align-items:center;gap:.7rem;padding:.55rem;border-radius:12px;background:rgba(255,255,255,.04);border-left:3px solid var(--tc)}
.td-driver img{width:58px;height:58px;border-radius:50%;object-fit:cover;object-position:top center;background:#1b2230;flex:0 0 auto}
.td-num{font-size:.7rem;font-weight:900;color:var(--tc);font-style:italic}
.td-name{font-size:.85rem;color:#dce6f2;line-height:1.15}
.td-name b{display:block;font-size:.95rem;color:#fff;letter-spacing:.05em}
.td-pts{font-size:.68rem;color:#8ea0b5}

@media (max-width:900px){
  .tr-panels{grid-template-columns:1fr}
  .ov-head,.ov-row{grid-template-columns:40px 1.5fr 1.2fr 1fr}
  .ov-head span:nth-child(n+5),.ov-row>span:nth-child(n+5){display:none}
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# VIEW C: INDIVIDUAL DRIVER PAGE
# ==========================================
if st.session_state.selected_team and st.session_state.selected_driver:
    team_name = st.session_state.selected_team
    target_last_name = st.session_state.selected_driver
    team_drivers = teams.get(team_name, [])
    driver = next((d for d in team_drivers if d['last_name'] == target_last_name), None)

    nav_left, nav_right = st.columns([4, 1])
    with nav_left:
        from_team = st.session_state.get("driver_origin") == "team"
        if st.button(f"← Back to {team_short_name(team_name)}" if from_team else "← All teams", key="back_from_driver"):
            if not from_team:
                st.session_state.selected_team = None
            st.session_state.selected_driver = None
            st.rerun()
    with nav_right:
        st.markdown(f"<div style='text-align:right;color:#AAB7C9;font-weight:800;padding-top:8px'>{st.session_state.global_season} SEASON</div>", unsafe_allow_html=True)
    if driver:
        st.markdown(f"<div style='margin:0 0 16px'><div style='color:var(--f1-accent);font-size:.78rem;letter-spacing:.18em;font-weight:800'>DRIVER INTELLIGENCE / {html.escape(team_name.upper())}</div><div style='color:#F7F9FD;font-size:clamp(1.7rem,2.6vw,2.6rem);font-weight:900;line-height:1.2'>{html.escape(driver['first_name'])} <span style='color:var(--f1-accent)'>{html.escape(driver['last_name'].upper())}</span></div></div>", unsafe_allow_html=True)
    selected_season = st.session_state.global_season

    if driver:
        team_color = f"#{driver['team_colour']}"
        logo_path = get_team_logo(team_name)
        logo_base64 = get_local_image_base64(PROJECT_ROOT / logo_path)
        img_src = get_local_image_base64(PROJECT_ROOT / clean_driver_filename(driver['first_name'], driver['last_name'], selected_season)) or driver.get('headshot_url') or DEFAULT_HEADSHOT
        
        col_img, col_radar, col_scorecard = st.columns([1.08, 1.22, 1.70], gap="medium")
        
        with col_img:
            watermark_html = f'<img src="{logo_base64}" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80%; max-height: 200px; object-fit: contain; opacity: 0.12; z-index: 0; pointer-events:none;">' if logo_base64 else ''
            st.markdown(f"""
            <div style="position: relative; background: linear-gradient(135deg, {team_color}30 0%, rgba(20, 24, 33, 0.95) 100%); border-top: 5px solid {team_color}; border-radius: 12px; padding: 20px; text-align: center; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.4);">
                {watermark_html}
                <div style="position: relative; z-index: 1;">
                    <span style="font-size: 3.5rem; font-weight: 900; font-style: italic; color: {team_color}; line-height: 1;">{driver["driver_number"]}</span><br>
                    <img src="{img_src}" style="width: 95%; max-height: 260px; object-fit: contain; margin-top: 10px;">
                    <h2 style="margin: 15px 0 0 0; color: #fff; font-size: 1.4rem;">{driver["first_name"]} <b style="font-size: 1.2em;">{driver["last_name"].upper()}</b></h2>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Biography uses a season-specific driver record, never hardcoded details.
            bio = fetch_driver_biography(selected_season, driver["first_name"], driver["last_name"])
            birth_raw = bio.get("dateOfBirth", "")
            birth_date = pd.to_datetime(birth_raw, errors="coerce")
            birth_display = birth_date.strftime("%d %b %Y") if pd.notna(birth_date) else "Unavailable"
            birth_year = str(birth_date.year) if pd.notna(birth_date) else "Unavailable"
            nationality = bio.get("nationality") or "Unavailable"
            driver_code = bio.get("code") or driver.get("name_acronym") or "Unavailable"
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#1c2431,#111721);border:1px solid #354052;
            border-left:4px solid {team_color};border-radius:12px;padding:20px;margin-top:12px;
            min-height:304px; display:flex; flex-direction:column; justify-content:space-between;">
    <div>
        <div style="font-size:.76rem;letter-spacing:.14em;font-weight:800;color:{team_color};margin-bottom:16px;">DRIVER INFORMATION</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px 10px;">
            <div><small style="color:#A7B3C5;">Nationality</small><div style="font-weight:800;color:white;">{nationality}</div></div>
            <div><small style="color:#A7B3C5;">Birth year</small><div style="font-weight:800;color:white;">{birth_year}</div></div>
            <div><small style="color:#A7B3C5;">Date of birth</small><div style="font-weight:800;color:white;">{birth_display}</div></div>
            <div><small style="color:#A7B3C5;">Driver code</small><div style="font-weight:800;color:white;">{driver_code}</div></div>
            <div style="grid-column:1/-1;"><small style="color:#A7B3C5;">{selected_season} constructor</small><div style="font-weight:800;color:white;">{team_name}</div></div>
        </div>
    </div>
    <div style="font-size:.7rem;color:#9AA6B8;margin-top:20px;">Biographical data: Jolpica F1 API</div>
</div>
            """, unsafe_allow_html=True)

        with col_radar:
            ratings = calculate_driver_ratings(driver['last_name'], season=selected_season)
            categories = ['Pace', 'Qualifying', 'Racecraft', 'Consistency', 'Experience']
            values = [ratings['Pace'], ratings['Qualifying'], ratings['Racecraft'], ratings['Consistency'], ratings['Experience']]
                
            fig_rad = go.Figure(go.Scatterpolar(
                r=values + [values[0]], 
                theta=categories + [categories[0]], 
                fill='toself', 
                line=dict(color=team_color, width=3), 
                fillcolor=f"{team_color}35", 
                marker=dict(size=6, color=team_color),
                hovertemplate="<b>%{theta} Rating: %{r}</b><extra></extra>"
            ))
            
            # Title embedded directly inside the Plotly container
            fig_rad.update_layout(
                title=dict(
                    text="<b>DRIVER PERFORMANCE RATINGS</b>",
                    font=dict(color="#F3F6FC", size=14),
                    x=0.5,           # Centers the title horizontally
                    xanchor="center",
                    y=0.98           # Pushes the title to the very top of the container
                ),
                polar=dict(
                    bgcolor="rgba(20, 24, 33, 0.6)", 
                    radialaxis=dict(visible=True, range=[40, 100], gridcolor="#383E4B"), 
                    angularaxis=dict(gridcolor="#383E4B", tickfont=dict(color="#FFF"))
                ), 
                paper_bgcolor="rgba(0,0,0,0)", 
                showlegend=False, 
                height=320, 
                margin=dict(l=40, r=40, t=40, b=40) # Increased top margin (t) to make room for the title
            )
            
            st.plotly_chart(fig_rad, width="stretch")
            
            # 1. Row-by-Row Custom HTML Metrics (Moved up)
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; gap: 14px; padding: 0 10px; margin-top: -10px; margin-bottom: 12px;">
                <div title="Methodology: Modeled from average race finishing position." style="display: flex; justify-content: space-between; align-items: center; background: rgba(20,24,33,0.6); padding: 11px 15px; border-radius: 6px; border-left: 3px solid {team_color}; cursor: help;">
                    <span style="font-size: 0.8rem; color: #A0AAB8; font-weight: bold; border-bottom: 1px dotted #A0AAB8;">Pace</span>
                    <span style="font-size: 1.2rem; color: #FFF; font-weight: 900;">{ratings['Pace']}</span>
                </div>
                <div title="Methodology: Modeled from average starting grid position." style="display: flex; justify-content: space-between; align-items: center; background: rgba(20,24,33,0.6); padding: 11px 15px; border-radius: 6px; border-left: 3px solid {team_color}; cursor: help;">
                    <span style="font-size: 0.8rem; color: #A0AAB8; font-weight: bold; border-bottom: 1px dotted #A0AAB8;">Qualifying</span>
                    <span style="font-size: 1.2rem; color: #FFF; font-weight: 900;">{ratings['Qualifying']}</span>
                </div>
                <div title="Methodology: Delta of positions gained vs lost during races." style="display: flex; justify-content: space-between; align-items: center; background: rgba(20,24,33,0.6); padding: 11px 15px; border-radius: 6px; border-left: 3px solid {team_color}; cursor: help;">
                    <span style="font-size: 0.8rem; color: #A0AAB8; font-weight: bold; border-bottom: 1px dotted #A0AAB8;">Racecraft</span>
                    <span style="font-size: 1.2rem; color: #FFF; font-weight: 900;">{ratings['Racecraft']}</span>
                </div>
                <div title="Methodology: Season finish-rate heuristic (DNF statuses estimated)." style="display: flex; justify-content: space-between; align-items: center; background: rgba(20,24,33,0.6); padding: 11px 15px; border-radius: 6px; border-left: 3px solid {team_color}; cursor: help;">
                    <span style="font-size: 0.8rem; color: #A0AAB8; font-weight: bold; border-bottom: 1px dotted #A0AAB8;">Consistency</span>
                    <span style="font-size: 1.2rem; color: #FFF; font-weight: 900;">{ratings['Consistency']}</span>
                </div>
                <div title="Methodology: Heuristic based on available season race records." style="display: flex; justify-content: space-between; align-items: center; background: rgba(20,24,33,0.6); padding: 11px 15px; border-radius: 6px; border-left: 3px solid {team_color}; cursor: help;">
                    <span style="font-size: 0.8rem; color: #A0AAB8; font-weight: bold; border-bottom: 1px dotted #A0AAB8;">Experience</span>
                    <span style="font-size: 1.2rem; color: #FFF; font-weight: 900;">{ratings['Experience']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Season Win Rate Bar (Moved down)
            radar_stats = get_driver_season_summary(driver['last_name'], selected_season)
            gps_count = radar_stats['gps']
            win_count = radar_stats['wins']
            win_pct = round((win_count / gps_count * 100), 1) if gps_count > 0 else 0
            
            st.markdown(f"""
            <div style="padding: 0 10px; margin-top: 0px;">
                <div style="background: rgba(20,24,33,0.6); border: 1px solid #2A303C; border-radius: 8px; padding: 14px 15px;">
                    <div style="font-size: 0.7rem; color: #A0AAB8; letter-spacing: 1px; margin-bottom: 6px; font-weight: bold;">SEASON WIN RATE</div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="color: {team_color}; font-size: 1.6rem; font-weight: 900; line-height: 1;">{win_pct}%</span>
                        <div style="flex: 1; height: 6px; background: #2A303C; border-radius: 3px; overflow: hidden;">
                            <div style="width: {win_pct}%; height: 100%; background: {team_color};"></div>
                        </div>
                    </div>
                    <div style="font-size: 0.7rem; color: #888; margin-top: 6px;">{win_count} wins &bull; {gps_count - win_count} non-wins &bull; {gps_count} races</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top Tracks Mini-Card for the empty space
            top_tracks = get_top_tracks(driver['last_name'])
            if top_tracks:
                tracks_html = "".join([
                    f"<div style='display: flex; justify-content: space-between; font-size: 0.8rem; padding: 4px 0; border-bottom: 1px solid #2A303C;'>"
                    f"<span style='color: #E8EDF5;'>{t['name']}</span>"
                    f"<span style='color: {team_color}; font-weight: bold;'>Avg Finish: P{t['avg_finish']:.1f}</span>"
                    f"</div>" for t in top_tracks
                ])
                st.markdown(f"""
                <div style="background: rgba(20,24,33,0.6); border: 1px solid #354052; border-left: 3px solid {team_color}; border-radius: 8px; padding: 12px 15px; margin-top: 15px;">
                    <div style="font-size: 0.7rem; color: #A0AAB8; letter-spacing: 1px; font-weight: bold; margin-bottom: 8px; text-transform: uppercase;">Top Strongest Tracks</div>
                    {tracks_html}
                </div>
                """, unsafe_allow_html=True)
            
        with col_scorecard:
            stats = get_driver_season_summary(driver['last_name'], selected_season)
            dnf_pct = round((stats['dnfs'] / stats['gps'] * 100), 1) if stats['gps'] > 0 else 0
            
            # SVGs color-matched to the specific team
            svg_gp = f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{team_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>'
            svg_win = f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{team_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path><path d="M4 22h16"></path><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"></path><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"></path><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"></path></svg>'
            svg_pod = f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{team_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"></circle><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline></svg>'
            svg_pole = f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{team_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="13" r="8"></circle><polyline points="12 9 12 13 14 15"></polyline><line x1="16" y1="4" x2="14" y2="6"></line><line x1="8" y1="4" x2="10" y2="6"></line><line x1="12" y1="2" x2="12" y2="4"></line></svg>'
            svg_pts = f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{team_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline><polyline points="16 7 22 7 22 13"></polyline></svg>'
            svg_dnf = f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{team_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'

            # 1. BigDataF1 Style Grid Scorecard with Icons
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px;">
                <div style="background: rgba(20,24,33,0.6); border: 1px solid #2A303C; border-radius: 8px; padding: 12px 5px; text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 900; color: #FFF; line-height: 1;">{stats['gps']}</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 4px; font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; font-weight: bold;">
                        {svg_gp} GPs
                    </div>
                </div>
                <div style="background: rgba(20,24,33,0.6); border: 1px solid #2A303C; border-radius: 8px; padding: 12px 5px; text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 900; color: #FFF; line-height: 1;">{stats['wins']}</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 4px; font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; font-weight: bold;">
                        {svg_win} Wins
                    </div>
                </div>
                <div style="background: rgba(20,24,33,0.6); border: 1px solid #2A303C; border-radius: 8px; padding: 12px 5px; text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 900; color: #FFF; line-height: 1;">{stats['podiums']}</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 4px; font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; font-weight: bold;">
                        {svg_pod} Podiums
                    </div>
                </div>
                <div style="background: rgba(20,24,33,0.6); border: 1px solid #2A303C; border-radius: 8px; padding: 12px 5px; text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 900; color: #FFF; line-height: 1;">{stats['poles']}</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 4px; font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; font-weight: bold;">
                        {svg_pole} Poles
                    </div>
                </div>
                <div style="background: rgba(20,24,33,0.6); border: 1px solid #2A303C; border-radius: 8px; padding: 12px 5px; text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 900; color: #FFF; line-height: 1;">{stats['points']}</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 4px; font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; font-weight: bold;">
                        {svg_pts} Points
                    </div>
                </div>
                <div style="background: rgba(20,24,33,0.6); border: 1px solid #2A303C; border-radius: 8px; padding: 12px 5px; text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 900; color: #FFF; line-height: 1;">{stats['dnfs']}</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 4px; font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; font-weight: bold;">
                        {svg_dnf} DNFs ({dnf_pct}%)
                    </div>
                </div>
            </div>
            
            <!-- 2. Restored Season Summary Bar -->
            <div style="background: rgba(20,24,33,0.9); border: 1px solid #2A303C; border-radius: 8px; padding: 14px 15px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <div style="color: #888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;">{selected_season} Season</div>
                <div style="color: {team_color}; font-size: 1.4rem; font-weight: 900;">{stats['wdc']}</div>
                <div style="color: #FFF; font-size: 1.3rem; font-weight: 900;">{stats['points']} <span style="font-size: 0.75rem; color: #888;">PTS</span></div>
                <div style="color: #FFF; font-size: 1.3rem; font-weight: 900;">{stats['wins']} <span style="font-size: 0.75rem; color: #888;">WINS</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. H2H VS TEAMMATES Section (Appended directly underneath)
            teammate = next((d for d in team_drivers if d['last_name'] != driver['last_name'] and d.get('role', 'main') == 'main'), None)
            
            h2h_html = ""
            if teammate:
                h2h = calculate_h2h_stats(driver['last_name'], teammate['last_name'], selected_season)
                if h2h:
                    q_ahead, q_behind = h2h['quali']
                    r_ahead, r_behind = h2h['race']
                    q_tot, r_tot = (q_ahead + q_behind), (r_ahead + r_behind)
                    q_pct = round((q_ahead / q_tot * 100), 2) if q_tot > 0 else 0
                    r_pct = round((r_ahead / r_tot * 100), 2) if r_tot > 0 else 0
                    
                    h2h_html = f"""
                    <div style="background: rgba(20,24,33,0.6); border: 1px solid #2A303C; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                            <span style="color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;">H2H VS TEAMMATES</span>
                            <span style="color: {team_color}; font-size: 0.75rem; font-weight: bold;">vs {teammate['last_name'].upper()}</span>
                        </div>
                        <div style="display: flex; gap: 20px;">
                            <div style="flex: 1;">
                                <div style="font-size: 0.7rem; color: #A0AAB8; letter-spacing: 1px; margin-bottom: 5px;">RACE</div>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span style="color: {team_color}; font-size: 1.6rem; font-weight: 900;">{r_pct}%</span>
                                    <div style="flex: 1; height: 6px; background: #2A303C; border-radius: 3px; overflow: hidden;">
                                        <div style="width: {r_pct}%; height: 100%; background: {team_color};"></div>
                                    </div>
                                </div>
                                <div style="font-size: 0.7rem; color: #888; margin-top: 5px;">{r_ahead} ahead &bull; {r_behind} behind &bull; {r_tot} races</div>
                            </div>
                            <div style="flex: 1; border-left: 1px solid #2A303C; padding-left: 20px;">
                                <div style="font-size: 0.7rem; color: #A0AAB8; letter-spacing: 1px; margin-bottom: 5px;">QUALIFYING</div>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span style="color: {team_color}; font-size: 1.6rem; font-weight: 900;">{q_pct}%</span>
                                    <div style="flex: 1; height: 6px; background: #2A303C; border-radius: 3px; overflow: hidden;">
                                        <div style="width: {q_pct}%; height: 100%; background: {team_color};"></div>
                                    </div>
                                </div>
                                <div style="font-size: 0.7rem; color: #888; margin-top: 5px;">{q_ahead} ahead &bull; {q_behind} behind &bull; {q_tot} sessions</div>
                            </div>
                        </div>
                    </div>
                    """
                else:
                    h2h_html = f'<div style="background: rgba(20,24,33,0.6); border: 1px solid #2A303C; border-radius: 8px; padding: 15px; font-size: 0.8rem; color: #888;">H2H data unavailable for {selected_season}.</div>'
            st.markdown(h2h_html, unsafe_allow_html=True)

            # All standings are displayed in a fixed-height scrollable leaderboard.
            render_scrollable_wdc(selected_season, driver["last_name"])

        st.markdown("---")
        st.subheader("📊 Season Performance Trend")
        if st.button("↻ Refresh complete season results from Jolpica", key=f"refresh_races_{selected_season}"):
            with st.spinner("Downloading all race-result pages and updating the offline cache…"):
                fetch_api_results.clear()
                refreshed = fetch_api_results(selected_season, refresh=True)
                fetch_api_results.clear()
            if refreshed:
                st.success(f"Saved {len(refreshed)} available Grand Prix rounds for {selected_season}.")
                st.rerun()
            else:
                st.error("Refresh did not return results. Existing saved data, if any, was retained.")
        finish_chart, race_count = build_driver_finish_trend(selected_season, driver["last_name"], team_color, teammate["last_name"] if teammate else None)
        if finish_chart is not None:
            st.plotly_chart(finish_chart, width="stretch")
        else:
            st.info("Race-by-race classified finishing positions are unavailable for this driver and season.")

        st.markdown("### More Season Analytics")
        tab_points, tab_grid, tab_laps = st.tabs(["🏆 Points per Grand Prix", "📈 Grid vs Finish", "⏱️ Lap-Time Consistency"])
        with tab_points:
            points_chart = build_points_per_race_chart(selected_season, driver["last_name"], team_color)
            if points_chart is not None:
                st.plotly_chart(points_chart, width="stretch")
                st.caption("Official race-result points by round, including any points credited in the race results. Sprint-only points may not be represented.")
            else:
                st.info("Points-by-race data are not available for this season and driver.")
        with tab_grid:
            grid_chart, compared_races = build_grid_vs_finish_chart(selected_season, driver["last_name"], team_color)
            if grid_chart is not None:
                st.plotly_chart(grid_chart, width="stretch")
                st.caption(f"{compared_races} races with a valid starting grid. Positive bars indicate places gained from grid to official classified position. Retirements and penalties can affect classification; this is not an overtaking count.")
            else:
                st.info("Grid-to-classification data are unavailable for this driver and season.")
        with tab_laps:
            show_lap_consistency_chart(selected_season, driver, team_drivers, team_color)

        st.markdown("---")
        st.markdown("### Tyre Strategy & Performance")
        render_tyre_analysis(selected_season, driver, team_color)

# ==========================================
# VIEW D: TEAM PAGE  /  VIEW B: TEAM OVERVIEW + DETAILED CARDS
# ==========================================
elif st.session_state.selected_team:
    render_team_page(st.session_state.selected_team, teams, st.session_state.global_season)

else:
    c_title, c_select = st.columns([7, 3])
    with c_title:
        st.title("🏎️ F1 Driver Statistics & Constructors")
    with c_select:
        season_options = [2024, 2025, 2026]

        def update_season():
            st.session_state.global_season = st.session_state.temp_season_sel
            st.session_state.selected_team = None
            st.session_state.selected_driver = None
            st.session_state.expanded_team = None

        st.selectbox(
            "Data Filter (Historical Season)",
            season_options,
            index=season_options.index(st.session_state.global_season),
            key="temp_season_sel",
            on_change=update_season,
        )

    st.markdown(
        "<p class='for1sight-caption'>Every constructor at a glance: expand a team for its detailed card, "
        "open a driver for individual statistics, or open the team dashboard for the full season. "
        "Use the separate Driver Comparison page to compare two drivers.</p>",
        unsafe_allow_html=True,
    )

    if not teams:
        st.error(f"No driver roster could be loaded for {st.session_state.global_season}.")
        with st.expander("Data paths and troubleshooting", expanded=True):
            st.code(
                f"Project root: {PROJECT_ROOT}\n"
                f"Warehouse: {WAREHOUSE_DIR}\n"
                f"Working directory: {Path.cwd()}"
            )
        if st.button("Retry roster loading"):
            get_season_roster.clear()
            st.rerun()

    season = st.session_state.global_season
    stats = get_team_season_stats(season)
    rank_info = build_team_rank_info(stats, fetch_constructor_standings(season))

    ctl_layout, ctl_sort, ctl_refresh = st.columns([2.2, 2.4, 1.4], vertical_alignment="bottom")
    with ctl_layout:
        layout = st.segmented_control("Layout", ["Overview", "Detailed cards"], default="Overview",
                                      key="team_layout_mode") or "Overview"
    with ctl_sort:
        sort_mode = st.segmented_control("Order teams by", ["Standings", "A–Z"], default="Standings",
                                         key="team_sort_mode") or "Standings"
    with ctl_refresh:
        if st.button("↻ Refresh standings", key="refresh_constructor_standings", width="stretch"):
            fetch_constructor_standings.clear()
            fetch_sprint_points.clear()
            fetch_constructor_standings(season, refresh=True)
            fetch_sprint_points(season, refresh=True)
            fetch_constructor_standings.clear()
            fetch_sprint_points.clear()
            get_team_season_stats.clear()
            st.rerun()

    names = list(teams.keys())
    if sort_mode == "Standings":
        names.sort(key=lambda n: (rank_info[n]["rank"] if n in rank_info else 999, n))
    else:
        names.sort()

    expanded = st.session_state.get("expanded_team")
    if layout == "Overview":
        head_col, _ = st.columns([14, 1.7])
        head_col.markdown(overview_head_html(), unsafe_allow_html=True)

    with st.container(key="ovlist"):
        for team_name in names:
            if layout != "Overview":
                render_team_card(team_name, season, teams, stats, rank_info)
                continue
            all_drivers = teams[team_name]
            main_drivers = [d for d in all_drivers if d.get("role", "main") == "main"] or all_drivers[:2]
            if not main_drivers:
                continue
            color = team_colour_for(team_name, main_drivers)
            is_open = expanded == (season, team_name)
            row_col, btn_col = st.columns([14, 1.7], vertical_alignment="center", gap="small")
            row_col.markdown(overview_row_html(team_name, season, stats, rank_info, main_drivers, color),
                             unsafe_allow_html=True)
            if btn_col.button("Hide ▴" if is_open else "Details ▾", key=f"ov_toggle_{season}_{_akey(team_name)}",
                              width="stretch"):
                st.session_state.expanded_team = None if is_open else (season, team_name)
                st.rerun()
            if is_open:
                render_team_card(team_name, season, teams, stats, rank_info)

    st.caption(
        "Points include sprint results. Points-race sparklines share one scale so teams are directly comparable. "
        "Average grid/finish only counts races the car finished."
    )
