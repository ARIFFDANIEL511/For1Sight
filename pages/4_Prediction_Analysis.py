import base64
import hashlib
import mimetypes
import textwrap
import difflib
import html
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from future_race_predictor import (
    FEATURE_COLUMNS,
    TARGET_SEASON_ROSTER,
    build_future_feature_frame,
    fill_remaining_by_history,
    get_current_driver_roster,
    get_race_catalog,
    get_roster_as_of,
    predict_event,
    validate_warehouse,
)

st.set_page_config(
    page_title="Prediction Analysis | For1Sight",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE_PATH = ROOT / "for1sight_warehouse" / "prediction_training_v2_8_prerace.parquet"
# Support both project layouts:
#   A) models/v2_8/  (older/local layout)
#   B) v2_8/         (uploaded V2.8 project ZIP layout)
_MODEL_DIR_CANDIDATES = [
    ROOT / "models" / "v2_8",
    ROOT / "v2_8",
]
MODEL_DIR = next(
    (candidate for candidate in _MODEL_DIR_CANDIDATES
     if (candidate / "gradient_boosting_deployment_v2_8.joblib").is_file()
     and (candidate / "random_forest_deployment_v2_8.joblib").is_file()
     and (candidate / "logistic_regression_deployment_v2_8.joblib").is_file()),
    _MODEL_DIR_CANDIDATES[0],
)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression_deployment_v2_8.joblib",
    "Random Forest": "random_forest_deployment_v2_8.joblib",
    "Gradient Boosting": "gradient_boosting_deployment_v2_8.joblib",
}
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
PREDICTION_YEAR = 2027
HISTORICAL_START_YEAR = 2024
HISTORICAL_END_YEAR = 2026

# ---------------------------------------------------------------------------
# Hybrid F1 analytics styling: motorsport red + restrained analytics blue.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --f1-red: #e10600;
        --f1-blue: #55b8f2;
        --panel: #11151d;
        --panel-2: #171d28;
        --line: rgba(148,163,184,.20);
        --muted: #9aa8ba;
    }
    .block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(ellipse at 8% 0%, rgba(225,6,0,.12), transparent 32%),
          radial-gradient(ellipse at 95% 12%, rgba(40,130,190,.10), transparent 30%);
    }
    .hero {
        padding: 1.35rem 1.55rem;
        border: 1px solid rgba(225,6,0,.38);
        border-left: 5px solid var(--f1-red);
        border-radius: 16px;
        background: linear-gradient(115deg, rgba(225,6,0,.13), rgba(20,29,43,.94) 48%, rgba(33,78,111,.20));
        margin-bottom: 1.15rem;
    }
    .hero-kicker {font-size:.72rem; letter-spacing:.16em; color:#ff7772; font-weight:800;}
    .hero-title {font-size:2.05rem; line-height:1.15; font-weight:850; margin:.28rem 0 .35rem;}
    .hero-sub {color:#b7c3d2; font-size:.94rem;}
    .section-head {
        display:flex; align-items:center; gap:.65rem;
        margin:1.4rem 0 .8rem; font-size:1.22rem; font-weight:800;
    }
    .section-num {
        display:inline-flex; align-items:center; justify-content:center;
        width:2rem; height:2rem; border-radius:8px;
        background:rgba(225,6,0,.16); border:1px solid rgba(225,6,0,.45);
        color:#ff6a65; font-size:.9rem;
    }
    .section-note {color:var(--muted); font-size:.84rem; margin-top:-.35rem; margin-bottom:.8rem;}
    .metric-card {
        background:linear-gradient(145deg, rgba(25,34,48,.98), rgba(15,20,29,.98));
        border:1px solid var(--line); border-radius:13px; padding:1rem 1.05rem;
        min-height:112px;
    }
    .metric-label {font-size:.74rem; color:#9eacbd; text-transform:uppercase; letter-spacing:.08em; font-weight:750;}
    .metric-value {font-size:1.65rem; font-weight:850; color:#f4f7fb; margin-top:.25rem;}
    .metric-foot {font-size:.76rem; color:#91a0b2; margin-top:.15rem;}
    .pill {
        display:inline-block; padding:.22rem .58rem; border-radius:999px;
        border:1px solid rgba(85,184,242,.35); background:rgba(85,184,242,.10);
        color:#9bd8ff; font-size:.72rem; font-weight:750; letter-spacing:.04em;
    }
    .subtle-card {
        background:rgba(18,24,34,.78); border:1px solid var(--line);
        border-radius:13px; padding:1rem 1.1rem;
    }
    div[data-testid="stForm"] {
        background:rgba(17,22,31,.76); border:1px solid var(--line);
        padding:1rem; border-radius:14px;
    }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
        background:linear-gradient(90deg,#e10600,#b90000); border:0; font-weight:800;
    }
    div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover {
        background:linear-gradient(90deg,#ff241d,#d20a04); border:0;
    }
    [data-testid="stMetric"] {
        background:rgba(17,22,31,.72); border:1px solid var(--line);
        padding:.75rem .9rem; border-radius:12px;
    }
    [data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:10px; overflow:hidden;}
    .footnote {font-size:.76rem; color:#9aa8ba;}
    
    .for1sight-driver-card {
        --team-accent: #55b8f2;
        height: 100%;
        min-width: 0;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,.22);
        border-top: 3px solid var(--team-accent);
        border-radius: 12px;
        background: linear-gradient(155deg, rgba(20,27,38,.98), rgba(7,10,15,.98));
        box-shadow: 0 5px 16px rgba(0,0,0,.18);
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }
    .for1sight-driver-card:hover {
        transform: translateY(-3px);
        border-color: var(--team-accent);
        box-shadow: 0 9px 22px rgba(0,0,0,.30);
    }
    .for1sight-driver-photo {
        position: relative;
        height: 148px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background: linear-gradient(145deg,#182333,#0c1119);
    }
    .driver-portrait {
        display: block;
        width: 100%;
        height: 148px;
        object-fit: contain;
        object-position: center bottom;
    }
    .driver-portrait-placeholder {
        width: 100%;
        height: 148px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #55b8f2;
        font-size: 2rem;
    }
    .for1sight-driver-number {
        position: absolute;
        top: 8px;
        right: 8px;
        padding: 4px 9px;
        border-radius: 7px;
        color: #fff;
        background: var(--team-accent);
        font-size: .78rem;
        font-weight: 850;
        line-height: 1.2;
        box-shadow: 0 2px 8px rgba(0,0,0,.25);
    }
    .for1sight-driver-details {
        padding: 9px 10px 10px;
        border-left: 3px solid var(--team-accent);
    }
    .for1sight-driver-name {
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        color: #f8fafc;
        font-size: .91rem;
        font-weight: 750;
        line-height: 1.25;
        margin-bottom: 8px;
    }
    .for1sight-driver-team-row {
        display: flex;
        align-items: center;
        gap: 9px;
        min-height: 34px;
    }
    .for1sight-driver-logo-wrap {
        width: 68px;
        height: 34px;
        flex: 0 0 68px;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        overflow: hidden;
    }
    .driver-team-logo {
        display: block;
        max-width: 68px;
        max-height: 32px;
        width: auto;
        height: auto;
        object-fit: contain;
    }
    .driver-team-logo-fallback {
        color: var(--team-accent);
        font-size: .8rem;
        font-weight: 900;
        letter-spacing: .08em;
    }
    .for1sight-driver-team-name {
        min-width: 0;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        color: #9fb0c4;
        font-size: .74rem;
        line-height: 1.2;
    }

    .prediction-podium-card {
        position: relative;
        display: flex;
        align-items: center;
        gap: 14px;
        width: 100%;
        min-width: 0;
        min-height: 132px;
        padding: 12px 14px;
        overflow: hidden;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,.22);
        box-shadow: 0 8px 22px rgba(0,0,0,.22);
        margin-bottom: 10px;
    }
    .prediction-podium-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 5px;
        background: var(--podium-accent, #D4AF37);
    }
    .prediction-podium-photo-wrap {
        width: 86px;
        height: 106px;
        flex: 0 0 86px;
        display: flex;
        align-items: flex-end;
        justify-content: center;
        overflow: hidden;
        border-radius: 10px;
        background: rgba(7,11,17,.35);
        border: 1px solid rgba(255,255,255,.16);
    }
    .prediction-podium-photo {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center bottom;
    }
    .prediction-podium-photo-fallback {
        display:flex;
        align-items:center;
        justify-content:center;
        width:100%;
        height:100%;
        font-size:2rem;
    }
    .prediction-podium-meta {
        min-width: 0;
        flex: 1 1 auto;
        overflow: hidden;
    }
    .prediction-podium-topline {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:8px;
        margin-bottom:5px;
    }
    .prediction-podium-kicker {
        color: rgba(255,255,255,.88);
        font-size: .68rem;
        font-weight: 850;
        letter-spacing: .10em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .prediction-podium-rank {
        display:inline-flex;
        align-items:center;
        padding:3px 8px;
        border-radius:999px;
        background:rgba(0,0,0,.24);
        color:#fff;
        font-size:.67rem;
        font-weight:900;
        letter-spacing:.06em;
        white-space:nowrap;
    }
    .prediction-podium-identity {
        display:flex;
        align-items:baseline;
        gap:9px;
        min-width:0;
        margin-bottom:3px;
    }
    .prediction-podium-code {
        flex:0 0 auto;
        color:#fff;
        font-size:1.75rem;
        line-height:1;
        font-weight:950;
        white-space:nowrap;
    }
    .prediction-podium-name {
        min-width:0;
        color:#fff;
        font-size:.92rem;
        font-weight:750;
        overflow:hidden;
        white-space:nowrap;
        text-overflow:ellipsis;
    }
    .prediction-podium-team-row {
        display:flex;
        align-items:center;
        gap:7px;
        min-width:0;
        min-height:23px;
        margin:3px 0 4px;
    }
    .prediction-podium-team-logo {
        display:block;
        max-width:48px;
        max-height:21px;
        width:auto;
        height:auto;
        object-fit:contain;
        flex:0 0 auto;
    }
    .prediction-podium-team {
        min-width:0;
        color:rgba(255,255,255,.82);
        font-size:.76rem;
        overflow:hidden;
        white-space:nowrap;
        text-overflow:ellipsis;
    }
    .prediction-podium-score {
        color:#fff;
        font-size:.80rem;
        font-weight:850;
        white-space:nowrap;
    }
    .prediction-podium-foot {
        color:rgba(255,255,255,.68);
        font-size:.66rem;
        margin-top:2px;
    }
    .movement-driver-card {
        display:flex;
        align-items:center;
        gap:11px;
        min-height:91px;
        padding:10px 11px;
        border-radius:12px;
        margin-bottom:9px;
        background:linear-gradient(145deg,rgba(25,34,48,.98),rgba(15,20,29,.98));
    }
    .movement-driver-photo {
        width:57px;
        height:69px;
        flex:0 0 57px;
        display:flex;
        align-items:flex-end;
        justify-content:center;
        overflow:hidden;
        border-radius:8px;
        background:rgba(0,0,0,.22);
    }
    .movement-driver-photo img {
        width:100%;
        height:100%;
        object-fit:contain;
        object-position:center bottom;
    }
    .movement-driver-info {min-width:0;flex:1;}
    .movement-driver-heading {
        display:flex;
        align-items:baseline;
        flex-wrap:wrap;
        gap:7px;
        margin:2px 0;
    }
    .movement-driver-code {font-size:1.35rem;font-weight:950;color:#fff;}
    .movement-driver-delta {font-size:.95rem;font-weight:900;}
    .movement-driver-name {
        color:#edf3fa;font-size:.78rem;font-weight:700;
        overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
    }
    .movement-driver-route {color:#aab8c8;font-size:.70rem;}
    .movement-kpi {
        background:linear-gradient(145deg,rgba(25,34,48,.98),rgba(15,20,29,.98));
        border:1px solid rgba(148,163,184,.20);
        border-radius:10px;
        padding:9px 10px;
        min-height:72px;
    }
    .movement-kpi-label {
        color:#aab8c8;font-size:.66rem;font-weight:750;
        letter-spacing:.04em;line-height:1.25;
    }
    .movement-kpi-value {
        color:#f7fafc;font-size:1.25rem;font-weight:900;
        line-height:1.2;margin-top:5px;
    }


</style>
    """,
    unsafe_allow_html=True,
)


def section_heading(number: str, title: str, note: str = ""):
    st.markdown(
        f'<div class="section-head"><span class="section-num">{number}</span>'
        f'<span>{title}</span></div>',
        unsafe_allow_html=True,
    )
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, foot: str = ""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading For1Sight warehouse...")
def load_warehouse(path: str, modified_ns: int):
    return pd.read_parquet(path, engine="pyarrow")


@st.cache_data(show_spinner=False)
def load_fixed_roster(path: str, modified_ns: int, roster_year: int):
    """Cache the fixed-season roster; the warehouse itself is already cached."""
    warehouse = load_warehouse(path, modified_ns)
    return get_current_driver_roster(warehouse, roster_year=roster_year)


@st.cache_data(show_spinner=False)
def load_cached_roster_features(
    path: str,
    modified_ns: int,
    target_year: int,
    driver_codes: tuple[str, ...],
):
    """Prepare pre-race historical driver/team features once per roster and target year.

    V2.8's get_roster_as_of() intentionally calculates history strictly before
    target_year; the selected reference round is not a model feature. Therefore
    changing the selected Grand Prix should not repeat this calculation.
    """
    warehouse = load_warehouse(path, modified_ns)
    return get_roster_as_of(
        warehouse,
        target_year=int(target_year),
        target_round=None,
        driver_codes=list(driver_codes),
    )


@st.cache_data(show_spinner=False)
def load_cached_historical_catalog(path: str, modified_ns: int):
    """Cache the combined 2024–2026 historical event catalogue."""
    warehouse = load_warehouse(path, modified_ns)
    return build_historical_event_catalog(warehouse)


@st.cache_resource(show_spinner="Loading the V2.8 Gradient Boosting deployment model...")
def load_models(model_dir: str, artifact_signature: tuple = ()):
    """Load only the lead V2.8 Gradient Boosting model and verify its feature schema."""
    folder = Path(model_dir)
    models = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = folder / filename
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing V2.8 model artifact: {model_path}")
        model = joblib.load(model_path)
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None and list(model_features) != list(FEATURE_COLUMNS):
            raise ValueError(
                f"{model_name} feature schema does not match the V2.8 eight-feature contract."
            )
        models[model_name] = model

    metadata = {}
    if MODEL_METADATA_PATH.is_file():
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return models, metadata


@st.cache_data(show_spinner="Loading 2026 driver details from FastF1...")
def load_current_season_driver_details(year: int, latest_round: int):
    """Load driver profile details, trying Race first and then other available sessions."""
    import fastf1
    cache_dir = ROOT / "cache" / "fastf1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    errors = []
    for session_name in ("R", "Q", "FP3", "FP2", "FP1"):
        try:
            session = fastf1.get_session(int(year), int(latest_round), session_name)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            results = session.results.copy()
            wanted = ["Abbreviation", "FullName", "TeamName", "DriverNumber", "HeadshotUrl"]
            available = [column for column in wanted if column in results.columns]
            if "Abbreviation" not in available:
                errors.append(f"{session_name}: no driver abbreviation field")
                continue
            records = results[available].fillna("").to_dict("records")
            if records:
                return records
            errors.append(f"{session_name}: no driver result rows")
        except Exception as exc:
            errors.append(f"{session_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "FastF1 could not load driver profiles for this event. " + " | ".join(errors[-3:])
    )


def _asset_key(value: str) -> str:
    """Normalize names so local asset filenames match despite spaces/punctuation."""
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


@st.cache_data(show_spinner=False)
def find_local_asset(folder_name: str, display_name: str, asset_year: int | None = None):
    """Find local assets, including season subfolders such as headshots/2026/."""
    base = ROOT / "assets" / folder_name
    if not base.is_dir() or not display_name:
        return None

    wanted = _asset_key(display_name)
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Prefer the requested season folder, then search all nested folders.
    search_roots = []
    if asset_year is not None:
        season_folder = base / str(int(asset_year))
        if season_folder.is_dir():
            search_roots.append(season_folder)
    search_roots.append(base)

    candidates = []
    seen = set()
    for search_root in search_roots:
        for item in search_root.rglob("*"):
            if item.is_file() and item.suffix.lower() in valid_extensions:
                resolved = str(item.resolve())
                if resolved not in seen:
                    candidates.append(item)
                    seen.add(resolved)

    # If multiple seasons contain the same driver, prefer the selected season.
    if asset_year is not None:
        candidates.sort(
            key=lambda item: (
                0 if str(int(asset_year)) in item.parts else 1,
                len(item.parts),
                item.name.lower(),
            )
        )

    keyed = {_asset_key(item.stem): item for item in candidates}
    if wanted in keyed:
        return keyed[wanted]

    if folder_name == "logos":
        team_aliases = {
            "scuderiaferrari": ["ferrari"],
            "mercedesamgf1team": ["mercedes"],
            "redbullracing": ["redbull"],
            "mclaren": ["mclaren"],
            "astonmartinf1team": ["astonmartin"],
            "bwtalpinef1team": ["alpine"],
            "alpinef1team": ["alpine"],
            "haasf1team": ["haas"],
            "stakef1teamkicksauber": ["sauber", "kicksauber"],
            "kicksauber": ["sauber", "kicksauber"],
            "audif1team": ["audi"],
            "rbf1team": ["rb", "racingbulls", "visacashapprb"],
            "racingbulls": ["racingbulls", "rb"],
            "cadillacf1team": ["cadillac"],
            "williams": ["williams"],
        }
        for alias in team_aliases.get(wanted, []):
            if alias in keyed:
                return keyed[alias]
        for key, path in keyed.items():
            if len(key) >= 4 and (wanted.endswith(key) or key.endswith(wanted)):
                return path

    if folder_name == "headshots" and keyed:
        # Handles small filename variations, e.g. georgerussel.jpg vs
        # the FastF1 display name "George Russell".
        close = difflib.get_close_matches(wanted, list(keyed.keys()), n=1, cutoff=0.88)
        if close:
            return keyed[close[0]]

    return None


def display_driver_name(code: str, driver_lookup: dict):
    info = driver_lookup.get(str(code).upper(), {})
    full_name = info.get("FullName") or info.get("full_name")
    if full_name:
        return full_name
    return str(code).upper()


def display_driver_option(code: str, driver_lookup: dict):
    """Compact selector label: full driver name and racing number only."""
    info = driver_lookup.get(str(code).upper(), {})
    name = info.get("FullName") or str(code).upper()
    number = info.get("DriverNumber")
    suffix = f" · #{number}" if number else ""
    return f"{name}{suffix}"


def team_accent_colour(team: str) -> str:
    """Return a consistent constructor colour for starting-grid row accents."""
    team_key = _asset_key(str(team))
    team_colours = [
        (("ferrari",), "#E10600"),
        (("mercedes",), "#00D2BE"),
        (("redbull",), "#3671C6"),
        (("mclaren",), "#FF8700"),
        (("astonmartin",), "#00665E"),
        (("alpine",), "#0093CC"),
        (("williams",), "#005AFF"),
        (("haas",), "#B6BABD"),
        (("audi", "sauber", "kicksauber"), "#D71920"),
        (("racingbulls", "visacashapprb", "rbf1", "rbf1team"), "#6692FF"),
        (("cadillac",), "#D6B46A"),
    ]
    for aliases, colour in team_colours:
        if any(alias in team_key for alias in aliases):
            return colour
    return "#55B8F2"


def render_starting_grid_row(position: int, code: str, driver_lookup: dict, constructor: str):
    """Render one confirmed grid entry as a compact, full-width row."""
    info = driver_lookup.get(str(code).upper(), {})
    full_name = str(info.get("FullName") or str(code).upper())
    number = str(info.get("DriverNumber") or "—")
    team = str(constructor or info.get("TeamName") or "Team details unavailable")
    accent = team_accent_colour(team)

    with st.container(border=True):
        stripe_col, position_col, name_col, team_col, number_col = st.columns(
            [0.12, 0.65, 3.2, 2.1, 0.8],
            vertical_alignment="center",
        )
        with stripe_col:
            st.markdown(
                f"<div style='width:6px;height:38px;border-radius:5px;background:{accent};margin:auto;'></div>",
                unsafe_allow_html=True,
            )
        with position_col:
            st.markdown(f"**P{position}**")
        with name_col:
            st.markdown(f"**{full_name}**")
        with team_col:
            st.caption(team)
        with number_col:
            st.markdown(f"**#{number}**")


def render_driver_card(code: str, driver_lookup: dict, constructor_fallback: str = "", asset_year: int | None = None):
    info = driver_lookup.get(str(code).upper(), {})
    name = str(info.get("FullName") or str(code).upper())
    team = str(info.get("TeamName") or constructor_fallback or "Team details unavailable")
    number = str(info.get("DriverNumber") or "—")

    local_headshot = find_local_asset("headshots", name, asset_year=asset_year)
    local_logo = find_local_asset("logos", team, asset_year=asset_year)

    # Team accents are visual identifiers only; prediction calculations are untouched.
    team_key = _asset_key(team)
    team_colours = [
        (("ferrari",), "#E10600"),
        (("mercedes",), "#00D2BE"),
        (("redbull",), "#3671C6"),
        (("mclaren",), "#FF8700"),
        (("astonmartin",), "#00665E"),
        (("alpine",), "#0093CC"),
        (("williams",), "#005AFF"),
        (("haas",), "#B6BABD"),
        (("audi", "sauber", "kicksauber"), "#D71920"),
        (("racingbulls", "visacashapprb", "rbf1"), "#6692FF"),
        (("cadillac",), "#D6B46A"),
    ]
    accent = "#55B8F2"
    for aliases, colour in team_colours:
        if any(alias in team_key for alias in aliases):
            accent = colour
            break

    safe_name = html.escape(name)
    safe_team = html.escape(team)
    safe_number = html.escape(number)

    if local_headshot:
        headshot_mime = mimetypes.guess_type(local_headshot.name)[0] or "image/jpeg"
        headshot_b64 = base64.b64encode(local_headshot.read_bytes()).decode("ascii")
        portrait_html = (
            f'<img class="driver-portrait" src="data:{headshot_mime};base64,{headshot_b64}" '
            f'alt="{safe_name}">'
        )
    else:
        portrait_html = '<div class="driver-portrait-placeholder">🏎️</div>'

    if local_logo:
        logo_mime = mimetypes.guess_type(local_logo.name)[0] or "image/jpeg"
        logo_b64 = base64.b64encode(local_logo.read_bytes()).decode("ascii")
        logo_html = (
            f'<img class="driver-team-logo" src="data:{logo_mime};base64,{logo_b64}" '
            f'alt="{safe_team} logo">'
        )
    else:
        logo_html = '<span class="driver-team-logo-fallback">F1</span>'

    st.markdown(
        f"""
        <div class="for1sight-driver-card" style="--team-accent:{accent};">
            <div class="for1sight-driver-photo">
                {portrait_html}
                <span class="for1sight-driver-number">#{safe_number}</span>
            </div>
            <div class="for1sight-driver-details">
                <div class="for1sight-driver-name" title="{safe_name}">{safe_name}</div>
                <div class="for1sight-driver-team-row">
                    <div class="for1sight-driver-logo-wrap">{logo_html}</div>
                    <div class="for1sight-driver-team-name" title="{safe_team}">{safe_team}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_podium_card(
    code: str,
    position_label: str,
    score: float,
    driver_lookup: dict,
    constructor: str,
    asset_year: int | None = None,
    position: int = 1,
):
    """Render compact horizontal podium cards with medal-based P1/P2/P3 colours."""
    info = driver_lookup.get(str(code).upper(), {})
    full_name = str(info.get("FullName") or str(code).upper())
    team = str(info.get("TeamName") or constructor or "Team details unavailable")
    local_headshot = find_local_asset("headshots", full_name, asset_year=asset_year)
    local_logo = find_local_asset("logos", team, asset_year=asset_year)

    medal_styles = {
        1: {
            "name": "GOLD",
            "accent": "#D4AF37",
            "background": "linear-gradient(110deg,rgba(155,112,16,.98),rgba(88,65,16,.96) 48%,rgba(28,25,18,.98))",
        },
        2: {
            "name": "SILVER",
            "accent": "#C0C0C0",
            "background": "linear-gradient(110deg,rgba(115,127,141,.98),rgba(61,70,82,.96) 48%,rgba(24,29,37,.98))",
        },
        3: {
            "name": "BRONZE",
            "accent": "#CD7F32",
            "background": "linear-gradient(110deg,rgba(151,83,34,.98),rgba(91,53,29,.96) 48%,rgba(30,23,20,.98))",
        },
    }
    medal = medal_styles.get(int(position), medal_styles[3])
    safe_code = html.escape(str(code).upper())
    safe_name = html.escape(full_name)
    safe_team = html.escape(team)

    if local_headshot:
        headshot_mime = mimetypes.guess_type(local_headshot.name)[0] or "image/jpeg"
        headshot_b64 = base64.b64encode(local_headshot.read_bytes()).decode("ascii")
        portrait_html = (
            f'<img class="prediction-podium-photo" src="data:{headshot_mime};base64,'
            f'{headshot_b64}" alt="{safe_name}">'
        )
    else:
        portrait_html = '<div class="prediction-podium-photo-fallback">🏎️</div>'

    if local_logo:
        logo_mime = mimetypes.guess_type(local_logo.name)[0] or "image/png"
        logo_b64 = base64.b64encode(local_logo.read_bytes()).decode("ascii")
        logo_html = (
            f'<img class="prediction-podium-team-logo" src="data:{logo_mime};base64,'
            f'{logo_b64}" alt="{safe_team} logo">'
        )
    else:
        logo_html = '<span class="prediction-podium-team">F1</span>'

    st.markdown(
        f"""
        <div class="prediction-podium-card"
             style="--podium-accent:{medal['accent']};background:{medal['background']};
                    border-color:{hex_to_rgba(medal['accent'], .68)};">
            <div class="prediction-podium-photo-wrap">{portrait_html}</div>
            <div class="prediction-podium-meta">
                <div class="prediction-podium-topline">
                    <div class="prediction-podium-kicker">PREDICTED {int(position)} · {medal['name']}</div>
                    <div class="prediction-podium-rank">{medal['name']} MEDAL</div>
                </div>
                <div class="prediction-podium-identity">
                    <div class="prediction-podium-code">{safe_code}</div>
                    <div class="prediction-podium-name" title="{safe_name}">{safe_name}</div>
                </div>
                <div class="prediction-podium-team-row">
                    {logo_html}
                    <div class="prediction-podium-team" title="{safe_team}">{safe_team}</div>
                </div>
                <div class="prediction-podium-score">Relative model score: {float(score):.2%}</div>
                <div class="prediction-podium-foot">Gradient Boosting prediction output</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def find_v28_model_dir():
    """Return a V2.8 folder only when all three deployment artifacts exist."""
    required_paths = [MODEL_DIR / filename for filename in MODEL_FILES.values()]
    return MODEL_DIR if all(path.is_file() for path in required_paths) else None


def model_artifact_signature(model_dir: Path) -> tuple:
    """Cache key that changes whenever any deployment artifact or metadata changes."""
    paths = [model_dir / filename for filename in MODEL_FILES.values()]
    paths.append(model_dir / "model_metadata.json")
    return tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths if path.is_file()
    )


def show_grid_table(grid):
    return pd.DataFrame(
        {"Grid": [f"P{i}" for i in range(1, len(grid) + 1)], "Driver": grid}
    )


def build_historical_event_catalog(warehouse: pd.DataFrame) -> pd.DataFrame:
    """Build selectable model event references from the 2024–2026 warehouse.

    V2.8 uses event_norm as a categorical model feature. Therefore, retain one
    catalogue row per event_norm rather than merging differently named Grand Prix
    events that may happen at the same physical venue.
    """
    required = {"year", "round", "event_name", "event_norm"}
    missing = sorted(required - set(warehouse.columns))
    if missing:
        raise ValueError("Cannot build historical circuit catalogue; missing columns: " + ", ".join(missing))

    catalog = warehouse[["year", "round", "event_name", "event_norm"]].copy()
    catalog["year"] = pd.to_numeric(catalog["year"], errors="coerce")
    catalog["round"] = pd.to_numeric(catalog["round"], errors="coerce")
    catalog["event_name"] = catalog["event_name"].fillna("").astype(str).str.strip()
    catalog["event_norm"] = catalog["event_norm"].fillna("").astype(str).str.strip()
    catalog = catalog.loc[
        catalog["year"].between(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR)
        & catalog["event_name"].ne("")
        & catalog["event_norm"].ne("")
    ].copy()
    if catalog.empty:
        raise ValueError("No Grand Prix records were found for the 2024–2026 historical period.")

    catalog = catalog.sort_values(["year", "round"], kind="mergesort")
    seasons_by_event = (
        catalog.groupby("event_norm")["year"]
        .apply(lambda values: sorted({int(value) for value in values.dropna()}))
        .to_dict()
    )
    # Keep the latest event label/round for the model's event_norm category.
    catalog = catalog.drop_duplicates("event_norm", keep="last").copy()
    catalog["historical_seasons"] = catalog["event_norm"].map(
        lambda key: seasons_by_event.get(key, [])
    )
    catalog["historical_seasons_label"] = catalog["historical_seasons"].map(
        lambda years: ", ".join(str(year) for year in years)
    )
    return catalog.sort_values("event_name").reset_index(drop=True)


def scenario_signature(event, year, round_no, grid):
    payload = "|".join([str(event), str(year), str(round_no), *[str(x) for x in grid]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_scrollable_prediction_chart(
    result: pd.DataFrame,
    model_name: str,
    roster: pd.DataFrame,
    driver_lookup: dict,
    value_column: str,
    metric_label: str,
    chart_title: str,
):
    """
    Render all 22 drivers in a fixed-height scrollable bar chart.
    Bar lengths are scaled to the largest value for readability;
    displayed values retain their actual model output.
    """
    chart_data = result.sort_values(
        value_column, ascending=False, kind="mergesort"
    ).copy()

    values = pd.to_numeric(chart_data[value_column], errors="coerce").fillna(0.0)
    max_value = max(float(values.max()), 1e-12)
    bars = []

    for rank, (_, row) in enumerate(chart_data.iterrows(), start=1):
        code = str(row["driver_code"]).upper()
        try:
            constructor = str(roster.loc[code, "constructor"])
        except (KeyError, TypeError):
            constructor = "Unknown"
        accent = team_accent_colour(constructor)
        value = max(0.0, float(row[value_column]))
        width = max(1.2, min(100.0, value / max_value * 100.0))
        selected_style = (
            "border:2px solid #fff;box-shadow:0 0 0 1px rgba(255,255,255,.45);"
            if rank == 1 else ""
        )
        full_name = str(driver_lookup.get(code, {}).get("FullName") or code)
        display_name = full_name.split()[-1].upper() if full_name else code
        safe_code = html.escape(display_name)
        safe_team = html.escape(constructor)
        value_text = f"{value:.1%}"
        bars.append(
            f"<div class='entry' title='{safe_code} · {safe_team} · {value_text}'>"
            f"<span class='rank'>P{rank}</span>"
            f"<div class='track' style='{selected_style}'>"
            f"<div class='fill' style='width:{width:.2f}%;background:{accent};"
            f"opacity:{1.0 if rank == 1 else .80};'></div>"
            f"<div class='barlabel'><span class='driver'>{safe_code}</span>"
            f"<span class='value'>· {value_text}</span></div>"
            f"</div></div>"
        )

    safe_title = html.escape(chart_title)
    safe_model = html.escape(model_name)
    markup = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box}}
    body{{margin:0;background:#111722;color:#f5f7fc;font:12px Arial,sans-serif}}
    .card{{border:1px solid #303848;border-radius:12px;padding:15px 12px;background:#111722}}
    h3{{font-size:12px;letter-spacing:.07em;margin:0 0 8px;color:#f2f5fa}}
    p{{color:#a4afc0;font-size:11px;margin:0 0 12px}}
    .scroll{{height:242px;overflow-y:auto;padding-right:7px;
      scrollbar-color:#657085 #151a23;scrollbar-width:thin}}
    .entry{{display:flex;gap:7px;align-items:center;margin-bottom:8px;height:24px}}
    .rank{{width:25px;flex-shrink:0;font-size:10px;color:#c3cfdf;text-align:right}}
    .track{{height:23px;flex:1;position:relative;background:#202734;border-radius:3px;overflow:hidden}}
    .fill{{height:100%;border-radius:2px;min-width:3px}}
    .barlabel{{position:absolute;inset:0;display:flex;align-items:center;gap:5px;
      padding-left:7px;color:white;font-size:10px;font-weight:800;
      text-shadow:0 1px 3px #000,0 0 3px #000;white-space:nowrap}}
    .barlabel .value{{font-size:9px;font-weight:650}}
    .foot{{margin-top:8px;color:#9daabd;font-size:10px}}
    </style></head><body><div class='card'>
    <h3>{safe_title}</h3>
    <p>{safe_model} · Scroll to view all {len(bars)} drivers · P1 outlined in white</p>
    <div class='scroll'>{''.join(bars)}</div>
    <div class='foot'>{html.escape(metric_label)} · Constructor colours</div>
    </div></body></html>"""
    st.components.v1.html(markup, height=430, scrolling=False)


def render_result_chart(
    result: pd.DataFrame, model_name: str, roster: pd.DataFrame, driver_lookup: dict
):
    render_scrollable_prediction_chart(
        result=result,
        model_name=model_name,
        roster=roster,
        driver_lookup=driver_lookup,
        value_column="race_win_probability",
        metric_label="Normalised relative model score",
        chart_title="RELATIVE MODEL SCORE",
    )


def render_probability_chart(
    result: pd.DataFrame, model_name: str, roster: pd.DataFrame, driver_lookup: dict
):
    render_scrollable_prediction_chart(
        result=result,
        model_name=model_name,
        roster=roster,
        driver_lookup=driver_lookup,
        value_column="raw_win_score",
        metric_label="Uncalibrated model-estimated win probability",
        chart_title="MODEL-ESTIMATED WIN PROBABILITY",
    )


def hex_to_rgba(hex_colour: str, alpha: float = 1.0) -> str:
    colour = str(hex_colour or "#55B8F2").strip().lstrip("#")
    if len(colour) != 6:
        return f"rgba(85,184,242,{alpha})"
    try:
        r = int(colour[0:2], 16)
        g = int(colour[2:4], 16)
        b = int(colour[4:6], 16)
    except ValueError:
        return f"rgba(85,184,242,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"



def prepare_position_movement(
    result: pd.DataFrame,
    roster: pd.DataFrame,
    driver_lookup: dict,
) -> pd.DataFrame:
    """Add predicted position movement and team/driver display metadata."""
    data = result.copy()
    data["starting_grid"] = pd.to_numeric(data["starting_grid"], errors="coerce")
    data["predicted_rank"] = pd.to_numeric(data["predicted_rank"], errors="coerce")
    data = data.dropna(subset=["starting_grid", "predicted_rank"]).copy()
    data["starting_grid"] = data["starting_grid"].astype(int)
    data["predicted_rank"] = data["predicted_rank"].astype(int)
    data["positions_gained"] = data["starting_grid"] - data["predicted_rank"]
    data["movement"] = np.select(
        [data["positions_gained"] > 0, data["positions_gained"] < 0],
        ["Gain", "Loss"],
        default="Same",
    )
    data["driver_code"] = data["driver_code"].astype(str).str.upper()
    data["driver_name"] = data["driver_code"].map(
        lambda code: str(driver_lookup.get(code, {}).get("FullName") or code)
    )
    data["constructor"] = data["driver_code"].map(
        lambda code: str(roster.loc[code, "constructor"])
        if code in roster.index else "Unknown"
    )
    data["team_colour"] = data["constructor"].map(team_accent_colour)
    return data


def build_start_to_finish_chart(
    result: pd.DataFrame,
    roster: pd.DataFrame,
    driver_lookup: dict,
    title: str = "Starting Grid to Race Finish",
    view: str = "All 22 drivers",
):
    """Create a Sankey chart with true grid/finish labels and team-coloured flows."""
    data = prepare_position_movement(result, roster, driver_lookup)
    if data.empty:
        return None

    if view == "Top 10 predicted finishers":
        data = data.nsmallest(10, "predicted_rank")
    elif view == "Top 15 predicted finishers":
        data = data.nsmallest(15, "predicted_rank")

    data = data.sort_values("starting_grid", kind="mergesort").reset_index(drop=True)
    start_positions = sorted(data["starting_grid"].unique().tolist())
    finish_positions = sorted(data["predicted_rank"].unique().tolist())
    start_index = {position: idx for idx, position in enumerate(start_positions)}
    finish_index = {position: idx for idx, position in enumerate(finish_positions)}

    start_labels = []
    for position in start_positions:
        row = data.loc[data["starting_grid"] == position].iloc[0]
        start_labels.append(f"P{position} · {row['driver_code']}")

    finish_labels = []
    for position in finish_positions:
        row = data.loc[data["predicted_rank"] == position].iloc[0]
        finish_labels.append(f"P{position} · {row['driver_code']}")

    labels = start_labels + finish_labels
    node_count = len(labels)
    y_start = (
        np.linspace(0.015, 0.985, len(start_positions)).tolist()
        if len(start_positions) > 1 else [0.5]
    )
    y_finish = (
        np.linspace(0.015, 0.985, len(finish_positions)).tolist()
        if len(finish_positions) > 1 else [0.5]
    )
    x_positions = [0.015] * len(start_positions) + [0.985] * len(finish_positions)
    y_positions = y_start + y_finish

    node_colours = ["rgba(65,80,103,0.95)"] * node_count
    node_lines = ["rgba(255,255,255,0.22)"] * node_count
    sources, targets, values, link_colours, customdata = [], [], [], [], []

    for _, row in data.iterrows():
        start_pos = int(row["starting_grid"])
        finish_pos = int(row["predicted_rank"])
        colour = str(row["team_colour"])
        sources.append(start_index[start_pos])
        targets.append(len(start_positions) + finish_index[finish_pos])
        # Constant visual weight: each driver represents one car. A value >1
        # thickens the Sankey ribbons without implying a larger quantity.
        values.append(2.8)
        link_colours.append(hex_to_rgba(colour, 0.58))
        customdata.append([
            row["driver_name"], row["driver_code"], row["constructor"],
            f"P{start_pos}", f"P{finish_pos}",
            f"{int(row['positions_gained']):+d}",
        ])
        node_colours[start_index[start_pos]] = hex_to_rgba(colour, 0.95)
        node_colours[len(start_positions) + finish_index[finish_pos]] = hex_to_rgba(colour, 0.95)

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=13,
                thickness=17,
                line=dict(color=node_lines, width=1),
                label=labels,
                color=node_colours,
                x=x_positions,
                y=y_positions,
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colours,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]} (%{customdata[1]})</b><br>"
                    "Constructor: %{customdata[2]}<br>"
                    "Starting grid: %{customdata[3]}<br>"
                    "Predicted finish: %{customdata[4]}<br>"
                    "Positions gained/lost: %{customdata[5]}<extra></extra>"
                ),
            ),
        )]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#F2F5FA"), x=0.02),
        font=dict(size=11, color="#DCE6F2"),
        paper_bgcolor="rgba(17,21,29,1)",
        plot_bgcolor="rgba(17,21,29,1)",
        margin=dict(l=18, r=18, t=58, b=22),
        height=max(700, 155 + len(data) * 29),
    )
    return fig


def render_position_movement_insights(
    result: pd.DataFrame,
    roster: pd.DataFrame,
    driver_lookup: dict,
    asset_year: int | None = None,
):
    """Render position movers, a centred movement donut and compact KPI cards."""
    movement = prepare_position_movement(result, roster, driver_lookup)
    if movement.empty:
        st.info("Position movement insights are unavailable for this prediction.")
        return

    gainers = movement[movement["positions_gained"] > 0].sort_values(
        "positions_gained", ascending=False, kind="mergesort"
    )
    losers = movement[movement["positions_gained"] < 0].sort_values(
        "positions_gained", ascending=True, kind="mergesort"
    )
    gain_row = gainers.iloc[0] if not gainers.empty else None
    loss_row = losers.iloc[0] if not losers.empty else None

    def movement_card(row, title, colour, empty_text):
        if row is None:
            st.markdown(
                f"""
                <div class="movement-driver-card" style="border:1px solid {colour};">
                    <div class="movement-driver-photo"><span style="font-size:1.7rem;">🏎️</span></div>
                    <div class="movement-driver-info">
                        <div style="font-size:.68rem;color:#aab8c8;font-weight:850;letter-spacing:.08em;">{title}</div>
                        <div class="movement-driver-name" style="margin-top:8px;">{empty_text}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        code_raw = str(row["driver_code"]).upper()
        code = html.escape(code_raw)
        name = html.escape(str(row["driver_name"]))
        team = html.escape(str(row["constructor"]))
        photo = find_local_asset(
            "headshots",
            str(driver_lookup.get(code_raw, {}).get("FullName") or row["driver_name"]),
            asset_year=asset_year,
        )
        if photo:
            photo_mime = mimetypes.guess_type(photo.name)[0] or "image/jpeg"
            photo_b64 = base64.b64encode(photo.read_bytes()).decode("ascii")
            photo_html = (
                f'<img src="data:{photo_mime};base64,{photo_b64}" alt="{name}">'
            )
        else:
            photo_html = '<span style="font-size:1.7rem;">🏎️</span>'

        delta = int(row["positions_gained"])
        delta_label = f"+{delta}" if delta > 0 else str(delta)
        route = f"P{int(row['starting_grid'])} → P{int(row['predicted_rank'])}"
        st.markdown(
            f"""
            <div class="movement-driver-card"
                 style="border:1px solid {hex_to_rgba(colour,.62)};
                        border-left:4px solid {colour};">
                <div class="movement-driver-photo">{photo_html}</div>
                <div class="movement-driver-info">
                    <div style="font-size:.68rem;color:#aab8c8;font-weight:850;letter-spacing:.08em;">{title}</div>
                    <div class="movement-driver-heading">
                        <span class="movement-driver-code">{code}</span>
                        <span class="movement-driver-delta" style="color:{colour};">{delta_label} POS</span>
                    </div>
                    <div class="movement-driver-name" title="{name}">{name}</div>
                    <div class="movement-driver-route">{team} · {route}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    movement_card(
        gain_row, "BIGGEST POSITION GAIN", "#20C997",
        "No driver predicted to gain positions",
    )
    movement_card(
        loss_row, "BIGGEST POSITION LOSS", "#F05A67",
        "No driver predicted to lose positions",
    )

    counts = movement["movement"].value_counts()
    gain_count = int(counts.get("Gain", 0))
    loss_count = int(counts.get("Loss", 0))
    same_count = int(counts.get("Same", 0))
    total_drivers = int(len(movement))

    distribution = pd.DataFrame({
        "Movement": ["Gained positions", "Lost positions", "Same position"],
        "Drivers": [gain_count, loss_count, same_count],
    })

    pie = go.Figure(go.Pie(
        labels=distribution["Movement"],
        values=distribution["Drivers"],
        hole=0.66,
        sort=False,
        marker=dict(
            colors=["#20C997", "#F05A67", "#8291A5"],
            line=dict(color="#11151d", width=3),
        ),
        textinfo="percent",
        textposition="inside",
        hovertemplate="%{label}: %{value} drivers (%{percent})<extra></extra>",
    ))
    pie.update_layout(
        height=230,
        margin=dict(l=2, r=2, t=4, b=30),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.20,
            xanchor="center", x=0.5, font=dict(size=9, color="#DCE6F2"),
        ),
        annotations=[dict(
            text=f"<b>{total_drivers}</b><br><span style='font-size:10px'>DRIVERS</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=19, color="#F2F5FA"),
        )],
        paper_bgcolor="rgba(17,21,29,0)",
        plot_bgcolor="rgba(17,21,29,0)",
        font=dict(color="#DCE6F2"),
    )
    st.markdown("**Driver position movement distribution**")
    st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})

    average_movement = float(movement["positions_gained"].abs().mean())
    kpis = [
        ("DRIVERS GAINED", f"{gain_count}/{total_drivers}"),
        ("DRIVERS LOST", f"{loss_count}/{total_drivers}"),
        ("UNCHANGED", f"{same_count}/{total_drivers}"),
        ("AVG. POSITIONS MOVED", f"{average_movement:.1f}"),
    ]
    kpi_cols = st.columns(2, gap="small")
    for col, (label, value) in zip(kpi_cols * 2, kpis):
        with col:
            st.markdown(
                f"""
                <div class="movement-kpi">
                    <div class="movement-kpi-label">{label}</div>
                    <div class="movement-kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# V3 redesign layer: visual helpers only. Prediction logic is unchanged.
# ---------------------------------------------------------------------------
_LIGHT_KEYFRAMES = "".join(
    f"@keyframes f1light{i}{{0%,{6 + i * 9}%{{background:#35100e;box-shadow:none}}"
    f"{12 + i * 9}%,78%{{background:#ff1e1e;box-shadow:0 0 12px rgba(255,30,30,.9)}}"
    f"84%,100%{{background:#35100e;box-shadow:none}}}}"
    for i in range(5)
)
_LIGHT_RULES = "".join(
    f".startlights span:nth-child({i + 1}){{animation:f1light{i} 5s infinite}}"
    for i in range(5)
)

_V3_CSS = """
.hero-v3{position:relative;overflow:hidden}
.hero-v3::after{content:"";position:absolute;right:-30px;top:0;bottom:0;width:240px;opacity:.07;
  background:repeating-conic-gradient(#fff 0% 25%,transparent 0% 50%) 0 0/28px 28px;
  transform:skewX(-14deg);pointer-events:none}
.hero-top{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap}
.startlights{display:flex;gap:8px;padding:9px 12px;border-radius:12px;
  background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.10)}
.startlights span{width:16px;height:16px;border-radius:50%;background:#35100e}
.hero-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem;margin-top:1rem}
.hero-stat{padding:.55rem .8rem;border-radius:10px;background:rgba(255,255,255,.045);
  border:1px solid rgba(255,255,255,.09)}
.hero-stat b{display:block;font-size:1.25rem;font-weight:900;color:#fff}
.hero-stat span{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:#9fb0c4;font-weight:700}

.stepper{display:flex;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 1rem}
.step{display:flex;align-items:center;gap:.6rem;flex:1 1 180px;padding:.55rem .8rem;border-radius:12px;
  border:1px solid var(--line);background:rgba(17,22,31,.72);color:#9aa8ba;font-size:.84rem;font-weight:750}
.step small{display:block;font-weight:500;color:#8ea0b5;font-size:.7rem}
.step-dot{width:1.6rem;height:1.6rem;border-radius:50%;display:inline-flex;align-items:center;
  justify-content:center;font-size:.78rem;font-weight:900;background:rgba(148,163,184,.15);flex:0 0 auto}
.step.done{border-color:rgba(32,201,151,.5);color:#d8fff1}
.step.done .step-dot{background:#20c997;color:#07130f}
.step.active{border-color:rgba(225,6,0,.7);color:#fff;
  box-shadow:0 0 0 1px rgba(225,6,0,.25),0 0 18px rgba(225,6,0,.18)}
.step.active .step-dot{background:#e10600;color:#fff}

.circuit-card{display:flex;align-items:center;gap:1rem;padding:.9rem 1.1rem;border-radius:14px;
  border:1px solid var(--line);border-left:4px solid var(--f1-blue);
  background:linear-gradient(120deg,rgba(33,78,111,.22),rgba(17,22,31,.85))}
.circuit-flag{font-size:2.6rem;line-height:1}
.circuit-name{font-size:1.15rem;font-weight:850;color:#f4f7fb}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.45rem}

.grid-stage{display:grid;grid-template-columns:1fr 1fr;column-gap:2.2rem;max-width:760px;
  margin:.4rem auto 1rem;padding:1rem 1.2rem;border-radius:14px;border:1px solid var(--line);
  background:repeating-linear-gradient(0deg,rgba(255,255,255,.02) 0 2px,transparent 2px 4px),#0d1118}
.grid-slot{display:flex;align-items:center;gap:.6rem;height:50px;padding:0 .8rem;border-radius:8px;
  background:rgba(255,255,255,.045);border-left:5px solid var(--team-accent,#55b8f2);min-width:0}
.grid-slot:nth-child(odd){margin-bottom:24px}
.grid-slot:nth-child(even){margin-top:24px}
.grid-slot:first-child{box-shadow:0 0 0 1px rgba(212,175,55,.65)}
.grid-pos{width:2.2rem;font-weight:900;color:#9aa8ba;font-size:.82rem}
.grid-code{font-size:1.1rem;font-weight:950;color:#fff}
.grid-name{flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
  color:#9fb0c4;font-size:.72rem;font-weight:700}
.grid-num{color:#dce6f2;font-weight:850;font-size:.8rem}

.cons-row{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;padding:.6rem .8rem;margin-bottom:.45rem;
  border-radius:10px;border:1px solid var(--line);background:rgba(17,22,31,.72)}
.cons-model{min-width:150px;font-weight:800;color:#f4f7fb;font-size:.85rem}
.cons-chip{display:inline-block;padding:.18rem .55rem;margin-right:.35rem;border-radius:999px;
  font-size:.74rem;font-weight:800;background:rgba(255,255,255,.07);color:#dce6f2}
.cons-chip.win{background:rgba(212,175,55,.2);color:#ffe08a;border:1px solid rgba(212,175,55,.5)}
.cons-badge{margin-left:auto;font-size:.66rem;font-weight:900;letter-spacing:.08em;
  padding:.2rem .55rem;border-radius:6px}
.cons-badge.ok{background:rgba(32,201,151,.16);color:#6ff0c8}
.cons-badge.diff{background:rgba(240,90,103,.16);color:#ff9aa3}

.empty-state{text-align:center;padding:2rem 1rem;border:1px dashed rgba(148,163,184,.35);
  border-radius:14px;background:rgba(17,22,31,.5);color:#9aa8ba;margin-top:1rem}
.empty-state .big{font-size:2.2rem}
.empty-state b{color:#f4f7fb}
.pipeline{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem;margin-bottom:.8rem}
.pipe-node{padding:.35rem .7rem;border-radius:8px;background:rgba(85,184,242,.10);
  border:1px solid rgba(85,184,242,.3);color:#bfe6ff;font-size:.76rem;font-weight:750}
.pipe-arrow{color:#6f7f93}

@media (max-width:640px){
  .grid-stage{grid-template-columns:1fr}
  .grid-slot:nth-child(n){margin:0 0 6px}
}
@media (prefers-reduced-motion:reduce){
  .startlights span{animation:none !important;background:#ff1e1e}
}
"""
st.markdown(
    "<style>" + _V3_CSS + _LIGHT_KEYFRAMES + _LIGHT_RULES + "</style>",
    unsafe_allow_html=True,
)


def _flat(markup: str) -> str:
    """Collapse indented HTML to one line so Markdown never treats it as a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


_EVENT_FLAGS = [
    ("bahrain", "🇧🇭"), ("saudi", "🇸🇦"), ("australia", "🇦🇺"), ("japan", "🇯🇵"),
    ("chin", "🇨🇳"), ("miami", "🇺🇸"), ("lasvegas", "🇺🇸"), ("unitedstates", "🇺🇸"),
    ("emilia", "🇮🇹"), ("imola", "🇮🇹"), ("ital", "🇮🇹"), ("monaco", "🇲🇨"),
    ("canad", "🇨🇦"), ("spanish", "🇪🇸"), ("spain", "🇪🇸"), ("madrid", "🇪🇸"),
    ("barcelona", "🇪🇸"), ("austria", "🇦🇹"), ("british", "🇬🇧"), ("hungar", "🇭🇺"),
    ("belgi", "🇧🇪"), ("dutch", "🇳🇱"), ("netherland", "🇳🇱"), ("azerbaijan", "🇦🇿"),
    ("singapore", "🇸🇬"), ("mexic", "🇲🇽"), ("saopaulo", "🇧🇷"), ("brazil", "🇧🇷"),
    ("qatar", "🇶🇦"), ("abudhabi", "🇦🇪"),
]


def event_flag(event_name: str) -> str:
    """Decorative flag for a Grand Prix name; falls back to a chequered flag."""
    key = _asset_key(event_name)
    for needle, flag in _EVENT_FLAGS:
        if needle in key:
            return flag
    return "🏁"


def render_stepper(slot, circuit_done: bool, grid_done: bool, predicted: bool):
    """Three-step progress tracker rendered into a placeholder at the top of the page."""
    steps = [
        ("1", "Circuit", "Pick a Grand Prix", circuit_done),
        ("2", "Starting grid", "Build your 22-car grid", grid_done),
        ("3", "Predict", "Run all three models", predicted),
    ]
    first_open = next((i for i, step in enumerate(steps) if not step[3]), None)
    parts = []
    for i, (num, title, sub, done) in enumerate(steps):
        state = "done" if done else ("active" if i == first_open else "")
        dot = "✓" if done else num
        parts.append(
            f'<div class="step {state}"><span class="step-dot">{dot}</span>'
            f'<span>{title}<small>{sub}</small></span></div>'
        )
    slot.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_staggered_grid(grid, driver_lookup: dict, roster_asof: pd.DataFrame):
    """Track-style two-column staggered grid (pole on the left, P2 slightly behind)."""
    slots = []
    for position, code in enumerate(grid, start=1):
        code_up = str(code).upper()
        info = driver_lookup.get(code_up, {})
        number = html.escape(str(info.get("DriverNumber") or "—"))
        full_name = str(info.get("FullName") or code_up)
        surname = html.escape(full_name.split()[-1].upper())
        try:
            team = str(roster_asof.loc[code, "constructor"])
        except (KeyError, TypeError):
            team = "Unknown"
        accent = team_accent_colour(team)
        slots.append(
            f'<div class="grid-slot" style="--team-accent:{accent};" title="{html.escape(team)}">'
            f'<span class="grid-pos">P{position}</span>'
            f'<span class="grid-code">{html.escape(code_up)}</span>'
            f'<span class="grid-name">{surname}</span>'
            f'<span class="grid-num">#{number}</span></div>'
        )
    st.markdown(f'<div class="grid-stage">{"".join(slots)}</div>', unsafe_allow_html=True)


def compute_model_consensus(model_predictions: dict, model_names: list) -> dict:
    """Compare the winners/podiums of the available models (lead model = first name)."""
    rows = []
    for name in model_names:
        frame = model_predictions[name].sort_values("predicted_rank").reset_index(drop=True)
        top3 = [str(code).upper() for code in frame["driver_code"].head(3)]
        first = float(frame.loc[0, "race_win_probability"])
        second = float(frame.loc[1, "race_win_probability"]) if len(frame) > 1 else 0.0
        rows.append({"model": name, "winner": top3[0], "top3": top3, "margin": first - second})
    lead_winner = rows[0]["winner"]
    agree = sum(1 for row in rows if row["winner"] == lead_winner)
    overlap = len(set.intersection(*[set(row["top3"]) for row in rows])) if rows else 0
    return {"rows": rows, "lead_winner": lead_winner, "agree": agree,
            "total": len(rows), "overlap": overlap}


def render_consensus_table(consensus: dict):
    rows_html = []
    for row in consensus["rows"]:
        chips = "".join(
            f'<span class="cons-chip {"win" if i == 0 else ""}">P{i + 1} {html.escape(code)}</span>'
            for i, code in enumerate(row["top3"])
        )
        agrees = row["winner"] == consensus["lead_winner"]
        badge = (
            '<span class="cons-badge ok">SAME WINNER</span>' if agrees
            else '<span class="cons-badge diff">DIFFERENT WINNER</span>'
        )
        rows_html.append(
            f'<div class="cons-row"><div class="cons-model">{html.escape(row["model"])}</div>'
            f'<div>{chips}</div>{badge}</div>'
        )
    st.markdown("".join(rows_html), unsafe_allow_html=True)


def build_consensus_chart(model_predictions: dict, model_names: list, top_n: int = 8):
    """Grouped bars: relative score of the leading drivers, side by side for each model."""
    frames = []
    for name in model_names:
        part = model_predictions[name][["driver_code", "race_win_probability"]].copy()
        part["Model"] = name
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    leaders = (
        long.sort_values("race_win_probability", ascending=False)
        .groupby("Model").head(top_n)["driver_code"].unique().tolist()
    )
    long = long[long["driver_code"].isin(leaders)].copy()
    order = (
        long.groupby("driver_code")["race_win_probability"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    fig = px.bar(
        long, x="driver_code", y="race_win_probability", color="Model", barmode="group",
        category_orders={"driver_code": order, "Model": model_names},
        color_discrete_map={
            "Gradient Boosting": "#E10600",
            "Random Forest": "#55B8F2",
            "Logistic Regression": "#F5C542",
        },
        labels={"driver_code": "Driver", "race_win_probability": "Relative model score"},
    )
    fig.update_layout(
        template="plotly_dark", height=380, margin=dict(l=8, r=8, t=18, b=8),
        yaxis_tickformat=".0%", xaxis_title="", legend_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_empty_results_state(grid_ready: bool):
    if grid_ready:
        icon, title, body = "🏁", "Grid locked in. Ready to race?", \
            "Press <b>Predict Race – All Three Models</b> above to see the podium, charts and model consensus."
    else:
        icon, title, body = "🚦", "Your prediction will appear here", \
            "Choose a circuit, build a starting grid (manual, top-5 + historical fill, or random), then run the models."
    st.markdown(
        f'<div class="empty-state"><div class="big">{icon}</div><b>{title}</b><br>{body}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    _flat(f"""
    <div class="hero hero-v3">
      <div class="hero-top">
        <div>
          <div class="hero-kicker">FOR1SIGHT · BIG DATA ANALYTICS</div>
          <div class="hero-title">🏎️ Race Prediction Lab</div>
          <div class="hero-sub">Pick a Grand Prix, set a hypothetical 2027 starting grid and watch three
          machine-learning models call the podium.</div>
        </div>
        <div class="startlights" title="Lights out and away we go!"><span></span><span></span><span></span><span></span><span></span></div>
      </div>
      <div class="hero-stats">
        <div class="hero-stat"><b>2027</b><span>Prediction season</span></div>
        <div class="hero-stat"><b>22</b><span>Drivers on the grid</span></div>
        <div class="hero-stat"><b>3</b><span>ML models compared</span></div>
        <div class="hero-stat"><b>{len(FEATURE_COLUMNS)}</b><span>Model features</span></div>
      </div>
    </div>
    """),
    unsafe_allow_html=True,
)
stepper_slot = st.empty()

if not WAREHOUSE_PATH.exists():
    st.error(f"Prediction warehouse not found: {WAREHOUSE_PATH}")
    st.info("Place prediction_training_v2_8_prerace.parquet inside the project's for1sight_warehouse folder.")
    st.stop()

model_dir = find_v28_model_dir()
if model_dir is None:
    st.error("The V2.8 deployment model files could not be located.")
    st.markdown("The following Gradient Boosting model file is required inside `models/v2_8/`:")
    st.code(
        "gradient_boosting_deployment_v2_8.joblib\n"
        "model_metadata.json  (optional)"
    )
    st.caption("Expected location: models/v2_8/ or v2_8/")
    st.stop()

try:
    warehouse = load_warehouse(str(WAREHOUSE_PATH), WAREHOUSE_PATH.stat().st_mtime_ns)
    validate_warehouse(warehouse)
    models, manifest = load_models(
        str(model_dir),
        model_artifact_signature(model_dir),
    )
    roster, roster_year = load_fixed_roster(
        str(WAREHOUSE_PATH), WAREHOUSE_PATH.stat().st_mtime_ns,
        TARGET_SEASON_ROSTER,
    )
    race_catalog = get_race_catalog(warehouse)
except Exception as exc:
    st.error(f"Unable to initialise prediction workspace: {exc}")
    st.stop()

if len(roster) < 2 or race_catalog.empty:
    st.error("The warehouse does not contain enough driver or race information.")
    st.stop()

st.markdown(
    '<span class="pill">V2.8 · FUTURE RACE SIMULATOR</span> '
    '<span class="pill">GRADIENT BOOSTING · LEAD MODEL</span> '
    '<span class="pill">FIXED 2026 DRIVER ROSTER</span>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 01 Historical Grand Prix / circuit reference selection
# ---------------------------------------------------------------------------
section_heading(
    "01",
    "Select Grand Prix / Circuit Reference",
    "Choose from the combined historical event catalogue used during 2024–2026. The prediction target is fixed internally to 2027.",
)

try:
    historical_catalog = load_cached_historical_catalog(
        str(WAREHOUSE_PATH), WAREHOUSE_PATH.stat().st_mtime_ns
    )
except Exception as exc:
    st.error(f"Unable to build the 2024–2026 Grand Prix catalogue: {exc}")
    st.stop()

race_labels = historical_catalog["event_name"].astype(str).tolist()
selected_event = st.selectbox(
    "Historical Grand Prix / circuit reference",
    race_labels,
    key="prediction_event_v28_final",
    help="The event name is passed to the model as a categorical feature (event_norm).",
)
selected_row = historical_catalog.loc[
    historical_catalog["event_name"].astype(str).eq(selected_event)
].iloc[0]
PREDICTION_YEAR = 2027
event_norm = str(selected_row["event_norm"])
reference_year = int(selected_row["year"])
reference_round = int(selected_row["round"]) if pd.notna(selected_row["round"]) else 1
historical_seasons_label = str(selected_row["historical_seasons_label"])

st.markdown(
    _flat(f"""
    <div class="circuit-card">
      <div class="circuit-flag">{event_flag(selected_event)}</div>
      <div>
        <div class="circuit-name">{html.escape(selected_event)}</div>
        <div class="chips">
          <span class="pill">SEEN IN {html.escape(historical_seasons_label)}</span>
          <span class="pill">TARGET 2027</span>
          <span class="pill">{len(historical_catalog)} EVENTS IN CATALOGUE</span>
        </div>
        <div class="footnote" style="margin-top:.45rem;">The selected event category is passed to the V2.8 model.</div>
      </div>
    </div>
    """),
    unsafe_allow_html=True,
)

# Fixed 2026 roster: use the latest single event in the latest warehouse season.
roster = roster.copy()
roster["driver_code_norm"] = roster["driver_code_norm"].astype(str).str.upper()
roster = roster.drop_duplicates("driver_code_norm")
codes = roster["driver_code_norm"].tolist()
roster["constructor"] = roster["constructor"].fillna("Unknown").astype(str)
roster = roster.set_index("driver_code_norm", drop=False)

latest_round_for_roster = int(pd.to_numeric(roster["round"], errors="coerce").dropna().max())
driver_lookup = {}
try:
    details = load_current_season_driver_details(TARGET_SEASON_ROSTER, latest_round_for_roster)
    for item in details:
        code = str(item.get("Abbreviation", "")).upper()
        if code:
            driver_lookup[code] = item
except ImportError:
    st.info("Install FastF1 (`pip install fastf1`) to show driver full names, numbers and headshots.")
except Exception as exc:
    st.warning(
        "FastF1 driver profile data could not be loaded. Driver codes and warehouse team names "
        f"will be used as a fallback. Details: {exc}"
    )

# FastF1 is cached: the first profile retrieval may take longer, while subsequent
# Streamlit reruns reuse the cached driver details.

# Retain only the fixed 2026 entry list; no separate editable entry-list section.
if len(codes) != 22:
    st.warning(
        f"The fixed 2026 warehouse roster contains {len(codes)} drivers, not 22. "
        "The standard-grid simulator requires the 22-driver 2026 lineup. Update the warehouse "
        "with the correct 2026 event roster before proceeding."
    )
    st.stop()

roster_asof = load_cached_roster_features(
    str(WAREHOUSE_PATH),
    WAREHOUSE_PATH.stat().st_mtime_ns,
    int(PREDICTION_YEAR),
    tuple(codes),
)
roster_asof["driver_code_norm"] = roster_asof["driver_code_norm"].astype(str).str.upper()
roster_asof = roster_asof.set_index("driver_code_norm", drop=False)
roster_key = hashlib.sha1("|".join(codes).encode("utf-8")).hexdigest()[:10]
roster_asof["constructor"] = roster_asof["constructor"].fillna("Unknown").astype(str)

section_heading(
    "02",
    "2026 driver lineup",
    "The 22 drivers are fixed to the 2026 roster. Future-season driver changes are outside this project scope.",
)
local_headshot_count = sum(
    find_local_asset("headshots", display_driver_name(code, driver_lookup), asset_year=TARGET_SEASON_ROSTER) is not None
    for code in codes
)
local_logo_count = sum(
    find_local_asset(
        "logos",
        str(driver_lookup.get(code, {}).get("TeamName") or roster_asof.loc[code, "constructor"]),
        asset_year=TARGET_SEASON_ROSTER,
    ) is not None
    for code in codes
)
st.caption(
    f"Local assets detected: {local_headshot_count}/22 driver headshots · "
    f"{local_logo_count}/22 team logos. Local images are preferred over online images."
)

with st.expander("View all 2026 driver cards", expanded=False):
    card_cols = st.columns(4)
    for i, code in enumerate(codes):
        with card_cols[i % 4]:
            render_driver_card(
                code,
                driver_lookup,
                str(roster_asof.loc[code, "constructor"]),
                asset_year=TARGET_SEASON_ROSTER,
            )

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 03 Starting grid
# ---------------------------------------------------------------------------
section_heading(
    "03",
    "Starting grid simulator",
    "Choose drivers for each position, fill the remaining places using historical form, or randomise the complete grid.",
)
mode = st.radio(
    "Grid configuration",
    [
        "Manual grid",
        "Choose P1–P5 + historical fill",
        "Randomise entire grid",
    ],
    horizontal=True,
    key="prediction_grid_mode",
)

if "for1sight_scenario_grid" not in st.session_state:
    st.session_state.for1sight_scenario_grid = None

if mode == "Manual grid":
    st.markdown("**Assign one unique driver to every starting position**")
    st.caption(
        "Choose all 22 drivers first, then submit once. The page will not refresh "
        "after every individual selection."
    )
    selections = []
    with st.form(key=f"manual_grid_form_v28_{roster_key}"):
        for row_start in range(0, 22, 4):
            selector_cols = st.columns(4)
            for offset, column in enumerate(selector_cols):
                i = row_start + offset
                if i >= 22:
                    continue
                with column:
                    with st.container(border=True):
                        st.markdown(f"**P{i + 1}**")
                        selected_code = st.selectbox(
                            f"Driver for P{i + 1}",
                            ["— Select driver —"] + codes,
                            format_func=lambda value: (
                                "— Select driver —"
                                if value == "— Select driver —"
                                else display_driver_option(value, driver_lookup)
                            ),
                            key=f"manual_grid_simple_v28_{roster_key}_{i}",
                            label_visibility="collapsed",
                        )
                        selections.append(selected_code)
        submitted_manual_grid = st.form_submit_button(
            "Use this starting grid",
            type="primary",
            use_container_width=True,
        )

    if submitted_manual_grid:
        if any(x == "— Select driver —" for x in selections):
            st.error("Please assign all 22 starting positions.")
        elif len(set(selections)) != 22:
            st.error("Duplicate drivers detected. Each driver can appear only once.")
        else:
            st.session_state.for1sight_scenario_grid = selections
            st.success("Manual starting grid confirmed.")

elif mode == "Choose P1–P5 + historical fill":
    st.markdown("**Choose your preferred top-five drivers**")
    st.caption(
        "Choose all five positions, then submit once. P6–P22 will be filled "
        "using historical driver and constructor performance."
    )
    top = []
    with st.form(key=f"top5_grid_form_v28_{roster_key}"):
        top_cols = st.columns(5)
        for i in range(5):
            with top_cols[i]:
                with st.container(border=True):
                    st.markdown(f"**P{i + 1}**")
                    selected_code = st.selectbox(
                        f"Driver for P{i + 1}",
                        codes,
                        index=i,
                        format_func=lambda value: display_driver_option(value, driver_lookup),
                        key=f"top5_driver_simple_v28_{roster_key}_{i}",
                        label_visibility="collapsed",
                    )
                    top.append(selected_code)
        submitted_top_five = st.form_submit_button(
            "Fill P6–P22 using historical form",
            type="primary",
            use_container_width=True,
        )

    if submitted_top_five:
        if len(set(top)) != 5:
            st.error("Please select five different drivers for P1–P5.")
        else:
            try:
                st.session_state.for1sight_scenario_grid = fill_remaining_by_history(
                    roster_asof.reset_index(drop=True), top
                )
                st.success(
                    "P1–P5 preserved. P6–P22 filled using historical driver and constructor performance."
                )
            except Exception as exc:
                st.error(f"Unable to fill remaining positions: {exc}")

else:
    st.markdown(
        '<div class="subtle-card">Generate a random hypothetical grid. '
        'This is a scenario simulation, not an official qualifying order.</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "🎲 Randomise all 22 starting positions",
        type="primary",
        use_container_width=True,
        key="randomise_grid_simple_v28",
    ):
        st.session_state.for1sight_scenario_grid = list(
            np.random.default_rng().permutation(codes)
        )

grid = st.session_state.get("for1sight_scenario_grid")
if grid:
    if len(grid) != 22 or len(set(grid)) != 22 or set(grid) != set(codes):
        st.warning(
            "The saved grid no longer matches the current 22-driver entry list. Generate a new grid."
        )
        st.session_state.for1sight_scenario_grid = None
        grid = None

if grid:
    st.markdown("#### Your starting grid")
    grid_view = st.radio(
        "Grid view",
        ["🏁 Track view", "📋 List view"],
        horizontal=True,
        key="grid_view_v3",
        label_visibility="collapsed",
    )
    st.caption(
        "Confirmed hypothetical starting order. Colours show the constructor; numbers are racing numbers."
    )
    if grid_view.startswith("🏁"):
        render_staggered_grid(grid, driver_lookup, roster_asof)
    else:
        # A single vertical list guarantees P1–P22 remain in numerical order.
        for index, code in enumerate(grid, start=1):
            constructor = str(roster_asof.loc[code, "constructor"])
            render_starting_grid_row(index, code, driver_lookup, constructor)

    if st.button(
        "Clear starting grid and configure again",
        use_container_width=True,
        key="clear_grid_simple_v28",
    ):
        st.session_state.for1sight_scenario_grid = None
        st.rerun()

    current_signature = scenario_signature(
        selected_event, PREDICTION_YEAR, reference_round, grid
    )
    if st.button(
        "🏁 Predict Race – All Three Models",
        type="primary",
        use_container_width=True,
        key="predict_race_gradient_boosting_v28",
    ):
        try:
            with st.spinner("Generating V2.8 pre-race features and running all three models..."):
                feature_frame = build_future_feature_frame(
                    warehouse=warehouse,
                    event_norm=event_norm,
                    race_round=int(reference_round),
                    grid_codes=grid,
                    target_year=int(PREDICTION_YEAR),
                )
                outputs = predict_event(models, feature_frame, grid)
                required_models = {"Gradient Boosting", "Random Forest", "Logistic Regression"}
                missing_models = required_models.difference(outputs)
                if missing_models:
                    raise ValueError(
                        "The V2.8 prediction engine did not return: "
                        + ", ".join(sorted(missing_models))
                    )
            st.session_state.for1sight_model_predictions_v28 = outputs
            st.session_state.for1sight_gb_prediction = outputs["Gradient Boosting"]
            st.session_state.for1sight_prediction_context_v28 = {
                "event": selected_event,
                "event_norm": event_norm,
                "prediction_year": int(PREDICTION_YEAR),
                "reference_year": int(reference_year),
                "historical_seasons": historical_seasons_label,
                "reference_round": int(reference_round),
                "grid": list(grid),
                "signature": current_signature,
            }
            _winner_code = str(outputs["Gradient Boosting"].iloc[0]["driver_code"]).upper()
            st.success("All three V2.8 model predictions completed successfully.")
            st.toast(f"Lights out! Predicted winner: {display_driver_name(_winner_code, driver_lookup)}", icon="🏁")
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

# ---------------------------------------------------------------------------
# 04 Prediction results – three V2.8 models
# ---------------------------------------------------------------------------
model_predictions = st.session_state.get("for1sight_model_predictions_v28")
# Backward compatibility for a session created before the three-model update.
if not isinstance(model_predictions, dict):
    legacy_result = st.session_state.get("for1sight_gb_prediction")
    model_predictions = (
        {"Gradient Boosting": legacy_result}
        if isinstance(legacy_result, pd.DataFrame) else {}
    )

context = st.session_state.get("for1sight_prediction_context_v28", {})
active_grid = st.session_state.get("for1sight_scenario_grid")
active_signature = (
    scenario_signature(selected_event, PREDICTION_YEAR, reference_round, active_grid)
    if active_grid and len(active_grid) == 22
    else None
)
results_are_current = bool(
    model_predictions
    and context.get("signature")
    and context.get("signature") == active_signature
)

if model_predictions and not results_are_current:
    st.info(
        "The displayed prediction belongs to a previous circuit or starting-grid scenario. "
        "Run the prediction again to refresh the results."
    )

render_stepper(
    stepper_slot,
    circuit_done=True,
    grid_done=bool(active_grid),
    predicted=results_are_current,
)

if not model_predictions:
    render_empty_results_state(bool(active_grid))

if results_are_current:
    available_models = [
        name for name in ("Gradient Boosting", "Random Forest", "Logistic Regression")
        if isinstance(model_predictions.get(name), pd.DataFrame)
    ]
    lead_result = model_predictions.get("Gradient Boosting")
    if not isinstance(lead_result, pd.DataFrame):
        lead_result = model_predictions[available_models[0]]

    lead_result = lead_result.sort_values("predicted_rank").reset_index(drop=True)
    consensus = compute_model_consensus(model_predictions, available_models)

    st.markdown("---")
    section_heading(
        "04",
        "Predicted Podium",
        f"{context['event']} · 2027 prediction scenario · Historical reference: {context['historical_seasons']}",
    )

    # Podium stage: P2 | P1 | P3, with P1 standing tallest.
    podium_rows = lead_result.head(3).reset_index(drop=True)
    podium_cols = st.columns(3, gap="medium")
    for column, rank_index, spacer in zip(podium_cols, (1, 0, 2), (26, 0, 52)):
        if rank_index >= len(podium_rows):
            continue
        driver_row = podium_rows.iloc[rank_index]
        driver_code = str(driver_row["driver_code"]).upper()
        try:
            constructor = str(roster_asof.loc[driver_code, "constructor"])
        except Exception:
            constructor = "Unknown"
        with column:
            st.markdown(f"<div style='height:{spacer}px'></div>", unsafe_allow_html=True)
            render_prediction_podium_card(
                code=driver_code,
                position_label=f"P{rank_index + 1}",
                score=float(driver_row["race_win_probability"]),
                driver_lookup=driver_lookup,
                constructor=constructor,
                asset_year=TARGET_SEASON_ROSTER,
                position=rank_index + 1,
            )

    # Headline KPIs
    winner_row = lead_result.iloc[0]
    winner_code = str(winner_row["driver_code"]).upper()
    winner_name = html.escape(display_driver_name(winner_code, driver_lookup))
    try:
        winner_team = html.escape(str(roster_asof.loc[winner_code, "constructor"]))
    except Exception:
        winner_team = "Unknown"
    margin = consensus["rows"][0]["margin"]
    kpi_cols = st.columns(4, gap="small")
    with kpi_cols[0]:
        metric_card("Predicted winner", winner_name, winner_team)
    with kpi_cols[1]:
        metric_card("Win-score margin", f"{margin:.1%}", "P1 vs P2 relative score (lead model)")
    with kpi_cols[2]:
        metric_card(
            "Model agreement",
            f"{consensus['agree']}/{consensus['total']} models",
            f"name {html.escape(consensus['lead_winner'])} as winner · {consensus['overlap']}/3 podium drivers shared",
        )
    with kpi_cols[3]:
        metric_card(
            "Winner starts from",
            f"P{int(winner_row['starting_grid'])}",
            "Predicted winner's slot on your grid",
        )

    section_heading(
        "05",
        "Driver Prediction Visualisation",
        "Switch between relative scores, raw model-estimated win probability and a side-by-side model comparison.",
    )

    score_tab, probability_tab, consensus_tab = st.tabs(
        ["📊 Relative Model Score", "🏆 Winning Probability", "🤝 Model Consensus"]
    )

    with score_tab:
        score_model = st.selectbox(
            "Select machine-learning model",
            available_models,
            index=available_models.index("Gradient Boosting") if "Gradient Boosting" in available_models else 0,
            key="relative_score_model_v28",
        )
        score_result = model_predictions[score_model].sort_values("predicted_rank").reset_index(drop=True)
        st.caption(
            "Normalised scores sum to 100% across the 22 drivers. "
            "They indicate relative ranking strength, not an independent probability of winning."
        )
        render_result_chart(score_result, score_model, roster_asof, driver_lookup)

    with probability_tab:
        probability_model = st.selectbox(
            "Select machine-learning model",
            available_models,
            index=available_models.index("Gradient Boosting") if "Gradient Boosting" in available_models else 0,
            key="winning_probability_model_v28",
        )
        probability_result = model_predictions[probability_model].sort_values("raw_win_score", ascending=False).reset_index(drop=True)
        st.caption(
            "These values are taken directly from each model's positive-class predict_proba output. "
            "They are not calibrated and may not sum to 100% across drivers."
        )
        render_probability_chart(probability_result, probability_model, roster_asof, driver_lookup)

    with consensus_tab:
        st.caption(
            "Do the three models see the race the same way? Each row shows a model's predicted top three; "
            "the chart compares relative scores for the leading drivers."
        )
        render_consensus_table(consensus)
        st.plotly_chart(
            build_consensus_chart(model_predictions, available_models),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    section_heading(
        "06",
        "Starting Grid to Race Finish",
        "The Sankey chart shows how each car is predicted to move; the right panel summarises the biggest movers.",
    )
    sankey_col, insights_col = st.columns([0.62, 0.38], gap="large")
    with sankey_col:
        flow_view = st.selectbox(
            "Drivers shown in flow chart",
            ["All 22 drivers", "Top 15 predicted finishers", "Top 10 predicted finishers"],
            index=0,
            key="start_finish_flow_view_v28",
        )
        flow_chart = build_start_to_finish_chart(
            lead_result, roster_asof, driver_lookup, view=flow_view
        )
        if flow_chart is not None:
            st.plotly_chart(
                flow_chart,
                use_container_width=True,
                config={"displayModeBar": True, "displaylogo": False},
            )
        else:
            st.info("The start-to-finish flow chart could not be generated for this scenario.")
    with insights_col:
        render_position_movement_insights(
            lead_result, roster_asof, driver_lookup,
            asset_year=TARGET_SEASON_ROSTER,
        )

    with st.expander("View complete prediction data", expanded=False):
        table_model = st.selectbox(
            "Model for detailed table",
            available_models,
            index=available_models.index("Gradient Boosting") if "Gradient Boosting" in available_models else 0,
            key="prediction_detail_model_v28",
        )
        table_result = model_predictions[table_model].sort_values("predicted_rank").reset_index(drop=True)
        display = table_result[["predicted_rank", "driver_code", "starting_grid", "race_win_probability", "raw_win_score"]].copy()
        display = display.rename(columns={
            "predicted_rank": "Predicted position",
            "driver_code": "Driver",
            "starting_grid": "Starting grid",
            "race_win_probability": "Relative model score",
            "raw_win_score": "Raw model win probability",
        })
        st.dataframe(
            display.style.format({
                "Relative model score": "{:.2%}",
                "Raw model win probability": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    section_heading(
        "07",
        "Download Prediction Results",
        "Export the selected model's complete prediction output and scenario details.",
    )
    export_model = st.selectbox(
        "Model to export",
        available_models,
        index=available_models.index("Gradient Boosting") if "Gradient Boosting" in available_models else 0,
        key="prediction_export_model_v28",
    )
    export = model_predictions[export_model].copy()
    export.insert(0, "prediction_season", context["prediction_year"])
    export.insert(1, "historical_event_reference", context["event"])
    export.insert(2, "historical_seasons_available", context["historical_seasons"])
    export.insert(3, "model", export_model)
    st.download_button(
        "⬇️ Download 2027 race prediction (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"for1sight_v28_{_asset_key(export_model)}_prediction_2027.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# 08 Model evidence, methodology and artefacts (tabs keep the page short)
# ---------------------------------------------------------------------------
st.markdown("---")
section_heading(
    "08",
    "Model Evidence & Methodology",
    "How well have these models performed historically, and how does the simulator work?",
)
evidence_tab, method_tab, artifact_tab = st.tabs(
    ["📊 2025 Validation", "🧭 How it works", "🗂️ Artifacts"]
)

with evidence_tab:
    st.caption(
        "Independent 2025 validation results across 24 races. These metrics evaluate historical "
        "model performance; they are not scores from the current hypothetical 2027 scenario."
    )
    validation_df = pd.DataFrame([
        {"Model": "Gradient Boosting", "Top-1 Accuracy": 62.50, "Top-3 Accuracy": 91.67},
        {"Model": "Logistic Regression", "Top-1 Accuracy": 45.83, "Top-3 Accuracy": 83.33},
        {"Model": "Random Forest", "Top-1 Accuracy": 45.83, "Top-3 Accuracy": 91.67},
    ])
    best = validation_df.sort_values("Top-1 Accuracy", ascending=False).iloc[0]
    best_cols = st.columns(3, gap="small")
    with best_cols[0]:
        metric_card("Best Top-1 model", html.escape(str(best["Model"])), "Picked the exact winner most often")
    with best_cols[1]:
        metric_card("Best Top-1 accuracy", f"{best['Top-1 Accuracy']:.1f}%", "Winner correct in 2025 validation")
    with best_cols[2]:
        metric_card("Validation sample", "24 races", "2025 season")

    validation_col, validation_chart_col = st.columns([1, 1.25])
    with validation_col:
        st.dataframe(
            validation_df.style.format({
                "Top-1 Accuracy": "{:.2f}%",
                "Top-3 Accuracy": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Top-1 = predicted winner is the real winner. Top-3 = real winner is in the predicted top three.")
    with validation_chart_col:
        validation_long = validation_df.melt(
            id_vars="Model",
            value_vars=["Top-1 Accuracy", "Top-3 Accuracy"],
            var_name="Metric",
            value_name="Accuracy",
        )
        validation_chart = px.bar(
            validation_long,
            x="Model",
            y="Accuracy",
            color="Metric",
            barmode="group",
            text=validation_long["Accuracy"].map(lambda value: f"{value:.2f}%"),
            color_discrete_sequence=["#e10600", "#55b8f2"],
        )
        validation_chart.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=8, r=8, t=20, b=8),
            yaxis_title="Accuracy (%)",
            xaxis_title="",
            yaxis_range=[0, 100],
            legend_title="Validation metric",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(validation_chart, use_container_width=True, config={"displayModeBar": False})

with method_tab:
    st.markdown(
        '<div class="pipeline">'
        '<span class="pipe-node">Circuit reference</span><span class="pipe-arrow">→</span>'
        '<span class="pipe-node">2026 roster (22)</span><span class="pipe-arrow">→</span>'
        '<span class="pipe-node">Starting grid</span><span class="pipe-arrow">→</span>'
        f'<span class="pipe-node">{len(FEATURE_COLUMNS)} pre-race features</span><span class="pipe-arrow">→</span>'
        '<span class="pipe-node">3 models</span><span class="pipe-arrow">→</span>'
        '<span class="pipe-node">P1–P22 ranking</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **Workflow**
        1. Select a Grand Prix event reference from the combined 2024–2026 historical catalogue.
        2. Use the fixed 22-driver lineup from the 2026 warehouse season.
        3. Configure a hypothetical starting grid manually, through Top-5 + historical fill, or by randomisation.
        4. Generate the eight V2.8 model features using historical records strictly before the 2027 target season.
        5. Run the saved deployment models (Gradient Boosting leads) to rank all 22 drivers.
        6. Review the predicted winner, podium and complete P1–P22 ranking.

        **Important interpretation notes**
        - The prediction target is fixed internally to 2027; no prediction-season selector is shown.
        - The 2026 driver lineup is fixed because the project dataset covers 2024–2026.
        - The selected event is passed using its existing `event_norm` category because that is part of the trained V2.8 feature schema.
        - Different Grand Prix event categories are not automatically merged just because they may use the same physical circuit; the current warehouse does not provide a separate circuit identifier in the validated schema.
        - The simulator does not use 2027 practice, qualifying or live race information.
        - Relative model scores are for ranking and are not calibrated real-world winning probabilities.
        - The Top-5 fill method is a historical-performance heuristic, not an ML prediction.
        - Historical model evaluation metrics are separate from the interactive 2027 scenario output.
        """
    )

with artifact_tab:
    st.write(f"**Warehouse:** `{WAREHOUSE_PATH.relative_to(ROOT)}`")
    st.write(f"**Model folder:** `{model_dir.relative_to(ROOT)}`")
    st.write(f"**Prediction target:** {PREDICTION_YEAR}")
    st.write(f"**Fixed driver roster:** {TARGET_SEASON_ROSTER}")
    st.write(f"**Model feature count:** {len(FEATURE_COLUMNS)}")
    st.write("**Interactive prediction models:** Gradient Boosting, Random Forest and Logistic Regression")
    st.write("**Historical validation models:** Gradient Boosting, Logistic Regression, Random Forest")
