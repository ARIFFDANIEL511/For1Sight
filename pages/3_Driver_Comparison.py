"""For1Sight — Driver Comparison dashboard.

Place this file in the project's pages/ directory and keep for1sight_shared.py
in the project root. The page is offline-first for race results, qualifying data,
standings snapshots, driver images and the local Parquet lap warehouse.
"""
from __future__ import annotations

import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Driver Comparison | For1Sight", page_icon="🏁", layout="wide")

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "pages" else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import for1sight_shared as F  # noqa: E402
from telemetry_reader import (  # noqa: E402
    TelemetryDataError,
    load_telemetry_audit,
    load_telemetry_lap_samples,
    load_telemetry_pair_candidates,
)
from telemetry_replay import build_interactive_track_replay  # noqa: E402

SEASONS = [2024, 2025, 2026]
WAREHOUSE = F.WAREHOUSE_DIR

# -----------------------------------------------------------------------------
# Visual system (compact dashboard theme)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;800;900&display=swap');
    :root { --panel:#111826; --panel2:#151d2b; --line:#273245; --muted:#8fa0b6; --red:#ff304d; }
    .stApp {
        background:
          radial-gradient(ellipse at 80% 0%, rgba(255,48,77,.13) 0%, transparent 38%),
          linear-gradient(145deg, #080b11 0%, #0f1520 58%, #090d14 100%);
        color: #f3f5fa; font-family: 'Titillium Web', sans-serif;
    }
    html, body, [class*="css"], button, input { font-family: 'Titillium Web', sans-serif; }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { max-width: 1600px; padding: 2.6rem 1.6rem 2rem; }
    [data-testid="stVerticalBlock"] { gap: .6rem; }
    [data-testid="stPlotlyChart"] {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 12px; overflow: hidden;
    }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
    .stButton > button { background:#202a38; color:#f4f7fb; border:1px solid #3a4658; border-radius:9px; font-weight:800; }

    /* ---- inputs ---- */
    [data-testid="stSelectbox"] label p { color:#9fb0c6; font-size:.62rem; font-weight:800; letter-spacing:.12em; }
    [data-baseweb="select"] > div { min-height: 36px; }

    /* ---- tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { height:38px; padding:0 14px; font-weight:800; letter-spacing:.04em; color:var(--muted); }
    .stTabs [aria-selected="true"] { color:#fff !important; background:rgba(255,48,77,.12); border-radius:8px 8px 0 0; }
    .stTabs [data-baseweb="tab-highlight"] { background:var(--red) !important; }

    /* ---- page hero ---- */
    .dash-hero {
        display:flex; justify-content:space-between; align-items:center; gap:14px;
        padding:12px 18px; border:1px solid var(--line); border-left:4px solid var(--red);
        border-radius:12px; background:linear-gradient(110deg, rgba(255,48,77,.12), rgba(17,24,38,.95) 55%);
    }
    .dash-kicker { color:var(--red); font-size:.64rem; font-weight:900; letter-spacing:.22em; }
    .dash-title { color:#fff; font-size:1.55rem; font-weight:900; line-height:1.1; }
    .dash-sub { color:var(--muted); font-size:.8rem; margin-top:2px; }
    .dash-chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.68rem; font-weight:800;
        letter-spacing:.08em; color:#cfd8e6; border:1px solid var(--line); background:rgba(255,255,255,.04); }

    /* ---- panel headings (replace big in-chart titles) ---- */
    .panel-head { display:flex; align-items:baseline; gap:8px; margin:4px 0 0; padding-left:8px; border-left:3px solid var(--red); }
    .panel-title { font-size:.76rem; font-weight:900; letter-spacing:.12em; text-transform:uppercase; color:#e8eef7; }
    .panel-sub { font-size:.68rem; color:var(--muted); }
    .pair-legend { display:flex; flex-wrap:wrap; gap:18px; align-items:center; font-size:.78rem; color:#d5deeb; padding:2px 2px; }
    .pair-legend .lg { display:inline-flex; align-items:center; gap:7px; font-weight:700; }
    .pair-legend .lg i { display:inline-block; width:24px; height:4px; border-radius:2px; }
    .f1-muted { color:#9aa7b9; font-size:.74rem; }
    .f1-section { font-size:.9rem; font-weight:900; letter-spacing:.08em; margin:.1rem 0 .3rem; }

    /* ---- compact driver profile ---- */
    .driver-profile {
        --team:#888888; position:relative; isolation:isolate; overflow:hidden;
        display:grid; grid-template-columns:92px minmax(0,1fr) 216px; gap:14px; align-items:center;
        padding:12px 16px; border:1px solid color-mix(in srgb, var(--team) 42%, #273245);
        border-left:4px solid var(--team); border-radius:14px;
        background:linear-gradient(120deg, color-mix(in srgb, var(--team) 16%, #141b28) 0%, #101722 66%);
    }
    .driver-profile::before {
        content:""; position:absolute; z-index:-1; inset:6% 30% 6% 30%;
        background-image:var(--logo-image, none); background-position:center;
        background-repeat:no-repeat; background-size:contain; opacity:.07; filter:grayscale(1); pointer-events:none;
    }
    .profile-portrait { height:104px; width:92px; object-fit:contain; object-position:center bottom; filter:drop-shadow(0 4px 8px rgba(0,0,0,.3)); }
    .dp-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .driver-slot { display:inline-flex; padding:2px 8px; border-radius:20px; font-size:.58rem; font-weight:900; letter-spacing:.16em; color:#fff; background:var(--team); }
    .driver-number { color:var(--team); font-size:1.35rem; line-height:1; font-weight:900; font-style:italic; }
    .driver-name { color:#f7f9fc; font-size:1.3rem; line-height:1.1; font-weight:900; margin-top:5px; }
    .driver-team { color:#aebace; font-size:.78rem; margin-top:2px; }
    .winrate-row { display:flex; align-items:center; gap:8px; margin-top:8px; }
    .winrate-label { color:#9fb0c6; font-size:.58rem; letter-spacing:.14em; font-weight:900; }
    .winrate-value { color:var(--team); font-size:1.05rem; font-weight:900; min-width:48px; }
    .winrate-track { height:6px; flex:1; background:#2b3544; border-radius:10px; overflow:hidden; }
    .winrate-fill { height:100%; border-radius:10px; background:var(--team); }
    .winrate-detail { color:#8493a8; font-size:.64rem; margin-top:3px; }
    .profile-metrics { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:6px; }
    .profile-metric { background:rgba(8,12,19,.5); border:1px solid rgba(160,177,199,.10); border-radius:8px; padding:6px 9px; }
    .profile-metric-head { display:flex; align-items:center; gap:5px; color:#9fb0c6; font-size:.56rem; letter-spacing:.1em; font-weight:800; }
    .profile-metric-value { color:var(--team); font-size:1.05rem; line-height:1.15; font-weight:900; margin-top:2px; }
    @media(max-width:1150px){ .driver-profile{grid-template-columns:80px 1fr;} .profile-metrics{grid-column:1 / -1;} }

    /* ---- head-to-head scoreboard cards ---- */
    .compare-card { background:linear-gradient(145deg,#1a2330,#121923); border:1px solid #2b3647; border-radius:12px; padding:9px 12px; height:100%; }
    .compare-card-head { display:flex; align-items:center; gap:6px; color:#a6b3c5; font-size:.62rem; font-weight:900; letter-spacing:.11em; }
    .compare-card-values { display:flex; justify-content:space-between; align-items:baseline; gap:6px; margin-top:6px; }
    .compare-value { font-size:1.3rem; line-height:1.1; font-weight:900; white-space:nowrap; }
    .compare-vs { color:#7f8ca0; font-size:.64rem; font-weight:800; }
    .compare-split { display:flex; height:4px; border-radius:4px; overflow:hidden; background:#243044; margin-top:7px; }
    .compare-split > div { height:100%; }
    .compare-card-foot { color:#8493a8; font-size:.62rem; line-height:1.25; margin-top:5px; }
    .legend-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }

    /* ---- circuit / telemetry ---- */
    .circuit-hero { padding:10px 16px; border:1px solid #2d3a4d; border-left:4px solid var(--red); border-radius:12px;
        background:linear-gradient(115deg, rgba(27,39,56,.96), rgba(12,18,29,.96)); }
    .circuit-eyebrow { color:#ff5369; font-size:.58rem; font-weight:900; letter-spacing:.22em; text-transform:uppercase; }
    .circuit-hero-title { color:#f5f7fb; font-size:1.25rem; font-weight:900; line-height:1.15; }
    .circuit-hero-subtitle { color:#aab8ca; font-size:.74rem; margin-top:2px; }
    .tele-row { display:flex; gap:8px; align-items:stretch; flex-wrap:wrap; margin-bottom:6px; }
    .tele-id { display:flex; flex-direction:column; justify-content:center; min-width:150px; border-left:3px solid var(--c); padding:2px 10px; }
    .tele-code { color:#fff; font-size:.95rem; font-weight:900; letter-spacing:.08em; }
    .tele-name { color:#aab8ca; font-size:.7rem; }
    .tele-chip { flex:1; min-width:108px; background:rgba(18,27,40,.72); border:1px solid #2a3749; border-radius:9px; padding:6px 10px; }
    .tele-chip .l { color:#93a2b7; font-size:.56rem; letter-spacing:.1em; font-weight:800; text-transform:uppercase; }
    .tele-chip .v { color:#fff; font-size:1.08rem; font-weight:900; line-height:1.2; }
    [data-testid="stMetric"] { background:rgba(18,27,40,.72); border:1px solid #2a3749; border-radius:10px; padding:8px 12px; }
    [data-testid="stMetricLabel"] p { color:#93a2b7; font-size:.62rem; font-weight:700; }
    [data-testid="stMetricValue"] { color:#f5f7fb; font-size:1.2rem; font-weight:900; }
    hr { border-color:#273245 !important; margin:.6rem 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() in {"", "None", "nan"}:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    return int(number) if number is not None and number.is_integer() else None


def clean_hex(value: Any, fallback: str = "888888") -> str:
    raw = str(value or fallback).strip().lstrip("#")
    return "#" + (raw.upper() if re.fullmatch(r"[0-9a-fA-F]{6}", raw) else fallback.upper())


def hex_rgb(color: str) -> tuple[int, int, int]:
    c = clean_hex(color).lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def rgba(color: str, alpha: float) -> str:
    r, g, b = hex_rgb(color)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def lighten(color: str, amount: float = .34) -> str:
    r, g, b = hex_rgb(color)
    return "#" + "".join(f"{round(v + (255-v)*amount):02X}" for v in (r, g, b))


def icon_svg(name: str, color: str = "#AAB5C5", size: int = 16) -> str:
    # Minimal inline line icons; no external icon package is required.
    paths = {
        "trophy": '<path d="M8 4h8v4a4 4 0 0 1-8 0V4Z"/><path d="M8 6H5v2a4 4 0 0 0 4 4M16 6h3v2a4 4 0 0 1-4 4M12 12v4m-4 3h8m-7-3h6"/>',
        "points": '<path d="M4 19V5m0 14h17"/><path d="m7 15 4-4 3 2 6-7"/><path d="M17 6h3v3"/>',
        "flag": '<path d="M5 21V4"/><path d="M5 5c5-3 8 3 14 0v10c-6 3-9-3-14 0"/>',
        "medal": '<circle cx="12" cy="8" r="5"/><path d="m8.5 12-2 9 5.5-3 5.5 3-2-9"/>',
        "timer": '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l3 2M9 2h6m-3 0v3"/>',
        "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 10h18"/>',
        "gauge": '<path d="M4 18a9 9 0 1 1 16 0"/><path d="m12 13 4-5M7 18h10"/>',
        "qualifying": '<path d="M4 19V5m0 14h17"/><path d="m7 15 4-5 3 2 5-7"/>',
        "check": '<path d="m5 12 4 4L19 6"/>',
    }
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{paths.get(name, paths["gauge"])}</svg>')


def driver_id_from_result(result: dict) -> str:
    driver = result.get("Driver") or {}
    return str(driver.get("driverId") or driver.get("code") or "").strip()


def numeric_race_position(result: dict) -> int | None:
    # positionOrder is the numeric official classification order; position is fallback.
    return safe_int(result.get("positionOrder")) or safe_int(result.get("position"))


def is_actual_start(status: str) -> bool:
    text = (status or "").strip().lower()
    return not any(term in text for term in ("did not start", "withdrew", "did not qualify", "did not prequalify"))


def is_classified_finish(status: str) -> bool:
    text = (status or "").strip().lower()
    return text == "finished" or bool(re.match(r"^\+\s*\d+\s+lap", text))


def short_event_name(name: str) -> str:
    label = re.sub(r"\s+Grand Prix$", " GP", str(name), flags=re.I)
    label = re.sub(r"\s+Grand Prix\s*$", " GP", label, flags=re.I)
    return label


def panel_head(title: str, sub: str = "") -> None:
    """Slim heading rendered above a chart (replaces large in-figure titles)."""
    st.markdown(
        f"<div class='panel-head'><span class='panel-title'>{html.escape(title)}</span>"
        f"<span class='panel-sub'>{html.escape(sub)}</span></div>",
        unsafe_allow_html=True,
    )


def pair_legend(name_a: str, color_a: str, name_b: str, color_b: str, dashed_b: bool = False) -> None:
    """One shared legend for a whole row of charts, so individual charts need none."""
    swatch_b = (f"background:repeating-linear-gradient(90deg,{color_b} 0 6px,transparent 6px 10px)"
                if dashed_b else f"background:{color_b}")
    st.markdown(
        f"<div class='pair-legend'>"
        f"<span class='lg'><i style='background:{color_a}'></i>{html.escape(name_a)}</span>"
        f"<span class='lg'><i style='{swatch_b}'></i>{html.escape(name_b)}</span></div>",
        unsafe_allow_html=True,
    )


def plot_theme(fig: go.Figure, title: str = "", height: int = 280, legend: bool = False) -> go.Figure:
    """Compact chart styling. `title` is accepted for backwards compatibility but the
    heading is drawn above the chart with panel_head() to save vertical space."""
    fig.update_layout(
        title=None, height=height, autosize=True,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,18,28,.55)",
        font=dict(family="Titillium Web, sans-serif", color="#E8EDF5", size=10),
        margin=dict(l=44, r=14, t=14, b=34),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hoverlabel=dict(bgcolor="#172131", bordercolor="#526176", font=dict(color="#FFFFFF", size=11)),
    )
    axis_style = dict(gridcolor="rgba(94,112,137,.20)", zerolinecolor="#485468", linecolor="#344258",
                      tickfont=dict(size=9), title_font=dict(size=10, color="#AAB7C8"))
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def event_axis(fig: go.Figure, race_meta: list[dict], names: bool = False) -> None:
    """Round numbers by default (compact); full Grand Prix names for wide charts."""
    pairs = [(safe_int(r.get("round")), short_event_name(r.get("raceName", f"Round {r.get('round')}")))
             for r in race_meta if safe_int(r.get("round")) is not None]
    if names:
        fig.update_xaxes(type="linear", tickmode="array", tickvals=[p[0] for p in pairs],
                         ticktext=[p[1] for p in pairs], tickangle=-45, automargin=True,
                         tickfont=dict(size=8), title=None)
        fig.update_layout(margin=dict(l=44, r=14, t=14, b=74))
    else:
        step = 1 if len(pairs) <= 14 else 2
        fig.update_xaxes(type="linear", tickmode="array", tickvals=[p[0] for p in pairs][::step],
                         ticktext=[str(p[0]) for p in pairs][::step], tickangle=0,
                         title="Round", title_standoff=2)


def render_plotly_chart(fig: go.Figure, *, scroll_zoom: bool = True, extra_height: int = 12) -> None:
    """Render a Plotly figure with a small height allowance (prevents an inner scrollbar)."""
    figure_height = fig.layout.height
    try:
        figure_height = int(figure_height) if figure_height is not None else 300
    except (TypeError, ValueError):
        figure_height = 300
    st.plotly_chart(
        fig,
        width="stretch",
        height=max(200, figure_height + extra_height),
        config={"displaylogo": False, "scrollZoom": scroll_zoom, "displayModeBar": "hover"},
    )

# -----------------------------------------------------------------------------
# Offline-first season caches
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def load_qualifying_results(season: int, refresh: bool = False) -> list[dict]:
    cache = WAREHOUSE / f"qualifying_results_{int(season)}.json"
    if cache.exists() and not refresh:
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("complete") and isinstance(payload.get("races"), list):
                return payload["races"]
        except (OSError, ValueError, TypeError):
            pass

    by_round: dict[int, dict] = {}
    offset, page_size, expected_total = 0, 100, None
    try:
        while True:
            response = requests.get(
                f"https://api.jolpi.ca/ergast/f1/{int(season)}/qualifying.json",
                params={"limit": page_size, "offset": offset}, timeout=25,
            )
            response.raise_for_status()
            mrdata = response.json()["MRData"]
            total, limit = int(mrdata["total"]), int(mrdata["limit"])
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise ValueError("Qualifying results changed during download; retry refresh.")
            if limit <= 0:
                raise ValueError("Jolpica returned an invalid pagination limit.")
            for race in mrdata.get("RaceTable", {}).get("Races", []):
                rnd = int(race["round"])
                target = by_round.setdefault(rnd, {k: v for k, v in race.items() if k != "QualifyingResults"})
                target.setdefault("QualifyingResults", []).extend(race.get("QualifyingResults", []))
            offset += limit
            if offset >= total:
                break

        cleaned = []
        for rnd in sorted(by_round):
            race = by_round[rnd]
            unique = {}
            for result in race.get("QualifyingResults", []):
                key = driver_id_from_result(result)
                if key:
                    unique[key] = result
            race["QualifyingResults"] = list(unique.values())
            cleaned.append(race)
        WAREHOUSE.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(".tmp")
        temp.write_text(json.dumps({"complete": True, "season": int(season), "races": cleaned}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(cache)
        return cleaned
    except (requests.RequestException, ValueError, KeyError, TypeError, OSError):
        # Offline fallback: retain the last complete snapshot if it exists.
        if cache.exists():
            try:
                payload = json.loads(cache.read_text(encoding="utf-8"))
                return payload.get("races", [])
            except (OSError, ValueError, TypeError):
                pass
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def load_driver_standings(season: int, refresh: bool = False) -> list[dict]:
    cache = WAREHOUSE / f"driver_standings_{int(season)}.json"
    if cache.exists() and not refresh:
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("standings") is not None:
                return payload["standings"]
        except (OSError, ValueError, TypeError):
            pass
    try:
        response = requests.get(
            f"https://api.jolpi.ca/ergast/f1/{int(season)}/driverStandings.json",
            params={"limit": 100}, timeout=20,
        )
        response.raise_for_status()
        lists = response.json().get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        standings = lists[0].get("DriverStandings", []) if lists else []
        WAREHOUSE.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(".tmp")
        temp.write_text(json.dumps({"season": int(season), "standings": standings}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(cache)
        return standings
    except (requests.RequestException, ValueError, KeyError, TypeError, OSError):
        if cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8")).get("standings", [])
            except (OSError, ValueError, TypeError):
                pass
        return []


# -----------------------------------------------------------------------------
# Build season driver catalog from actual race results
# -----------------------------------------------------------------------------
def build_driver_catalog(season: int, races: list[dict]) -> list[dict]:
    roster = F.get_season_roster(season) or {}
    roster_by_name: dict[tuple[str, str], dict] = {}
    for team_name, entries in roster.items():
        for item in entries:
            key = (str(item.get("first_name", "")).casefold(), str(item.get("last_name", "")).casefold())
            if all(key):
                roster_by_name[key] = {**item, "team_name": team_name}

    appearances: dict[str, list[dict]] = defaultdict(list)
    driver_info: dict[str, dict] = {}
    for race in races:
        for result in race.get("Results", []):
            d = result.get("Driver") or {}
            key = driver_id_from_result(result)
            if not key:
                continue
            constructor = (result.get("Constructor") or {}).get("name", "Unknown constructor")
            appearances[key].append({"team": constructor, "round": safe_int(race.get("round")) or 0})
            name_key = (str(d.get("givenName", "")).casefold(), str(d.get("familyName", "")).casefold())
            roster_item = roster_by_name.get(name_key, {})
            if key not in driver_info:
                team_counts = Counter()
                driver_info[key] = {
                    "driver_id": key,
                    "first_name": d.get("givenName", "Unknown"),
                    "last_name": d.get("familyName", "Driver"),
                    "driver_number": result.get("number") or d.get("permanentNumber") or roster_item.get("driver_number"),
                    "name_acronym": d.get("code") or roster_item.get("name_acronym") or str(d.get("familyName", "DRV"))[:3].upper(),
                    "nationality": d.get("nationality") or roster_item.get("nationality", ""),
                    "team_name": constructor,
                    "team_colour": clean_hex(roster_item.get("team_colour") or F.get_fallback_color(constructor)),
                    "headshot_url": roster_item.get("headshot_url", ""),
                }
    # If a driver represented multiple teams, use the constructor with the most race entries.
    for key, records in appearances.items():
        team = Counter(x["team"] for x in records).most_common(1)[0][0]
        info = driver_info[key]
        info["team_name"] = team
        # Use the matching season roster colour if it exists; otherwise the known team palette.
        for (first, last), item in roster_by_name.items():
            if first == info["first_name"].casefold() and last == info["last_name"].casefold():
                if item.get("team_name", "").casefold() == team.casefold():
                    info["team_colour"] = clean_hex(item.get("team_colour") or F.get_fallback_color(team))
                    info["headshot_url"] = item.get("headshot_url") or info.get("headshot_url", "")
                    break
        info["starts_recorded"] = len(records)
    return sorted(driver_info.values(), key=lambda d: (d["last_name"].casefold(), d["first_name"].casefold()))


def get_driver_records(races: list[dict], driver_id: str) -> pd.DataFrame:
    rows = []
    for race in races:
        result = next((r for r in race.get("Results", []) if driver_id_from_result(r) == driver_id), None)
        if not result:
            continue
        driver = result.get("Driver") or {}
        constructor = result.get("Constructor") or {}
        rows.append({
            "round": safe_int(race.get("round")),
            "event": race.get("raceName", "Grand Prix"),
            "date": race.get("date", ""),
            "position": numeric_race_position(result),
            "grid": safe_int(result.get("grid")),
            "points": safe_float(result.get("points")) or 0.0,
            "status": str(result.get("status") or ""),
            "actual_start": is_actual_start(str(result.get("status") or "")),
            "classified_finish": is_classified_finish(str(result.get("status") or "")),
            "constructor": constructor.get("name", ""),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("round") if not df.empty else df


def result_lookup(races: list[dict]) -> dict[tuple[int, str], dict]:
    lookup = {}
    for race in races:
        rnd = safe_int(race.get("round"))
        if rnd is None:
            continue
        for result in race.get("Results", []):
            key = driver_id_from_result(result)
            if key:
                lookup[(rnd, key)] = result
    return lookup


def qualifying_lookup(qualifying: list[dict]) -> dict[tuple[int, str], dict]:
    lookup = {}
    for race in qualifying:
        rnd = safe_int(race.get("round"))
        if rnd is None:
            continue
        for result in race.get("QualifyingResults", []):
            key = driver_id_from_result(result)
            if key:
                lookup[(rnd, key)] = result
    return lookup


def driver_standings_lookup(standings: list[dict]) -> dict[str, dict]:
    return {str((item.get("Driver") or {}).get("driverId")): item for item in standings if (item.get("Driver") or {}).get("driverId")}

# -----------------------------------------------------------------------------
# Driver stats and normalized performance indices
# -----------------------------------------------------------------------------
def build_qualifying_frame(qualifying: list[dict], driver_id: str) -> pd.DataFrame:
    rows = []
    for race in qualifying:
        rnd = safe_int(race.get("round"))
        result = next((x for x in race.get("QualifyingResults", []) if driver_id_from_result(x) == driver_id), None)
        if not result:
            continue
        rows.append({"round": rnd, "event": race.get("raceName", "Grand Prix"), "position": safe_int(result.get("position")),
                     "q1": result.get("Q1"), "q2": result.get("Q2"), "q3": result.get("Q3")})
    df = pd.DataFrame(rows)
    return df.sort_values("round") if not df.empty else df


def summary_for_driver(driver: dict, records: pd.DataFrame, standings_map: dict[str, dict], qualifying_df: pd.DataFrame) -> dict:
    driver_id = driver["driver_id"]
    official = standings_map.get(driver_id, {})
    starts = int(records["actual_start"].sum()) if not records.empty else 0
    valid_positions = records["position"].dropna() if not records.empty else pd.Series(dtype=float)
    wins = int((valid_positions == 1).sum())
    podiums = int((valid_positions <= 3).sum())
    gp_points = float(records["points"].sum()) if not records.empty else 0.0
    official_points = safe_float(official.get("points"))
    points = official_points if official_points is not None else gp_points
    wdc = safe_int(official.get("position"))
    poles = int((qualifying_df["position"] == 1).sum()) if not qualifying_df.empty else 0
    win_rate = (wins / starts * 100.0) if starts else 0.0
    return {
        "starts": starts, "wins": wins, "podiums": podiums, "points": points,
        "gp_points": gp_points, "wdc": wdc, "poles": poles, "win_rate": win_rate,
        "avg_finish": float(valid_positions.mean()) if len(valid_positions) else None,
        "dnfs": int((~records["classified_finish"] & records["actual_start"]).sum()) if not records.empty else 0,
    }


def build_season_indices(all_drivers: list[dict], records_by_id: dict[str, pd.DataFrame], qframes: dict[str, pd.DataFrame], field_size: int) -> dict[str, dict]:
    # All axes are normalized relative to the same season field (0–100).
    max_position = max(field_size - 1, 1)
    points_per_start = {}
    for d in all_drivers:
        df = records_by_id.get(d["driver_id"], pd.DataFrame())
        starts = int(df["actual_start"].sum()) if not df.empty else 0
        points = float(df["points"].sum()) if not df.empty else 0.0
        points_per_start[d["driver_id"]] = points / starts if starts else 0.0
    best_pps = max(points_per_start.values(), default=0.0)

    output = {}
    for d in all_drivers:
        key = d["driver_id"]
        df = records_by_id.get(key, pd.DataFrame())
        qdf = qframes.get(key, pd.DataFrame())
        finish_positions = df["position"].dropna().astype(float) if not df.empty else pd.Series(dtype=float)
        q_positions = qdf["position"].dropna().astype(float) if not qdf.empty else pd.Series(dtype=float)
        grids = df["grid"].dropna().astype(float) if not df.empty else pd.Series(dtype=float)
        race_mean = float(finish_positions.mean()) if len(finish_positions) else None
        q_mean = float(q_positions.mean()) if len(q_positions) else None
        # Lower average position maps to a higher index.
        race_index = max(0.0, min(100.0, 100.0 * (field_size - race_mean) / max_position)) if race_mean is not None else None
        quali_index = max(0.0, min(100.0, 100.0 * (field_size - q_mean) / max_position)) if q_mean is not None else None
        # Consistency rewards a smaller standard deviation of finishing position.
        finish_std = float(finish_positions.std(ddof=0)) if len(finish_positions) >= 2 else None
        consistency = max(0.0, min(100.0, 100.0 * (1.0 - min(finish_std / max(field_size / 2.0, 1.0), 1.0)))) if finish_std is not None else None
        # 50 is neutral (no net grid-to-finish gain); gains > 50, losses < 50.
        # Exclude pit-lane/unknown grid values (0) from this index.
        gain_rows = df.dropna(subset=["grid", "position"]).copy() if not df.empty else pd.DataFrame()
        if not gain_rows.empty:
            gain_rows = gain_rows[(gain_rows["grid"] > 0) & (gain_rows["position"] > 0)]
        deltas = (gain_rows["grid"] - gain_rows["position"]) if not gain_rows.empty else pd.Series(dtype=float)
        mean_gain = float(deltas.mean()) if len(deltas) else None
        position_gain = max(0.0, min(100.0, 50.0 + 50.0 * mean_gain / max_position)) if mean_gain is not None else None
        starts_mask = df["actual_start"] if not df.empty else pd.Series(dtype=bool)
        finish_rate = (float(df.loc[starts_mask, "classified_finish"].mean()) * 100.0) if len(starts_mask) and starts_mask.any() else None
        pps = points_per_start[key]
        points_efficiency = (100.0 * pps / best_pps) if best_pps > 0 else None
        output[key] = {
            "Qualifying Pace": quali_index,
            "Race Result": race_index,
            "Race Consistency": consistency,
            "Position Gain": position_gain,
            "Reliability": finish_rate,
            "GP Points / Start": points_efficiency,
        }
    return output


def index_methodology_text() -> str:
    return (
        "All radar values are transparent 0–100 season-relative indices, not official FIA ratings. "
        "Qualifying Pace and Race Result convert average position against the season's maximum field size "
        "(P1 = 100; last place = 0). Race Consistency is 100 × [1 − min(population SD of classified positions ÷ half the field size, 1)]. "
        "Position Gain is centred at 50 (no net grid-to-finish change); gains score above 50 and losses below 50, "
        "scaled by the field size. Reliability is the percentage of recorded starts classified as Finished or +1 Lap etc. "
        "GP Points / Start is the driver's race-result points per start divided by the highest such rate in the season × 100. "
        "Missing data is shown as unavailable, not zero. Sprint points are not included in GP Points / Start."
    )

# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------
def profile_card(driver: dict, label: str, accent: str, variant: str, stats: dict) -> None:
    name = f"{driver['first_name']} {driver['last_name']}"
    team = driver.get("team_name", "Unknown constructor")
    team_logo_rel = F.get_team_logo(team)
    logo_src = F.get_local_image_base64(ROOT / team_logo_rel) if team_logo_rel else ""
    portrait_path = ROOT / F.clean_driver_filename(driver["first_name"], driver["last_name"], st.session_state.compare_season)
    portrait_src = F.get_local_image_base64(portrait_path) or driver.get("headshot_url") or F.DEFAULT_HEADSHOT
    if logo_src:
        watermark_style = f"--logo-image:url('data:image/png;base64,{logo_src.split(',', 1)[-1]}')"
    else:
        watermark_style = "--logo-image:none"
    win_rate = stats["win_rate"]
    wdc_value = f"P{stats['wdc']}" if stats["wdc"] is not None else "—"
    points_value = f"{stats['points']:,.0f}" if float(stats["points"]).is_integer() else f"{stats['points']:,.1f}"
    metrics = [
        ("trophy", "WDC", wdc_value),
        ("points", "POINTS", points_value),
        ("flag", "WINS", str(stats["wins"])),
        ("medal", "PODIUMS", str(stats["podiums"])),
    ]
    metric_html = "".join(
        f"<div class='profile-metric'><div class='profile-metric-head'>{icon_svg(icon, accent, 12)}<span>{text}</span></div>"
        f"<div class='profile-metric-value'>{value}</div></div>"
        for icon, text, value in metrics
    )
    number = html.escape(str(driver.get("driver_number") or "—"))
    card = (
        f"<div class='driver-profile profile-{variant}' style=\"--team:{accent}; {watermark_style}\">"
        f"<img class='profile-portrait' src=\"{html.escape(portrait_src, quote=True)}\" alt=\"{html.escape(name, quote=True)}\">"
        f"<div class='dp-main'>"
        f"<div class='dp-top'><span class='driver-slot'>{label}</span><span class='driver-number'>#{number}</span></div>"
        f"<div class='driver-name'>{html.escape(name)}</div>"
        f"<div class='driver-team'>{html.escape(team)}</div>"
        f"<div class='winrate-row'><span class='winrate-label'>WIN RATE</span>"
        f"<span class='winrate-value'>{win_rate:.1f}%</span>"
        f"<div class='winrate-track'><div class='winrate-fill' style='width:{max(0, min(100, win_rate)):.1f}%'></div></div></div>"
        f"<div class='winrate-detail'>{stats['wins']} wins · {stats['starts'] - stats['wins']} non-wins · {stats['starts']} starts</div>"
        f"</div>"
        f"<div class='profile-metrics'>{metric_html}</div>"
        f"</div>"
    )
    st.markdown(card, unsafe_allow_html=True)


def overview_card(title: str, icon: str, value_a: str, value_b: str, accent_a: str, accent_b: str,
                  foot: str, share_a: float | None = None) -> None:
    split = ""
    if share_a is not None:
        pct = max(0.0, min(100.0, share_a * 100))
        split = (f"<div class='compare-split'><div style='width:{pct:.1f}%;background:{accent_a}'></div>"
                 f"<div style='width:{100 - pct:.1f}%;background:{accent_b}'></div></div>")
    st.markdown(
        f"<div class='compare-card'>"
        f"<div class='compare-card-head'>{icon_svg(icon, '#AAB5C5', 13)} <span>{html.escape(title.upper())}</span></div>"
        f"<div class='compare-card-values'>"
        f"<span class='compare-value' style='color:{accent_a}'>{html.escape(value_a)}</span>"
        f"<span class='compare-vs'>VS</span>"
        f"<span class='compare-value' style='color:{accent_b}'>{html.escape(value_b)}</span></div>"
        f"{split}<div class='compare-card-foot'>{html.escape(foot)}</div></div>",
        unsafe_allow_html=True,
    )


def make_radar(indices_a: dict, indices_b: dict, name_a: str, name_b: str, color_a: str, color_b: str) -> go.Figure:
    categories = list(indices_a.keys())
    vals_a = [indices_a.get(k) if indices_a.get(k) is not None else np.nan for k in categories]
    vals_b = [indices_b.get(k) if indices_b.get(k) is not None else np.nan for k in categories]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a + vals_a[:1], theta=categories + categories[:1], fill="toself", name=name_a,
        line=dict(color=color_a, width=2.5), fillcolor=rgba(color_a, .19),
        marker=dict(size=5, color=color_a, symbol="circle"),
        hovertemplate="%{theta}<br>Index: %{r:.1f}/100<extra>" + name_a + "</extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b + vals_b[:1], theta=categories + categories[:1], fill="toself", name=name_b,
        line=dict(color=color_b, width=2.5, dash="dash"), fillcolor=rgba(color_b, .12),
        marker=dict(size=5, color=color_b, symbol="diamond"),
        hovertemplate="%{theta}<br>Index: %{r:.1f}/100<extra>" + name_b + "</extra>",
    ))
    fig.update_layout(
        height=310, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,18,27,.45)",
        font=dict(family="Titillium Web, sans-serif", color="#E8EDF5", size=9),
        margin=dict(l=52, r=52, t=18, b=18), showlegend=False,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75, 100],
                            gridcolor="#344052", linecolor="#465267", tickfont=dict(size=8, color="#AAB5C5")),
            angularaxis=dict(gridcolor="#344052", linecolor="#465267", tickfont=dict(size=9, color="#DCE3ED")),
        ),
    )
    return fig


def points_display(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"

# -----------------------------------------------------------------------------
# HDFS Silver/Gold race telemetry analytics
# -----------------------------------------------------------------------------


def _driver_display_name(driver: dict) -> str:
    """Return the driver's readable full name for telemetry labels."""
    return f"{driver.get('first_name', '')} {driver.get('last_name', '')}".strip()


def _driver_code(driver: dict) -> str:
    """Return the driver's three-letter telemetry identifier."""
    return str(driver.get("name_acronym") or driver.get("last_name", "DRV")[:3]).strip().upper()


def _telemetry_progress_percent(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    """Use ETL relative distance; fall back transparently to lap distance."""
    relative = pd.to_numeric(frame.get("rel_distance"), errors="coerce")
    valid = relative.notna()
    if valid.mean() >= 0.80 and relative[valid].between(-0.01, 1.01).mean() >= 0.80:
        return relative.clip(0, 1) * 100.0, "ETL relative distance"

    distance = pd.to_numeric(frame.get("distance_m"), errors="coerce")
    valid_distance = distance.dropna()
    if len(valid_distance) >= 2 and valid_distance.max() > valid_distance.min():
        progress = (distance - valid_distance.min()) / (valid_distance.max() - valid_distance.min()) * 100.0
        return progress, "normalised lap distance fallback"

    sample_index = pd.to_numeric(frame.get("sample_index"), errors="coerce")
    if sample_index.notna().sum() >= 2 and sample_index.max() > sample_index.min():
        progress = (sample_index - sample_index.min()) / (sample_index.max() - sample_index.min()) * 100.0
        return progress, "sample-index fallback"

    return pd.Series(np.nan, index=frame.index), "unavailable"


def _telemetry_metric_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    speed = pd.to_numeric(frame.get("speed_kmh"), errors="coerce")
    throttle = pd.to_numeric(frame.get("throttle_pct"), errors="coerce")
    rpm = pd.to_numeric(frame.get("rpm"), errors="coerce")
    brake = pd.to_numeric(frame.get("brake"), errors="coerce")
    brake_valid = brake.dropna()
    return {
        "top_speed": float(speed.max()) if speed.notna().any() else np.nan,
        "average_speed": float(speed.mean()) if speed.notna().any() else np.nan,
        "average_throttle": float(throttle.mean()) if throttle.notna().any() else np.nan,
        "maximum_rpm": float(rpm.max()) if rpm.notna().any() else np.nan,
        "braking_samples": int((brake_valid > 0).sum()),
        "braking_pct": float((brake_valid > 0).mean() * 100.0) if len(brake_valid) else np.nan,
    }



def _telemetry_numeric_profile(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Return one row per progress value without averaging discrete states.

    Continuous telemetry (GPS, speed, throttle and RPM) is averaged when
    duplicate progress values occur. Gear and brake are states, so the last
    recorded sample at that progress is retained instead of creating values
    such as gear 5.5 or brake 0.4.
    """
    keep = ["progress_pct", *columns]
    available = [column for column in keep if column in frame.columns]
    if "progress_pct" not in available:
        return pd.DataFrame(columns=keep)

    profile = frame[available].copy()
    for column in available:
        profile[column] = pd.to_numeric(profile[column], errors="coerce")
    profile = profile.dropna(subset=["progress_pct"])
    if profile.empty:
        return profile

    # Preserve telemetry sample order for duplicate progress values.
    if "sample_index" in frame.columns:
        profile["__sample_order"] = pd.to_numeric(
            frame.loc[profile.index, "sample_index"], errors="coerce"
        )
        profile = profile.sort_values(
            ["progress_pct", "__sample_order"], kind="mergesort"
        ).drop(columns=["__sample_order"])
    else:
        profile = profile.sort_values("progress_pct", kind="mergesort")

    discrete = {"gear", "brake", "drs"}
    aggregate_columns = [column for column in columns if column in profile.columns]
    aggregations = {
        column: (
            lambda values: values.dropna().iloc[-1] if values.notna().any() else np.nan
        ) if column in discrete else "mean"
        for column in aggregate_columns
    }
    result = profile.groupby("progress_pct", as_index=False, sort=True).agg(aggregations)
    for column in discrete.intersection(result.columns):
        result[column] = pd.to_numeric(result[column], errors="coerce").round()
    return result


def _telemetry_interpolate(
    profile: pd.DataFrame, column: str, grid: np.ndarray
) -> np.ndarray | None:
    """Interpolate one telemetry channel only inside its recorded progress range."""
    if column not in profile.columns or profile.empty:
        return None
    valid = profile[["progress_pct", column]].dropna().sort_values("progress_pct")
    if len(valid) < 2:
        return None
    xp = valid["progress_pct"].to_numpy(dtype=float)
    fp = valid[column].to_numpy(dtype=float)
    unique = np.concatenate(([True], np.diff(xp) > 1e-9))
    xp, fp = xp[unique], fp[unique]
    if len(xp) < 2 or np.any(np.diff(xp) <= 0):
        return None
    if grid.min() < xp.min() or grid.max() > xp.max():
        return None
    return np.interp(grid, xp, fp)


def _audit_flag_for_driver(
    quality: pd.DataFrame, driver_code: str, column: str
) -> bool | None:
    """Read a boolean quality flag without treating missing values as valid."""
    if quality.empty or column not in quality.columns or "driver_code" not in quality.columns:
        return None
    rows = quality[quality["driver_code"].astype(str).str.upper() == driver_code.upper()]
    if rows.empty:
        return None
    value = rows.iloc[0][column]
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _gps_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Return valid x/y coordinates indexed by normalised lap progress."""
    profile = _telemetry_numeric_profile(frame, ("x", "y"))
    if not {"x", "y"}.issubset(profile.columns):
        return pd.DataFrame(columns=["progress_pct", "x", "y"])
    profile = profile.dropna(subset=["x", "y"]).sort_values("progress_pct")
    if len(profile) < 10 or profile["progress_pct"].nunique() < 10:
        return pd.DataFrame(columns=["progress_pct", "x", "y"])
    return profile.reset_index(drop=True)


def _matched_gps_separation(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    quality: pd.DataFrame,
    code_a: str,
    code_b: str,
) -> tuple[pd.DataFrame, str | None]:
    """Calculate Euclidean x/y separation at matched normalised lap progress."""
    map_a = _audit_flag_for_driver(quality, code_a, "track_map_usable")
    map_b = _audit_flag_for_driver(quality, code_b, "track_map_usable")
    if map_a is not True or map_b is not True:
        return pd.DataFrame(), "Gold quality audit does not confirm track-map usability for both drivers."
    gps_a, gps_b = _gps_profile(frame_a), _gps_profile(frame_b)
    if gps_a.empty or gps_b.empty:
        return pd.DataFrame(), "At least one driver has insufficient valid GPS x/y samples."
    lower = max(float(gps_a["progress_pct"].min()), float(gps_b["progress_pct"].min()))
    upper = min(float(gps_a["progress_pct"].max()), float(gps_b["progress_pct"].max()))
    if upper <= lower:
        return pd.DataFrame(), "The two GPS traces do not have an overlapping lap-progress range."
    point_count = max(100, min(500, min(len(gps_a), len(gps_b))))
    grid = np.linspace(lower, upper, point_count)
    ax = _telemetry_interpolate(gps_a, "x", grid)
    ay = _telemetry_interpolate(gps_a, "y", grid)
    bx = _telemetry_interpolate(gps_b, "x", grid)
    by = _telemetry_interpolate(gps_b, "y", grid)
    if any(value is None for value in (ax, ay, bx, by)):
        return pd.DataFrame(), "GPS coordinates could not be aligned reliably across the selected lap."
    separation = np.hypot(ax - bx, ay - by)
    result = pd.DataFrame({
        "progress_pct": grid,
        "separation_m": separation,
        "driver_a_x": ax,
        "driver_a_y": ay,
        "driver_b_x": bx,
        "driver_b_y": by,
    })
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["separation_m"])
    if len(result) < 20:
        return pd.DataFrame(), "Too few matched GPS positions remain for a reliable comparison."
    return result, None


def _telemetry_layout(fig: go.Figure, title: str = "", height: int = 250) -> go.Figure:
    """Compact telemetry chart body. Headings are drawn above with panel_head()."""
    fig.update_layout(
        title=None, height=height, autosize=True, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,18,28,.88)",
        font=dict(family="Titillium Web, sans-serif", color="#DCE5F1", size=10),
        margin=dict(l=52, r=14, t=12, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#172131", bordercolor="#526176",
                        font=dict(family="Titillium Web, sans-serif", color="#FFFFFF", size=11)),
    )
    fig.update_xaxes(
        title="Lap progress (%)", range=[0, 100], gridcolor="rgba(94,112,137,.22)", zeroline=False,
        showline=True, linecolor="#344258", ticks="outside", tickfont=dict(size=9, color="#AAB7C8"),
        title_font=dict(size=10, color="#B8C5D6"), title_standoff=2,
    )
    fig.update_yaxes(
        gridcolor="rgba(94,112,137,.22)", zerolinecolor="#526176", showline=True, linecolor="#344258",
        ticks="outside", tickfont=dict(size=9, color="#AAB7C8"), title_font=dict(size=10, color="#B8C5D6"),
        title_standoff=2,
    )
    return fig


def _circuit_chart_heading(icon: str, title: str, subtitle: str = "") -> None:
    panel_head(title, subtitle)


def _fmt(value, template: str) -> str:
    return template.format(value) if pd.notna(value) else "—"


def _telemetry_stat_rows(rows: list[tuple[str, str, str, dict]]) -> None:
    """One compact row per driver: identity + five metric chips."""
    html_rows = []
    for code, name, accent, s in rows:
        chips = [
            ("Top speed", _fmt(s["top_speed"], "{:.1f} km/h")),
            ("Avg speed", _fmt(s["average_speed"], "{:.1f} km/h")),
            ("Avg throttle", _fmt(s["average_throttle"], "{:.1f}%")),
            ("Max RPM", _fmt(s["maximum_rpm"], "{:,.0f}")),
            ("Braking", _fmt(s["braking_pct"], "{:.1f}%")),
        ]
        chip_html = "".join(f"<div class='tele-chip'><div class='l'>{l}</div><div class='v'>{v}</div></div>" for l, v in chips)
        html_rows.append(
            f"<div class='tele-row'><div class='tele-id' style='--c:{html.escape(accent)}'>"
            f"<span class='tele-code'>{html.escape(code)}</span><span class='tele-name'>{html.escape(name)}</span></div>"
            f"{chip_html}</div>"
        )
    st.markdown("".join(html_rows), unsafe_allow_html=True)


def render_race_telemetry_section(
    season: int,
    driver_a: dict,
    driver_b: dict,
    color_a: str,
    color_b: str,
) -> None:
    """Validated same-lap Race telemetry from the HDFS warehouse, laid out as a compact dashboard."""
    code_a, code_b = _driver_code(driver_a), _driver_code(driver_b)
    name_a, name_b = _driver_display_name(driver_a), _driver_display_name(driver_b)

    try:
        with st.spinner("Reading Gold comparison candidates from HDFS..."):
            candidates = load_telemetry_pair_candidates(season, code_a, code_b)
    except TelemetryDataError as exc:
        st.warning("Race telemetry is not available in this app session yet.")
        st.caption(str(exc))
        st.info(
            "The lap-time and sector analysis remains available independently. Start HDFS, "
            "confirm the Race telemetry Silver/Gold outputs exist, and check FOR1SIGHT_HDFS_URI "
            "if your NameNode is not localhost:9000."
        )
        return

    if candidates.empty:
        st.info(
            f"No Gold comparison candidate was found for {code_a} vs {code_b} in {season}. "
            "The pair table lists drivers who have common Race laps that passed the "
            "driving-style telemetry quality checks. Try another driver pair or season."
        )
        return

    event_options = candidates["event_name"].astype(str).tolist()
    default_event_index = next(
        (i for i, value in enumerate(event_options) if "australian grand prix" in value.casefold()), 0,
    )

    with st.container(border=True):
        filter_cols = st.columns([2, 1, 3], gap="medium")
        with filter_cols[0]:
            selected_event = st.selectbox(
                "TELEMETRY GRAND PRIX", event_options, index=default_event_index,
                key=f"telemetry_event_{season}_{code_a}_{code_b}",
            )
        selected_pair = candidates[candidates["event_name"].astype(str) == selected_event].iloc[0]
        common_laps = []
        for token in str(selected_pair.get("common_laps", "")).split(","):
            try:
                common_laps.append(int(token.strip()))
            except (TypeError, ValueError):
                continue
        if not common_laps:
            st.warning("The selected Gold comparison candidate has no readable common-lap list.")
            return
        lap_index = common_laps.index(1) if 1 in common_laps else 0
        with filter_cols[1]:
            selected_lap = st.selectbox(
                "COMMON RACE LAP", common_laps, index=lap_index,
                key=f"telemetry_lap_{season}_{code_a}_{code_b}_{selected_pair['event_slug']}",
                format_func=lambda value: f"Lap {value}",
            )
        with filter_cols[2]:
            st.markdown(
                f"<div class='f1-muted' style='padding-top:1.55rem'>Gold validates "
                f"{int(selected_pair['common_lap_count']):,} common laps for {code_a} and {code_b} · session: Race</div>",
                unsafe_allow_html=True,
            )

    try:
        with st.spinner(f"Loading lap {selected_lap} Silver telemetry samples..."):
            quality = load_telemetry_audit(season, str(selected_pair["event_slug"]), code_a, code_b, selected_lap)
            samples = load_telemetry_lap_samples(season, str(selected_pair["event_slug"]), code_a, code_b, selected_lap)
    except TelemetryDataError as exc:
        st.error("The selected telemetry records could not be loaded.")
        st.caption(str(exc))
        if exc.__cause__ is not None:
            st.caption(f"Cause: {type(exc.__cause__).__name__}: {exc.__cause__}")
        return

    if samples.empty:
        st.warning("No validated Silver samples were returned for this event, driver pair and lap.")
        return

    driver_frames = {
        code_a: samples[samples["driver_code"].astype(str).str.upper() == code_a].copy(),
        code_b: samples[samples["driver_code"].astype(str).str.upper() == code_b].copy(),
    }
    missing = [code for code, frame in driver_frames.items() if frame.empty]
    if missing:
        st.warning("Telemetry samples are missing for: " + ", ".join(missing))
        return

    for frame in driver_frames.values():
        frame.sort_values("sample_index", inplace=True)
        frame["progress_pct"], frame.attrs["progress_method"] = _telemetry_progress_percent(frame)

    summaries = {code: _telemetry_metric_summary(driver_frames[code]) for code in (code_a, code_b)}
    driver_specs = ((code_a, name_a, color_a, "solid"), (code_b, name_b, color_b, "dash"))

    tab_trace, tab_replay = st.tabs(["Telemetry traces", "Circuit replay"])

    # ------------------------------------------------------------------ traces
    with tab_trace:
        panel_head("Selected-lap summary", f"{selected_event} · lap {selected_lap}")
        _telemetry_stat_rows([
            (code_a, name_a, color_a, summaries[code_a]),
            (code_b, name_b, color_b, summaries[code_b]),
        ])
        pair_legend(name_a, color_a, name_b, color_b, dashed_b=True)

        # --- build every figure first, then place them in a 2-column grid
        speed_fig = go.Figure()
        throttle_fig = go.Figure()
        brake_fig = go.Figure()
        rpm_fig = go.Figure()
        gear_fig = go.Figure()
        for code, name, color, dash in driver_specs:
            frame = driver_frames[code]
            x = frame["progress_pct"]
            speed_fig.add_trace(go.Scatter(
                x=x, y=pd.to_numeric(frame["speed_kmh"], errors="coerce"), mode="lines", name=name,
                line=dict(color=color, width=2.6, dash=dash),
                customdata=frame[["distance_m", "gear", "rpm"]].to_numpy(),
                hovertemplate=(f"<b>{name}</b><br>Lap progress: %{{x:.1f}}%<br>Speed: %{{y:.1f}} km/h"
                               "<br>Distance: %{customdata[0]:.1f} m<br>Gear: %{customdata[1]}"
                               "<br>RPM: %{customdata[2]:,.0f}<extra></extra>"),
            ))
            throttle_fig.add_trace(go.Scatter(
                x=x, y=pd.to_numeric(frame["throttle_pct"], errors="coerce"), mode="lines", name=name,
                line=dict(color=color, width=2.2, dash=dash),
                hovertemplate=f"<b>{name}</b><br>Lap progress: %{{x:.1f}}%<br>Throttle: %{{y:.1f}}%<extra></extra>",
            ))
            brake_fig.add_trace(go.Scatter(
                x=x, y=pd.to_numeric(frame["brake"], errors="coerce"), mode="lines", name=name,
                line=dict(color=color, width=2.0, dash=dash, shape="hv"),
                hovertemplate=(f"<b>{name}</b><br>Lap progress: %{{x:.1f}}%"
                               "<br>Brake state: %{y:.0f} (1 = on, 0 = off)<extra></extra>"),
            ))
            rpm_fig.add_trace(go.Scatter(
                x=x, y=pd.to_numeric(frame["rpm"], errors="coerce"), mode="lines", name=name,
                line=dict(color=color, width=2.1, dash=dash),
                hovertemplate=f"<b>{name}</b><br>Lap progress: %{{x:.1f}}%<br>RPM: %{{y:,.0f}}<extra></extra>",
            ))
            gear_fig.add_trace(go.Scatter(
                x=x, y=pd.to_numeric(frame["gear"], errors="coerce"), mode="lines", name=name,
                line=dict(color=color, width=2.0, dash=dash, shape="hv"),
                hovertemplate=f"<b>{name}</b><br>Lap progress: %{{x:.1f}}%<br>Gear: %{{y:.0f}}<extra></extra>",
            ))

        _telemetry_layout(speed_fig, height=250)
        speed_fig.update_yaxes(title="km/h", rangemode="tozero")
        _telemetry_layout(throttle_fig, height=250)
        throttle_fig.update_yaxes(title="Throttle (%)", range=[0, 105], dtick=25)
        _telemetry_layout(brake_fig, height=250)
        brake_fig.update_yaxes(title="Brake", range=[-0.1, 1.15], tickvals=[0, 1], ticktext=["Off", "On"])
        _telemetry_layout(gear_fig, height=250)
        gear_fig.update_yaxes(title="Gear", range=[0.5, 8.5], tickvals=list(range(1, 9)), dtick=1)
        _telemetry_layout(rpm_fig, height=250)
        rpm_fig.update_yaxes(title="RPM")

        separation, separation_error = _matched_gps_separation(
            driver_frames[code_a], driver_frames[code_b], quality, code_a, code_b
        )

        row1 = st.columns(2, gap="small")
        with row1[0]:
            panel_head("Speed profile", f"{code_a} vs {code_b} · km/h")
            render_plotly_chart(speed_fig)
        with row1[1]:
            panel_head("Throttle application", "accelerator input")
            render_plotly_chart(throttle_fig)

        row2 = st.columns(2, gap="small")
        with row2[0]:
            panel_head("Braking zones", "recorded on / off state")
            render_plotly_chart(brake_fig)
        with row2[1]:
            panel_head("Gear selection", "recorded gear")
            render_plotly_chart(gear_fig)

        row3 = st.columns(2, gap="small")
        with row3[0]:
            panel_head("Engine RPM", "engine speed")
            render_plotly_chart(rpm_fig)
        with row3[1]:
            if separation_error:
                panel_head("GPS separation", "matched lap progress")
                st.info(separation_error)
                st.caption("Requires both Gold track-map quality flags and enough overlapping GPS x/y samples.")
            else:
                mean_sep = float(separation["separation_m"].mean())
                median_sep = float(separation["separation_m"].median())
                p95_sep = float(separation["separation_m"].quantile(.95))
                max_sep = float(separation["separation_m"].max())
                panel_head("GPS separation",
                           f"mean {mean_sep:.1f} m · median {median_sep:.1f} m · P95 {p95_sep:.1f} m · max {max_sep:.1f} m")
                dev_fig = go.Figure()
                dev_fig.add_trace(go.Scatter(
                    x=separation["progress_pct"], y=separation["separation_m"], mode="lines",
                    name=f"{code_a}–{code_b}", line=dict(color="#F4C95D", width=2.4),
                    fill="tozeroy", fillcolor="rgba(244,201,93,.12)",
                    hovertemplate="Lap progress: %{x:.1f}%<br>Separation: %{y:.1f} m<extra></extra>",
                ))
                dev_fig.add_hline(y=mean_sep, line_dash="dash", line_color="#DCE3ED", line_width=1)
                _telemetry_layout(dev_fig, height=250)
                dev_fig.update_yaxes(title="metres", rangemode="tozero")
                render_plotly_chart(dev_fig)

    # ------------------------------------------------------------------ replay
    with tab_replay:
        ctrl = st.columns([1, 4], gap="small")
        with ctrl[0]:
            replay_speed = st.selectbox(
                "REPLAY SPEED", [0.5, 1.0, 2.0], index=1, format_func=lambda value: f"{value:g}×",
                key=f"telemetry_replay_speed_{season}_{code_a}_{code_b}_{selected_pair['event_slug']}_{selected_lap}",
            )
        with ctrl[1]:
            st.markdown(
                "<div class='f1-muted' style='padding-top:1.55rem'>Animated driver positions on the GPS reference track · "
                "colour-coded matched-progress separation · drivers aligned by normalised lap progress, not session clock.</div>",
                unsafe_allow_html=True,
            )
        map_quality_a = _audit_flag_for_driver(quality, code_a, "track_map_usable")
        map_quality_b = _audit_flag_for_driver(quality, code_b, "track_map_usable")
        replay_profiles = {
            code: _telemetry_numeric_profile(driver_frames[code], ("x", "y", "speed_kmh", "throttle_pct", "brake", "gear", "rpm"))
            for code in (code_a, code_b)
        }
        if map_quality_a is not True or map_quality_b is not True:
            st.warning("Interactive replay is disabled because Gold does not confirm track-map quality for both drivers.")
        else:
            try:
                replay_fig = build_interactive_track_replay(
                    replay_profiles[code_a], replay_profiles[code_b],
                    code_a=code_a, code_b=code_b, name_a=name_a, name_b=name_b,
                    color_a=color_a, color_b=color_b, playback_speed=float(replay_speed), frame_count=120,
                )
                render_plotly_chart(replay_fig, scroll_zoom=True, extra_height=40)
            except ValueError as exc:
                st.info(f"Interactive replay is unavailable for this lap: {exc}")

    # ------------------------------------------------------------------ notes
    methods = sorted({
        str(value) for frame in driver_frames.values()
        for value in frame.get("rel_distance_source", pd.Series(dtype=str)).dropna().unique()
    })
    progress_methods = sorted({str(frame.attrs.get("progress_method", "unknown")) for frame in driver_frames.values()})
    with st.expander("Methodology & chart guide", expanded=False):
        st.markdown(
            "- **Speed / throttle / RPM / gear:** plotted against normalised lap progress for the same Race lap.\n"
            "- **Brake state:** the recorded 0/1 channel, not brake pressure.\n"
            "- **GPS separation:** Euclidean distance between the two recorded x/y positions at matched progress; "
            "not a time gap or lateral racing-line offset.\n\n"
            f"Distance source in Silver: {', '.join(methods) if methods else 'not recorded'} · "
            f"Alignment: {', '.join(progress_methods) if progress_methods else 'unknown'}.\n\n"
            "These are descriptive comparisons. Car setup, tyres, fuel load, traffic, weather and strategy influence the traces."
        )


def render_circuit_performance_tab(
    season: int,
    driver_a: dict,
    driver_b: dict,
    color_a: str,
    color_b: str,
) -> None:
    """Compact HDFS telemetry and circuit replay dashboard."""
    st.markdown(
        "<div class='circuit-hero'><div class='circuit-eyebrow'>For1Sight / Track Intelligence</div>"
        "<div class='circuit-hero-title'>Circuit Performance</div>"
        "<div class='circuit-hero-subtitle'>Race telemetry and interactive circuit replay · compare two drivers at the selected Grand Prix</div></div>",
        unsafe_allow_html=True,
    )
    render_race_telemetry_section(season, driver_a, driver_b, color_a, color_b)


# -----------------------------------------------------------------------------
# Page header and selectors
# -----------------------------------------------------------------------------
st.markdown(
    "<div class='dash-hero'><div><div class='dash-kicker'>FOR1SIGHT / RACE INTELLIGENCE</div>"
    "<div class='dash-title'>Driver Comparison</div>"
    "<div class='dash-sub'>Head-to-head performance analytics · compare any two drivers across the selected season</div></div>"
    "<div><span class='dash-chip'>SEASON H2H</span> <span class='dash-chip'>TELEMETRY</span></div></div>",
    unsafe_allow_html=True,
)

if "compare_season" not in st.session_state:
    st.session_state.compare_season = int(st.session_state.get("global_season", 2026))
if st.session_state.compare_season not in SEASONS:
    st.session_state.compare_season = SEASONS[-1]

with st.container(border=True):
    filter_cols = st.columns([.9, 1.55, 1.55], gap="medium")
    with filter_cols[0]:
        season = st.selectbox("SEASON", SEASONS, index=SEASONS.index(st.session_state.compare_season), key="compare_season_select")
    if season != st.session_state.compare_season:
        st.session_state.compare_season = season
    with st.spinner(f"Loading {season} season race data..."):
        races = F.fetch_api_results(season)
    if not races:
        st.error(f"No race results are available for {season}. Check the offline race_results_{season}.json cache or the Jolpica connection.")
        st.stop()
    all_drivers = build_driver_catalog(season, races)
    if len(all_drivers) < 2:
        st.error("At least two drivers are required in the selected season.")
        st.stop()
    options = list(range(len(all_drivers)))

    def label_for(index: int) -> str:
        item = all_drivers[index]
        return f"{item['first_name']} {item['last_name']}  ·  {item['team_name']}"

    previous_a = st.session_state.get(f"compare_driver_a_{season}", 0)
    previous_b = st.session_state.get(f"compare_driver_b_{season}", 1 if len(options) > 1 else 0)
    try:
        previous_a = int(previous_a)
    except (ValueError, TypeError):
        previous_a = 0
    previous_a = previous_a if previous_a in options else 0
    if f"compare_driver_a_{season}" in st.session_state and st.session_state[f"compare_driver_a_{season}"] not in options:
        st.session_state[f"compare_driver_a_{season}"] = previous_a
    with filter_cols[1]:
        index_a = st.selectbox("DRIVER A", options, index=previous_a, format_func=label_for, key=f"compare_driver_a_{season}")
    candidate_b = [i for i in options if i != index_a]
    try:
        previous_b = int(previous_b)
    except (ValueError, TypeError):
        previous_b = candidate_b[0]
    if previous_b not in candidate_b:
        previous_b = candidate_b[0]
    if f"compare_driver_b_{season}" in st.session_state and st.session_state[f"compare_driver_b_{season}"] not in candidate_b:
        st.session_state[f"compare_driver_b_{season}"] = previous_b
    with filter_cols[2]:
        index_b = st.selectbox("DRIVER B", candidate_b, index=candidate_b.index(previous_b), format_func=label_for, key=f"compare_driver_b_{season}")

    driver_a, driver_b = all_drivers[index_a], all_drivers[index_b]

# Pull reusable season snapshots once.
with st.spinner("Preparing comparison statistics and charts..."):
    qualifying_races = load_qualifying_results(season)
    standings = load_driver_standings(season)

records_by_id = {d["driver_id"]: get_driver_records(races, d["driver_id"]) for d in all_drivers}
qframes = {d["driver_id"]: build_qualifying_frame(qualifying_races, d["driver_id"]) for d in all_drivers}
standings_map = driver_standings_lookup(standings)
summary_a = summary_for_driver(driver_a, records_by_id[driver_a["driver_id"]], standings_map, qframes[driver_a["driver_id"]])
summary_b = summary_for_driver(driver_b, records_by_id[driver_b["driver_id"]], standings_map, qframes[driver_b["driver_id"]])

# When both drivers share a constructor, keep the team identity but separate the traces.
base_a = clean_hex(driver_a.get("team_colour") or F.get_fallback_color(driver_a["team_name"]))
base_b = clean_hex(driver_b.get("team_colour") or F.get_fallback_color(driver_b["team_name"]))
same_team = driver_a["team_name"].casefold() == driver_b["team_name"].casefold()
color_a = base_a
color_b = lighten(base_a, .43) if same_team else base_b

# The season comparison remains intact in Tab 1. Tab 2 reuses the same season
# and driver selections but analyses one event/session from the local Silver data.
tab_season, tab_circuit = st.tabs(["Season Head-to-Head", "Circuit Performance"])

with tab_season:
    full_a = f"{driver_a['first_name']} {driver_a['last_name']}"
    full_b = f"{driver_b['first_name']} {driver_b['last_name']}"
    dash_b = "dash" if same_team else "solid"

    # ---- driver profile cards -------------------------------------------------
    profile_cols = st.columns(2, gap="small")
    with profile_cols[0]:
        profile_card(driver_a, "DRIVER A", color_a, "a", summary_a)
    with profile_cols[1]:
        profile_card(driver_b, "DRIVER B", color_b, "b", summary_b)

    # ---- indices + head-to-head counts ---------------------------------------
    max_field_size = max((len(r.get("Results", [])) for r in races), default=20)
    max_field_size = max(max_field_size, 2)
    indices = build_season_indices(all_drivers, records_by_id, qframes, max_field_size)
    indices_a = indices.get(driver_a["driver_id"], {})
    indices_b = indices.get(driver_b["driver_id"], {})

    results_lookup = result_lookup(races)
    qual_lookup = qualifying_lookup(qualifying_races)
    common_rounds = sorted({rnd for rnd, key in results_lookup if key == driver_a["driver_id"]} & {rnd for rnd, key in results_lookup if key == driver_b["driver_id"]})
    race_ahead = race_behind = race_ties = race_compared = 0
    for rnd in common_rounds:
        pa = numeric_race_position(results_lookup.get((rnd, driver_a["driver_id"]), {}))
        pb = numeric_race_position(results_lookup.get((rnd, driver_b["driver_id"]), {}))
        if pa is None or pb is None:
            continue
        race_compared += 1
        if pa < pb:
            race_ahead += 1
        elif pb < pa:
            race_behind += 1
        else:
            race_ties += 1

    common_q_rounds = sorted({rnd for rnd, key in qual_lookup if key == driver_a["driver_id"]} & {rnd for rnd, key in qual_lookup if key == driver_b["driver_id"]})
    qual_ahead = qual_behind = qual_ties = qual_compared = 0
    for rnd in common_q_rounds:
        qa = safe_int(qual_lookup[(rnd, driver_a["driver_id"])].get("position"))
        qb = safe_int(qual_lookup[(rnd, driver_b["driver_id"])].get("position"))
        if qa is None or qb is None:
            continue
        qual_compared += 1
        if qa < qb:
            qual_ahead += 1
        elif qb < qa:
            qual_behind += 1
        else:
            qual_ties += 1

    points_gap = summary_a["points"] - summary_b["points"]
    podium_gap = summary_a["podiums"] - summary_b["podiums"]

    def _share(a: float, b: float) -> float | None:
        return a / (a + b) if (a + b) > 0 else None

    overview_cols = st.columns(4, gap="small")
    with overview_cols[0]:
        overview_card("Race head-to-head", "flag", str(race_ahead), str(race_behind), color_a, color_b,
                      f"{race_ties} ties · {race_compared} comparable races", _share(race_ahead, race_behind))
    with overview_cols[1]:
        overview_card("Qualifying head-to-head", "qualifying", str(qual_ahead), str(qual_behind), color_a, color_b,
                      f"{qual_ties} ties · {qual_compared} comparable sessions", _share(qual_ahead, qual_behind))
    with overview_cols[2]:
        overview_card("Championship points", "points", points_display(summary_a["points"]), points_display(summary_b["points"]),
                      color_a, color_b, f"A − B = {points_gap:+,.1f} points", _share(summary_a["points"], summary_b["points"]))
    with overview_cols[3]:
        overview_card("Podiums", "medal", str(summary_a["podiums"]), str(summary_b["podiums"]), color_a, color_b,
                      f"A − B = {podium_gap:+d} podiums", _share(summary_a["podiums"], summary_b["podiums"]))

    pair_legend(full_a, color_a, full_b, color_b, dashed_b=same_team)

    # ---- radar + index breakdown ---------------------------------------------
    radar_col, breakdown_col = st.columns(2, gap="small")
    with radar_col:
        panel_head("Performance radar", "0–100 season-relative indices")
        render_plotly_chart(make_radar(indices_a, indices_b, full_a, full_b, color_a, color_b))

    with breakdown_col:
        panel_head("Index breakdown", "same indices, side by side")
        index_categories = list(indices_a.keys())
        values_a = [float(indices_a[k]) if indices_a.get(k) is not None and pd.notna(indices_a.get(k)) else None for k in index_categories]
        values_b = [float(indices_b[k]) if indices_b.get(k) is not None and pd.notna(indices_b.get(k)) else None for k in index_categories]
        index_bar_fig = go.Figure()
        for values, name, color, tag in ((values_a, full_a, color_a, "Driver A"), (values_b, full_b, color_b, "Driver B")):
            index_bar_fig.add_trace(go.Bar(
                y=index_categories, x=values, orientation="h", name=name,
                marker=dict(color=color, line=dict(color="#F1F4F9" if (same_team and tag == "Driver B") else color, width=1)),
                text=[f"{v:.0f}" if v is not None else "" for v in values],
                textposition="inside", insidetextanchor="end", textfont=dict(color="#FFFFFF", size=9),
                hovertemplate="<b>%{y}</b><br>Index: %{x:.1f}/100<extra>" + tag + "</extra>",
            ))
        plot_theme(index_bar_fig, height=310)
        index_bar_fig.update_layout(barmode="group", bargap=.22, bargroupgap=.08, hovermode="closest",
                                    margin=dict(l=118, r=14, t=14, b=34))
        index_bar_fig.update_xaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100])
        index_bar_fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        render_plotly_chart(index_bar_fig)

    # ---- race-by-race finishing positions (full width, named GPs) --------------
    race_meta = [{"round": safe_int(r.get("round")), "raceName": r.get("raceName", "Grand Prix")} for r in races if safe_int(r.get("round")) is not None]
    df_a = records_by_id[driver_a["driver_id"]]
    df_b = records_by_id[driver_b["driver_id"]]

    panel_head("Race-by-race finishing position", "official classification · lower is better · hover for details")
    track_fig = go.Figure()
    for df, name, color, marker, dash in [
        (df_a, full_a, color_a, "circle", "solid"),
        (df_b, full_b, color_b, "diamond", dash_b),
    ]:
        if df.empty:
            continue
        track_fig.add_trace(go.Scatter(
            x=df["round"], y=df["position"], mode="lines+markers", name=name, connectgaps=False,
            line=dict(color=color, width=2.6, dash=dash),
            marker=dict(color=color, size=7, symbol=marker, line=dict(color="#0b1018", width=1)),
            customdata=df[["event", "status", "grid", "points"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>Classified position: %{y}<br>Grid: %{customdata[2]}<br>Points: %{customdata[3]}<br>Status: %{customdata[1]}<extra>%{fullData.name}</extra>",
        ))
    plot_theme(track_fig, height=300)
    event_axis(track_fig, race_meta, names=True)
    track_fig.update_yaxes(title="Position", autorange="reversed", dtick=4)
    render_plotly_chart(track_fig)

    # ---- three compact charts in one row -----------------------------------------
    c_points, c_quali, c_gain = st.columns(3, gap="small")

    with c_points:
        panel_head("Cumulative points", "race points only (no sprints)")
        points_fig = go.Figure()
        for df, name, color, dash, marker in [
            (df_a, full_a, color_a, "solid", "circle"),
            (df_b, full_b, color_b, dash_b, "diamond"),
        ]:
            if df.empty:
                continue
            timeline = pd.DataFrame({"round": [x["round"] for x in race_meta]})
            merged = timeline.merge(df[["round", "points"]], on="round", how="left").fillna({"points": 0})
            merged["cumulative"] = merged["points"].cumsum()
            points_fig.add_trace(go.Scatter(
                x=merged["round"], y=merged["cumulative"], mode="lines+markers", name=name,
                line=dict(color=color, width=2.4, dash=dash), marker=dict(color=color, size=5, symbol=marker),
            ))
        plot_theme(points_fig, height=270)
        event_axis(points_fig, race_meta)
        points_fig.update_yaxes(title="Points")
        render_plotly_chart(points_fig)

    with c_quali:
        panel_head("Qualifying position", "by round · lower is better")
        quali_fig = go.Figure()
        for qdf, name, color, dash, marker in [
            (qframes[driver_a["driver_id"]], full_a, color_a, "solid", "circle"),
            (qframes[driver_b["driver_id"]], full_b, color_b, dash_b, "diamond"),
        ]:
            if not qdf.empty:
                quali_fig.add_trace(go.Scatter(
                    x=qdf["round"], y=qdf["position"], mode="lines+markers", name=name,
                    line=dict(color=color, width=2.2, dash=dash), marker=dict(color=color, size=6, symbol=marker),
                    customdata=qdf[["event"]].to_numpy(),
                    hovertemplate="<b>%{customdata[0]}</b><br>Qualifying P%{y}<extra>%{fullData.name}</extra>",
                ))
        if quali_fig.data:
            plot_theme(quali_fig, height=270)
            event_axis(quali_fig, race_meta)
            quali_fig.update_yaxes(title="Position", autorange="reversed", dtick=4)
            render_plotly_chart(quali_fig)
        else:
            st.info("Qualifying classifications are not available in the local cache or from Jolpica right now.")

    with c_gain:
        panel_head("Positions gained / lost", "grid minus finish")
        gain_fig = go.Figure()
        for df, name, color in [(df_a, full_a, color_a), (df_b, full_b, color_b)]:
            valid = df.dropna(subset=["grid", "position"]).copy()
            valid = valid[(valid["grid"] > 0) & (valid["position"] > 0)]
            if not valid.empty:
                valid["gain"] = valid["grid"] - valid["position"]
                gain_fig.add_trace(go.Bar(
                    x=valid["round"], y=valid["gain"], name=name, marker_color=color,
                    customdata=valid[["event", "grid", "position"]].to_numpy(),
                    hovertemplate="<b>%{customdata[0]}</b><br>Grid P%{customdata[1]} → Finish P%{customdata[2]}<br>Change: %{y:+} positions<extra>%{fullData.name}</extra>",
                ))
        if gain_fig.data:
            plot_theme(gain_fig, height=270)
            event_axis(gain_fig, race_meta)
            gain_fig.update_layout(barmode="group", bargap=.18)
            gain_fig.update_yaxes(title="Positions", zeroline=True, zerolinecolor="#AAB5C5")
            render_plotly_chart(gain_fig)
        else:
            st.info("Starting-grid and finishing-position records are not available for both drivers.")

    # ---- notes (collapsed) -----------------------------------------------------
    with st.expander("Round key, index methodology & data notes", expanded=False):
        st.markdown("**Round key** · " + " · ".join(
            f"`R{r['round']}` {short_event_name(r['raceName'])}" for r in race_meta))
        st.markdown("---")
        st.write(index_methodology_text())
        st.caption(
            "Indices are calculated against the selected season's full driver field and are not a prediction or an official F1 rating. "
            "WDC points use the official standings snapshot when available; progression charts use Grand Prix result points (sprints excluded). "
            "Missing sessions are not treated as zero. Grid penalties and pit-lane starts can affect position-change interpretation."
        )


with tab_circuit:
    with st.container(key="circuit_performance_shell"):
        render_circuit_performance_tab(season, driver_a, driver_b, color_a, color_b)
