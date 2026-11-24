"""
For1Sight - Main Overview Dashboard (redesigned)

Sections
  1. Hero header + data-source status
  2. Next Grand Prix card (live countdown, weekend schedule) + latest race podium
  3. KPI strip (season + model headline numbers)
  4. Tabs
       - Championship : drivers, constructors, title-fight progression
       - Model Report : 2026 final-test results (models vs. grid baseline)
       - Project      : model card, pipeline, page navigation
Everything fails soft: if an API or a local file is missing, that block shows a
friendly note instead of crashing the page.
"""

import base64
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="For1Sight | Overview",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
API = "https://api.jolpi.ca/ergast/f1"
DISPLAY_TZ = ZoneInfo("UTC")  # change to e.g. ZoneInfo("Asia/Kuala_Lumpur") for local times
TZ_LABEL = "UTC"

# ======================================================================
# 1. THEME
# ======================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:ital,wght@0,400;0,600;0,700;0,900;1,700&display=swap');

:root{
  --red:#E10600; --blue:#55B8F2; --gold:#FFD166; --green:#2ECC71;
  --panel:rgba(18,22,31,.86); --panel2:rgba(26,32,45,.92);
  --line:rgba(148,163,184,.18); --muted:#9AA8BA; --text:#EEF2F8;
}
html, body, [class*="css"], .stMarkdown, .stTabs { font-family:'Titillium Web',sans-serif; }
.stApp{
  background:
    radial-gradient(ellipse at 0% 0%, rgba(225,6,0,.22), transparent 38%),
    radial-gradient(ellipse at 100% 8%, rgba(85,184,242,.09), transparent 34%),
    linear-gradient(160deg,#050507 0%,#0B0D14 55%,#150508 100%);
  background-attachment:fixed;
}
[data-testid="stHeader"]{background:transparent !important;}
.block-container{max-width:1480px; padding-top:1.4rem; padding-bottom:3rem;}
[data-testid="stSidebar"]{background:rgba(10,12,18,.96) !important; border-right:1px solid var(--line) !important;}
[data-testid="stSidebarNav"] span{font-size:1.02rem !important; color:#fff !important;}

/* ---------- hero ---------- */
.hero{display:flex; align-items:center; justify-content:space-between; gap:1.2rem;
  padding:1.1rem 1.5rem; border:1px solid var(--line); border-left:5px solid var(--red);
  border-radius:16px; background:linear-gradient(110deg,rgba(225,6,0,.16),rgba(18,22,31,.92) 55%);
  margin-bottom:1.1rem;}
.hero-left{display:flex; align-items:center; gap:1.1rem;}
.hero-logo{height:38px; filter:brightness(0) invert(1);}
.hero-title{font-style:italic; font-weight:900; font-size:2.5rem; line-height:1; letter-spacing:2px; text-transform:uppercase; color:#fff;}
.hero-title span{color:var(--red);}
.hero-sub{color:var(--muted); font-size:.9rem; letter-spacing:.14em; text-transform:uppercase; margin-top:.25rem;}
.pills{display:flex; gap:.5rem; flex-wrap:wrap; justify-content:flex-end;}
.pill{display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem; border-radius:999px;
  font-size:.74rem; font-weight:700; letter-spacing:.06em; border:1px solid var(--line); color:#cfd8e6; background:rgba(255,255,255,.04);}
.dot{width:8px; height:8px; border-radius:50%; display:inline-block;}
.dot.ok{background:var(--green); box-shadow:0 0 8px var(--green);}
.dot.bad{background:#F36C79;}
.dot.warn{background:var(--gold);}

/* ---------- cards ---------- */
.card{background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.3rem; height:100%;}
.card-title{font-size:.72rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; color:#ff7772; margin-bottom:.5rem;}
.race-name{font-size:2rem; font-weight:900; line-height:1.1; color:#fff;}
.race-meta{color:var(--muted); font-size:.95rem; margin-top:.3rem;}
.sess-row{display:flex; justify-content:space-between; padding:.38rem 0; border-bottom:1px dashed rgba(148,163,184,.14); font-size:.88rem;}
.sess-row:last-child{border-bottom:none;}
.sess-row b{color:#fff; font-weight:600;} .sess-row span{color:var(--muted);}
.sess-row.race b,.sess-row.race span{color:var(--gold); font-weight:700;}
.progress{height:7px; border-radius:6px; background:rgba(255,255,255,.08); overflow:hidden; margin-top:.8rem;}
.progress>div{height:100%; background:linear-gradient(90deg,var(--red),#ff6a5c);}
.podium-row{display:flex; align-items:center; gap:.8rem; padding:.55rem .2rem; border-bottom:1px solid rgba(148,163,184,.12);}
.podium-row:last-child{border-bottom:none;}
.pos{width:34px; height:34px; border-radius:9px; display:flex; align-items:center; justify-content:center; font-weight:900; color:#0b0d14;}
.p1{background:#FFD166;} .p2{background:#C9D1DC;} .p3{background:#D08A52;}
.podium-name{font-weight:700; color:#fff; font-size:1.02rem;}
.podium-team{font-size:.78rem; color:var(--muted);}
.podium-time{margin-left:auto; font-size:.84rem; color:#cfd8e6; font-variant-numeric:tabular-nums;}
.bar-team{width:4px; align-self:stretch; border-radius:3px;}

/* ---------- KPI ---------- */
.kpi{background:var(--panel2); border:1px solid var(--line); border-radius:14px; padding:.85rem 1.05rem; position:relative; overflow:hidden;}
.kpi::before{content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent,#E10600);}
.kpi-label{font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:700;}
.kpi-value{font-size:1.75rem; font-weight:900; color:#fff; line-height:1.15; margin-top:.15rem;}
.kpi-sub{font-size:.8rem; color:#b8c3d3; margin-top:.1rem;}

/* ---------- section heads ---------- */
.sec{display:flex; align-items:center; gap:.7rem; margin:1.3rem 0 .6rem;}
.sec::before{content:""; width:4px; height:22px; background:var(--red); border-radius:2px;}
.sec h3{margin:0; font-size:1.25rem; font-weight:700; color:#fff;}
.sec small{color:var(--muted); margin-left:.2rem;}
.callout{border:1px solid var(--line); border-left:4px solid var(--gold); background:rgba(255,209,102,.07);
  border-radius:10px; padding:.75rem 1rem; color:#e6dcc2; font-size:.92rem; margin:.4rem 0 .9rem;}
.callout.info{border-left-color:var(--blue); background:rgba(85,184,242,.07); color:#cfe6f6;}
.chip{display:inline-block; padding:.2rem .6rem; margin:.15rem .2rem .15rem 0; border-radius:8px; font-size:.78rem;
  background:rgba(85,184,242,.12); border:1px solid rgba(85,184,242,.30); color:#bfe3fa; font-family:monospace;}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{gap:.4rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:44px; padding:0 1.1rem; border-radius:10px 10px 0 0; color:var(--muted); font-weight:700; letter-spacing:.04em;}
.stTabs [aria-selected="true"]{color:#fff !important; background:rgba(225,6,0,.14);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--red) !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ======================================================================
# 2. HELPERS
# ======================================================================
def b64_image(path: Path) -> str:
    if path.exists():
        ext = path.suffix.lower().lstrip(".")
        mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    return ""


LOGO_SRC = b64_image(ROOT / "assets" / "icons" / "f1_logo.png") or \
    "https://upload.wikimedia.org/wikipedia/commons/3/33/F1_logo.svg"


def team_color(name: str) -> str:
    t = str(name).lower()
    table = [
        ("red bull", "#3671C6"), ("ferrari", "#E8002D"), ("mclaren", "#FF8000"),
        ("mercedes", "#27F4D2"), ("aston", "#229971"), ("alpine", "#0093CC"),
        ("williams", "#64C4FF"), ("haas", "#B6BABD"), ("cadillac", "#C9CED6"),
        ("audi", "#E6E6E6"), ("sauber", "#52E252"), ("racing bulls", "#6692FF"),
        ("alphatauri", "#6692FF"),
    ]
    for key, color in table:
        if key in t:
            return color
    if re.search(r"\brb\b", t):
        return "#6692FF"
    return "#888888"


def esc(x) -> str:
    return html.escape(str(x))


def style_fig(fig, height=420, title=None):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=16, color="#fff"), x=0.01) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,24,33,.35)",
        font=dict(color="#E8EDF5", family="Titillium Web, sans-serif"),
        margin=dict(l=10, r=20, t=50 if title else 20, b=10),
        hoverlabel=dict(bgcolor="#1B2230", font_color="#fff"),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2A3242", zeroline=False)
    fig.update_yaxes(gridcolor="#2A3242", zeroline=False)
    return fig


def section(title: str, sub: str = ""):
    st.markdown(f"<div class='sec'><h3>{esc(title)}</h3><small>{esc(sub)}</small></div>",
                unsafe_allow_html=True)


def kpi(label, value, sub="", accent="#E10600"):
    return (f"<div class='kpi' style='--accent:{accent}'><div class='kpi-label'>{esc(label)}</div>"
            f"<div class='kpi-value'>{esc(value)}</div><div class='kpi-sub'>{esc(sub)}</div></div>")


def parse_dt(date_str, time_str):
    if not date_str:
        return None
    t = (time_str or "00:00:00Z").replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(f"{date_str}T{t}")
    except ValueError:
        return None


def fmt_dt(dt):
    return dt.astimezone(DISPLAY_TZ).strftime("%a %d %b · %H:%M") if dt else "TBC"


# ======================================================================
# 3. DATA LAYER (live API, cached) - all fail-soft
# ======================================================================
def _get(url, params=None):
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    return r.json()["MRData"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_schedule():
    try:
        return _get(f"{API}/current.json", {"limit": 100})["RaceTable"]["Races"]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_driver_standings():
    try:
        return _get(f"{API}/current/driverStandings.json")["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_constructor_standings():
    try:
        return _get(f"{API}/current/constructorStandings.json")["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_last_race():
    try:
        races = _get(f"{API}/current/last/results.json")["RaceTable"]["Races"]
        return races[0] if races else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_progression(season: int, top_codes: tuple):
    """Cumulative race points per round for selected drivers.
    Uses the project's offline-first results cache from for1sight_shared if available."""
    try:
        from for1sight_shared import fetch_api_results
        races = fetch_api_results(season)
    except Exception:
        return pd.DataFrame()
    rows = []
    for race in races:
        for res in race.get("Results", []):
            code = res.get("Driver", {}).get("code")
            if code in top_codes:
                rows.append({"round": int(race["round"]), "race": race.get("raceName", ""),
                             "code": code, "points": float(res.get("points") or 0),
                             "team": res["Constructor"]["name"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["code", "round"])
    df["cum"] = df.groupby("code")["points"].cumsum()
    return df


# ---------- local model artefacts ----------
def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_model_report():
    base = _first_existing([ROOT / "race_winner_v2_7", ROOT / "models" / "race_winner_v2_7"])
    out = {"ok": False}
    if not base:
        return out
    stage4 = sorted(base.glob("stage4_final_test_2026_*"))
    if not stage4:
        return out
    d = stage4[-1]
    try:
        out["comparison"] = pd.read_csv(d / "model_comparison_final_test_2026.csv")
        out["events"] = {
            "Logistic Regression": pd.read_csv(d / "logistic_regression_event_predictions_2026.csv"),
            "Random Forest": pd.read_csv(d / "random_forest_event_predictions_2026.csv"),
            "Gradient Boosting": pd.read_csv(d / "gradient_boosting_event_predictions_2026.csv"),
        }
        out["baseline"] = pd.read_csv(d / "starting_grid_baseline_2026.csv")
        mf = d / "run_manifest.json"
        out["manifest"] = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
        out["ok"] = True
    except Exception:
        out["ok"] = False
    return out


@st.cache_data(show_spinner=False)
def load_model_metadata():
    p = _first_existing([ROOT / "v2_8" / "model_metadata.json", ROOT / "models" / "v2_8" / "model_metadata.json"])
    if not p:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ======================================================================
# 4. LOAD
# ======================================================================
with st.sidebar:
    st.markdown("### ⚙️ Dashboard")
    if st.button("🔄 Refresh live data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Times shown in {TZ_LABEL}")
    st.caption("Live data: Jolpica (Ergast-compatible) API, cached for 1 hour.")

schedule = get_schedule()
drivers_raw = get_driver_standings()
constructors_raw = get_constructor_standings()
last_race = get_last_race()
report = load_model_report()
meta = load_model_metadata()

now = datetime.now(timezone.utc)
season = int(schedule[0]["season"]) if schedule else now.year

next_race, completed = None, 0
for race in schedule:
    dt = parse_dt(race.get("date"), race.get("time"))
    if dt and dt >= now:
        next_race = race
        break
    completed += 1

df_d = pd.DataFrame([{
    "pos": int(d["position"]), "code": d["Driver"].get("code", ""),
    "name": f"{d['Driver']['givenName']} {d['Driver']['familyName']}",
    "team": d["Constructors"][-1]["name"] if d.get("Constructors") else "",
    "points": float(d["points"]), "wins": int(d["wins"]),
} for d in drivers_raw]) if drivers_raw else pd.DataFrame()

df_c = pd.DataFrame([{
    "pos": int(c["position"]), "team": c["Constructor"]["name"],
    "points": float(c["points"]), "wins": int(c["wins"]),
} for c in constructors_raw]) if constructors_raw else pd.DataFrame()

# ======================================================================
# 5. HERO
# ======================================================================
live_ok = bool(schedule and drivers_raw)
model_ok = report.get("ok", False)
pills = [
    f"<span class='pill'><span class='dot {'ok' if live_ok else 'bad'}'></span>Live API {'online' if live_ok else 'unreachable'}</span>",
    f"<span class='pill'><span class='dot {'ok' if model_ok else 'bad'}'></span>Model artefacts {'loaded' if model_ok else 'missing'}</span>",
    f"<span class='pill'>Model {esc(meta.get('version', 'n/a'))}</span>",
    f"<span class='pill'>Season {season}</span>",
]
st.markdown(f"""
<div class='hero'>
  <div class='hero-left'>
    <img class='hero-logo' src='{LOGO_SRC}'>
    <div>
      <div class='hero-title'>For<span>1</span>Sight</div>
      <div class='hero-sub'>Formula 1 analytics &amp; race-winner prediction · Executive overview</div>
    </div>
  </div>
  <div class='pills'>{''.join(pills)}</div>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# 6. NEXT RACE + LAST RACE
# ======================================================================
left, mid, right = st.columns([1.45, 1, 1.2], gap="medium")

SESSION_KEYS = [("FirstPractice", "Practice 1"), ("SecondPractice", "Practice 2"),
                ("ThirdPractice", "Practice 3"), ("SprintQualifying", "Sprint Qualifying"),
                ("Sprint", "Sprint"), ("Qualifying", "Qualifying")]

with left:
    if next_race:
        circuit = next_race["Circuit"]
        loc = circuit["Location"]
        race_dt = parse_dt(next_race.get("date"), next_race.get("time"))
        pct = 100 * completed / max(len(schedule), 1)
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>📍 Next Grand Prix · Round {esc(next_race['round'])} of {len(schedule)}</div>
          <div class='race-name'>{esc(next_race['raceName'])}</div>
          <div class='race-meta'>{esc(circuit['circuitName'])} · {esc(loc['locality'])}, {esc(loc['country'])}</div>
          <div class='progress'><div style='width:{pct:.0f}%'></div></div>
          <div class='race-meta' style='font-size:.8rem'>Season progress: {completed} of {len(schedule)} rounds completed</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='card'>
          <div class='card-title'>🏁 Season status</div>
          <div class='race-name'>Season complete</div>
          <div class='race-meta'>All rounds have been run. Final standings are shown below.</div>
        </div>""", unsafe_allow_html=True)

with mid:
    if next_race:
        rows = []
        for key, label in SESSION_KEYS:
            s = next_race.get(key)
            if s:
                rows.append(f"<div class='sess-row'><b>{label}</b><span>{fmt_dt(parse_dt(s.get('date'), s.get('time')))}</span></div>")
        rows.append(f"<div class='sess-row race'><b>Race</b><span>{fmt_dt(race_dt)}</span></div>")
        st.markdown(f"<div class='card'><div class='card-title'>🗓️ Weekend schedule</div>{''.join(rows)}</div>",
                    unsafe_allow_html=True)
        if race_dt:
            tiles = "".join(
                "<div style='flex:1;background:#171d28;border:1px solid #2a3242;border-radius:12px;"
                "padding:8px 0;text-align:center'><div id='" + i + "' style='font-size:26px;font-weight:900'>--</div>"
                "<div style='font-size:10px;letter-spacing:.14em;color:#9AA8BA'>" + lab + "</div></div>"
                for i, lab in [("d", "DAYS"), ("h", "HRS"), ("m", "MIN"), ("s", "SEC")]
            )
            target_iso = race_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            countdown_html = (
                "<div style=\"font-family:'Titillium Web',Arial,sans-serif;color:#fff;display:flex;gap:10px;margin-top:2px\">"
                + tiles + "</div><script>"
                "const target = new Date('" + target_iso + "').getTime();"
                "function tick(){let diff=Math.max(0,target-Date.now())/1000;"
                "const v={d:Math.floor(diff/86400),h:Math.floor(diff%86400/3600),m:Math.floor(diff%3600/60),s:Math.floor(diff%60)};"
                "for(const k in v){document.getElementById(k).textContent=String(v[k]).padStart(2,'0');}}"
                "tick();setInterval(tick,1000);</script>"
            )
            components.html(countdown_html, height=78)
    else:
        st.markdown("<div class='card'><div class='card-title'>🗓️ Weekend schedule</div>"
                    "<div class='race-meta'>No upcoming race on the calendar.</div></div>", unsafe_allow_html=True)

with right:
    if last_race and last_race.get("Results"):
        rows = []
        for res in last_race["Results"][:3]:
            drv = res["Driver"]
            team = res["Constructor"]["name"]
            t = res.get("Time", {}).get("time") or res.get("status", "")
            rows.append(
                f"<div class='podium-row'><div class='pos p{res['position']}'>{esc(res['position'])}</div>"
                f"<div class='bar-team' style='background:{team_color(team)}'></div>"
                f"<div><div class='podium-name'>{esc(drv['givenName'])} {esc(drv['familyName'])}</div>"
                f"<div class='podium-team'>{esc(team)} · from P{esc(res.get('grid', '-'))}</div></div>"
                f"<div class='podium-time'>{esc(t)}</div></div>")
        st.markdown(f"<div class='card'><div class='card-title'>🏆 Latest result · {esc(last_race['raceName'])}</div>{''.join(rows)}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'><div class='card-title'>🏆 Latest result</div>"
                    "<div class='race-meta'>Race results unavailable.</div></div>", unsafe_allow_html=True)

# ======================================================================
# 7. KPI STRIP
# ======================================================================
st.write("")
k = st.columns(6, gap="small")

if not df_d.empty:
    lead = df_d.iloc[0]
    gap = lead.points - df_d.iloc[1].points if len(df_d) > 1 else 0
    k[0].markdown(kpi("Drivers' leader", lead["name"].split()[-1], lead["team"], team_color(lead["team"])), unsafe_allow_html=True)
    k[1].markdown(kpi("Leader points", f"{lead.points:g}", f"{int(lead.wins)} wins", team_color(lead["team"])), unsafe_allow_html=True)
    k[2].markdown(kpi("Lead over P2", f"{gap:g} pts", df_d.iloc[1]["name"].split()[-1] if len(df_d) > 1 else "", "#FFD166"), unsafe_allow_html=True)
else:
    for i, lab in enumerate(["Drivers' leader", "Leader points", "Lead over P2"]):
        k[i].markdown(kpi(lab, "—", "standings unavailable", "#555"), unsafe_allow_html=True)

if not df_c.empty:
    c0 = df_c.iloc[0]
    k[3].markdown(kpi("Constructors' leader", c0["team"].replace(" F1 Team", ""), f"{c0.points:g} pts", team_color(c0["team"])), unsafe_allow_html=True)
else:
    k[3].markdown(kpi("Constructors' leader", "—", "standings unavailable", "#555"), unsafe_allow_html=True)

k[4].markdown(kpi("Rounds completed", f"{completed}/{len(schedule) or '—'}", "current season", "#55B8F2"), unsafe_allow_html=True)

if model_ok:
    comp = report["comparison"]
    models_only = comp[~comp["Model"].str.contains("baseline", case=False)]
    best = models_only.sort_values("Top-1 accuracy (%)", ascending=False).iloc[0]
    k[5].markdown(kpi("Best model Top-1", f"{best['Top-1 accuracy (%)']:.1f}%", f"{best['Model']} · {int(best['Events evaluated'])} races", "#2ECC71"), unsafe_allow_html=True)
else:
    k[5].markdown(kpi("Best model Top-1", "—", "model report not found", "#555"), unsafe_allow_html=True)

# ======================================================================
# 8. TABS
# ======================================================================
st.write("")
tab_champ, tab_model, tab_proj = st.tabs(["🏆 Championship", "🤖 Model Report", "🧭 Project"])

# ---------------------------------------------------------------- tab 1
with tab_champ:
    if df_d.empty and df_c.empty:
        st.info("Championship data is unavailable right now. Check your connection and use “Refresh live data”.")
    else:
        c1, c2 = st.columns([1.25, 1], gap="large")

        with c1:
            section("Drivers' Championship", "top 10")
            if not df_d.empty:
                top = df_d.head(10).iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=top["points"], y=top["name"], orientation="h",
                    marker=dict(color=[team_color(t) for t in top["team"]]),
                    text=[f"{p:g}" for p in top["points"]], textposition="outside",
                    customdata=top[["team", "wins", "pos"]].to_numpy(),
                    hovertemplate="<b>%{y}</b><br>P%{customdata[2]} · %{customdata[0]}<br>%{x:g} pts · %{customdata[1]} wins<extra></extra>",
                ))
                style_fig(fig, 460)
                fig.update_xaxes(range=[0, top["points"].max() * 1.15])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

        with c2:
            section("Constructors' Championship")
            if not df_c.empty:
                tc = df_c.iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=tc["points"], y=tc["team"], orientation="h",
                    marker=dict(color=[team_color(t) for t in tc["team"]]),
                    text=[f"{p:g}" for p in tc["points"]], textposition="outside",
                    hovertemplate="<b>%{y}</b><br>%{x:g} pts<extra></extra>",
                ))
                style_fig(fig, 460)
                fig.update_xaxes(range=[0, tc["points"].max() * 1.15])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

        if not df_d.empty:
            section("Title fight", "cumulative race points · top 5 drivers")
            codes = tuple(df_d.head(5)["code"])
            prog = get_progression(season, codes)
            if prog.empty:
                st.info("Points progression needs the season results cache from `for1sight_shared.fetch_api_results`.")
            else:
                fig = go.Figure()
                for code, g in prog.groupby("code"):
                    fig.add_trace(go.Scatter(
                        x=g["round"], y=g["cum"], mode="lines+markers", name=code,
                        line=dict(color=team_color(g["team"].iloc[-1]), width=3),
                        marker=dict(size=6),
                        customdata=g[["race"]].to_numpy(),
                        hovertemplate="<b>" + code + "</b> · %{customdata[0]}<br>Round %{x} · %{y:g} pts<extra></extra>",
                    ))
                style_fig(fig, 380)
                fig.update_xaxes(title="Round", dtick=1)
                fig.update_yaxes(title="Cumulative points")
                st.plotly_chart(fig, width="stretch")
                st.caption("Race-result points only; sprint points are not included, so totals can sit slightly below the official standings.")

# ---------------------------------------------------------------- tab 2
with tab_model:
    if not model_ok:
        st.warning("Model report not found. Expected `race_winner_v2_7/stage4_final_test_2026_*/` next to dashboard.py "
                   "(or under `models/`).")
    else:
        comp = report["comparison"].copy()
        events = report["events"]
        base = report["baseline"]
        manifest = report["manifest"]

        n_events = int(comp["Events evaluated"].iloc[0])
        base_row = comp[comp["Model"].str.contains("baseline", case=False)].iloc[0]
        mdl = comp[~comp["Model"].str.contains("baseline", case=False)]
        best = mdl.sort_values(["Top-1 accuracy (%)", "Winner log loss (lower better)"],
                               ascending=[False, True]).iloc[0]

        section("2026 final test", f"{n_events} unseen races · models trained on 2024 only")

        # honest framing - generated from the numbers, not hard-coded
        diff = best["Top-1 accuracy (%)"] - base_row["Top-1 accuracy (%)"]
        if diff < 0:
            st.markdown(
                f"<div class='callout'><b>Reading the results honestly:</b> the simple “pole-sitter wins” baseline "
                f"scored {base_row['Top-1 accuracy (%)']:.1f}% Top-1, ahead of the best model "
                f"({best['Model']}, {best['Top-1 accuracy (%)']:.1f}%). The models' value is in calibrated "
                f"<i>probabilities</i> (log loss, Brier) rather than beating the grid on a single pick. "
                f"Calibration verified: <b>{'yes' if meta.get('calibration_verified') else 'no'}</b>.</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='callout info'><b>{esc(best['Model'])}</b> beats the pole-sitter baseline by "
                f"{diff:+.1f} percentage points Top-1 over {n_events} unseen races.</div>", unsafe_allow_html=True)

        m1, m2 = st.columns([1.1, 1], gap="large")
        with m1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Top-1 accuracy", x=comp["Model"], y=comp["Top-1 accuracy (%)"],
                                 marker_color="#E10600", text=comp["Top-1 accuracy (%)"].round(1), textposition="outside"))
            fig.add_trace(go.Bar(name="Top-3 accuracy", x=comp["Model"], y=comp["Top-3 accuracy (%)"],
                                 marker_color="#55B8F2", text=comp["Top-3 accuracy (%)"].round(1), textposition="outside"))
            style_fig(fig, 400, "Winner-pick accuracy (%)")
            fig.update_layout(barmode="group", bargap=0.25)
            fig.update_yaxes(range=[0, 110])
            st.plotly_chart(fig, width="stretch")

        with m2:
            prob_fig = go.Figure()
            palette = {"Logistic Regression": "#9AA8BA", "Random Forest": "#FFD166", "Gradient Boosting": "#E10600"}
            for name, df in events.items():
                short = df["event_name"].str.replace(" Grand Prix", "", regex=False)
                prob_fig.add_trace(go.Scatter(
                    x=short, y=df["winner_probability"], mode="lines+markers", name=name,
                    line=dict(color=palette[name], width=2.4), marker=dict(size=6),
                    hovertemplate="<b>%{x}</b><br>P(actual winner) = %{y:.0%}<extra>" + name + "</extra>"))
            style_fig(prob_fig, 400, "Probability given to the actual winner")
            prob_fig.update_yaxes(tickformat=".0%", range=[0, 1])
            prob_fig.update_xaxes(tickangle=-45)
            st.plotly_chart(prob_fig, width="stretch")

        section("Race-by-race scorecard", "green = winner picked correctly")
        hm_models = dict(events)
        hm_models["Pole-sitter baseline"] = base
        ref = events["Gradient Boosting"]
        order = list(ref["event_id"])
        z, hover = [], []
        for name, df in hm_models.items():
            d = df.set_index("event_id")
            z.append([1 if (e in d.index and bool(d.loc[e, "top1_correct"])) else 0 for e in order])
            hover.append([
                f"{name}<br>Predicted {d.loc[e, 'predicted_winner']} · Actual {d.loc[e, 'actual_winner']}"
                if e in d.index else "n/a" for e in order])
        fig = go.Figure(go.Heatmap(
            z=z, x=[n.replace(" Grand Prix", "") for n in ref["event_name"]], y=list(hm_models.keys()),
            colorscale=[[0, "#5A1620"], [1, "#1F8F55"]], showscale=False, xgap=3, ygap=3,
            text=hover, hovertemplate="%{text}<extra></extra>"))
        style_fig(fig, 270)
        fig.update_xaxes(tickangle=-45, showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        st.plotly_chart(fig, width="stretch")

        section("Probabilistic quality", "lower log loss / Brier is better")
        table = mdl[["Model", "Winner log loss (lower better)", "Multiclass Brier (lower better)",
                     "Mean reciprocal rank", "Driver-row ROC-AUC"]].copy()
        table.columns = ["Model", "Log loss", "Brier", "MRR", "ROC-AUC"]
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in table.columns[1:]})
        if manifest:
            st.caption(f"Run {manifest.get('run_id', '')} · train {manifest.get('training_season')} → "
                       f"validate {manifest.get('validation_season')} → test {manifest.get('final_test_season')} · "
                       f"test used for model selection: {manifest.get('test_used_for_model_selection')}")

# ---------------------------------------------------------------- tab 3
with tab_proj:
    p1, p2 = st.columns([1.2, 1], gap="large")
    with p1:
        section("Deployed model card", meta.get("version", ""))
        if meta:
            a, b, c = st.columns(3)
            a.markdown(kpi("Training races", meta.get("training_races", "—"), f"{meta.get('training_records', '—')} driver rows", "#55B8F2"), unsafe_allow_html=True)
            b.markdown(kpi("Features", meta.get("feature_count", "—"), "pre-race only", "#55B8F2"), unsafe_allow_html=True)
            seasons = meta.get("deployment_training_seasons", [])
            c.markdown(kpi("Trained on", "–".join(str(s) for s in (seasons[0], seasons[-1])) if seasons else "—", "deployment seasons", "#55B8F2"), unsafe_allow_html=True)
            st.write("")
            st.markdown("**Input features**")
            st.markdown("".join(f"<span class='chip'>{esc(f)}</span>" for f in meta.get("features", [])), unsafe_allow_html=True)
            st.markdown("**Algorithms**")
            st.markdown("".join(f"<span class='chip'>{esc(a)}</span>" for a in meta.get("algorithms", {})), unsafe_allow_html=True)
            st.caption(f"Target: {meta.get('target_description', '')} · Preprocessing: "
                       f"{meta.get('preprocessing', {}).get('numeric_scaling', '')} + "
                       f"{meta.get('preprocessing', {}).get('categorical_encoding', '')} · "
                       f"Calibration verified: {meta.get('calibration_verified')}")
        else:
            st.info("`v2_8/model_metadata.json` not found.")

    with p2:
        section("Explore the app")
        pages = [
            ("pages/1_Historical_Performance.py", "Historical Performance", "Past race results & weather", "🏁"),
            ("pages/2_Drives_&_Teams.py", "Drivers & Teams", "Season profiles, form and tyres", "🧑‍✈️"),
            ("pages/3_Driver Comparison.py", "Driver Comparison", "Head-to-head and telemetry", "⚔️"),
            ("pages/4_Prediction_Analysis.py", "Prediction Analysis", "Race-winner probabilities", "🔮"),
        ]
        for path, label, desc, icon in pages:
            try:
                st.page_link(path, label=f"{label} — {desc}", icon=icon, use_container_width=True)
            except Exception:
                st.markdown(f"{icon} **{label}** — {desc}")

        st.markdown("<div class='callout info' style='margin-top:1rem'>Pipeline: Jolpica / OpenF1 / FastF1 → "
                    "warehouse (Parquet, CSV, JSON) → feature engineering → "
                    "Logistic Regression · Random Forest · Gradient Boosting → this dashboard.</div>",
                    unsafe_allow_html=True)

# ======================================================================
# 9. FOOTER
# ======================================================================
st.markdown(
    f"<div style='text-align:center;color:#6f7c90;font-size:.78rem;margin-top:2rem'>"
    f"For1Sight · Final Year Project · Data: Jolpica, OpenF1, FastF1 · Rendered {now.strftime('%d %b %Y %H:%M')} UTC</div>",
    unsafe_allow_html=True)
