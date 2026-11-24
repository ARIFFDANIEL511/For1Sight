"""Interactive Plotly circuit replay for For1Sight (V2.4.6).

The replay aligns two drivers by normalised lap progress. It is a spatial
comparison, not a session-clock-accurate race replay.

V2.4.3 improvements:
- compact 720 px replay canvas designed for typical 768 px laptop browser viewports;
- resized and repositioned scorecards so the complete replay remains balanced at the shorter height;
- compact top layout with a larger vertical circuit viewport;
- taller replay canvas and tighter margins to use more of the Streamlit page;
- animated driver scorecards arranged beside the map in a compact 2-column by 5-row grid;
- preserves discrete gear and brake states with stepwise interpolation;
- aggregates duplicate progress samples without averaging discrete channels;
- aspect-aware centred circuit viewport with equal X/Y scale;
- uses a white GPS-derived reference centreline with separation markers overlaid;
- matched-progress GPS separation connectors and a separation-coloured trace;
- live per-driver telemetry scorecards for speed, throttle, brake, gear and RPM;
- dynamic GPS-gap and speed-difference status strip;
- less cluttered first-sample markers and separated driver labels;
- compact top band for title and playback controls;
- compact legend above the circuit and progress slider directly below the circuit;
- reference-inspired dark scorecards with red icons, large values and small labels;
- scorecard values animate with the GPS replay in the left-hand panel.

GPS x/y coordinates in the For1Sight Silver warehouse are in metres. No GPS
coordinates are artificially shifted or converted. If the two paths overlap,
the map reports that faithfully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPLAY_CHANNELS = ("x", "y", "speed_kmh", "throttle_pct", "brake", "gear", "rpm")
DISCRETE_CHANNELS = frozenset({"brake", "gear", "drs"})

# -----------------------------------------------------------------------------
# Replay layout controls — adjust these values to resize the visualisation.
# -----------------------------------------------------------------------------
REPLAY_HEIGHT = 720                  # Compact laptop-friendly chart height, in pixels.
REPLAY_TOP_MARGIN = 45               # Tight top margin; title/buttons remain visible.
REPLAY_HEADER_Y = 1.00               # Vertical position of title and Play/Pause controls.
TRACK_DOMAIN_FACTOR = 0.59           # Higher = more horizontal room for the circuit.
TRACK_PADDING_RATIO = 0.035          # Lower = less empty space around the GPS trace.
TRACK_PADDING_MIN_METRES = 30.0      # Minimum GPS viewport padding, in metres.
TRACK_DOMAIN_MIN = 0.30
TRACK_DOMAIN_MAX = 0.92


def _last_valid_value(values: pd.Series) -> float:
    """Return the last recorded non-null state for duplicate progress values."""
    valid = values.dropna()
    return float(valid.iloc[-1]) if not valid.empty else np.nan


def _clean_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and aggregate a lap profile without averaging discrete states."""
    required = {"progress_pct", "x", "y"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Replay data is missing required fields: " + ", ".join(missing))

    columns = [c for c in ("progress_pct", *REPLAY_CHANNELS) if c in frame.columns]
    profile = frame[columns].copy()
    for column in columns:
        profile[column] = pd.to_numeric(profile[column], errors="coerce")
    profile = profile.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["progress_pct", "x", "y"]
    )
    if profile.empty:
        raise ValueError("No valid GPS positions are available for the selected driver.")

    aggregate_columns = [c for c in REPLAY_CHANNELS if c in profile.columns]
    # Stable sort retains original sample order when progress values tie.
    profile = profile.sort_values("progress_pct", kind="mergesort")
    aggregations = {
        column: (_last_valid_value if column in DISCRETE_CHANNELS else "mean")
        for column in aggregate_columns
    }
    profile = (
        profile.groupby("progress_pct", as_index=False, sort=True)
        .agg(aggregations)
        .sort_values("progress_pct")
        .reset_index(drop=True)
    )
    if len(profile) < 10 or profile["progress_pct"].nunique() < 10:
        raise ValueError("At least 10 distinct GPS positions are required for a replay.")

    # These are discrete source channels; do not expose fractional states.
    if "gear" in profile.columns:
        profile["gear"] = profile["gear"].round()
    if "brake" in profile.columns:
        profile["brake"] = profile["brake"].round()
    return profile


def _interpolate_continuous(profile: pd.DataFrame, column: str, grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate continuous channels inside the recorded range."""
    if column not in profile.columns:
        return np.full(len(grid), np.nan, dtype=float)
    valid = profile[["progress_pct", column]].dropna().sort_values("progress_pct")
    if len(valid) < 2:
        return np.full(len(grid), np.nan, dtype=float)
    xp = valid["progress_pct"].to_numpy(dtype=float)
    fp = valid[column].to_numpy(dtype=float)
    unique = np.concatenate(([True], np.diff(xp) > 1e-9))
    xp, fp = xp[unique], fp[unique]
    if len(xp) < 2:
        return np.full(len(grid), np.nan, dtype=float)
    result = np.interp(grid, xp, fp)
    result[(grid < xp.min()) | (grid > xp.max())] = np.nan
    return result


def _interpolate_stepwise(profile: pd.DataFrame, column: str, grid: np.ndarray) -> np.ndarray:
    """Use sample-and-hold interpolation for gear, brake and other states."""
    if column not in profile.columns:
        return np.full(len(grid), np.nan, dtype=float)
    valid = profile[["progress_pct", column]].dropna().sort_values("progress_pct")
    if len(valid) < 2:
        return np.full(len(grid), np.nan, dtype=float)
    xp = valid["progress_pct"].to_numpy(dtype=float)
    fp = valid[column].to_numpy(dtype=float)
    unique = np.concatenate(([True], np.diff(xp) > 1e-9))
    xp, fp = xp[unique], fp[unique]
    if len(xp) < 2:
        return np.full(len(grid), np.nan, dtype=float)

    # Use the most recent recorded state at or before each replay position.
    indices = np.searchsorted(xp, grid, side="right") - 1
    indices = np.clip(indices, 0, len(fp) - 1)
    result = fp[indices].astype(float, copy=True)
    result[(grid < xp.min()) | (grid > xp.max())] = np.nan
    return result


def _interpolate(profile: pd.DataFrame, column: str, grid: np.ndarray) -> np.ndarray:
    """Dispatch each channel to a suitable interpolation method."""
    if column in DISCRETE_CHANNELS:
        return _interpolate_stepwise(profile, column, grid)
    return _interpolate_continuous(profile, column, grid)


def _safe_value(values: np.ndarray, index: int):
    value = float(values[index])
    return value if np.isfinite(value) else None


def _axis_viewport(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    """Return tight axis ranges and a centred X-domain sized for the larger map."""
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    xspan = max(xmax - xmin, 1.0)
    yspan = max(ymax - ymin, 1.0)
    # A smaller pad makes the actual circuit occupy more of the replay canvas.
    # Coordinates remain untouched; this changes only the viewing window.
    xpad = max(xspan * TRACK_PADDING_RATIO, TRACK_PADDING_MIN_METRES)
    ypad = max(yspan * TRACK_PADDING_RATIO, TRACK_PADDING_MIN_METRES)
    x_range = [xmin - xpad, xmax + xpad]
    y_range = [ymin - ypad, ymax + ypad]

    # Match the map domain to the enlarged vertical plotting area. This makes
    # the circuit use more physical pixels without stretching its geometry.
    # The factor is tuned for the V2.4 1600px canvas and its wider Streamlit page.
    data_aspect = xspan / yspan
    domain_width = float(np.clip(
        data_aspect * TRACK_DOMAIN_FACTOR, TRACK_DOMAIN_MIN, TRACK_DOMAIN_MAX
    ))
    x_domain = [(1.0 - domain_width) / 2.0, (1.0 + domain_width) / 2.0]
    return x_range, y_range, x_domain


def build_interactive_track_replay(
    profile_a: pd.DataFrame,
    profile_b: pd.DataFrame,
    *,
    code_a: str,
    code_b: str,
    name_a: str,
    name_b: str,
    color_a: str,
    color_b: str,
    playback_speed: float = 1.0,
    frame_count: int = 120,
) -> go.Figure:
    """Build an animated two-driver GPS replay with honest spatial diagnostics.

    Both traces are interpolated only within their overlapping normalised
    lap-progress interval. Continuous channels use linear interpolation;
    gear and brake use sample-and-hold interpolation. The map shows both
    drivers' first valid samples, matched-progress connectors and a coloured
    separation trace. Euclidean separation is not a lateral offset or time gap.
    """
    if playback_speed not in (0.5, 1.0, 2.0):
        raise ValueError("Playback speed must be 0.5x, 1x or 2x.")

    a = _clean_profile(profile_a)
    b = _clean_profile(profile_b)
    lower = max(float(a["progress_pct"].min()), float(b["progress_pct"].min()))
    upper = min(float(a["progress_pct"].max()), float(b["progress_pct"].max()))
    if upper <= lower:
        raise ValueError("The two drivers do not have an overlapping lap-progress range.")

    count = max(40, min(int(frame_count), 180))
    grid = np.linspace(lower, upper, count)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for code, profile in ((code_a, a), (code_b, b)):
        arrays[code] = {
            column: _interpolate(profile, column, grid)
            for column in REPLAY_CHANNELS
        }
        if not np.isfinite(arrays[code]["x"]).all() or not np.isfinite(arrays[code]["y"]).all():
            raise ValueError(f"{code} has missing GPS coordinates inside the replay interval.")

    ax, ay = arrays[code_a]["x"], arrays[code_a]["y"]
    bx, by = arrays[code_b]["x"], arrays[code_b]["y"]
    separation = np.hypot(ax - bx, ay - by)
    speed_a = arrays[code_a]["speed_kmh"]
    speed_b = arrays[code_b]["speed_kmh"]
    speed_gap = speed_a - speed_b  # positive means Driver A is faster

    marker_symbols = {code_a: "circle", code_b: "diamond"}
    text_positions = {code_a: "top left", code_b: "bottom right"}

    def marker_trace(code: str, name: str, color: str, idx: int) -> go.Scatter:
        values = arrays[code]
        other = code_b if code == code_a else code_a
        custom = [[
            float(grid[idx]),
            _safe_value(values["speed_kmh"], idx),
            _safe_value(values["throttle_pct"], idx),
            _safe_value(values["brake"], idx),
            _safe_value(values["gear"], idx),
            _safe_value(values["rpm"], idx),
            _safe_value(arrays[other]["speed_kmh"], idx),
            _safe_value(speed_gap, idx),
            float(separation[idx]),
        ]]
        return go.Scatter(
            x=[float(values["x"][idx])],
            y=[float(values["y"][idx])],
            mode="markers+text",
            name=f"{code} live position",
            text=[code],
            textposition=text_positions[code],
            textfont=dict(color="#FFFFFF", size=12, family="Titillium Web, sans-serif"),
            marker=dict(
                symbol=marker_symbols[code], size=17, color=color,
                line=dict(color="#FFFFFF", width=2),
            ),
            customdata=custom,
            hovertemplate=(
                f"<b>{name} ({code})</b><br>Lap progress: %{{customdata[0]:.1f}}%"
                "<br>Speed: %{customdata[1]:.1f} km/h"
                "<br>Throttle: %{customdata[2]:.1f}%"
                "<br>Brake state: %{customdata[3]:.0f}"
                "<br>Gear: %{customdata[4]:.0f}"
                "<br>RPM: %{customdata[5]:,.0f}"
                "<br>Other driver speed: %{customdata[6]:.1f} km/h"
                "<br>Speed gap (A − B): %{customdata[7]:+.1f} km/h"
                "<br>GPS position separation: %{customdata[8]:.1f} m<extra></extra>"
            ),
        )

    first_a = marker_trace(code_a, name_a, color_a, 0)
    first_b = marker_trace(code_b, name_b, color_b, 0)

    # Thin connector segments at regular progress intervals make spatial
    # separation visible without shifting either driver's real GPS coordinates.
    connector_indices = np.unique(np.linspace(0, count - 1, min(24, count)).round().astype(int))
    connector_x: list[float | None] = []
    connector_y: list[float | None] = []
    for idx in connector_indices:
        connector_x.extend([float(ax[idx]), float(bx[idx]), None])
        connector_y.extend([float(ay[idx]), float(by[idx]), None])

    midpoint_x = (ax + bx) / 2.0
    midpoint_y = (ay + by) / 2.0
    midpoint_custom = np.column_stack([grid, separation, speed_gap])

    # Ten live scorecards form a 2-column x 5-row grid on the left.
    # Each column belongs to one driver; each row is one telemetry channel.
    # The circuit and lap-progress slider use the right-hand panel.
    card_centres = [0.125, 0.295]
    card_rows = [0.735, 0.615, 0.495, 0.375, 0.255]
    card_specs = (
        ("speed_kmh", "SPEED · KM/H", "↗"),
        ("throttle_pct", "THROTTLE", "▰"),
        ("brake", "BRAKE", "●"),
        ("gear", "GEAR", "⚙"),
        ("rpm", "ENGINE RPM", "⟳"),
    )

    def _scorecard_value(values: dict[str, np.ndarray], key: str, idx: int) -> tuple[str, str]:
        value = _safe_value(values.get(key, np.full(count, np.nan)), idx)
        if value is None:
            return "—", "#AAB6C8"
        if key == "speed_kmh":
            return f"{value:.0f}", "#F4F6FA"
        if key == "throttle_pct":
            return f"{value:.0f}%", "#F4F6FA"
        if key == "brake":
            active = value >= 0.5
            return ("ON" if active else "OFF"), ("#FF5369" if active else "#70D6C1")
        if key == "gear":
            return f"{value:.0f}", "#F4F6FA"
        if key == "rpm":
            return f"{value:,.0f}", "#F4F6FA"
        return f"{value:.1f}", "#F4F6FA"

    def _rgba(hex_color: str, alpha: float = 0.22) -> str:
        raw = str(hex_color).lstrip("#")
        if len(raw) != 6:
            return f"rgba(255,255,255,{alpha})"
        red, green, blue = (int(raw[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({red},{green},{blue},{alpha})"

    def scorecard_annotations(idx: int) -> list[dict[str, object]]:
        annotations: list[dict[str, object]] = [dict(
            text=f"<b>Interactive circuit replay · {code_a} vs {code_b}</b>",
            x=0.015, y=REPLAY_HEADER_Y, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=17, color="#F1F4F9"),
        )]

        drivers = (
            (code_a, arrays[code_a], color_a, "DRIVER A"),
            (code_b, arrays[code_b], color_b, "DRIVER B"),
        )

        # Driver headings sit above their respective scorecard columns.
        for col, (code, _values, driver_color, driver_label) in enumerate(drivers):
            centre_x = card_centres[col]
            annotations.append(dict(
                name=f"driver_tag_{code}",
                text=f"<b>{code}</b><br><span style='font-size:8px'>{driver_label}</span>",
                x=centre_x, y=0.855, xref="paper", yref="paper",
                xanchor="center", yanchor="middle", showarrow=False,
                align="center", width=145, height=34,
                bgcolor=_rgba(driver_color, 0.24),
                bordercolor=driver_color, borderwidth=1, borderpad=4,
                font=dict(family="Titillium Web, sans-serif", size=13, color="#FFFFFF"),
            ))

        # 2 columns (drivers) x 5 rows (telemetry metrics).
        for col, (code, values, driver_color, _driver_label) in enumerate(drivers):
            centre_x = card_centres[col]
            for row, (key, label, icon) in enumerate(card_specs):
                row_y = card_rows[row]
                value_text, value_color = _scorecard_value(values, key, idx)
                icon_color = "#FF304D"
                raw_value = _safe_value(values.get(key, np.full(count, np.nan)), idx)
                if key == "brake" and raw_value is not None and raw_value >= 0.5:
                    border_color = "#FF5369"
                else:
                    border_color = _rgba(driver_color, 0.72)

                # Keep the icon in its own top-left annotation so it never
                # collides with the live numeric value or metric label.
                value_text_block = (
                    f"<span style='font-size:21px;color:{value_color};font-weight:900'>{value_text}</span>"
                    f"<br><span style='font-size:8px;color:#9AA7B9;font-weight:800;letter-spacing:1px'>{label}</span>"
                )
                annotations.append(dict(
                    name=f"scorecard_{code}_{key}",
                    text=value_text_block,
                    x=centre_x, y=row_y - 0.014, xref="paper", yref="paper",
                    xanchor="center", yanchor="middle", showarrow=False,
                    align="center", width=145, height=58,
                    bgcolor="rgba(17,23,34,0.98)",
                    bordercolor=border_color, borderwidth=1, borderpad=4,
                    font=dict(family="Titillium Web, sans-serif", size=10, color="#F4F6FA"),
                ))
                annotations.append(dict(
                    name=f"scorecard_icon_{code}_{key}",
                    text=f"<span style='font-size:16px;color:{icon_color};font-weight:900'>{icon}</span>",
                    x=centre_x, y=row_y + 0.027,
                    xref="paper", yref="paper",
                    xanchor="center", yanchor="middle", showarrow=False,
                    align="center", width=28, height=22,
                    font=dict(family="Titillium Web, sans-serif", size=16, color=icon_color),
                ))

        status = (
            f"LAP {grid[idx]:.1f}%   ·   GPS SEPARATION {separation[idx]:.1f} m   ·   "
            f"SPEED Δ ({code_a} − {code_b}) {speed_gap[idx]:+.1f} km/h"
        )
        annotations.append(dict(
            name="live_pair_status", text=status,
            x=0.21, y=0.125, xref="paper", yref="paper",
            xanchor="center", yanchor="middle", showarrow=False,
            font=dict(family="Titillium Web, sans-serif", size=9, color="#C4CEDB"),
        ))
        return annotations

    x_range, y_range, _default_x_domain = _axis_viewport(
        np.concatenate([ax, bx]), np.concatenate([ay, by])
    )
    # Reserve roughly the left 36% of the figure for the live scorecard grid.
    # Equal X/Y scaling is retained, so the GPS track is not distorted.
    x_domain = [0.405, 0.955]

    fig = go.Figure(data=[
        # One neutral reference path, reconstructed as the midpoint between
        # the two drivers at matched normalised lap progress. This is a
        # GPS-derived visual reference, not an official surveyed circuit map.
        go.Scatter(
            x=midpoint_x, y=midpoint_y, mode="lines",
            name="GPS reference track",
            line=dict(color="#F4F6FA", width=4.2),
            hoverinfo="skip",
        ),
        # Separation is overlaid directly on the white reference path.
        # Colour represents Euclidean GPS position separation at each
        # matched-progress sample; it is not a time gap or lateral offset.
        go.Scatter(
            x=midpoint_x, y=midpoint_y, mode="markers",
            name="GPS separation along lap",
            marker=dict(
                size=6.5, color=separation, colorscale="Turbo", showscale=True,
                cmin=0.0, cmax=max(float(np.max(separation)), 1.0),
                colorbar=dict(title=dict(text="Position<br>separation (m)"), thickness=12, len=0.58),
                line=dict(color="#F4F6FA", width=0.35), opacity=0.98,
            ),
            customdata=midpoint_custom,
            hovertemplate=(
                "Lap progress: %{customdata[0]:.1f}%"
                "<br>GPS position separation: %{customdata[1]:.1f} m"
                "<br>Speed gap (A − B): %{customdata[2]:+.1f} km/h<extra></extra>"
            ),
        ),
        first_a,
        first_b,
        go.Scatter(
            x=[float(ax[0]), float(bx[0])],
            y=[float(ay[0]), float(by[0])],
            mode="markers", name="First valid GPS samples",
            marker=dict(
                symbol=["star", "star-diamond"],
                size=[13, 13], color=[color_a, color_b],
                line=dict(color="#101722", width=1.2),
            ),
            customdata=[[code_a], [code_b]],
            hovertemplate="%{customdata[0]} · first valid GPS sample in selected lap<extra></extra>",
        ),
        go.Scatter(
            x=connector_x, y=connector_y, mode="lines", name="Matched-progress position connectors",
            line=dict(color="#F4C95D", width=1.25, dash="dot"),
            opacity=0.48, hoverinfo="skip",
        ),
        go.Scatter(
            x=[float((ax[0] + bx[0]) / 2.0), float((ax[0] + bx[0]) / 2.0)],
            y=[float((ay[0] + by[0]) / 2.0), float((ay[0] + by[0]) / 2.0)],
            mode="lines", name="Current matched-progress separation",
            line=dict(color="#F4C95D", width=2.5),
            hoverinfo="skip", showlegend=False,
        ),
        go.Scatter(
            x=[float(midpoint_x[0])], y=[float(midpoint_y[0])], mode="markers",
            name="Current separation midpoint",
            marker=dict(symbol="x", size=10, color="#F4C95D", line=dict(width=2)),
            customdata=[midpoint_custom[0].tolist()],
            hovertemplate=(
                "Lap progress: %{customdata[0]:.1f}%"
                "<br>GPS position separation: %{customdata[1]:.1f} m"
                "<br>Speed gap (A − B): %{customdata[2]:+.1f} km/h<extra></extra>"
            ),
            showlegend=False,
        ),
    ])

    frames = []
    frame_names = []
    for idx in range(count):
        frame_name = f"progress_{idx:03d}"
        frame_names.append(frame_name)
        frames.append(go.Frame(
            name=frame_name,
            data=[
                marker_trace(code_a, name_a, color_a, idx),
                marker_trace(code_b, name_b, color_b, idx),
                go.Scatter(
                    x=[float(ax[idx]), float(bx[idx])],
                    y=[float(ay[idx]), float(by[idx])],
                    mode="lines", line=dict(color="#F4C95D", width=2.5),
                    hoverinfo="skip",
                ),
                go.Scatter(
                    x=[float(midpoint_x[idx])], y=[float(midpoint_y[idx])],
                    mode="markers", marker=dict(symbol="x", size=10, color="#F4C95D", line=dict(width=2)),
                    customdata=[midpoint_custom[idx].tolist()],
                    hovertemplate=(
                        "Lap progress: %{customdata[0]:.1f}%"
                        "<br>GPS position separation: %{customdata[1]:.1f} m"
                        "<br>Speed gap (A − B): %{customdata[2]:+.1f} km/h<extra></extra>"
                    ),
                ),
            ],
            traces=[2, 3, 6, 7],
            layout=go.Layout(annotations=scorecard_annotations(idx)),
        ))

    fig.frames = frames
    slider_steps = []
    step_stride = max(1, count // 20)
    step_indices = list(range(0, count, step_stride))
    if step_indices[-1] != count - 1:
        step_indices.append(count - 1)
    for idx in step_indices:
        slider_steps.append(dict(
            method="animate", label=f"{grid[idx]:.0f}%",
            args=[[frame_names[idx]], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }],
        ))

    frame_duration = max(30, int(80 / playback_speed))
    mean_sep = float(np.mean(separation))
    max_sep = float(np.max(separation))
    fig.update_layout(
        annotations=scorecard_annotations(0),
        height=REPLAY_HEIGHT,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,18,27,.75)",
        font=dict(family="Titillium Web, sans-serif", color="#E8EDF5", size=11),
        margin=dict(l=58, r=58, t=REPLAY_TOP_MARGIN, b=28),
        hovermode="closest",
        dragmode="pan",
        uirevision=f"{code_a}_{code_b}_track_replay_v2_4_6",
        # Legend remains above the circuit, while the left panel is reserved
        # for the two-driver telemetry scorecard grid.
        legend=dict(
            orientation="h", y=0.92, x=0.405, xanchor="left", yanchor="bottom",
            bgcolor="rgba(0,0,0,0)", font=dict(size=8),
            tracegroupgap=5,
        ),
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.985, y=REPLAY_HEADER_Y, xanchor="right", yanchor="top",
            buttons=[
                dict(
                    label="▶ Play", method="animate",
                    args=[None, {
                        "fromcurrent": True, "mode": "immediate",
                        "frame": {"duration": frame_duration, "redraw": True},
                        "transition": {"duration": 0},
                    }],
                ),
                dict(
                    label="❚❚ Pause", method="animate",
                    args=[[None], {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    }],
                ),
            ],
        )],
        sliders=[dict(
            active=0, x=0.425, y=0.085, len=0.51,
            currentvalue=dict(prefix="Lap progress: ", font=dict(color="#E8EDF5", size=11)),
            pad=dict(t=16, b=0), steps=slider_steps,
            tickcolor="#9AA7B9", font=dict(color="#DCE3ED", size=9),
        )],
    )
    fig.update_xaxes(
        title="GPS X coordinate (m)", range=x_range, domain=x_domain,
        gridcolor="#303849", zeroline=False, constrain="domain",
    )
    fig.update_yaxes(
        title="GPS Y coordinate (m)", range=y_range, domain=[0.17, 0.88],
        gridcolor="#303849", scaleanchor="x", scaleratio=1,
        zeroline=False, constrain="domain",
    )
    return fig
